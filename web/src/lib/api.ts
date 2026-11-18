/**
 * Typed client for the NeuroClass API. Types are generated from the FastAPI
 * OpenAPI schema (npm run gen:api), so a contract change breaks the build.
 *
 * Identity: a random client id (anonymous ownership) and an optional bearer
 * token, both kept in localStorage and read only inside calls — never during
 * render, so prerendered HTML does not depend on them.
 */
import { useSyncExternalStore } from "react";
import type { components } from "./api-types";

export type Schemas = components["schemas"];
export type Analysis = Schemas["AnalysisOut"];
export type AnalysisSummary = Schemas["AnalysisSummary"];
export type Inference = Schemas["InferenceResult"];
export type Reliability = Schemas["ReliabilityAssessment"];
export type Explanation = Schemas["Explanation"];
export type Passage = Schemas["Passage"];
export type Statement = Schemas["VerifiedStatement"];
export type ResearchAnswer = Schemas["ResearchAnswer"];
export type ModelInfo = Schemas["ModelInfo"];
export type ClassName = Inference["predicted_class"];

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

const CLIENT_KEY = "neuroclass.clientId";
const TOKEN_KEY = "neuroclass.token";
const EMAIL_KEY = "neuroclass.email";
const ROLE_KEY = "neuroclass.role";

export const DISPLAY: Record<ClassName, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No tumor",
  pituitary: "Pituitary tumor",
};

export function clientId(): string {
  let id = localStorage.getItem(CLIENT_KEY);
  if (!id) {
    id = crypto.randomUUID().replace(/-/g, "") + crypto.randomUUID().replace(/-/g, "").slice(0, 8);
    localStorage.setItem(CLIENT_KEY, id);
  }
  return id;
}

export type Session = { email: string; role: string } | null;

export function getSession(): Session {
  const email = localStorage.getItem(EMAIL_KEY);
  const role = localStorage.getItem(ROLE_KEY);
  return localStorage.getItem(TOKEN_KEY) && email && role ? { email, role } : null;
}

export function setSession(token: string, email: string, role: string) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(EMAIL_KEY, email);
  localStorage.setItem(ROLE_KEY, role);
  window.dispatchEvent(new Event("neuroclass:session"));
}

export function clearSession() {
  [TOKEN_KEY, EMAIL_KEY, ROLE_KEY].forEach((k) => localStorage.removeItem(k));
  window.dispatchEvent(new Event("neuroclass:session"));
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public requestId?: string | null) {
    super(message);
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  h.set("X-Client-Id", clientId());
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) h.set("Authorization", `Bearer ${token}`);
  return h;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers: headers(init.headers) });
  } catch {
    throw new ApiError(0, "The NeuroClass API is unreachable. It may be starting up or offline.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    if (res.status === 401) clearSession();
    throw new ApiError(res.status, detail, res.headers.get("X-Request-ID"));
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

export const api = {
  ready: () => request<{ status: string; checks: Record<string, Record<string, unknown>> }>("/ready"),
  samples: () =>
    request<{ id: string; label: ClassName; source_file: string; license: string; split: string }[]>(
      "/samples",
    ),
  sampleImageUrl: (id: string) => `${API_URL}/samples/${id}/image`,
  model: () => request<ModelInfo>("/models/current"),
  manifest: () => request<Record<string, unknown>>("/models/manifest"),
  benchmarks: () => request<Benchmarks>("/metrics/benchmarks"),

  createAnalysis: (input: { sampleId?: string; file?: File; question?: string; reviewMode?: string },
                   idempotencyKey: string) => {
    const form = new FormData();
    if (input.sampleId) form.set("sample_id", input.sampleId);
    if (input.file) form.set("file", input.file);
    if (input.question) form.set("question", input.question);
    if (input.reviewMode) form.set("review_mode", input.reviewMode);
    return request<Analysis>("/analyses", {
      method: "POST",
      body: form,
      headers: { "Idempotency-Key": idempotencyKey },
    });
  },
  analysis: (id: string) => request<Analysis>(`/analyses/${id}`),
  history: () => request<AnalysisSummary[]>("/analyses?limit=100"),
  reviewQueue: () => request<AnalysisSummary[]>("/analyses/review-queue"),
  review: (id: string, decision: string, note: string) =>
    request<Analysis>(`/analyses/${id}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, note }),
    }),
  /** Grad-CAM and reports need the identity headers, so they are fetched as blobs. */
  blob: async (path: string) => {
    const res = await fetch(`${API_URL}${path}`, { headers: headers() });
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.blob();
  },
  ask: (question: string, sourceType?: string) =>
    request<ResearchAnswer>("/research/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 6, filters: sourceType ? { source_type: sourceType } : null }),
    }),
  signup: (email: string, password: string, reviewerCode?: string) =>
    request<{ access_token: string; role: string }>("/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, ...(reviewerCode ? { reviewer_code: reviewerCode } : {}) }),
    }),
  login: (email: string, password: string) =>
    request<{ access_token: string; role: string }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),
};

/** Server-sent progress events. The analysis endpoints need identity headers,
 * which EventSource cannot send, so this reads the stream with fetch. */
export async function streamEvents(
  id: string,
  onStep: (e: { step: string; status: string; duration_ms: number | null; detail: Record<string, unknown> }) => void,
  signal: AbortSignal,
): Promise<string | null> {
  const res = await fetch(`${API_URL}/analyses/${id}/events`, { headers: headers(), signal });
  if (!res.ok || !res.body) return null;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: string | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const event = /^event: (.*)$/m.exec(chunk)?.[1];
      const data = /^data: (.*)$/m.exec(chunk)?.[1];
      if (!event || !data) continue;
      const parsed = JSON.parse(data);
      if (event === "step") onStep(parsed);
      if (event === "status") final = parsed.status;
    }
  }
  return final;
}

// ---- benchmark payload (shape of the measured JSON files) -------------------
export type Benchmarks = {
  paper: {
    citation: string;
    doi: string;
    dataset: string;
    status: string;
    models: { model: string; params_m: number; reported_accuracy: number; images_scored: number }[];
  };
  measured: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
};

// ---- session as an external store (no setState-in-effect, no hydration mismatch)
function subscribeSession(cb: () => void) {
  window.addEventListener("neuroclass:session", cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener("neuroclass:session", cb);
    window.removeEventListener("storage", cb);
  };
}
const sessionKey = () => {
  const s = getSession();
  return s ? `${s.role}|${s.email}` : "";
};

/** Current session; the server snapshot is "signed out". */
export function useSession(): Session {
  const key = useSyncExternalStore(subscribeSession, sessionKey, () => "");
  if (!key) return null;
  const [role, ...email] = key.split("|");
  return { role, email: email.join("|") };
}
