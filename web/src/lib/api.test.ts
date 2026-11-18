import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, clientId, streamEvents } from "./api";

function sse(chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      chunks.forEach((x) => c.enqueue(enc.encode(x)));
      c.close();
    },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("api client", () => {
  it("creates one stable random client id", () => {
    const a = clientId();
    expect(a).toMatch(/^[0-9a-f]{40}$/);
    expect(clientId()).toBe(a);
  });

  it("sends identity headers and surfaces API error detail", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "analysis not found" }), {
      status: 404, headers: { "content-type": "application/json", "X-Request-ID": "r1" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(api.analysis("x")).rejects.toMatchObject({ status: 404, message: "analysis not found", requestId: "r1" });
    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("X-Client-Id")).toBe(clientId());
  });

  it("reports an unreachable API distinctly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));
    const err = await api.ready().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
  });

  it("parses server-sent events split across network chunks", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(sse([
      'id: 1\nevent: step\ndata: {"step":"validate","status":"completed","duration_ms":3,"detail":{}}\n\n',
      'id: 2\nevent: st', 'ep\ndata: {"step":"infer","status":"started","duration_ms":null,"detail":{}}\n\n',
      ': keep-alive\n\n',
      'event: status\ndata: {"status":"completed"}\n\n',
    ]))));
    const steps: string[] = [];
    const final = await streamEvents("a", (e) => steps.push(`${e.step}:${e.status}`), new AbortController().signal);
    expect(steps).toEqual(["validate:completed", "infer:started"]);
    expect(final).toBe("completed");
  });
});
