"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, ApiError, setSession } from "@/lib/api";
import Image from "next/image";
import { buttonClass } from "@/components/ui";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [reviewerCode, setReviewerCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = mode === "login" ? await api.login(email, password) : await api.signup(email, password, reviewerCode.trim() || undefined);
      setSession(res.access_token, email.toLowerCase(), res.role);
      router.push(res.role === "reviewer" ? "/review" : "/history");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto w-full max-w-sm py-6 sm:py-10">
      <div className="text-center">
        <Image src="/brand-mark.png" alt="" width={44} height={44} className="mx-auto rounded-[11px]" priority />
        <h1 className="mt-5 text-2xl font-semibold tracking-[-0.02em] text-ink">
          {mode === "login" ? "Sign in to NeuroClass" : "Create your account"}
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-ink-2">
          Keep your analyses across devices and receive reviewed reports.
        </p>
      </div>
      <div className="mt-8 rounded-xl border border-line bg-raised p-6 shadow-card">
        <form onSubmit={submit} className="grid gap-4">
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium text-ink">Email</span>
            <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)}
                   className="h-10 rounded-lg border border-line-strong bg-surface px-3 text-ink placeholder:text-muted focus:border-accent" />
          </label>
          <label className="grid gap-1.5 text-sm">
            <span className="flex items-baseline justify-between font-medium text-ink">Password <span className="text-xs font-normal text-muted">10+ characters</span></span>
            <input type="password" required minLength={10} autoComplete={mode === "login" ? "current-password" : "new-password"}
                   value={password} onChange={(e) => setPassword(e.target.value)}
                   className="h-10 rounded-lg border border-line-strong bg-surface px-3 text-ink placeholder:text-muted focus:border-accent" />
          </label>
          {mode === "signup" && (
            <details className="text-sm">
              <summary className="cursor-pointer text-ink-2 hover:text-ink">I have a reviewer access code</summary>
              <label className="mt-3 grid gap-1.5">
                <span className="text-ink-2">Reviewer access code</span>
                <input type="password" autoComplete="off" value={reviewerCode} onChange={(e) => setReviewerCode(e.target.value)}
                       className="h-10 rounded-lg border border-line-strong bg-surface px-3 text-ink placeholder:text-muted focus:border-accent" />
                <span className="text-xs text-muted">
                  Given by the operator. It only takes effect for an email the operator has listed.
                </span>
              </label>
            </details>
          )}
          {error && <p role="alert" className="text-sm text-critical-ink">{error}</p>}
          <button disabled={busy} className={buttonClass("primary", "md", "mt-1 w-full")}>
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
          <button type="button" onClick={() => setMode(mode === "login" ? "signup" : "login")}
                  className={buttonClass("ghost", "sm", "text-accent hover:text-accent-hover")}>
            {mode === "login" ? "New here? Create an account" : "Have an account? Sign in"}
          </button>
        </form>
      </div>
      <p className="mt-6 text-center text-xs leading-relaxed text-muted">
        Reviewer access is granted by the operator, not chosen at sign-up.
      </p>
    </div>
  );
}
