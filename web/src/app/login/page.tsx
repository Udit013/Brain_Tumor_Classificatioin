"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, ApiError, setSession } from "@/lib/api";
import { Card, PageHeader } from "@/components/ui";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = mode === "login" ? await api.login(email, password) : await api.signup(email, password);
      setSession(res.access_token, email.toLowerCase(), res.role);
      router.push(res.role === "reviewer" ? "/review" : "/history");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader eyebrow="Account" title={mode === "login" ? "Sign in" : "Create an account"}>
        An account keeps your analyses across devices and lets reviewers return flagged reports to you. Reviewer
        access is granted by the operator, not chosen at sign-up.
      </PageHeader>
      <Card className="max-w-md">
        <form onSubmit={submit} className="grid gap-4">
          <label className="grid gap-1 text-sm">
            <span className="text-ink-2">Email</span>
            <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)}
                   className="rounded-lg border border-line bg-surface px-3 py-2" />
          </label>
          <label className="grid gap-1 text-sm">
            <span className="text-ink-2">Password (at least 10 characters)</span>
            <input type="password" required minLength={10} autoComplete={mode === "login" ? "current-password" : "new-password"}
                   value={password} onChange={(e) => setPassword(e.target.value)}
                   className="rounded-lg border border-line bg-surface px-3 py-2" />
          </label>
          {error && <p role="alert" className="text-sm text-critical-ink">{error}</p>}
          <button disabled={busy} className="rounded-lg bg-accent px-4 py-2 font-medium text-accent-ink disabled:opacity-50">
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
          <button type="button" onClick={() => setMode(mode === "login" ? "signup" : "login")} className="text-sm text-accent">
            {mode === "login" ? "New here? Create an account" : "Have an account? Sign in"}
          </button>
        </form>
      </Card>
    </>
  );
}
