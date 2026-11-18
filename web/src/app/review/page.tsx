"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError, DISPLAY, useSession, type Analysis, type AnalysisSummary } from "@/lib/api";
import { ResultView } from "@/components/analysis/ResultView";
import { Card, Empty, ErrorState, LevelBadge, Notice, PageHeader, Spinner, pct } from "@/components/ui";

export default function ReviewPage() {
  const [queue, setQueue] = useState<AnalysisSummary[] | null>(null);
  const [selected, setSelected] = useState<Analysis | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const allowed = useSession()?.role === "reviewer";

  const load = useCallback(() => {
    api.reviewQueue().then(setQueue).catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (allowed) load();
  }, [allowed, load]);

  async function open(id: string) {
    setNote("");
    setSelected(await api.analysis(id));
  }

  async function decide(decision: "approve" | "reject" | "request_revision") {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await api.review(selected.id, decision, note);
      setSelected(null);
      load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The review could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader eyebrow="Human review" title="Review queue">
        Reports the reliability rules flagged wait here. Approve, reject or request a revision — a revision request
        sends your note back to the explanation step (at most twice).
      </PageHeader>
      <Notice tone="warning">
        A review records a researcher&apos;s assessment of an explanation report. It is not clinical validation and
        never turns a model output into a diagnosis.
      </Notice>
      <div className="mt-6">
        {!allowed && <Empty title="Reviewer access required">Sign in with an account the operator has granted the reviewer role.</Empty>}
        {error && <ErrorState error={error} />}
        {allowed && !queue && !error && <Spinner label="Loading queue…" />}
        {queue && queue.length === 0 && <Empty title="Nothing to review">Flagged reports will appear here.</Empty>}
        {queue && queue.length > 0 && !selected && (
          <ul className="grid gap-2">
            {queue.map((q) => (
              <li key={q.id}>
                <button onClick={() => open(q.id)} className="flex w-full flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-raised px-4 py-3 text-left hover:border-line-strong">
                  <span className="text-ink">
                    {q.predicted_class ? DISPLAY[q.predicted_class as keyof typeof DISPLAY] : "—"} · {pct(q.calibrated_confidence)}
                    <span className="ml-2 text-sm text-muted">{new Date(q.created_at).toLocaleString()}</span>
                  </span>
                  <LevelBadge level={q.reliability_level} />
                </button>
              </li>
            ))}
          </ul>
        )}
        {selected && (
          <div className="grid gap-6">
            <ResultView analysis={selected} />
            <Card title="Your decision">
              <label htmlFor="note" className="text-sm text-ink-2">Note (included in the report; for a revision, tell the explainer what to fix)</label>
              <textarea id="note" value={note} onChange={(e) => setNote(e.target.value)} rows={3} maxLength={4000}
                        className="mt-2 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm" />
              <div className="mt-3 flex flex-wrap gap-2">
                <button disabled={busy} onClick={() => decide("approve")} className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-accent-ink disabled:opacity-50">Approve report</button>
                <button disabled={busy} onClick={() => decide("request_revision")} className="rounded-md border border-line-strong px-3 py-2 text-sm hover:bg-sunken disabled:opacity-50">Request revision</button>
                <button disabled={busy} onClick={() => decide("reject")} className="rounded-md border border-critical/50 px-3 py-2 text-sm text-critical-ink hover:bg-critical/10 disabled:opacity-50">Reject report</button>
                <button onClick={() => setSelected(null)} className="ml-auto text-sm text-ink-2">Back to queue</button>
              </div>
            </Card>
          </div>
        )}
      </div>
    </>
  );
}
