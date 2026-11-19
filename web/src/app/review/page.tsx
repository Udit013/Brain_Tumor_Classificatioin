"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError, DISPLAY, useSession, type Analysis, type AnalysisSummary } from "@/lib/api";
import { ResultView } from "@/components/analysis/ResultView";
import { buttonClass, Card, Empty, ErrorState, LevelBadge, PageHeader, pct, Spinner } from "@/components/ui";

const HOW = [
  { title: "A signed-in user runs an analysis",
    body: "Anonymous analyses are never reviewed. Signed-in users can also ask for review of every result." },
  { title: "The rules flag it",
    body: "If the reliability level isn't standard (low confidence, unstable views, input shift), the workflow pauses before the report." },
  { title: "A reviewer decides",
    body: "Approve, reject, or request a revision with a note; a revision re-runs the explanation step (at most twice)." },
  { title: "The workflow resumes",
    body: "The decision and note go into the report the user sees. The pause survives restarts — it's saved in the database." },
];

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
      <PageHeader title="Review queue">
        Reports the reliability rules flagged wait here. Approve, reject or request a revision — a revision request
        sends your note back to the explanation step (at most twice).
      </PageHeader>
      <p className="-mt-4 mb-8 max-w-2xl text-[13px] leading-relaxed text-muted">
        A review records a researcher&apos;s assessment of an explanation report. It is not clinical validation and
        never turns a model output into a diagnosis.
      </p>
      <section aria-labelledby="how" className="mb-10">
        <h2 id="how" className="text-[15px] font-semibold tracking-[-0.01em] text-ink">How human review works</h2>
        <ol className="mt-4 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {HOW.map((h, i) => (
            <li key={h.title} className="bg-raised p-4">
              <span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
              <p className="mt-2 text-sm font-medium text-ink">{h.title}</p>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{h.body}</p>
            </li>
          ))}
        </ol>
      </section>
      <div>
        {!allowed && (
          <Empty title="Reviewer access required">
            Reviewers are invited by the operator: their email must be on the reviewer list and they enter the
            operator&apos;s reviewer code when creating an account.
          </Empty>
        )}
        {error && <ErrorState error={error} />}
        {allowed && !queue && !error && <Spinner label="Loading queue…" />}
        {queue && queue.length === 0 && <Empty title="Nothing to review">Flagged reports will appear here.</Empty>}
        {queue && queue.length > 0 && !selected && (
          <ul className="divide-y divide-line overflow-hidden rounded-xl border border-line bg-raised">
            {queue.map((q) => (
              <li key={q.id}>
                <button onClick={() => open(q.id)} className="flex w-full flex-wrap items-center justify-between gap-3 px-4 py-3.5 text-left hover:bg-sunken/60">
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
                <button disabled={busy} onClick={() => decide("approve")} className={buttonClass("primary", "md")}>Approve report</button>
                <button disabled={busy} onClick={() => decide("request_revision")} className={buttonClass("secondary", "md")}>Request revision</button>
                <button disabled={busy} onClick={() => decide("reject")} className={buttonClass("secondary", "md", "text-critical-ink hover:text-critical-ink")}>Reject report</button>
                <button onClick={() => setSelected(null)} className={buttonClass("ghost", "md", "ml-auto")}>Back to queue</button>
              </div>
            </Card>
          </div>
        )}
      </div>
    </>
  );
}
