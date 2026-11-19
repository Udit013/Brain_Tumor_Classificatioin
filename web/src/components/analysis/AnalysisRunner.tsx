"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError, streamEvents, type Analysis } from "@/lib/api";
import { buttonClass, ErrorState, Spinner } from "@/components/ui";
import { ReportLinks, ResultView, ReviewStatus, TraceView } from "./ResultView";

const STEPS = [
  { id: "validate", label: "Validate" },
  { id: "infer", label: "Classify" },
  { id: "assess", label: "Reliability" },
  { id: "retrieve", label: "Retrieve" },
  { id: "explain", label: "Explain" },
  { id: "verify", label: "Verify" },
  { id: "human_review", label: "Review" },
  { id: "report", label: "Report" },
];
const TERMINAL = new Set(["completed", "failed", "awaiting_review"]);

type StepState = Record<string, "started" | "completed" | "failed" | "interrupted" | undefined>;

export function AnalysisRunner({ id, localImage }: { id: string; localImage?: string | null }) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [steps, setSteps] = useState<StepState>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    let cancelled = false;

    async function load() {
      try {
        const a = await api.analysis(id);
        if (cancelled) return;
        setAnalysis(a);
        const fromTrace: StepState = {};
        a.trace.forEach((t) => { fromTrace[t.step] = t.status as StepState[string]; });
        setSteps(fromTrace);
        if (TERMINAL.has(a.status)) return;
        try {
          await streamEvents(id, (e) => setSteps((s) => ({ ...s, [e.step]: e.status as StepState[string] })), ctrl.signal);
        } catch {
          /* stream dropped: fall through to polling */
        }
        for (let i = 0; i < 240 && !cancelled; i++) {
          const latest = await api.analysis(id);
          if (cancelled) return;
          setAnalysis(latest);
          if (TERMINAL.has(latest.status)) return;
          await new Promise((r) => setTimeout(r, 1000));
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Could not load this analysis.");
      }
    }
    load();
    return () => { cancelled = true; ctrl.abort(); };
  }, [id]);

  if (error) return <ErrorState error={error} />;
  if (!analysis) return <Spinner label="Loading analysis…" />;

  const running = !TERMINAL.has(analysis.status);
  const done = STEPS.filter((s) => steps[s.id] === "completed").length;
  const total = analysis.inference?.latency_ms?.total;
  const statusText = analysis.status === "awaiting_review" ? "Awaiting review"
    : analysis.status === "completed" ? "Completed"
    : analysis.status === "failed" ? "Failed"
    : analysis.status === "queued" ? "Queued" : "Running";
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      <div className="rounded-xl border border-line bg-raised">
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
          <p className="flex items-center gap-2 text-sm">
            <span aria-hidden className={`h-2 w-2 rounded-full ${
              analysis.status === "completed" ? "bg-good"
                : analysis.status === "failed" ? "bg-critical"
                : analysis.status === "awaiting_review" ? "bg-warning" : "animate-pulse bg-accent"}`} />
            <span className="font-medium text-ink">{statusText}</span>
            <span className="text-muted">
              · {done} of {STEPS.length} steps{total ? ` · model ${Math.round(total)} ms` : ""}
            </span>
          </p>
          {(analysis.status === "completed" || analysis.status === "awaiting_review") && (
            <div className="flex flex-wrap items-center gap-2">
              <ReportLinks id={analysis.id} />
              <Link href={`/analyses/${analysis.id}`} className={buttonClass("ghost", "sm")}>Permalink</Link>
            </div>
          )}
        </div>
        <ol aria-label="Workflow progress" aria-live="polite"
            className="grid grid-cols-4 gap-px border-t border-line bg-line sm:grid-cols-8">
          {STEPS.map((s) => {
            const st = steps[s.id];
            const skipped = !running && !st;
            const bar = st === "completed" ? "bg-good"
              : st === "failed" ? "bg-critical"
              : st === "started" || st === "interrupted" ? "animate-pulse bg-accent"
              : "bg-line-strong/60";
            return (
              <li key={s.id} className="bg-raised px-2.5 pb-2.5 pt-2">
                <span aria-hidden className={`block h-1 rounded-full ${bar} ${skipped ? "opacity-40" : ""}`} />
                <span className={`mt-1.5 block truncate text-[11px] ${st ? "text-ink-2" : "text-muted"}`}>{s.label}</span>
                <span className="sr-only"> — {st ?? (skipped ? "not needed" : "pending")}</span>
              </li>
            );
          })}
        </ol>
      </div>

      {analysis.status === "failed" && (
        <ErrorState error={analysis.error ?? "The analysis failed."} />
      )}
      {running && !analysis.inference && <Spinner label={analysis.status === "queued" ? "Queued…" : "Analysing…"} />}

      <ReviewStatus analysis={analysis} />
      {(analysis.inference || analysis.explanation) && <ResultView analysis={analysis} localImage={localImage} />}
      <TraceView trace={analysis.trace} />
    </div>
  );
}
