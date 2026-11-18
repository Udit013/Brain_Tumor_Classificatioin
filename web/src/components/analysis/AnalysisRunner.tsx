"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError, streamEvents, type Analysis } from "@/lib/api";
import { ErrorState, Spinner } from "@/components/ui";
import { ReportLinks, ResultView, ReviewStatus, TraceView } from "./ResultView";

const STEPS = [
  { id: "validate", label: "Validate input" },
  { id: "infer", label: "Run classifier" },
  { id: "assess", label: "Assess reliability" },
  { id: "retrieve", label: "Retrieve literature" },
  { id: "explain", label: "Explain" },
  { id: "verify", label: "Verify claims" },
  { id: "human_review", label: "Human review" },
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
  return (
    <div className="grid gap-6">
      <ol aria-label="Workflow progress" aria-live="polite" className="flex flex-wrap gap-2">
        {STEPS.map((s) => {
          const st = steps[s.id];
          const skipped = !running && !st;
          const cls = st === "completed" ? "border-good/40 text-good-ink"
            : st === "failed" ? "border-critical/40 text-critical-ink"
            : st === "started" || st === "interrupted" ? "border-accent text-accent"
            : "border-line text-muted";
          const mark = st === "completed" ? "✓" : st === "failed" ? "✕" : st === "started" ? "…" : st === "interrupted" ? "⏸" : skipped ? "–" : "○";
          return (
            <li key={s.id} className={`rounded-full border px-3 py-1 text-xs font-medium ${cls}`}>
              <span aria-hidden className="mr-1">{mark}</span>
              {s.label}
              <span className="sr-only"> — {st ?? (skipped ? "not needed" : "pending")}</span>
            </li>
          );
        })}
      </ol>

      {analysis.status === "failed" && (
        <ErrorState error={analysis.error ?? "The analysis failed."} />
      )}
      {running && <Spinner label={analysis.status === "queued" ? "Queued…" : "Analysing…"} />}

      <ReviewStatus analysis={analysis} />
      {(analysis.inference || analysis.explanation) && <ResultView analysis={analysis} localImage={localImage} />}

      {(analysis.status === "completed" || analysis.status === "awaiting_review") && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-raised p-4">
          <ReportLinks id={analysis.id} />
          <Link href={`/analyses/${analysis.id}`} className="text-sm">Permanent link to this analysis</Link>
        </div>
      )}
      <TraceView trace={analysis.trace} />
    </div>
  );
}
