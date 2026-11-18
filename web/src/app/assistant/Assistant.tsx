"use client";

import { useState } from "react";
import { api, ApiError, type ResearchAnswer } from "@/lib/api";
import { ExplanationView } from "@/components/analysis/ResultView";
import { ErrorState, Spinner } from "@/components/ui";

const SUGGESTIONS = [
  "What accuracy did the original IEEE paper report for EfficientNetB3, and on how many test images?",
  "How does train/test leakage inflate accuracy in medical image classification?",
  "What are the limitations of Grad-CAM saliency maps for explaining MRI classifiers?",
  "How is temperature scaling used to calibrate deep network confidence?",
  "How well do brain tumour models generalise to data from other hospitals?",
];

type Turn = { question: string; answer?: ResearchAnswer; error?: string };

export function Assistant() {
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState<"" | "pmc_article" | "project_document">("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);

  async function ask(q: string) {
    const text = q.trim();
    if (text.length < 3 || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((t) => [{ question: text }, ...t]);
    try {
      const answer = await api.ask(text, scope || undefined);
      setTurns((t) => [{ question: text, answer }, ...t.slice(1)]);
    } catch (e) {
      setTurns((t) => [{ question: text, error: e instanceof ApiError ? e.message : "The question could not be answered." }, ...t.slice(1)]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6">
      <form
        onSubmit={(e) => { e.preventDefault(); ask(question); }}
        className="rounded-xl border border-line bg-raised p-4"
      >
        <label htmlFor="q" className="text-sm font-medium text-ink">Ask a research question</label>
        <div className="mt-2 flex flex-col gap-2 sm:flex-row">
          <input id="q" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={1000}
                 placeholder="e.g. Why can near-duplicate images inflate test accuracy?"
                 className="flex-1 rounded-lg border border-line bg-surface px-3 py-2" />
          <select aria-label="Sources to search" value={scope} onChange={(e) => setScope(e.target.value as typeof scope)}
                  className="rounded-lg border border-line bg-surface px-2 py-2 text-sm">
            <option value="">All sources (auto-routed)</option>
            <option value="pmc_article">Open-access papers only</option>
            <option value="project_document">NeuroClass documents only</option>
          </select>
          <button disabled={busy || question.trim().length < 3}
                  className="rounded-lg bg-accent px-4 py-2 font-medium text-accent-ink disabled:opacity-50">
            Ask
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" onClick={() => ask(s)} disabled={busy}
                    className="rounded-full border border-line px-3 py-1 text-left text-xs text-ink-2 hover:bg-sunken">
              {s}
            </button>
          ))}
        </div>
      </form>

      <div aria-live="polite" className="grid gap-6">
        {turns.map((t, i) => (
          <article key={turns.length - i} className="grid gap-3">
            <p className="font-medium text-ink">Q: {t.question}</p>
            {!t.answer && !t.error && <Spinner label="Searching the literature…" />}
            {t.error && <ErrorState error={t.error} />}
            {t.answer && (
              <>
                <ExplanationView explanation={t.answer.answer} title="Answer" />
                <p className="text-xs text-muted">
                  Retrieval {Math.round(t.answer.retrieval_ms)} ms · generation {Math.round(t.answer.generation_ms)} ms.
                  Answers come only from the indexed open-access papers and project documents; they are research
                  context, not medical advice.
                </p>
              </>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
