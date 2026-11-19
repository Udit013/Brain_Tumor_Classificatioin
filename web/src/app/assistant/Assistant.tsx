"use client";

import { useState } from "react";
import { api, ApiError, type ResearchAnswer } from "@/lib/api";
import { ExplanationView } from "@/components/analysis/ResultView";
import { buttonClass, ErrorState, Spinner } from "@/components/ui";

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
    <div className="grid gap-8">
      <form onSubmit={(e) => { e.preventDefault(); ask(question); }}
            className="rounded-xl border border-line bg-raised shadow-card focus-within:border-line-strong">
        <label htmlFor="q" className="sr-only">Ask a research question</label>
        <textarea id="q" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={1000} rows={2}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(question); } }}
                  placeholder="Ask a research question, e.g. why can near-duplicate images inflate test accuracy?"
                  className="block w-full resize-none rounded-t-xl bg-transparent px-4 pt-4 text-[15px] text-ink placeholder:text-muted focus:outline-none" />
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-3 pt-2">
          <select aria-label="Sources to search" value={scope} onChange={(e) => setScope(e.target.value as typeof scope)}
                  className="h-8 rounded-md border border-line bg-surface px-2 text-sm text-ink-2">
            <option value="">All sources, auto-routed</option>
            <option value="pmc_article">Open-access papers only</option>
            <option value="project_document">NeuroClass documents only</option>
          </select>
          <button disabled={busy || question.trim().length < 3} className={buttonClass("primary", "sm")}>
            Ask
          </button>
        </div>
      </form>

      {turns.length === 0 && (
        <section aria-labelledby="try">
          <h2 id="try" className="text-xs font-medium text-muted">Try asking</h2>
          <ul className="mt-2 divide-y divide-line border-y border-line">
            {SUGGESTIONS.map((q) => (
              <li key={q}>
                <button type="button" onClick={() => ask(q)} disabled={busy}
                        className="group flex w-full items-center justify-between gap-4 py-3 text-left text-sm text-ink-2 hover:text-ink">
                  {q}
                  <span aria-hidden className="text-muted transition-transform group-hover:translate-x-0.5">→</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div aria-live="polite" className="grid gap-10">
        {turns.map((t, i) => (
          <article key={turns.length - i} className="grid gap-4">
            <p className="text-lg font-medium leading-snug tracking-[-0.01em] text-ink">{t.question}</p>
            {!t.answer && !t.error && <Spinner label="Searching the literature…" />}
            {t.error && <ErrorState error={t.error} />}
            {t.answer && (
              <>
                <ExplanationView explanation={t.answer.answer} title="Answer" />
                <p className="text-xs text-muted">
                  Retrieval {Math.round(t.answer.retrieval_ms)} ms · generation {Math.round(t.answer.generation_ms)} ms ·
                  research context from indexed open-access papers and project documents, not medical advice.
                </p>
              </>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
