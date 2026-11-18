"use client";

import { useEffect, useMemo, useState } from "react";
import { api, DISPLAY, type Analysis, type Passage, type Statement } from "@/lib/api";
import { buttonClass, Card, LevelBadge, Notice, pct, Pill } from "@/components/ui";

const KIND_LABEL: Record<string, string> = {
  model_output: "Model output",
  limitation: "Limitation",
  literature: "Research context",
};

export function ResultView({ analysis, localImage }: { analysis: Analysis; localImage?: string | null }) {
  const inf = analysis.inference;
  const [gradcam, setGradcam] = useState<string | null>(null);

  useEffect(() => {
    if (!inf) return;
    let url: string | null = null;
    api.blob(`/analyses/${analysis.id}/gradcam.png`)
      .then((b) => setGradcam((url = URL.createObjectURL(b))))
      .catch(() => setGradcam(null));
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [analysis.id, inf]);

  const original = analysis.sample_id ? api.sampleImageUrl(analysis.sample_id) : localImage ?? null;

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-6">
      {inf && (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <Card title="Image and Grad-CAM">
            <div className="grid grid-cols-2 gap-3">
              <figure>
                {original ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img src={original} alt="Input MRI slice" className="aspect-square w-full rounded-lg border border-line bg-black object-contain" />
                ) : (
                  <div className="grid aspect-square place-items-center rounded-lg border border-line bg-sunken text-sm text-muted">
                    Upload not retained
                  </div>
                )}
                <figcaption className="mt-1 text-xs text-muted">Input as received</figcaption>
              </figure>
              <figure>
                {gradcam ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img src={gradcam} alt={`Grad-CAM heatmap for the predicted class ${DISPLAY[inf.predicted_class]}`} className="aspect-square w-full rounded-lg border border-line object-contain" />
                ) : (
                  <div className="aspect-square animate-pulse rounded-lg bg-sunken" />
                )}
                <figcaption className="mt-1 text-xs text-muted">Grad-CAM (256 × 256 model input)</figcaption>
              </figure>
            </div>
            <p className="mt-3 text-sm text-ink-2">
              Red areas increased the score for the predicted class most. This is not a tumor outline and does not
              show that those regions caused the prediction, or that it is correct.
            </p>
          </Card>

          <Card title="Model output" aside={<Pill>{inf.model_version}</Pill>}>
            <p className="text-sm text-ink-2">Model-predicted class</p>
            <p className="text-3xl font-semibold text-ink">{DISPLAY[inf.predicted_class]}</p>
            <p className="mt-1 text-ink-2">
              <span className="text-2xl font-semibold text-ink">{pct(inf.calibrated_confidence)}</span>{" "}
              calibrated confidence <span className="text-muted">(uncalibrated {pct(inf.raw_confidence)})</span>
            </p>
            <ProbabilityTable probabilities={inf.probabilities} predicted={inf.predicted_class} />
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-muted">Stability under small changes</dt>
                <dd className="text-ink">{inf.tta_agreement} of {inf.tta_views} views agree</dd>
              </div>
              <div>
                <dt className="text-muted">Input-shift heuristic</dt>
                <dd className="text-ink">
                  {inf.ood_flag ? "Flagged" : "Not flagged"}{" "}
                  <span className="text-muted">({inf.ood_distance.toFixed(0)} / {inf.ood_threshold.toFixed(0)})</span>
                </dd>
              </div>
              <div>
                <dt className="text-muted">Input</dt>
                <dd className="text-ink">{inf.input_quality.width}×{inf.input_quality.height} {inf.input_quality.format ?? ""}</dd>
              </div>
              <div>
                <dt className="text-muted">CPU latency</dt>
                <dd className="text-ink">{Math.round(inf.latency_ms.total ?? 0)} ms</dd>
              </div>
            </dl>
          </Card>
        </div>
      )}

      {analysis.reliability && (
        <Card title="Reliability assessment" aside={<LevelBadge level={analysis.reliability.level} />}>
          <p className="mb-3 text-sm text-ink-2">
            Deterministic rules (version {analysis.reliability.rules_version}) over this result and the model&apos;s
            measured evaluation. On the test set, results at the standard level were right far more often than
            results flagged for caution — see Benchmarks.
          </p>
          <ul className="grid gap-2">
            {analysis.reliability.findings.map((f) => (
              <li key={f.code} className="flex gap-3 text-sm">
                <span className="w-16 shrink-0 font-medium text-ink-2">{f.severity === "info" ? "Note" : "Caution"}</span>
                <span className="text-ink">{f.message}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {analysis.explanation && <ExplanationView explanation={analysis.explanation} />}
    </div>
  );
}

function ProbabilityTable({ probabilities, predicted }: {
  probabilities: Analysis["inference"] extends infer T ? T extends { probabilities: infer P } ? P : never : never;
  predicted: string;
}) {
  const rows = [...probabilities].sort((a, b) => b.calibrated - a.calibrated);
  return (
    <table className="mt-4 w-full text-sm">
      <caption className="sr-only">Calibrated class probabilities</caption>
      <thead>
        <tr className="text-left text-muted">
          <th className="pb-1 font-normal">Class</th>
          <th className="pb-1 font-normal">Calibrated probability</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((p) => (
          <tr key={p.label} title={`${DISPLAY[p.label]}: ${pct(p.calibrated)} calibrated, ${pct(p.raw)} uncalibrated`}>
            <td className={`py-1 pr-3 ${p.label === predicted ? "font-medium text-ink" : "text-ink-2"}`}>{DISPLAY[p.label]}</td>
            <td className="w-3/5 py-1">
              <div className="flex items-center gap-2">
                <div className="h-2.5 flex-1 rounded-r bg-sunken">
                  <div className="h-2.5 rounded-r bg-series-1" style={{ width: `${Math.max(p.calibrated * 100, 0.5)}%` }} />
                </div>
                <span className="w-14 text-right text-ink-2 tabular-nums">{pct(p.calibrated)}</span>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ExplanationView({ explanation, title = "Research interpretation" }: {
  explanation: Analysis["explanation"] & {}; title?: string;
}) {
  const passages = useMemo(() => new Map(explanation.passages.map((p) => [p.chunk_id, p])), [explanation.passages]);
  const [open, setOpen] = useState<string | null>(null);
  const shown = open ? passages.get(open) : null;
  return (
    <Card
      title={title}
      aside={<Pill tone={explanation.mode === "llm" ? "accent" : "neutral"}>
        {explanation.mode === "llm" ? `Generated · ${explanation.llm_model}` : "Extractive · no language model"}
      </Pill>}
    >
      <p className="text-ink">{explanation.summary}</p>
      {explanation.mode === "extractive" && (
        <p className="mt-2 text-sm text-muted">
          {explanation.fallback_reason ? `Language model not used (${explanation.fallback_reason}). ` : ""}
          Research context is quoted from the retrieved passages, ranked by relevance. Every statement below was
          checked against its sources.
        </p>
      )}
      <ul className="mt-4 grid gap-3">
        {explanation.statements.map((s, i) => (
          <StatementItem key={i} s={s} onCite={setOpen} active={open} />
        ))}
      </ul>
      {shown && <PassagePreview passage={shown} onClose={() => setOpen(null)} />}

      {explanation.removed_statements.length > 0 && (
        <details className="mt-4 rounded-lg border border-line px-4 py-3 text-sm">
          <summary className="cursor-pointer font-medium text-ink">
            {explanation.removed_statements.length} statement(s) removed by verification
          </summary>
          <ul className="mt-2 grid gap-2">
            {explanation.removed_statements.map((s, i) => (
              <li key={i}>
                <span className="text-ink-2 line-through">{s.text}</span>
                <span className="block text-xs text-muted">{s.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {explanation.passages.length > 0 && (
        <div className="mt-5">
          <h3 className="mb-2 text-sm font-semibold text-ink">Sources</h3>
          <ol className="grid gap-1.5 text-sm">
            {explanation.passages.map((p) => (
              <li key={p.chunk_id} className="text-ink-2">
                <button className="mr-1 font-mono text-xs text-accent underline-offset-2 hover:underline" onClick={() => setOpen(p.chunk_id)}>
                  [{p.chunk_id}]
                </button>
                <a href={p.source_url} target="_blank" rel="noreferrer">{p.title}</a>
                {p.year ? ` (${p.year})` : ""} — <span className="text-muted">{p.license}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </Card>
  );
}

function StatementItem({ s, onCite, active }: { s: Statement; onCite: (id: string) => void; active: string | null }) {
  return (
    <li className="flex gap-3 text-sm">
      <span className="w-28 shrink-0 text-xs font-medium uppercase tracking-wide text-muted">{KIND_LABEL[s.kind]}</span>
      <span className="text-ink">
        {s.text}{" "}
        {s.citations?.map((c) => (
          <button
            key={c}
            onClick={() => onCite(c)}
            aria-pressed={active === c}
            className="ml-0.5 rounded bg-sunken px-1 font-mono text-xs text-accent hover:underline"
            aria-label={`Show source passage ${c}`}
          >
            {c}
          </button>
        ))}
      </span>
    </li>
  );
}

function PassagePreview({ passage, onClose }: { passage: Passage; onClose: () => void }) {
  return (
    <aside aria-label="Source passage" className="mt-4 rounded-lg border border-line bg-sunken p-4 text-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-ink">{passage.title}</p>
          <p className="text-xs text-muted">
            {passage.section} · {passage.journal ?? (passage.source_type === "project_document" ? "Project document" : "")}
            {passage.year ? ` · ${passage.year}` : ""} · {passage.license}
          </p>
        </div>
        <button onClick={onClose} className="rounded px-2 text-ink-2 hover:bg-raised" aria-label="Close source passage">×</button>
      </div>
      <blockquote className="mt-2 border-l-2 border-line-strong pl-3 leading-relaxed text-ink-2">{passage.text}</blockquote>
      <a className="mt-2 inline-block" href={passage.source_url} target="_blank" rel="noreferrer">Open source</a>
    </aside>
  );
}

export function TraceView({ trace }: { trace: Analysis["trace"] }) {
  if (!trace.length) return null;
  const rows = trace.filter((t) => t.status !== "started");
  return (
    <Card title="Execution trace">
      <p className="mb-3 text-sm text-ink-2">
        Each step of the LangGraph workflow, as recorded by the server. No image data or text content is logged.
      </p>
      <div className="overflow-x-auto">
      <table className="w-full min-w-[28rem] text-sm">
        <thead>
          <tr className="text-left text-muted">
            <th className="pb-1 font-normal">Step</th>
            <th className="pb-1 font-normal">Status</th>
            <th className="pb-1 text-right font-normal">Time</th>
            <th className="pb-1 pl-4 font-normal">Detail</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => (
            <tr key={i} className="border-t border-line align-top">
              <td className="py-1.5 font-mono text-xs text-ink">{t.step}</td>
              <td className="py-1.5 text-ink-2">{t.status}</td>
              <td className="py-1.5 text-right text-ink-2 tabular-nums">{t.duration_ms == null ? "" : `${Math.round(t.duration_ms)} ms`}</td>
              <td className="max-w-md truncate py-1.5 pl-4 font-mono text-xs text-muted" title={JSON.stringify(t.detail)}>
                {Object.keys(t.detail ?? {}).length ? JSON.stringify(t.detail) : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </Card>
  );
}

export function ReportLinks({ id }: { id: string }) {
  const [error, setError] = useState<string | null>(null);
  async function download(format: "md" | "html") {
    setError(null);
    try {
      const blob = await api.blob(`/analyses/${id}/report?format=${format}`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neuroclass-report-${id.slice(0, 8)}.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("The report could not be downloaded.");
    }
  }
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button onClick={() => download("html")} className={buttonClass("primary", "sm")}>
        Download report (HTML)
      </button>
      <button onClick={() => download("md")} className={buttonClass("secondary", "sm")}>
        Markdown
      </button>
      {error && <span role="alert" className="text-sm text-critical-ink">{error}</span>}
    </div>
  );
}

export function ReviewStatus({ analysis }: { analysis: Analysis }) {
  if (analysis.status === "awaiting_review") {
    return (
      <Notice tone="warning">
        The reliability rules flagged this result, so the report is waiting for a research reviewer. A review records
        a researcher&apos;s assessment of the report; it is not clinical validation.
      </Notice>
    );
  }
  if (!analysis.reviews.length) return null;
  return (
    <Notice>
      {analysis.reviews.map((r, i) => (
        <span key={i} className="block">
          Reviewed by {r.reviewer}: <strong>{r.decision.replace("_", " ")}</strong>{r.note ? ` — ${r.note}` : ""}
        </span>
      ))}
      <span className="block text-ink-2">A research review, not clinical validation.</span>
    </Notice>
  );
}
