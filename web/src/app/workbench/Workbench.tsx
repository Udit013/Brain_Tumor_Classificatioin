"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError, DISPLAY, useSession, type ClassName } from "@/lib/api";
import { AnalysisRunner } from "@/components/analysis/AnalysisRunner";
import { buttonClass, ErrorState, Spinner } from "@/components/ui";

const MAX_BYTES = 10 * 1024 * 1024;
type Sample = { id: string; label: ClassName; source_file: string; license: string };

export function Workbench() {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [mode, setMode] = useState<"sample" | "upload">("sample");
  const [sampleId, setSampleId] = useState<string>("glioma");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [requestReview, setRequestReview] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<{ id: string; image: string | null } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLElement>(null);
  const signedIn = Boolean(useSession());
  const [reviewEnabled, setReviewEnabled] = useState(false);

  useEffect(() => {
    api.samples().then(setSamples).catch((e) => setSamplesError(e instanceof ApiError ? e.message : String(e)));
    api.ready().then((r) => setReviewEnabled(r.checks.review?.enabled === true)).catch(() => undefined);
  }, []);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function choose(f: File | undefined | null) {
    setError(null);
    if (!f) return;
    if (!["image/jpeg", "image/png"].includes(f.type)) return setError("Upload a JPEG or PNG image.");
    if (f.size > MAX_BYTES) return setError("The file is larger than 10 MB.");
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function run() {
    setError(null);
    setSubmitting(true);
    try {
      const a = await api.createAnalysis(
        { sampleId: mode === "sample" ? sampleId : undefined, file: mode === "upload" ? file ?? undefined : undefined,
          question: question.trim() || undefined, reviewMode: requestReview ? "always" : "auto" },
        crypto.randomUUID(),
      );
      setCurrent({ id: a.id, image: mode === "upload" ? preview : null });
      // On narrow screens the result sits below the setup panel: bring it into view.
      if (window.matchMedia("(max-width: 1023px)").matches) {
        requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The analysis could not be started.");
    } finally {
      setSubmitting(false);
    }
  }

  const canRun = mode === "sample" ? Boolean(sampleId) : Boolean(file);

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-8 lg:grid-cols-[22rem_minmax(0,1fr)] lg:items-start">
      <aside aria-label="Analysis setup" className="lg:sticky lg:top-20">
        <div className="overflow-hidden rounded-xl border border-line bg-raised">
          <div className="border-b border-line p-4">
            <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">Image</h2>
            <div role="tablist" aria-label="Input source" className="mt-3 grid grid-cols-2 rounded-lg bg-sunken p-0.5">
              {(["sample", "upload"] as const).map((m) => (
                <button
                  key={m}
                  role="tab"
                  aria-selected={mode === m}
                  onClick={() => setMode(m)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${mode === m ? "bg-raised text-ink shadow-card" : "text-ink-2 hover:text-ink"}`}
                >
                  {m === "sample" ? "Example images" : "Upload your own"}
                </button>
              ))}
            </div>

            {mode === "sample" && (
              <div className="mt-4">
                {samplesError && <ErrorState error={samplesError} />}
                {!samples && !samplesError && <Spinner label="Loading examples…" />}
                {samples && (
                  <fieldset>
                    <legend className="sr-only">Example image</legend>
                    <div className="grid grid-cols-2 gap-2">
                      {samples.map((s) => (
                        <label key={s.id} className={`group cursor-pointer rounded-lg border p-1.5 transition-colors ${sampleId === s.id ? "border-accent bg-accent-soft/40 ring-1 ring-accent" : "border-line hover:border-line-strong"}`}>
                          <input type="radio" name="sample" value={s.id} checked={sampleId === s.id}
                                 onChange={() => setSampleId(s.id)} className="sr-only" />
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={api.sampleImageUrl(s.id)} alt={`Example MRI labelled ${DISPLAY[s.label]}`} className="aspect-square w-full rounded-md bg-black object-contain" />
                          <span className="mt-1.5 flex items-baseline justify-between gap-2 px-0.5">
                            <span className="text-[13px] font-medium text-ink">Labelled: {DISPLAY[s.label]}</span>
                          </span>
                          <span className="block px-0.5 font-mono text-[11px] text-muted">{s.source_file}</span>
                        </label>
                      ))}
                    </div>
                    <p className="mt-3 text-xs leading-relaxed text-muted">
                      First file of each class in the test split, not picked for being classified correctly.
                      Masoud Nickparvar, Brain Tumor MRI Dataset, CC BY 4.0.
                    </p>
                  </fieldset>
                )}
              </div>
            )}

            {mode === "upload" && (
              <div className="mt-4 grid gap-3">
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files?.[0]); }}
                  className="grid place-items-center rounded-lg border border-dashed border-line-strong bg-sunken/50 px-4 py-8 text-center"
                >
                  {preview ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img src={preview} alt="Selected upload" className="mb-3 h-36 w-36 rounded-md bg-black object-contain" />
                  ) : (
                    <p className="text-sm text-ink">Drop a JPEG or PNG here</p>
                  )}
                  <button onClick={() => inputRef.current?.click()} className={buttonClass("secondary", "sm", "mt-2")}>
                    {preview ? "Choose another file" : "Choose a file"}
                  </button>
                  <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="sr-only"
                         aria-label="MRI image file" onChange={(e) => choose(e.target.files?.[0])} />
                </div>
                <ul className="grid gap-1 text-xs leading-relaxed text-muted">
                  <li>One 2-D brain MRI slice, up to 10 MB, at least 32 × 32 px.</li>
                  <li>The model only knows glioma, meningioma, pituitary tumor and no tumor; any other image still gets one of those labels.</li>
                  <li>Remove anything that identifies a person. Uploads are private and deleted after 24 hours.</li>
                </ul>
              </div>
            )}
          </div>

          <div className="border-b border-line p-4">
            <label htmlFor="question" className="flex items-baseline justify-between text-[15px] font-semibold tracking-[-0.01em] text-ink">
              Research question <span className="text-xs font-normal text-muted">Optional</span>
            </label>
            <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={1000}
                      rows={3} placeholder="How reliable is Grad-CAM for brain tumour MRI?"
                      className="mt-2 w-full resize-none rounded-lg border border-line-strong bg-surface px-3 py-2 text-sm placeholder:text-muted" />
            <p className="mt-1 text-xs text-muted">Added to the literature search for the explanation.</p>
            {signedIn && reviewEnabled && (
              <label className="mt-3 flex items-start gap-2 text-sm text-ink-2">
                <input type="checkbox" className="mt-0.5" checked={requestReview} onChange={(e) => setRequestReview(e.target.checked)} />
                Always send to a research reviewer (otherwise only when the reliability rules require it)
              </label>
            )}
          </div>

          <div className="p-4">
            <button onClick={run} disabled={!canRun || submitting} className={buttonClass("primary", "md", "w-full")}>
              {submitting ? "Starting…" : "Run analysis"}
            </button>
            {error && <p role="alert" className="mt-2 text-sm text-critical-ink">{error}</p>}
            <p className="mt-3 text-xs leading-relaxed text-muted">
              Results are an experimental model&apos;s output for research and teaching, not a diagnosis.
            </p>
          </div>
        </div>
      </aside>

      <section ref={resultRef} aria-label="Analysis result" className="min-w-0">
        {current ? (
          <AnalysisRunner key={current.id} id={current.id} localImage={current.image} />
        ) : (
          <div className="hidden min-h-[28rem] place-items-center rounded-xl border border-dashed border-line-strong px-6 py-16 text-center lg:grid">
            <div className="max-w-sm">
              <p className="font-medium text-ink">No analysis yet</p>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">
                Pick an example or upload a slice, then run it. You&apos;ll see the predicted class with a calibrated
                confidence, the Grad-CAM heatmap, a reliability level and a cited explanation, step by step.
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
