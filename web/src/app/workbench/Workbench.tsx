"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError, DISPLAY, useSession, type ClassName } from "@/lib/api";
import { AnalysisRunner } from "@/components/analysis/AnalysisRunner";
import { buttonClass, Card, ErrorState, Notice, Spinner } from "@/components/ui";

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
  const signedIn = Boolean(useSession());

  useEffect(() => {
    api.samples().then(setSamples).catch((e) => setSamplesError(e instanceof ApiError ? e.message : String(e)));
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
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The analysis could not be started.");
    } finally {
      setSubmitting(false);
    }
  }

  const canRun = mode === "sample" ? Boolean(sampleId) : Boolean(file);

  return (
    <div className="grid gap-8">
      <Card title="Image">
        <div role="tablist" aria-label="Input source" className="mb-4 inline-flex rounded-lg border border-line p-1">
          {(["sample", "upload"] as const).map((m) => (
            <button
              key={m}
              role="tab"
              aria-selected={mode === m}
              onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1.5 text-sm ${mode === m ? "bg-ink text-surface" : "text-ink-2 hover:bg-sunken"}`}
            >
              {m === "sample" ? "Example images" : "Upload your own"}
            </button>
          ))}
        </div>

        {mode === "sample" && (
          <>
            {samplesError && <ErrorState error={samplesError} />}
            {!samples && !samplesError && <Spinner label="Loading examples…" />}
            {samples && (
              <fieldset>
                <legend className="mb-3 text-sm text-ink-2">
                  One image per class from the dataset&apos;s test split, picked as the first file of each class — not
                  for being classified correctly. Masoud Nickparvar, Brain Tumor MRI Dataset, CC BY 4.0.
                </legend>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {samples.map((s) => (
                    <label key={s.id} className={`cursor-pointer rounded-xl border p-2 ${sampleId === s.id ? "border-accent ring-2 ring-accent/30" : "border-line hover:border-line-strong"}`}>
                      <input type="radio" name="sample" value={s.id} checked={sampleId === s.id}
                             onChange={() => setSampleId(s.id)} className="sr-only" />
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={api.sampleImageUrl(s.id)} alt={`Example MRI labelled ${DISPLAY[s.label]}`} className="aspect-square w-full rounded-lg bg-black object-contain" />
                      <span className="mt-2 block text-sm font-medium text-ink">Labelled: {DISPLAY[s.label]}</span>
                      <span className="block font-mono text-xs text-muted">{s.source_file}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            )}
          </>
        )}

        {mode === "upload" && (
          <div className="grid gap-4 sm:grid-cols-[1fr_auto]">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files?.[0]); }}
              className="grid place-items-center rounded-xl border-2 border-dashed border-line-strong px-6 py-10 text-center"
            >
              <p className="text-ink">Drag a JPEG or PNG here, or</p>
              <button onClick={() => inputRef.current?.click()} className={buttonClass("secondary", "sm", "mt-3")}>
                Choose a file
              </button>
              <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="sr-only"
                     aria-label="MRI image file" onChange={(e) => choose(e.target.files?.[0])} />
              <ul className="mt-4 text-left text-xs leading-relaxed text-ink-2">
                <li>• One 2-D brain MRI slice, JPEG or PNG, up to 10 MB, at least 32 × 32 px.</li>
                <li>• The model knows only glioma, meningioma, pituitary tumor and no tumor; other images still get one of those labels.</li>
                <li>• Remove anything that identifies a person. Uploads are private to you and deleted after 24 hours.</li>
              </ul>
            </div>
            {preview && (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img src={preview} alt="Selected upload" className="h-48 w-48 rounded-xl border border-line bg-black object-contain" />
            )}
          </div>
        )}
      </Card>

      <Card title="Research question" aside={<span className="text-xs text-muted">Optional</span>}>
        <label htmlFor="question" className="text-sm text-ink-2">
          Added to the literature search, e.g. “How reliable is Grad-CAM for brain tumour MRI?”
        </label>
        <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={1000}
                  rows={2} className="mt-2 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm" />
        {signedIn && (
          <label className="mt-3 flex items-center gap-2 text-sm text-ink-2">
            <input type="checkbox" checked={requestReview} onChange={(e) => setRequestReview(e.target.checked)} />
            Always send the report to a research reviewer (otherwise only when the reliability rules require it)
          </label>
        )}
      </Card>

      <div className="flex flex-wrap items-center gap-4">
        <button onClick={run} disabled={!canRun || submitting}
                className={buttonClass("primary", "lg")}>
          {submitting ? "Starting…" : "Run analysis"}
        </button>
        {error && <span role="alert" className="text-sm text-critical-ink">{error}</span>}
      </div>

      {current && (
        <section aria-label="Analysis result" className="grid gap-4">
          <Notice>
            Results are an experimental model&apos;s output for research and teaching. They are not a diagnosis.
          </Notice>
          <AnalysisRunner key={current.id} id={current.id} localImage={current.image} />
        </section>
      )}
    </div>
  );
}
