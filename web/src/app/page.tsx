import Image from "next/image";
import Link from "next/link";
import { ButtonLink } from "@/components/ui";

/*
 * Every figure on this page is measured (results/ and docs/EVALUATION.md); the
 * preview reproduces a real analysis of the meningioma example image
 * (Kaggle test split, CC BY 4.0) as returned by the API.
 */

const PREVIEW = {
  file: "Te-me_1.jpg",
  predicted: "Meningioma",
  confidence: "96.5%",
  probs: [
    { label: "Meningioma", value: 96.5 },
    { label: "Pituitary tumor", value: 1.8 },
    { label: "Glioma", value: 1.1 },
    { label: "No tumor", value: 0.6 },
  ],
  statement:
    "Paul et al. suggested a deep learning system for the purpose of classifying brain images containing several types of tumours, including meningioma, glioma, and pituitary.",
  citation: "PMC9933351",
  source: "Contemporary Oncology, 2022 · CC BY-NC-SA",
};

const METRICS = [
  { value: "93.4%", label: "accuracy on the full 1,600-image test set" },
  { value: "89.4%", label: "on test images with no near-copy in training" },
  { value: "41 ms", label: "per image with Grad-CAM, on CPU" },
  { value: "241", label: "openly licensed papers behind every citation" },
];

const STAGES = [
  {
    name: "Check",
    steps: ["Validate the upload", "Classify with 6 augmented views"],
    by: "Rules and the model",
  },
  {
    name: "Judge",
    steps: ["Score reliability against measured weaknesses"],
    by: "Deterministic rules",
  },
  {
    name: "Ground",
    steps: ["Retrieve literature", "Explain", "Verify every claim and number"],
    by: "Search, Gemini or quotes, rules",
  },
  {
    name: "Decide",
    steps: ["Pause for a reviewer when flagged", "Write the report"],
    by: "A person, then rules",
  },
];

const EXPLORE = [
  { href: "/workbench", title: "Workbench", body: "Analyse an example or your own slice." },
  { href: "/assistant", title: "Research assistant", body: "Ask the literature; every claim is cited." },
  { href: "/benchmarks", title: "Benchmarks", body: "The paper's table beside the audit and retraining." },
  { href: "/system", title: "System", body: "Live status, model checksums and data licences." },
];

function ProductPreview() {
  return (
    <figure className="relative">
      <div aria-hidden className="absolute -inset-x-6 -inset-y-8 -z-10 rounded-[28px] bg-[radial-gradient(60%_60%_at_70%_30%,var(--accent-soft),transparent)]" />
      <div className="overflow-hidden rounded-xl border border-line bg-raised shadow-raised">
        <div className="flex items-center justify-between border-b border-line px-4 py-2.5 text-xs">
          <span className="flex items-center gap-2 font-medium text-ink">
            <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-good" />
            Analysis · <span className="font-mono text-ink-2">{PREVIEW.file}</span>
          </span>
          <span className="text-muted">Completed</span>
        </div>
        <div className="grid gap-5 p-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] sm:p-5">
          <div className="grid grid-cols-2 gap-2">
            {[
              { src: "/showcase/meningioma-input.webp", label: "Input" },
              { src: "/showcase/meningioma-gradcam.webp", label: "Grad-CAM" },
            ].map((im) => (
              <div key={im.label}>
                <Image src={im.src} alt={`${im.label} for the meningioma example`} width={256} height={256}
                       className="aspect-square w-full rounded-md bg-black object-cover" priority />
                <p className="mt-1.5 text-[11px] text-muted">{im.label}</p>
              </div>
            ))}
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted">Model-predicted class</p>
            <p className="mt-1 flex items-baseline gap-2">
              <span className="text-xl font-semibold tracking-tight text-ink">{PREVIEW.predicted}</span>
              <span className="tabular text-sm text-ink-2">{PREVIEW.confidence} calibrated</span>
            </p>
            <ul className="mt-3 grid gap-1.5">
              {PREVIEW.probs.map((p) => (
                <li key={p.label} className="grid grid-cols-[6.5rem_minmax(0,1fr)_2.5rem] items-center gap-2 text-xs">
                  <span className="truncate text-ink-2">{p.label}</span>
                  <span className="h-1.5 rounded-full bg-sunken">
                    <span className="block h-1.5 rounded-full bg-series-1" style={{ width: `${Math.max(p.value, 1.5)}%` }} />
                  </span>
                  <span className="tabular text-right text-muted">{p.value.toFixed(1)}%</span>
                </li>
              ))}
            </ul>
            <p className="mt-4 inline-flex items-center gap-1.5 rounded-md border border-good/30 bg-good/10 px-2 py-0.5 text-xs font-medium text-good-ink">
              <span aria-hidden>●</span> Standard reliability
            </p>
            <p className="mt-1.5 text-xs text-muted">6 of 6 views agree · input not flagged</p>
          </div>
        </div>
        <div className="border-t border-line bg-sunken/60 px-4 py-3 sm:px-5">
          <p className="text-xs leading-relaxed text-ink-2">
            {PREVIEW.statement}
            <span className="ml-1.5 inline-flex rounded border border-line bg-raised px-1.5 py-px align-middle font-mono text-[10px] text-ink-2">
              {PREVIEW.citation}
            </span>
          </p>
          <p className="mt-1 text-[11px] text-muted">Verified against its source · {PREVIEW.source}</p>
        </div>
      </div>
      <figcaption className="mt-3 text-xs text-muted">
        Real output for an example from the dataset&apos;s test split (CC BY 4.0); one of its cited statements shown.
      </figcaption>
    </figure>
  );
}

export default function Home() {
  return (
    <div className="grid gap-28">
      <section className="grid items-center gap-14 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:gap-16">
        <div>
          <p className="text-sm text-ink-2">
            Research platform ·{" "}
            <a href="https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941" className="text-ink-2 underline decoration-line-strong underline-offset-4 hover:text-ink">
              built on an IEEE ICC-ROBINS 2024 study
            </a>
          </p>
          <h1 className="mt-5 text-[2.25rem] font-semibold leading-[1.06] tracking-[-0.03em] text-ink sm:text-[3.4rem]">
            A brain MRI classifier that shows its work.
          </h1>
          <p className="mt-5 max-w-[34rem] text-[17px] leading-relaxed text-ink-2">
            Upload a slice and see the predicted class with a calibrated confidence, where the model looked, how far
            to trust the result, and what the literature says — with every citation checked before it reaches you.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <ButtonLink href="/workbench" size="lg">Analyse an example</ButtonLink>
            <ButtonLink href="/benchmarks" size="lg" variant="ghost">
              Read the evaluation <span aria-hidden>→</span>
            </ButtonLink>
          </div>
          <p className="mt-10 max-w-[34rem] border-l-2 border-line-strong pl-3 text-[13px] leading-relaxed text-muted">
            For research and education only. Outputs are an experimental model&apos;s predictions, not diagnoses,
            with no clinical validation or regulatory clearance.
          </p>
        </div>
        <ProductPreview />
      </section>

      <section aria-label="Measured results" className="border-y border-line">
        <dl className="grid grid-cols-2 lg:grid-cols-4">
          {METRICS.map((m, i) => (
            <div key={m.label} className={`py-7 pr-6 ${i % 2 ? "pl-6" : "lg:pl-0"} ${i > 0 ? "lg:border-l lg:border-line lg:pl-6" : ""} ${i > 1 ? "border-t border-line lg:border-t-0" : ""} ${i === 1 ? "border-l border-line" : ""} ${i === 3 ? "border-l border-line" : ""}`}>
              <dd className="tabular text-[1.9rem] font-semibold tracking-[-0.02em] text-ink">{m.value}</dd>
              <dt className="mt-1 text-sm leading-snug text-ink-2">{m.label}</dt>
            </div>
          ))}
        </dl>
      </section>

      <section aria-labelledby="claim" className="grid gap-10 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-16">
        <div>
          <h2 id="claim" className="text-[1.75rem] font-semibold leading-tight tracking-[-0.02em] text-ink">
            The headline number, and what it actually measured
          </h2>
          <p className="mt-4 leading-relaxed text-ink-2">
            The 2024 paper reported 99.844% for EfficientNetB3. Re-running it showed that figure came from 640 of the
            1,311 test images, and that 44.6% of test images have a near-duplicate in training. The model was retrained
            on a leakage-safe split and measured on CPU with the exact preprocessing the platform serves.
          </p>
          <Link href="/benchmarks" className="mt-5 inline-flex items-center gap-1 text-sm font-medium no-underline">
            Methods and per-class results <span aria-hidden>→</span>
          </Link>
        </div>
        <div className="overflow-hidden rounded-xl border border-line bg-raised">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Reported versus measured accuracy</caption>
            <thead>
              <tr className="border-b border-line text-xs text-muted">
                <th scope="col" className="px-5 py-3 font-medium">Figure</th>
                <th scope="col" className="px-5 py-3 font-medium">Source</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              <tr>
                <td className="px-5 py-3.5 text-ink">EfficientNetB3, 640 test images</td>
                <td className="px-5 py-3.5 text-muted">Reported by the paper</td>
                <td className="tabular px-5 py-3.5 text-right text-ink-2">99.844%</td>
              </tr>
              {[
                ["Retrained model, full test set (1,600)", "93.4%"],
                ["Test images with no near-copy in training (886)", "89.4%"],
                ["Separate public MRI dataset (253)", "72.7%"],
              ].map(([fig, v]) => (
                <tr key={fig}>
                  <td className="px-5 py-3.5 text-ink">{fig}</td>
                  <td className="px-5 py-3.5 text-muted">Measured</td>
                  <td className="tabular px-5 py-3.5 text-right font-medium text-ink">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="workflow">
        <div className="max-w-2xl">
          <h2 id="workflow" className="text-[1.75rem] font-semibold leading-tight tracking-[-0.02em] text-ink">
            One analysis, four stages, every step traced
          </h2>
          <p className="mt-4 leading-relaxed text-ink-2">
            Each analysis is a stateful LangGraph run. The judging and checking are plain code rather than a language
            model, and a flagged result waits for a person before it is reported.
          </p>
        </div>
        <ol className="mt-10 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {STAGES.map((s, i) => (
            <li key={s.name} className="flex flex-col bg-raised p-5">
              <span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
              <h3 className="mt-3 font-semibold text-ink">{s.name}</h3>
              <ul className="mt-2 grid flex-1 gap-1 text-sm text-ink-2">
                {s.steps.map((t) => <li key={t}>{t}</li>)}
              </ul>
              <p className="mt-4 text-xs text-muted">{s.by}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="explore" className="grid gap-8 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-16">
        <div>
          <h2 id="explore" className="text-[1.75rem] font-semibold leading-tight tracking-[-0.02em] text-ink">Explore</h2>
          <p className="mt-4 leading-relaxed text-ink-2">
            The language-model, retrieval and agent features are new engineering; they were not part of the published
            study.
          </p>
        </div>
        <ul className="divide-y divide-line border-y border-line">
          {EXPLORE.map((e) => (
            <li key={e.href}>
              <Link href={e.href} className="group flex items-center justify-between gap-6 py-4 text-ink no-underline hover:text-ink">
                <span>
                  <span className="font-medium">{e.title}</span>
                  <span className="mt-0.5 block text-sm text-ink-2">{e.body}</span>
                </span>
                <span aria-hidden className="text-muted transition-transform group-hover:translate-x-0.5 group-hover:text-ink">→</span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
