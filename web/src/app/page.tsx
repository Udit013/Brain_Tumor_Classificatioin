import Link from "next/link";

const TIMELINE = [
  {
    when: "2024 · published",
    title: "IEEE ICC-ROBINS paper",
    body: "Four CNNs (custom CNN, VGG16, InceptionV3, EfficientNetB3) for four-class brain MRI classification on 7,023 images. Reported top accuracy: 99.844% for EfficientNetB3.",
    tag: "Reported by the paper",
  },
  {
    when: "2026 · audit",
    title: "What that number measured",
    body: "The 99.844% is 639 of 640 images — half the test set. 44.6% of test images have a near-duplicate in training, and the paper's recipe scored 85.7% on the novel ones.",
    tag: "Measured by this project",
  },
  {
    when: "2026 · retrained",
    title: "A model trained to fix it",
    body: "Leakage-safe validation, two-stage fine-tuning: 93.44% test accuracy, 89.4% on novel images, glioma recall 64% → 80%, 90.7% under image corruptions — and still 72.7% on an external MRI set.",
    tag: "Measured, single seed",
  },
  {
    when: "Now · platform",
    title: "A research workbench around it",
    body: "Explainability, deterministic reliability checks, evidence-grounded literature answers, a traced agent workflow and human review — with every number traceable.",
    tag: "New engineering",
  },
];

const FEATURES = [
  { href: "/workbench", title: "MRI workbench", body: "Run the model on an example or your own image: calibrated confidence, Grad-CAM, stability, input checks, and a reproducible report." },
  { href: "/assistant", title: "Research assistant", body: "Questions answered from openly licensed papers and the project's own reports. Every statement cites a passage; unsupported ones are removed." },
  { href: "/benchmarks", title: "Benchmarks", body: "The paper's reported table next to the audit, the retraining comparison, per-class errors, calibration and robustness." },
  { href: "/system", title: "Transparency", body: "Live subsystem status, model checksums and export verification, data licences, privacy and intended use." },
];

const STEPS = ["Validate", "Classify", "Assess reliability", "Retrieve", "Explain", "Verify claims", "Human review", "Report"];

export default function Home() {
  return (
    <div className="grid gap-16">
      <section className="grid gap-6 pt-4">
        <p className="text-sm font-medium uppercase tracking-wide text-muted">Research & explainability platform</p>
        <h1 className="max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-ink sm:text-5xl">
          A published brain MRI classifier, audited, retrained and made inspectable.
        </h1>
        <p className="max-w-3xl text-lg leading-relaxed text-ink-2">
          NeuroClass starts from a 2024 IEEE study and shows the whole path from a headline accuracy to a model whose
          strengths and failure modes are measured — then wraps it in tools for studying its predictions with evidence.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link href="/workbench" className="rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-ink no-underline hover:opacity-90">
            Open the workbench
          </Link>
          <Link href="/benchmarks" className="rounded-lg border border-line-strong px-5 py-2.5 font-medium text-ink no-underline hover:bg-sunken">
            See the evidence
          </Link>
        </div>
        <p className="max-w-3xl rounded-md border-l-4 border-l-warning bg-warning/10 px-4 py-3 text-sm text-ink">
          For research and education only. Outputs are an experimental model&apos;s predictions, not diagnoses; the
          model has no clinical validation or regulatory clearance.
        </p>
      </section>

      <section aria-labelledby="story" className="grid gap-6">
        <h2 id="story" className="text-2xl font-semibold">From paper to platform</h2>
        <ol className="grid gap-4 md:grid-cols-4">
          {TIMELINE.map((t) => (
            <li key={t.title} className="flex flex-col rounded-xl border border-line bg-raised p-5">
              <span className="text-xs font-medium uppercase tracking-wide text-muted">{t.when}</span>
              <span className="mt-2 font-semibold text-ink">{t.title}</span>
              <span className="mt-2 flex-1 text-sm leading-relaxed text-ink-2">{t.body}</span>
              <span className="mt-4 self-start rounded-full border border-line-strong px-2 py-0.5 text-xs text-ink-2">{t.tag}</span>
            </li>
          ))}
        </ol>
        <p className="text-sm text-ink-2">
          Measured figures come from the project&apos;s evaluation files and are shown with their methods on the{" "}
          <Link href="/benchmarks">benchmarks page</Link>. The language-model and retrieval features are new
          engineering; they were not part of the published study.
        </p>
      </section>

      <section aria-labelledby="workflow" className="grid gap-4">
        <h2 id="workflow" className="text-2xl font-semibold">One analysis, one traced workflow</h2>
        <p className="max-w-3xl text-ink-2">
          Each analysis is a stateful LangGraph run. Reliability and claim verification are deterministic code, not a
          language model; flagged results pause for a human reviewer and resume from a checkpoint.
        </p>
        <ol className="flex flex-wrap items-center gap-2 text-sm">
          {STEPS.map((s, i) => (
            <li key={s} className="flex items-center gap-2">
              <span className="rounded-full border border-line bg-raised px-3 py-1 text-ink">{s}</span>
              {i < STEPS.length - 1 && <span aria-hidden className="text-muted">→</span>}
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="explore" className="grid gap-4">
        <h2 id="explore" className="text-2xl font-semibold">Explore</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <Link key={f.href} href={f.href} className="group rounded-xl border border-line bg-raised p-5 no-underline hover:border-line-strong">
              <span className="font-semibold text-ink group-hover:underline">{f.title} →</span>
              <span className="mt-2 block text-sm leading-relaxed text-ink-2">{f.body}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
