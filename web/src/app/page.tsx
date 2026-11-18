import Link from "next/link";
import { ButtonLink } from "@/components/ui";

const EVIDENCE = [
  { value: "93.4%", label: "Accuracy on the full 1,600-image test set" },
  { value: "89.4%", label: "On test images with no near-copy in training" },
  { value: "72.7%", label: "On a separate public MRI dataset (n = 253)" },
];

const TIMELINE = [
  {
    when: "2024",
    title: "Published",
    body: "IEEE ICC-ROBINS study of four CNNs for four-class brain MRI classification. EfficientNetB3 reported 99.844%.",
  },
  {
    when: "2026",
    title: "Audited",
    body: "That figure was scored on 640 of 1,311 test images, and 44.6% of test images have a near-duplicate in training.",
  },
  {
    when: "2026",
    title: "Retrained",
    body: "Leakage-safe validation and a two-stage fine-tune: glioma recall 64% → 80%, accuracy under corruption 67% → 91%.",
  },
  {
    when: "Now",
    title: "Made inspectable",
    body: "Calibrated confidence, Grad-CAM, rule-based reliability, cited literature and human review for every prediction.",
  },
];

const PIPELINE: { step: string; by: string }[] = [
  { step: "Validate input", by: "rules" },
  { step: "Classify", by: "model" },
  { step: "Assess reliability", by: "rules" },
  { step: "Retrieve literature", by: "search" },
  { step: "Explain", by: "LLM or quotes" },
  { step: "Verify claims", by: "rules" },
  { step: "Human review", by: "person" },
  { step: "Report", by: "rules" },
];

const FEATURES = [
  { href: "/workbench", title: "Workbench", body: "Run the model on an example or your own slice and inspect confidence, Grad-CAM, reliability and a report.",
    icon: "M4 5h16v14H4z M4 15l4-4 4 4 3-3 5 5" },
  { href: "/assistant", title: "Research assistant", body: "Ask about the literature or the project. Every statement cites a passage; unsupported ones are removed.",
    icon: "M5 5h14v10H9l-4 4z" },
  { href: "/benchmarks", title: "Benchmarks", body: "The paper's reported table beside the audit, per-class errors, calibration and robustness.",
    icon: "M5 19V9 M12 19V5 M19 19v-7" },
  { href: "/system", title: "System", body: "Live status of every subsystem, model checksums, export verification and data licences.",
    icon: "M12 4a8 8 0 1 0 0 16a8 8 0 0 0 0-16z M12 8v4l3 2" },
];

export default function Home() {
  return (
    <div className="grid gap-24">
      <section className="grid items-center gap-12 pt-2 lg:grid-cols-[1.15fr_1fr]">
        <div>
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-line bg-raised px-3 py-1 text-xs font-medium text-ink-2 shadow-card">
            <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-accent" />
            Built on an IEEE ICC-ROBINS 2024 study
          </p>
          <h1 className="text-[2.125rem] font-semibold leading-[1.08] tracking-[-0.025em] text-ink sm:text-[3.25rem]">
            Brain MRI classification you can inspect, not just trust.
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-2">
            NeuroClass audits a published classifier, measures where it fails, and shows the evidence behind every
            prediction: calibrated confidence, Grad-CAM, reliability rules and cited research.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <ButtonLink href="/workbench" size="lg">Analyse an MRI slice</ButtonLink>
            <ButtonLink href="/benchmarks" size="lg" variant="secondary">See the evidence</ButtonLink>
          </div>
          <p className="mt-6 max-w-xl text-sm leading-relaxed text-muted">
            Research and education only. Outputs are an experimental model&apos;s predictions, not diagnoses; the
            model has no clinical validation or regulatory clearance.
          </p>
        </div>

        <figure className="rounded-2xl border border-line bg-raised p-6 shadow-raised sm:p-8">
          <figcaption className="flex items-center justify-between gap-4 text-sm">
            <span className="font-medium text-ink">What the headline number means</span>
            <span className="text-muted">EfficientNetB3</span>
          </figcaption>
          <div className="mt-6 flex items-baseline justify-between gap-4 border-b border-line pb-6">
            <div>
              <p className="text-xs font-medium text-muted">Reported by the paper</p>
              <p className="tabular mt-1 text-4xl font-semibold tracking-tight text-ink-2">99.844%</p>
              <p className="mt-1 text-sm text-ink-2">639 of 640 images — about half of the 1,311-image test set</p>
            </div>
          </div>
          <p className="mt-6 text-xs font-medium text-muted">Measured by this project</p>
          <dl className="mt-3 grid gap-4">
            {EVIDENCE.map((e) => (
              <div key={e.label} className="flex items-baseline gap-4">
                <dt className="order-2 text-sm leading-snug text-ink-2">{e.label}</dt>
                <dd className="tabular w-24 shrink-0 text-2xl font-semibold tracking-tight text-ink">{e.value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-6 text-xs leading-relaxed text-muted">
            Measured on CPU with the served preprocessing, retrained model, single seed.{" "}
            <Link href="/benchmarks">Methods</Link>
          </p>
        </figure>
      </section>

      <section aria-labelledby="story">
        <div className="mb-10 max-w-2xl">
          <h2 id="story" className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">From paper to platform</h2>
          <p className="mt-3 text-ink-2">
            The language-model, retrieval and agent features are new engineering — they were not part of the
            published study.
          </p>
        </div>
        <ol className="relative grid gap-8 md:grid-cols-4 md:gap-6">
          <span aria-hidden className="absolute left-0 right-0 top-[11px] hidden h-px bg-line md:block" />
          {TIMELINE.map((t, i) => (
            <li key={t.title} className="relative">
              <span className="relative z-10 grid h-6 w-6 place-items-center rounded-full border border-line-strong bg-surface text-xs font-semibold text-ink-2 tabular">
                {i + 1}
              </span>
              <p className="mt-4 text-xs font-medium text-muted">{t.when}</p>
              <h3 className="mt-1 font-semibold text-ink">{t.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">{t.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="workflow" className="rounded-2xl border border-line bg-raised p-6 shadow-card sm:p-10">
        <div className="grid gap-8 lg:grid-cols-[1fr_1.6fr] lg:gap-12">
          <div>
            <h2 id="workflow" className="text-2xl font-semibold tracking-tight text-ink">One analysis, one traced workflow</h2>
            <p className="mt-3 leading-relaxed text-ink-2">
              Each analysis is a stateful LangGraph run. Reliability and claim checks are deterministic code, not a
              language model. Flagged results pause for a reviewer and resume from a saved checkpoint.
            </p>
          </div>
          <ol className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
            {PIPELINE.map((p, i) => (
              <li key={p.step} className="bg-raised p-4">
                <span className="tabular text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
                <p className="mt-2 text-sm font-medium leading-snug text-ink">{p.step}</p>
                <p className="mt-1 text-xs text-muted">{p.by}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section aria-labelledby="explore">
        <h2 id="explore" className="mb-8 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Explore</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <Link key={f.href} href={f.href}
                  className="group flex gap-4 rounded-xl border border-line bg-raised p-5 text-ink no-underline shadow-card transition-colors hover:border-line-strong hover:text-ink">
              <span aria-hidden className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
                <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75"
                     strokeLinecap="round" strokeLinejoin="round"><path d={f.icon} /></svg>
              </span>
              <span>
                <span className="flex items-center gap-1 font-semibold">
                  {f.title}
                  <span aria-hidden className="text-muted transition-transform group-hover:translate-x-0.5">→</span>
                </span>
                <span className="mt-1 block text-sm leading-relaxed text-ink-2">{f.body}</span>
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
