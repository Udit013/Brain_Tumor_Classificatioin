import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { Benchmarks } from "./Benchmarks";

export const metadata: Metadata = { title: "Research & benchmarks" };

const SECTIONS = [
  ["paper", "What the paper reported"],
  ["audit", "What the audit found"],
  ["retrain", "Retrained vs paper recipe"],
  ["perclass", "Per-class results"],
  ["calib", "Calibration and reliability"],
  ["robust", "Robustness and generalisation"],
  ["rag", "Literature retrieval"],
] as const;

export default function BenchmarksPage() {
  return (
    <>
      <PageHeader title="From a published number to measured evidence">
        The original paper&apos;s reported results, what an audit of them found, and every measurement of the model
        this platform serves — including the ones that are unflattering.
      </PageHeader>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1fr)_13rem]">
        <div className="min-w-0">
          <Benchmarks />
        </div>
        <nav aria-label="On this page" className="hidden lg:block">
          <div className="sticky top-20">
            <p className="text-xs font-medium text-muted">On this page</p>
            <ol className="mt-3 grid gap-1 border-l border-line">
              {SECTIONS.map(([id, label], i) => (
                <li key={id}>
                  <a href={`#${id}`}
                     className="-ml-px flex gap-2 border-l border-transparent py-1 pl-3 text-[13px] text-ink-2 no-underline hover:border-ink-2 hover:text-ink">
                    <span className="font-mono text-[11px] leading-5 text-muted">{String(i + 1).padStart(2, "0")}</span>
                    {label}
                  </a>
                </li>
              ))}
            </ol>
          </div>
        </nav>
      </div>
    </>
  );
}
