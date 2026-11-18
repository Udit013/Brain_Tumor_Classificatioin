import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { Benchmarks } from "./Benchmarks";

export const metadata: Metadata = { title: "Research & benchmarks" };

export default function BenchmarksPage() {
  return (
    <>
      <PageHeader eyebrow="Research & benchmarks" title="From a published number to measured evidence">
        The original paper&apos;s reported results, what an audit of them found, and every measurement of the model
        this platform serves — including the ones that are unflattering.
      </PageHeader>
      <Benchmarks />
    </>
  );
}
