import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { System } from "./System";

export const metadata: Metadata = { title: "Model & system transparency" };

export default function SystemPage() {
  return (
    <>
      <PageHeader eyebrow="Transparency" title="Model and system">
        What is running, which exact model version, how it was verified, where the data comes from, and what the
        platform must not be used for.
      </PageHeader>
      <System />
    </>
  );
}
