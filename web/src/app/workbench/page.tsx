import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { Workbench } from "./Workbench";

export const metadata: Metadata = { title: "MRI analysis workbench" };

export default function WorkbenchPage() {
  return (
    <>
      <PageHeader eyebrow="Workbench" title="Analyse an MRI slice">
        Run the audited EfficientNetB3 model, inspect its Grad-CAM and reliability checks, and get an
        explanation grounded in openly licensed literature. Every step is traced and reproducible.
      </PageHeader>
      <Workbench />
    </>
  );
}
