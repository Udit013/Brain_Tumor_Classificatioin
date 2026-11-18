import { Suspense } from "react";
import { PageHeader, Spinner } from "@/components/ui";
import { AnalysisDetail } from "./AnalysisDetail";

export default function AnalysisPage() {
  return (
    <>
      <PageHeader eyebrow="Analysis" title="Analysis result" />
      <Suspense fallback={<Spinner label="Loading analysis…" />}>
        <AnalysisDetail />
      </Suspense>
    </>
  );
}
