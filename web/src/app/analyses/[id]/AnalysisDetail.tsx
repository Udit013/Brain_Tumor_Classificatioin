"use client";

import { useParams } from "next/navigation";
import { AnalysisRunner } from "@/components/analysis/AnalysisRunner";

export function AnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  return <AnalysisRunner id={id} />;
}
