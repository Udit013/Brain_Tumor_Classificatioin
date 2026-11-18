"use client";

import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { AnalysisRunner } from "@/components/analysis/AnalysisRunner";
import { api, ApiError } from "@/lib/api";
import { buttonClass } from "@/components/ui";

export function AnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function remove() {
    if (!window.confirm("Delete this analysis, its report, review notes and trace? This can't be undone.")) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteAnalysis(id);
      router.push("/history");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The analysis could not be deleted.");
      setBusy(false);
    }
  }

  return (
    <>
      <AnalysisRunner id={id} />
      <div className="mt-10 flex flex-wrap items-center gap-3 border-t border-line pt-6 text-sm">
        <button onClick={remove} disabled={busy}
                className={buttonClass("secondary", "sm", "text-critical-ink hover:text-critical-ink")}>
          {busy ? "Deleting…" : "Delete analysis"}
        </button>
        <span className="text-muted">Removes the result, report, reviews and workflow history for good.</span>
        {error && <span role="alert" className="text-critical-ink">{error}</span>}
      </div>
    </>
  );
}
