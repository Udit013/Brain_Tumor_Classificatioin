"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api, ApiError, DISPLAY, useSession, type AnalysisSummary } from "@/lib/api";
import { Empty, ErrorState, LevelBadge, Notice, PageHeader, Spinner, pct } from "@/components/ui";

export default function HistoryPage() {
  const [rows, setRows] = useState<AnalysisSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [temporary, setTemporary] = useState(false);
  const signedIn = Boolean(useSession());

  useEffect(() => {
    api.history().then(setRows).catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    api.ready()
      .then((r) => setTemporary(r.checks.database?.persistent === false))
      .catch(() => undefined);
  }, []);

  const shown = useMemo(() => (rows ?? []).filter((r) => !status || r.status === status), [rows, status]);

  return (
    <>
      <PageHeader title="Your analyses">
        {signedIn
          ? "Analyses saved to your account."
          : "Analyses started from this browser. Sign in to keep them across devices and to use human review."}
      </PageHeader>
      {temporary && (
        <div className="mb-4"><Notice tone="warning">
          This free deployment keeps history on a temporary disk, so it is cleared whenever the server restarts.
        </Notice></div>
      )}
      <div role="radiogroup" aria-label="Filter by status" className="mb-4 inline-flex flex-wrap rounded-lg bg-sunken p-0.5">
        {[["", "All"], ["completed", "Completed"], ["awaiting_review", "Awaiting review"], ["running", "Running"], ["failed", "Failed"]].map(([v, label]) => (
          <button key={v} role="radio" aria-checked={status === v} onClick={() => setStatus(v)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${status === v ? "bg-raised text-ink shadow-card" : "text-ink-2 hover:text-ink"}`}>
            {label}
          </button>
        ))}
      </div>
      {error && <ErrorState error={error} />}
      {!rows && !error && <Spinner label="Loading history…" />}
      {rows && shown.length === 0 && (
        <Empty title="No analyses yet">
          Start one in the <Link href="/workbench">workbench</Link>.
        </Empty>
      )}
      {shown.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-raised">
          <table className="w-full text-sm">
            <thead className="border-b border-line text-left text-xs text-muted">
              <tr>
                <th className="px-4 py-2.5 font-normal">Started</th>
                <th className="px-4 py-2.5 font-normal">Input</th>
                <th className="px-4 py-2.5 font-normal">Model-predicted class</th>
                <th className="px-4 py-2.5 font-normal">Confidence</th>
                <th className="px-4 py-2.5 font-normal">Reliability</th>
                <th className="px-4 py-2.5 font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id} className="border-b border-line last:border-0 hover:bg-sunken/60">
                  <td className="px-4 py-3">
                    <Link href={`/analyses/${r.id}`} className="whitespace-nowrap font-medium text-ink no-underline hover:underline">
                      {new Date(r.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-ink-2">{r.sample_id ? `Example: ${r.sample_id}` : "Upload"}</td>
                  <td className="px-4 py-3">{r.predicted_class ? DISPLAY[r.predicted_class as keyof typeof DISPLAY] : "—"}</td>
                  <td className="px-4 py-3 tabular-nums">{pct(r.calibrated_confidence)}</td>
                  <td className="px-4 py-3"><LevelBadge level={r.reliability_level} /></td>
                  <td className="px-4 py-3 capitalize text-ink-2">{r.status.replace("_", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
