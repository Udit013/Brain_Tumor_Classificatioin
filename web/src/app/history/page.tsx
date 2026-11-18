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
      <PageHeader eyebrow="History" title="Your analyses">
        {signedIn
          ? "Analyses saved to your account."
          : "Analyses started from this browser. Sign in to keep them across devices and to use human review."}
      </PageHeader>
      {temporary && (
        <div className="mb-4"><Notice tone="warning">
          This free deployment keeps history on a temporary disk, so it is cleared whenever the server restarts.
        </Notice></div>
      )}
      <div className="mb-4 flex items-center gap-2">
        <label htmlFor="status" className="text-sm text-ink-2">Status</label>
        <select id="status" value={status} onChange={(e) => setStatus(e.target.value)}
                className="rounded-md border border-line bg-surface px-2 py-1 text-sm">
          <option value="">All</option>
          <option value="completed">Completed</option>
          <option value="awaiting_review">Awaiting review</option>
          <option value="running">Running</option>
          <option value="failed">Failed</option>
        </select>
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
            <thead className="border-b border-line text-left text-muted">
              <tr>
                <th className="px-4 py-2 font-normal">Started</th>
                <th className="px-4 py-2 font-normal">Input</th>
                <th className="px-4 py-2 font-normal">Model-predicted class</th>
                <th className="px-4 py-2 font-normal">Confidence</th>
                <th className="px-4 py-2 font-normal">Reliability</th>
                <th className="px-4 py-2 font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id} className="border-b border-line last:border-0 hover:bg-sunken">
                  <td className="px-4 py-2">
                    <Link href={`/analyses/${r.id}`} className="whitespace-nowrap">
                      {new Date(r.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
                    </Link>
                  </td>
                  <td className="px-4 py-2 text-ink-2">{r.sample_id ? `Example: ${r.sample_id}` : "Upload"}</td>
                  <td className="px-4 py-2">{r.predicted_class ? DISPLAY[r.predicted_class as keyof typeof DISPLAY] : "—"}</td>
                  <td className="px-4 py-2 tabular-nums">{pct(r.calibrated_confidence)}</td>
                  <td className="px-4 py-2"><LevelBadge level={r.reliability_level} /></td>
                  <td className="px-4 py-2 capitalize text-ink-2">{r.status.replace("_", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
