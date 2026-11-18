"use client";

/**
 * Small SVG charts following the dataviz method: validated two-series palette
 * (series-1 current, series-2 previous), one value axis, thin marks with 4px
 * rounded data ends, 2px gaps, recessive solid gridlines, legend + selective
 * direct labels, a hover tooltip per mark, and a table view for every chart.
 */
import { useState, type ReactNode } from "react";

export type Series = { key: string; label: string; color: string };

function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="flex flex-wrap gap-4 text-sm text-ink-2" aria-label="Legend">
      {series.map((s) => (
        <li key={s.key} className="flex items-center gap-2">
          <span aria-hidden className="h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} />
          {s.label}
        </li>
      ))}
    </ul>
  );
}

function ChartFrame({ title, caption, legend, table, children }: {
  title: string; caption?: string; legend?: Series[]; table: ReactNode; children: ReactNode;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className="grid gap-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <figcaption>
          <span className="block font-medium text-ink">{title}</span>
          {caption && <span className="block text-sm text-ink-2">{caption}</span>}
        </figcaption>
        <button onClick={() => setAsTable((v) => !v)} className="rounded-md border border-line px-2 py-1 text-xs text-ink-2 hover:bg-sunken"
                aria-pressed={asTable}>
          {asTable ? "Show chart" : "Show as table"}
        </button>
      </div>
      {legend && legend.length > 1 && !asTable && <Legend series={legend} />}
      {asTable ? <div className="overflow-x-auto">{table}</div> : children}
    </figure>
  );
}

const pctLabel = (v: number) => `${(v * 100).toFixed(1)}%`;

/** Grouped vertical bars on a single 0–100% axis. */
export function GroupedBars({ title, caption, categories, series, values }: {
  title: string; caption?: string; categories: string[]; series: Series[];
  values: Record<string, number[]>; // series key -> value per category (0..1)
}) {
  const [hover, setHover] = useState<{ x: number; text: string } | null>(null);
  const W = 640, H = 260, left = 36, bottom = 44, top = 18;
  const plotH = H - bottom - top;
  const groupW = (W - left) / categories.length;
  const barW = Math.min(36, (groupW - 24) / series.length);
  const y = (v: number) => top + plotH * (1 - v);
  const table = (
    <table className="w-full text-sm">
      <thead><tr className="text-left text-muted"><th className="font-normal">Metric</th>{series.map((s) => <th key={s.key} className="font-normal">{s.label}</th>)}</tr></thead>
      <tbody>{categories.map((c, i) => (
        <tr key={c} className="border-t border-line"><td className="py-1 text-ink">{c}</td>{series.map((s) => <td key={s.key} className="tabular-nums">{pctLabel(values[s.key][i])}</td>)}</tr>
      ))}</tbody>
    </table>
  );
  return (
    <ChartFrame title={title} caption={caption} legend={series} table={table}>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="max-h-72 w-full" role="img" aria-label={`${title}: bar chart; use "Show as table" for values`}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={left} x2={W} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
              <text x={left - 6} y={y(t) + 4} textAnchor="end" className="fill-muted text-[11px]">{t * 100}%</text>
            </g>
          ))}
          {categories.map((c, ci) => {
            const gx = left + ci * groupW + (groupW - barW * series.length - 2 * (series.length - 1)) / 2;
            return (
              <g key={c}>
                {series.map((s, si) => {
                  const v = values[s.key][ci];
                  const x = gx + si * (barW + 2);
                  const h = Math.max(plotH * v, 1);
                  const yy = top + plotH - h;
                  return (
                    <g key={s.key}
                       onMouseEnter={() => setHover({ x: ((x + barW / 2) / W) * 100, text: `${s.label} · ${c}: ${pctLabel(v)}` })}
                       onMouseLeave={() => setHover(null)}>
                      <rect x={x - 4} y={top} width={barW + 8} height={plotH} fill="transparent" />
                      <path d={`M${x},${top + plotH} V${yy + 4} Q${x},${yy} ${x + 4},${yy} H${x + barW - 4} Q${x + barW},${yy} ${x + barW},${yy + 4} V${top + plotH} Z`}
                            fill={s.color} />
                      <text x={x + barW / 2} y={yy - 5} textAnchor="middle" className="fill-ink-2 text-[11px]">{(v * 100).toFixed(1)}</text>
                    </g>
                  );
                })}
                <text x={left + ci * groupW + groupW / 2} y={H - bottom + 18} textAnchor="middle" className="fill-ink text-[12px]">{c}</text>
              </g>
            );
          })}
        </svg>
        {hover && (
          <div role="tooltip" className="pointer-events-none absolute top-0 z-10 -translate-x-1/2 rounded-md border border-line bg-raised px-2.5 py-1.5 text-xs text-ink shadow-sm"
               style={{ left: `${hover.x}%` }}>{hover.text}</div>
        )}
      </div>
    </ChartFrame>
  );
}

/** Confusion matrix heatmap: cell shade = share of the true class (single-hue ramp). */
export function ConfusionMatrix({ title, caption, labels, matrix }: {
  title: string; caption?: string; labels: string[]; matrix: number[][];
}) {
  const [hover, setHover] = useState<string | null>(null);
  const ramp = ["var(--seq-100)", "var(--seq-250)", "var(--seq-400)", "var(--seq-550)", "var(--seq-700)"];
  const table = (
    <table className="text-sm">
      <thead><tr><th className="pr-3 text-left font-normal text-muted">True \ Predicted</th>{labels.map((l) => <th key={l} className="px-2 font-normal text-muted">{l}</th>)}</tr></thead>
      <tbody>{matrix.map((row, i) => (
        <tr key={i} className="border-t border-line"><td className="pr-3 text-ink">{labels[i]}</td>{row.map((v, j) => <td key={j} className="px-2 text-center tabular-nums">{v}</td>)}</tr>
      ))}</tbody>
    </table>
  );
  return (
    <ChartFrame title={title} caption={caption} table={table}>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-0.5 text-sm" aria-label={title}>
          <thead>
            <tr>
              <th className="px-2 text-left text-xs font-normal text-muted">True ↓ / Predicted →</th>
              {labels.map((l) => <th key={l} className="px-2 pb-1 text-xs font-normal text-ink-2">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const total = row.reduce((a, b) => a + b, 0) || 1;
              return (
                <tr key={i}>
                  <th className="pr-3 text-right text-xs font-normal text-ink-2">{labels[i]}</th>
                  {row.map((v, j) => {
                    const share = v / total;
                    const step = share === 0 ? -1 : Math.min(4, Math.floor(share * 5));
                    const high = step >= 3;    // high end of the ramp: text token flips per mode
                    const tip = `${labels[i]} predicted as ${labels[j]}: ${v} of ${total} (${(share * 100).toFixed(1)}%)`;
                    return (
                      <td key={j} title={tip} onMouseEnter={() => setHover(tip)} onMouseLeave={() => setHover(null)}
                          className={`h-14 w-20 rounded text-center tabular-nums ${i === j ? "font-semibold" : ""}`}
                          style={{ background: step < 0 ? "var(--surface-sunken)" : ramp[step],
                                   color: high ? "var(--seq-ink-high)" : "var(--seq-ink-low)" }}>
                        {v}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-2 h-5 text-xs text-ink-2" aria-live="polite">{hover ?? "Shade = share of each true class. Hover a cell for counts."}</p>
      </div>
    </ChartFrame>
  );
}

/** Small multiples: accuracy vs corruption severity, one panel per corruption. */
export function RobustnessPanels({ title, caption, series, data }: {
  title: string; caption?: string; series: Series[];
  data: Record<string, Record<string, number[]>>; // corruption -> series key -> accuracy per severity
}) {
  const [hover, setHover] = useState<string | null>(null);
  const corruptions = Object.keys(data);
  const W = 220, H = 140, l = 30, b = 22, t = 10, r = 8;
  const x = (i: number) => l + (i * (W - l - r)) / 4;
  const y = (v: number) => t + (H - t - b) * (1 - v);
  const table = (
    <table className="w-full text-sm">
      <thead><tr className="text-left text-muted"><th className="font-normal">Corruption</th><th className="font-normal">Model</th>{[1, 2, 3, 4, 5].map((s) => <th key={s} className="font-normal">Sev {s}</th>)}</tr></thead>
      <tbody>{corruptions.flatMap((c) => series.map((s) => (
        <tr key={c + s.key} className="border-t border-line"><td>{c.replace("_", " ")}</td><td>{s.label}</td>{data[c][s.key].map((v, i) => <td key={i} className="tabular-nums">{pctLabel(v)}</td>)}</tr>
      )))}</tbody>
    </table>
  );
  return (
    <ChartFrame title={title} caption={caption} legend={series} table={table}>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {corruptions.map((c) => (
          <div key={c}>
            <p className="mb-1 text-sm capitalize text-ink">{c.replace("_", " ")}</p>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${c}: accuracy by severity`}>
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={l} x2={W - r} y1={y(v)} y2={y(v)} stroke="var(--line)" />
                  <text x={l - 4} y={y(v) + 4} textAnchor="end" className="fill-muted text-[10px]">{v * 100}%</text>
                </g>
              ))}
              {[0, 1, 2, 3, 4].map((i) => <text key={i} x={x(i)} y={H - 6} textAnchor="middle" className="fill-muted text-[10px]">{i + 1}</text>)}
              {series.map((s) => {
                const vals = data[c][s.key];
                return (
                  <g key={s.key}>
                    <polyline fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" points={vals.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
                    {vals.map((v, i) => (
                      <g key={i} onMouseEnter={() => setHover(`${c.replace("_", " ")}, severity ${i + 1} — ${s.label}: ${pctLabel(v)}`)} onMouseLeave={() => setHover(null)}>
                        <circle cx={x(i)} cy={y(v)} r={10} fill="transparent" />
                        <circle cx={x(i)} cy={y(v)} r={4} fill={s.color} stroke="var(--surface-raised)" strokeWidth={2} />
                      </g>
                    ))}
                  </g>
                );
              })}
            </svg>
          </div>
        ))}
      </div>
      <p className="h-5 text-xs text-ink-2" aria-live="polite">{hover ?? "x: corruption severity 1–5 · y: accuracy. Hover a point for its value."}</p>
    </ChartFrame>
  );
}

export function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-line bg-raised p-4">
      <p className="text-sm text-ink-2">{label}</p>
      <p className="mt-1 text-3xl font-semibold text-ink">{value}</p>
      {sub && <p className="mt-1 text-sm text-muted">{sub}</p>}
    </div>
  );
}
