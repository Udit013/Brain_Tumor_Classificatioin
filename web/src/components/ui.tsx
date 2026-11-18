import type { ReactNode } from "react";

export function Card({ title, children, className = "", aside }: {
  title?: ReactNode; children: ReactNode; className?: string; aside?: ReactNode;
}) {
  return (
    <section className={`rounded-xl border border-line bg-raised p-5 ${className}`}>
      {(title || aside) && (
        <div className="mb-3 flex items-start justify-between gap-4">
          {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-3xl">
      {eyebrow && <p className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">{eyebrow}</p>}
      <h1 className="text-3xl font-semibold tracking-tight text-ink sm:text-4xl">{title}</h1>
      {children && <div className="mt-3 text-lg leading-relaxed text-ink-2">{children}</div>}
    </header>
  );
}

const LEVELS = {
  standard: { icon: "●", label: "Standard", cls: "text-good-ink border-good/40 bg-good/10" },
  caution: { icon: "▲", label: "Caution", cls: "text-warning-ink border-warning/50 bg-warning/10" },
  unreliable: { icon: "■", label: "Unreliable", cls: "text-critical-ink border-critical/40 bg-critical/10" },
} as const;

/** Reliability level: status colour + shape icon + text, never colour alone. */
export function LevelBadge({ level }: { level: keyof typeof LEVELS | string | null | undefined }) {
  const l = LEVELS[(level ?? "") as keyof typeof LEVELS];
  if (!l) return <span className="text-muted">—</span>;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-sm font-medium ${l.cls}`}>
      <span aria-hidden>{l.icon}</span>
      {l.label}
    </span>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" }) {
  const cls = tone === "accent" ? "border-accent/40 text-accent" : "border-line-strong text-ink-2";
  return <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>{children}</span>;
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warning" }) {
  const cls = tone === "warning" ? "border-l-warning bg-warning/10" : "border-l-accent bg-sunken";
  return <div role="note" className={`rounded-md border-l-4 px-4 py-3 text-sm leading-relaxed text-ink ${cls}`}>{children}</div>;
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-md border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-ink">
      <p className="font-medium text-critical-ink">Something went wrong</p>
      <p className="mt-1">{error}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-2 rounded-md border border-line-strong px-3 py-1 text-sm hover:bg-sunken">
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line-strong px-6 py-10 text-center">
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="mt-1 text-sm text-ink-2">{children}</div>}
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-ink-2" role="status">
      <span aria-hidden className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line-strong border-t-accent" />
      {label}
    </span>
  );
}

export const DISCLAIMER =
  "Research and education only. Outputs are an experimental model's predictions, not diagnoses, and must not inform anyone's care.";

export function pct(x: number | null | undefined, digits = 1) {
  return x == null ? "—" : `${(x * 100).toFixed(digits)}%`;
}
