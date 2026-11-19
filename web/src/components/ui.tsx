import Link from "next/link";
import type { ButtonHTMLAttributes, ComponentProps, ReactNode } from "react";

/* ------------------------------------------------------------------ buttons */
const BUTTON_BASE =
  "inline-flex items-center justify-center gap-2 rounded-lg font-medium no-underline transition-colors " +
  "disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap";
const BUTTON_VARIANTS = {
  primary: "bg-accent text-accent-ink shadow-card hover:bg-accent-hover hover:text-accent-ink",
  secondary: "border border-line-strong bg-raised text-ink shadow-card hover:bg-sunken hover:text-ink",
  ghost: "text-ink-2 hover:bg-sunken hover:text-ink",
} as const;
const BUTTON_SIZES = { sm: "h-8 px-3 text-sm", md: "h-10 px-4 text-sm", lg: "h-11 px-5 text-[15px]" } as const;
type Variant = keyof typeof BUTTON_VARIANTS;
type Size = keyof typeof BUTTON_SIZES;

export function buttonClass(variant: Variant = "primary", size: Size = "md", extra = "") {
  return `${BUTTON_BASE} ${BUTTON_VARIANTS[variant]} ${BUTTON_SIZES[size]} ${extra}`;
}

export function Button({ variant = "primary", size = "md", className = "", ...props }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size }) {
  return <button className={buttonClass(variant, size, className)} {...props} />;
}

export function ButtonLink({ variant = "primary", size = "md", className = "", ...props }:
  ComponentProps<typeof Link> & { variant?: Variant; size?: Size }) {
  return <Link className={buttonClass(variant, size, className)} {...props} />;
}

/* ------------------------------------------------------------------ layout */
export function Card({ title, children, className = "", aside }: {
  title?: ReactNode; children: ReactNode; className?: string; aside?: ReactNode;
}) {
  return (
    <section className={`rounded-xl border border-line bg-raised p-5 sm:p-6 ${className}`}>
      {(title || aside) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
          {title && <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ eyebrow, title, children, actions }: {
  eyebrow?: string; title: string; children?: ReactNode; actions?: ReactNode;
}) {
  return (
    <header className="mb-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        {eyebrow && <p className="mb-3 text-[13px] font-medium text-muted">{eyebrow}</p>}
        <h1 className="text-[1.875rem] font-semibold leading-[1.15] tracking-[-0.025em] text-ink sm:text-[2.25rem]">{title}</h1>
        {children && <div className="mt-3 text-[15px] leading-relaxed text-ink-2 sm:text-base">{children}</div>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
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
  const cls = tone === "accent" ? "border-transparent bg-accent-soft text-accent" : "border-line bg-sunken text-ink-2";
  return <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${cls}`}>{children}</span>;
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warning" }) {
  const cls = tone === "warning"
    ? "border-warning/40 bg-warning/10"
    : "border-line bg-sunken";
  const icon = tone === "warning" ? "!" : "i";
  return (
    <div role="note" className={`flex gap-3 rounded-lg border px-4 py-3 text-sm leading-relaxed text-ink ${cls}`}>
      <span aria-hidden className={`mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full text-[10px] font-bold ${
        tone === "warning" ? "bg-warning text-black" : "bg-ink-2 text-raised"}`}>{icon}</span>
      <div>{children}</div>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-md border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-ink">
      <p className="font-medium text-critical-ink">Something went wrong</p>
      <p className="mt-1">{error}</p>
      {onRetry && (
        <Button variant="secondary" size="sm" onClick={onRetry} className="mt-3">Try again</Button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line-strong bg-raised/50 px-6 py-12 text-center">
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
