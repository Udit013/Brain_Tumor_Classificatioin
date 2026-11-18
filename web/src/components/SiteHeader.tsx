"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense, useState } from "react";
import { clearSession, useSession } from "@/lib/api";

const NAV = [
  { href: "/workbench", label: "Workbench" },
  { href: "/assistant", label: "Research assistant" },
  { href: "/history", label: "History" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/system", label: "System" },
];

function linkClass(active: boolean) {
  return `rounded-md px-3 py-2 text-sm font-medium no-underline transition-colors ${
    active ? "bg-sunken text-ink hover:text-ink" : "text-ink-2 hover:bg-sunken hover:text-ink"}`;
}

// The pathname is unknown while a dynamic route (/analyses/[id]) prerenders, so
// only this leaf reads it; the fallback is the same link without the highlight.
function ActiveNavLink({ href, label, onNavigate }: { href: string; label: string; onNavigate: () => void }) {
  const active = usePathname()?.startsWith(href) ?? false;
  return (
    <Link href={href} aria-current={active ? "page" : undefined} onClick={onNavigate} className={linkClass(active)}>
      {label}
    </Link>
  );
}

function NavLink(props: { href: string; label: string; onNavigate: () => void }) {
  return (
    <Suspense fallback={<Link href={props.href} onClick={props.onNavigate} className={linkClass(false)}>{props.label}</Link>}>
      <ActiveNavLink {...props} />
    </Suspense>
  );
}

export function SiteHeader() {
  const session = useSession();
  const [open, setOpen] = useState(false);

  const links = session?.role === "reviewer" ? [...NAV, { href: "/review", label: "Review queue" }] : NAV;
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-surface/85 backdrop-blur-md">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded focus:bg-raised focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5 font-semibold tracking-tight text-ink no-underline hover:text-ink">
          <span aria-hidden className="grid h-8 w-8 place-items-center rounded-lg bg-ink text-[11px] font-bold tracking-wide text-surface">NC</span>
          NeuroClass
        </Link>
        <button
          className="inline-flex h-9 items-center gap-2 rounded-lg border border-line bg-raised px-3 text-sm font-medium text-ink md:hidden"
          aria-expanded={open}
          aria-controls="site-nav"
          onClick={() => setOpen((o) => !o)}
        >
          <svg aria-hidden viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round">
            {open ? <path d="M5 5l10 10M15 5L5 15" /> : <path d="M3 6h14M3 10h14M3 14h14" />}
          </svg>
          Menu
        </button>
        <nav id="site-nav" aria-label="Main" className={`${open ? "flex" : "hidden"} absolute left-0 right-0 top-16 z-20 flex-col gap-1 border-b border-line bg-raised px-4 py-3 shadow-raised md:static md:flex md:flex-row md:items-center md:gap-1 md:border-0 md:bg-transparent md:p-0 md:shadow-none`}>
          {links.map((l) => <NavLink key={l.href} href={l.href} label={l.label} onNavigate={() => setOpen(false)} />)}
          <span aria-hidden className="mx-2 hidden h-5 w-px bg-line md:block" />
          {session ? (
            <button onClick={clearSession} className="rounded-md px-3 py-2 text-left text-sm font-medium text-ink-2 hover:bg-sunken hover:text-ink" title={session.email}>
              Sign out <span className="font-normal text-muted">· {session.role}</span>
            </button>
          ) : (
            <Link href="/login" onClick={() => setOpen(false)}
                  className="inline-flex h-9 items-center rounded-lg border border-line-strong bg-raised px-3.5 text-sm font-medium text-ink no-underline shadow-card hover:bg-sunken hover:text-ink">
              Sign in
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}
