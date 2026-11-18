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
  return `rounded-md px-3 py-2 text-sm no-underline ${active ? "bg-sunken font-medium text-ink" : "text-ink-2 hover:bg-sunken hover:text-ink"}`;
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
    <header className="border-b border-line bg-raised/80 backdrop-blur">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded focus:bg-raised focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold text-ink no-underline">
          <span aria-hidden className="grid h-7 w-7 place-items-center rounded-md bg-ink text-xs font-bold text-surface">NC</span>
          NeuroClass
        </Link>
        <button
          className="rounded-md border border-line px-3 py-1.5 text-sm sm:hidden"
          aria-expanded={open}
          aria-controls="site-nav"
          onClick={() => setOpen((o) => !o)}
        >
          Menu
        </button>
        <nav id="site-nav" aria-label="Main" className={`${open ? "flex" : "hidden"} absolute left-0 right-0 top-14 z-20 flex-col border-b border-line bg-raised px-4 py-2 sm:static sm:flex sm:flex-row sm:items-center sm:gap-1 sm:border-0 sm:bg-transparent sm:p-0`}>
          {links.map((l) => <NavLink key={l.href} href={l.href} label={l.label} onNavigate={() => setOpen(false)} />)}
          {session ? (
            <button onClick={clearSession} className="rounded-md px-3 py-2 text-left text-sm text-ink-2 hover:bg-sunken" title={session.email}>
              Sign out <span className="text-muted">({session.role})</span>
            </button>
          ) : (
            <Link href="/login" className="rounded-md px-3 py-2 text-sm text-ink-2 no-underline hover:bg-sunken">Sign in</Link>
          )}
        </nav>
      </div>
    </header>
  );
}
