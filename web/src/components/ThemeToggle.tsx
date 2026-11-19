"use client";

import { useSyncExternalStore } from "react";
import { THEME_KEY as KEY } from "@/lib/theme";

/*
 * System / light / dark. The choice is stored in localStorage and applied as
 * data-theme on <html>; "system" removes the attribute so the CSS follows
 * prefers-color-scheme. THEME_SCRIPT (in the root layout) applies the stored
 * choice before first paint, so a forced theme never flashes.
 */
export type Theme = "system" | "light" | "dark";
const ORDER: Theme[] = ["system", "light", "dark"];
const EVENT = "nc-theme-change";


function read(): Theme {
  try {
    const t = localStorage.getItem(KEY);
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

function subscribe(cb: () => void) {
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb);
  return () => { window.removeEventListener(EVENT, cb); window.removeEventListener("storage", cb); };
}

function apply(t: Theme) {
  try {
    if (t === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, t);
  } catch { /* private mode: still applies for this page */ }
  if (t === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  window.dispatchEvent(new Event(EVENT));
}

const LABEL: Record<Theme, string> = { system: "System theme", light: "Light theme", dark: "Dark theme" };

function Icon({ theme }: { theme: Theme }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (theme === "light") {
    return (
      <svg aria-hidden viewBox="0 0 20 20" className="h-4 w-4" {...common}>
        <circle cx="10" cy="10" r="3.2" />
        <path d="M10 2.5v1.8M10 15.7v1.8M2.5 10h1.8M15.7 10h1.8M4.7 4.7l1.3 1.3M14 14l1.3 1.3M4.7 15.3 6 14M14 6l1.3-1.3" />
      </svg>
    );
  }
  if (theme === "dark") {
    return (
      <svg aria-hidden viewBox="0 0 20 20" className="h-4 w-4" {...common}>
        <path d="M15.5 12.3A6.2 6.2 0 0 1 7.7 4.5a6.2 6.2 0 1 0 7.8 7.8Z" />
      </svg>
    );
  }
  return (
    <svg aria-hidden viewBox="0 0 20 20" className="h-4 w-4" {...common}>
      <rect x="2.75" y="3.75" width="14.5" height="10" rx="1.5" />
      <path d="M7 17h6M10 13.75V17" />
    </svg>
  );
}

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, read, () => "system" as Theme);
  const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
  return (
    <button
      type="button"
      onClick={() => apply(next)}
      title={`${LABEL[theme]} — switch to ${LABEL[next].toLowerCase()}`}
      aria-label={`${LABEL[theme]}. Switch to ${LABEL[next].toLowerCase()}`}
      className="inline-flex h-8 w-8 items-center justify-center rounded-md text-ink-2 transition-colors hover:bg-sunken hover:text-ink"
    >
      <Icon theme={theme} />
    </button>
  );
}
