import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SiteHeader } from "@/components/SiteHeader";
import Image from "next/image";
import { DISCLAIMER } from "@/components/ui";
import { THEME_SCRIPT } from "@/lib/theme";

const FOOTER = [
  { title: "Product", links: [
    { label: "Workbench", href: "/workbench" }, { label: "Research assistant", href: "/assistant" },
    { label: "History", href: "/history" }, { label: "Human review", href: "/review" } ] },
  { title: "Evidence", links: [
    { label: "Benchmarks", href: "/benchmarks" }, { label: "System status", href: "/system" },
    { label: "IEEE paper", href: "https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941" } ] },
  { title: "Developers", links: [
    { label: "API reference", href: "https://udit013-brain-tumor-mri-classifier.hf.space/api/docs" },
    { label: "Source code", href: "https://github.com/Udit013/Brain_Tumor_Classificatioin" },
    { label: "Model card", href: "https://huggingface.co/Udit013/brain-tumor-efficientnetb3" } ] },
];
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "NeuroClass — MRI research & explainability platform", template: "%s · NeuroClass" },
  description:
    "Research workbench built on an IEEE 2024 brain MRI classification study: audited model, explainability, evidence-grounded literature retrieval and human review. Not for clinical use.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <head>
        {/* Apply a stored light/dark choice before first paint (no flash). */}
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col">
        <SiteHeader />
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pt-12 sm:px-6 sm:pt-20">
          {children}
        </main>
        <footer className="mt-24 border-t border-line">
          <div className="mx-auto grid max-w-6xl gap-10 px-4 py-12 text-sm sm:px-6 md:grid-cols-[minmax(0,1.4fr)_repeat(3,minmax(0,1fr))]">
            <div className="max-w-sm">
              <p className="flex items-center gap-2 font-semibold text-ink">
                <Image src="/brand-mark.png" alt="" width={22} height={22} className="rounded-[6px]" />
                NeuroClass
              </p>
              <p className="mt-3 leading-relaxed text-muted">{DISCLAIMER}</p>
            </div>
            {FOOTER.map((col) => (
              <nav key={col.title} aria-label={col.title}>
                <p className="font-medium text-ink">{col.title}</p>
                <ul className="mt-3 grid gap-2">
                  {col.links.map((l) => (
                    <li key={l.href}>
                      <a href={l.href} className="text-muted no-underline hover:text-ink">{l.label}</a>
                    </li>
                  ))}
                </ul>
              </nav>
            ))}
          </div>
        </footer>
      </body>
    </html>
  );
}
