import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SiteHeader } from "@/components/SiteHeader";
import { DISCLAIMER } from "@/components/ui";
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
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <SiteHeader />
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-12 sm:px-6 sm:py-16">
          {children}
        </main>
        <footer className="border-t border-line bg-raised/40">
          <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-8 text-sm sm:flex-row sm:items-start sm:justify-between sm:px-6">
            <div className="max-w-xl">
              <p className="font-medium text-ink">NeuroClass</p>
              <p className="mt-1 leading-relaxed text-ink-2">{DISCLAIMER}</p>
            </div>
            <nav aria-label="Footer" className="flex shrink-0 flex-wrap gap-x-5 gap-y-2 text-ink-2">
              <a className="whitespace-nowrap text-ink-2 no-underline hover:text-ink hover:underline" href="https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941">IEEE paper</a>
              <a className="whitespace-nowrap text-ink-2 no-underline hover:text-ink hover:underline" href="https://github.com/Udit013/Brain_Tumor_Classificatioin">Source code</a>
              <a className="whitespace-nowrap text-ink-2 no-underline hover:text-ink hover:underline" href="https://udit013-brain-tumor-mri-classifier.hf.space/api/docs">API</a>
            </nav>
          </div>
        </footer>
      </body>
    </html>
  );
}
