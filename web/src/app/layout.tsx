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
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-10">
          {children}
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-sm text-ink-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="max-w-2xl">{DISCLAIMER}</p>
            <p>
              <a href="https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941">IEEE ICC-ROBINS 2024</a>
              {" · "}
              <a href="https://github.com/Udit013/Brain_Tumor_Classificatioin">Source code</a>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
