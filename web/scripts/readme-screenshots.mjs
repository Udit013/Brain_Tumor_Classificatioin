// Regenerates the README screenshots: BASE=https://neuroclass-lab.vercel.app OUT=../docs/images node scripts/readme-screenshots.mjs
import { chromium } from "@playwright/test";

const BASE = process.env.BASE ?? "http://localhost:3000";
const OUT = process.env.OUT;
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, colorScheme: "light" });
const p = await ctx.newPage();
const settle = (ms = 1200) => p.waitForTimeout(ms);

// 1. Landing
await p.goto(BASE + "/"); await settle();
await p.screenshot({ path: `${OUT}/landing.png` });

// 2. Workbench with a completed analysis
await p.goto(BASE + "/workbench"); await settle();
await p.getByRole("radio", { name: /Labelled: Meningioma/ }).check({ force: true });
await p.getByRole("button", { name: "Run analysis" }).click();
await p.getByText("Completed", { exact: true }).first().waitFor({ timeout: 90000 });
await settle(2000);
await p.screenshot({ path: `${OUT}/workbench.png` });
const href = await p.getByRole("link", { name: "Permalink" }).getAttribute("href");

// 3. Analysis page in dark mode, at the reliability + explanation sections
await p.emulateMedia({ colorScheme: "dark" });
await p.goto(BASE + href); await settle(2500);
await p.getByRole("heading", { name: "Reliability assessment" }).scrollIntoViewIfNeeded();
await p.evaluate(() => {
  const h = [...document.querySelectorAll("h2")].find((x) => x.textContent.includes("Reliability assessment"));
  window.scrollTo(0, h.getBoundingClientRect().top + window.scrollY - 90);
});
await settle(600);
await p.screenshot({ path: `${OUT}/analysis-dark.png` });

// 4. Research assistant with an answer (light)
await p.emulateMedia({ colorScheme: "light" });
await p.goto(BASE + "/assistant"); await settle();
await p.getByLabel("Ask a research question").fill("What share of NeuroClass test images have a near-duplicate in the training set?");
await p.getByRole("button", { name: "Ask", exact: true }).click();
await p.getByText("Answer", { exact: true }).waitFor({ timeout: 90000 });
await settle(1500);
await p.screenshot({ path: `${OUT}/assistant.png` });

// 5. Benchmarks, at the audit section with its chart
await p.goto(BASE + "/benchmarks"); await settle(2500);
await p.evaluate(() => {
  const h = document.getElementById("audit");
  window.scrollTo(0, h.getBoundingClientRect().top + window.scrollY - 90);
});
await settle(800);
await p.screenshot({ path: `${OUT}/benchmarks.png` });

await browser.close();
console.log("done");
