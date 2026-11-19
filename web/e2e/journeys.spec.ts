import { expect, test } from "@playwright/test";

test("landing page separates the paper's claim from measured results", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("shows its work");
  const table = page.getByRole("table", { name: "Reported versus measured accuracy" });
  await expect(table.getByText("Reported by the paper")).toBeVisible();
  await expect(table.getByText("Measured").first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Analyse an example", exact: true })).toBeVisible();
  await expect(page.getByText(/not diagnoses/).first()).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
});

test("workbench: analyse an example end to end and download the report", async ({ page }) => {
  await page.goto("/workbench");
  await page.getByRole("radio", { name: /Labelled: Meningioma/ }).check({ force: true });
  await page.getByRole("button", { name: "Run analysis" }).click();
  const result = page.getByRole("region", { name: "Analysis result" });
  await expect(result.getByText("Model-predicted class", { exact: true })).toBeVisible();
  await expect(result.getByText("Meningioma").first()).toBeVisible();
  await expect(result.getByRole("img", { name: /Grad-CAM heatmap/ })).toBeVisible();
  await expect(result.getByText("Reliability assessment")).toBeVisible();
  await expect(result.getByText(/Research interpretation/)).toBeVisible();
  const download = page.waitForEvent("download");
  await result.getByRole("button", { name: "Download report (HTML)" }).click();
  expect((await download).suggestedFilename()).toMatch(/^neuroclass-report-.*\.html$/);

  // The owner can delete it; it then disappears from history (and the test leaves no data behind).
  await page.goto("/history");
  await page.getByRole("table").getByRole("link").first().click();
  await expect(page.getByRole("button", { name: "Delete analysis" })).toBeVisible();
  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: "Delete analysis" }).click();
  await expect(page).toHaveURL(/\/history$/);
});

test("workbench rejects an unsupported upload before contacting the model", async ({ page }) => {
  await page.goto("/workbench");
  await page.getByRole("tab", { name: "Upload your own" }).click();
  await page.getByLabel("MRI image file").setInputFiles({ name: "notes.gif", mimeType: "image/gif", buffer: Buffer.from("GIF89a") });
  await expect(page.getByRole("main").getByRole("alert")).toContainText("JPEG or PNG");
});

test("research assistant answers with verified, cited statements", async ({ page }) => {
  await page.goto("/assistant");
  await page.getByLabel("Ask a research question").fill("What are the limitations of Grad-CAM saliency maps?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(page.getByText("Answer", { exact: true })).toBeVisible();
  const cite = page.getByRole("button", { name: /Show source passage/ }).first();
  await cite.click();
  await expect(page.getByRole("complementary", { name: "Source passage" })).toContainText(/CC|MIT/);
});

test("benchmarks show reported and measured numbers with table views", async ({ page }) => {
  await page.goto("/benchmarks");
  await expect(page.getByText("Reported, not re-validated.")).toBeVisible();
  await expect(page.getByRole("cell", { name: "640 of 1,311" })).toBeVisible();
  await page.getByRole("button", { name: "Show as table" }).first().click();
  await expect(page.getByRole("cell", { name: "Novel images" })).toBeVisible();
});

test("history lists analyses from this browser only", async ({ page, request }) => {
  await page.goto("/history");
  await expect(page.getByRole("heading", { name: "Your analyses" })).toBeVisible();
  const other = await request.get(`${process.env.NEUROCLASS_E2E_API ?? "http://localhost:8000"}/analyses`, {
    headers: { "X-Client-Id": "playwright-other-client-0123456789" },
  });
  expect(await other.json()).toEqual([]);
});
