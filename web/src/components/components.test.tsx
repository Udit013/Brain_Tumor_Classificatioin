import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Analysis } from "@/lib/api";
import { ExplanationView, ResultView } from "./analysis/ResultView";
import { GroupedBars } from "./charts";
import { LevelBadge } from "./ui";

vi.mock("@/lib/api", async (orig) => {
  const actual = await orig<typeof import("@/lib/api")>();
  return { ...actual, api: { ...actual.api, blob: vi.fn().mockRejectedValue(new Error("offline")) } };
});

const passage = {
  chunk_id: "PMC1#0", doc_id: "PMC1", title: "Saliency maps in MRI", section: "Discussion",
  text: "Saliency maps show where a network looked, not why it decided.", source_url: "https://example.org/PMC1",
  license: "CC BY", year: 2024, journal: "J Test", source_type: "pmc_article" as const, score: 1,
};

const analysis: Analysis = {
  id: "abc123", status: "completed", created_at: "2026-11-17T10:00:00Z", updated_at: "2026-11-17T10:00:01Z",
  sample_id: "glioma", question: null, model_version: "2026.10-finetune", error: null, owner: null, reviews: [], trace: [],
  inference: {
    model_version: "2026.10-finetune", predicted_class: "glioma", calibrated_confidence: 0.35, raw_confidence: 0.343,
    probabilities: [
      { label: "meningioma", calibrated: 0.255, raw: 0.25 },
      { label: "glioma", calibrated: 0.35, raw: 0.343 },
      { label: "notumor", calibrated: 0.212, raw: 0.22 },
      { label: "pituitary", calibrated: 0.183, raw: 0.187 },
    ],
    tta_agreement: 4, tta_views: 6, tta_entropy: 1.2, ood_distance: 44.2, ood_threshold: 71.4, ood_flag: false,
    input_quality: { width: 354, height: 442, mode: "RGB", format: "JPEG", grayscale_like: true,
                     mean_intensity: 50, std_intensity: 40, findings: [] },
    gradcam_png_base64: null, latency_ms: { total: 240 },
  },
  reliability: {
    level: "unreliable", requires_review: true, rules_version: "2026.11-r1",
    findings: [{ code: "very_low_confidence", severity: "caution", message: "Calibrated confidence is 35%." }],
  },
  explanation: {
    mode: "extractive", llm_model: null, summary: "Model-predicted class: glioma.", usage: {},
    statements: [
      { text: "Saliency maps show where a network looked, not why it decided.", kind: "literature",
        citations: ["PMC1#0"], status: "supported", support_score: 1, reason: null },
    ],
    removed_statements: [
      { text: "Grad-CAM proves the tumour causes it.", kind: "literature", citations: ["PMC9#9"],
        status: "unsupported", support_score: null, reason: "cites passages that were not retrieved: ['PMC9#9']" },
    ],
    passages: [passage],
  },
};

describe("ResultView", () => {
  it("shows the model-predicted class with probabilities sorted, and never says diagnosis", () => {
    const { container } = render(<ResultView analysis={analysis} />);
    expect(screen.getByText("Model-predicted class")).toBeInTheDocument();
    const rows = within(screen.getByRole("table", { name: /calibrated class probabilities/i }))
      .getAllByRole("row").slice(1).map((r) => r.textContent);
    expect(rows[0]).toMatch(/^Glioma35\.0%/);
    expect(rows[1]).toMatch(/^Meningioma25\.5%/);
    expect(container.textContent?.toLowerCase()).not.toMatch(/\bdiagnos(is|ed)\b(?! —)/);
    expect(screen.getByText(/not a tumor outline/i)).toBeInTheDocument();
  });

  it("opens the exact cited passage and lists removed statements with reasons", async () => {
    render(<ExplanationView explanation={analysis.explanation!} />);
    await userEvent.click(screen.getByRole("button", { name: "Show source passage PMC1#0" }));
    const preview = screen.getByRole("complementary", { name: "Source passage" });
    expect(within(preview).getByText(/not why it decided/)).toBeInTheDocument();
    expect(within(preview).getByText(/CC BY/)).toBeInTheDocument();
    expect(screen.getByText(/1 statement\(s\) removed by verification/)).toBeInTheDocument();
    expect(screen.getByText(/not retrieved/)).toBeInTheDocument();
  });
});

describe("LevelBadge", () => {
  it("encodes level with an icon and a label, not colour alone", () => {
    render(<LevelBadge level="caution" />);
    expect(screen.getByText("Caution")).toBeInTheDocument();
    expect(screen.getByText("▲")).toHaveAttribute("aria-hidden");
  });
});

describe("charts", () => {
  it("every chart has a table view with the exact values", async () => {
    render(<GroupedBars title="Test" categories={["A", "B"]}
                        series={[{ key: "p", label: "Previous", color: "red" }, { key: "c", label: "Current", color: "blue" }]}
                        values={{ p: [0.5, 0.6], c: [0.7, 0.8] }} />);
    expect(screen.getByRole("img", { name: /Test: bar chart/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show as table" }));
    const table = screen.getByRole("table");
    expect(table.textContent).toContain("70.0%");
    expect(table.textContent).toContain("60.0%");
  });
});
