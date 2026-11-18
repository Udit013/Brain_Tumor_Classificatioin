"use client";

import { useEffect, useState } from "react";
import { api, ApiError, type Benchmarks as B } from "@/lib/api";
import { ConfusionMatrix, GroupedBars, RobustnessPanels, StatTile, type Series } from "@/components/charts";
import { Card, ErrorState, Notice, Spinner, pct } from "@/components/ui";

const SERIES: Series[] = [
  { key: "previous", label: "Previous (paper recipe, reproduced)", color: "var(--series-2)" },
  { key: "current", label: "Current (retrained)", color: "var(--series-1)" },
];
const CLASS_LABELS = ["Glioma", "Meningioma", "No tumor", "Pituitary"];

export function Benchmarks() {
  const [data, setData] = useState<B | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.benchmarks().then(setData).catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  if (error) return <ErrorState error={error} />;
  if (!data) return <Spinner label="Loading benchmarks…" />;

  const m = data.measured;
  const cmp = m.comparison;
  const leak = m.leakage;
  const rel = m.reliability;
  const retr = m.retrieval;

  return (
    <div className="grid gap-10">
      {/* 1. paper ------------------------------------------------------------- */}
      <section aria-labelledby="paper" className="grid gap-4">
        <h2 id="paper" className="text-2xl font-semibold">1 · What the IEEE paper reported</h2>
        <Notice>
          <strong>Reported, not re-validated.</strong> {data.paper.citation} (DOI{" "}
          <a href={`https://doi.org/${data.paper.doi}`}>{data.paper.doi}</a>). Dataset: {data.paper.dataset}. These are the
          paper&apos;s own figures. Each notebook scored a fixed number of batches, so the models were evaluated on
          different subsets of the test set, and the test set was also used to pick checkpoints.
        </Notice>
        <div className="overflow-x-auto rounded-xl border border-line bg-raised">
          <table className="w-full text-sm">
            <caption className="sr-only">Published benchmark</caption>
            <thead className="border-b border-line text-left text-muted">
              <tr>
                <th className="px-4 py-2 font-normal">Model</th>
                <th className="px-4 py-2 font-normal">Parameters</th>
                <th className="px-4 py-2 font-normal">Reported accuracy</th>
                <th className="px-4 py-2 font-normal">Test images actually scored</th>
              </tr>
            </thead>
            <tbody>
              {data.paper.models.map((r) => (
                <tr key={r.model} className="border-b border-line last:border-0">
                  <td className="px-4 py-2 font-medium">{r.model}</td>
                  <td className="px-4 py-2 tabular-nums">{r.params_m}M</td>
                  <td className="px-4 py-2 tabular-nums">{(r.reported_accuracy * 100).toFixed(3)}%</td>
                  <td className="px-4 py-2 tabular-nums">{r.images_scored.toLocaleString()} of 1,311</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 2. audit ------------------------------------------------------------- */}
      {leak && cmp && (
        <section aria-labelledby="audit" className="grid gap-4">
          <h2 id="audit" className="text-2xl font-semibold">2 · What the audit found</h2>
          <p className="max-w-3xl text-ink-2">
            A perceptual- and content-hash comparison of every train/test image pair on the current Kaggle release
            (1,600 test images). Patient IDs are not available, so this is a lower bound on leakage.
          </p>
          <div className="grid gap-4 sm:grid-cols-3">
            <StatTile label="Test images with a near-duplicate in training" value={pct(leak.near_duplicate_rate)}
                      sub={`${leak.near_duplicates_in_test} of ${leak.n_test}`} />
            <StatTile label="Exact pixel duplicates across the split" value={String(leak.exact_duplicates_in_test)} />
            <StatTile label="Novel test images (no near-duplicate)" value={String(cmp.n_novel)}
                      sub="the more honest in-distribution test" />
          </div>
          <Card>
            <GroupedBars
              title="Accuracy on leaked vs novel test images"
              caption="Both models score much higher on test images that have a near-copy in training."
              categories={["Near-duplicate images", "Novel images"]}
              series={SERIES}
              values={{
                previous: [cmp.previous.near_duplicate_subset_accuracy, cmp.previous.novel_subset_accuracy],
                current: [cmp.current.near_duplicate_subset_accuracy, cmp.current.novel_subset_accuracy],
              }}
            />
          </Card>
        </section>
      )}

      {/* 3. retraining ---------------------------------------------------------- */}
      {cmp && (
        <section aria-labelledby="retrain" className="grid gap-4">
          <h2 id="retrain" className="text-2xl font-semibold">3 · Retrained model vs the paper&apos;s recipe</h2>
          <p className="max-w-3xl text-ink-2">
            Measured on CPU with the serving preprocessing. The current model was selected on a leakage-safe
            validation split and scored on the test set once. Results come from a single training seed.
          </p>
          <Card>
            <GroupedBars
              title="Headline metrics"
              categories={["Test accuracy", "Novel subset", "Glioma recall", "Under corruption", "External MRI set"]}
              series={SERIES}
              values={{
                previous: [cmp.previous.test_accuracy, cmp.previous.novel_subset_accuracy, cmp.previous.glioma_recall,
                           cmp.previous.mean_corrupted_accuracy, cmp.previous.external_accuracy],
                current: [cmp.current.test_accuracy, cmp.current.novel_subset_accuracy, cmp.current.glioma_recall,
                          cmp.current.mean_corrupted_accuracy, cmp.current.external_accuracy],
              }}
            />
          </Card>
          <div className="overflow-x-auto rounded-xl border border-line bg-raised">
            <table className="w-full text-sm">
              <caption className="sr-only">Paired significance</caption>
              <thead className="border-b border-line text-left text-muted">
                <tr><th className="px-4 py-2 font-normal">Paired comparison (exact McNemar)</th><th className="px-4 py-2 font-normal">Only current correct</th><th className="px-4 py-2 font-normal">Only previous correct</th><th className="px-4 py-2 font-normal">p-value</th></tr>
              </thead>
              <tbody>
                {Object.entries(cmp.significance as Record<string, { current_only_correct: number; previous_only_correct: number; exact_mcnemar_p: number }>).map(([k, s]) => (
                  <tr key={k} className="border-b border-line last:border-0">
                    <td className="px-4 py-2 capitalize">{k.replace("_", " ")}</td>
                    <td className="px-4 py-2 tabular-nums">{s.current_only_correct}</td>
                    <td className="px-4 py-2 tabular-nums">{s.previous_only_correct}</td>
                    <td className="px-4 py-2 tabular-nums">{s.exact_mcnemar_p.toExponential(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 4. per class ------------------------------------------------------------ */}
      {cmp && (
        <section aria-labelledby="perclass" className="grid gap-4">
          <h2 id="perclass" className="text-2xl font-semibold">4 · Per-class results (current model)</h2>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <ConfusionMatrix title="Confusion matrix, test set (n = 1,600)"
                               caption="Rows are the true class; the main error is gliomas predicted as other classes."
                               labels={CLASS_LABELS} matrix={cmp.current.confusion_matrix} />
            </Card>
            <Card title="Precision, recall and F1">
              <table className="w-full text-sm">
                <thead className="text-left text-muted"><tr><th className="font-normal">Class</th><th className="font-normal">Precision</th><th className="font-normal">Recall</th><th className="font-normal">F1</th></tr></thead>
                <tbody>
                  {(["glioma", "meningioma", "notumor", "pituitary"] as const).map((c, i) => {
                    const r = cmp.current.per_class[c];
                    return (
                      <tr key={c} className="border-t border-line">
                        <td className="py-1.5">{CLASS_LABELS[i]}</td>
                        <td className="tabular-nums">{pct(r.precision)}</td>
                        <td className="tabular-nums">{pct(r.recall)}</td>
                        <td className="tabular-nums">{r.f1.toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-3 text-sm text-ink-2">
                {cmp.current.glioma_to_notumor} of 400 glioma test images are predicted as “no tumor” — the most
                consequential error for any screening-like use (previous model: {cmp.previous.glioma_to_notumor}).
              </p>
            </Card>
          </div>
        </section>
      )}

      {/* 5. calibration & reliability ---------------------------------------------- */}
      {cmp && (
        <section aria-labelledby="calib" className="grid gap-4">
          <h2 id="calib" className="text-2xl font-semibold">5 · Calibration and reliability rules</h2>
          <div className="grid gap-4 sm:grid-cols-3">
            <StatTile label="Expected calibration error, current" value={cmp.current.ece_raw.toFixed(3)}
                      sub={`after temperature scaling ${cmp.current.ece_calibrated.toFixed(3)}`} />
            <StatTile label="Expected calibration error, previous" value={cmp.previous.ece_raw.toFixed(3)}
                      sub={`after temperature scaling ${cmp.previous.ece_calibrated.toFixed(3)}`} />
            <StatTile label="Calibration temperature (fit on validation)" value={cmp.current_temperature.toFixed(3)} />
          </div>
          {rel && (
            <Card title="Do the reliability levels mean anything?">
              <p className="mb-3 text-sm text-ink-2">
                The workbench assigns every result a level with fixed, deterministic rules (rules {rel.rules_version}).
                Accuracy by level, measured after the rules were fixed:
              </p>
              <table className="w-full text-sm">
                <thead className="text-left text-muted"><tr><th className="font-normal">Split</th><th className="font-normal">Level</th><th className="font-normal">Images</th><th className="font-normal">Share</th><th className="font-normal">Accuracy</th></tr></thead>
                <tbody>
                  {(["val", "test"] as const).flatMap((split) =>
                    (["standard", "caution", "unreliable"] as const).map((lv) => {
                      const r = rel.reliability_levels[split][lv];
                      return (
                        <tr key={split + lv} className="border-t border-line">
                          <td className="py-1.5">{split === "val" ? "Validation" : "Test"}</td>
                          <td className="capitalize">{lv}</td>
                          <td className="tabular-nums">{r.n}</td>
                          <td className="tabular-nums">{pct(r.coverage)}</td>
                          <td className="tabular-nums">{r.accuracy == null ? "—" : pct(r.accuracy)}</td>
                        </tr>
                      );
                    }))}
                </tbody>
              </table>
              <h3 className="mt-5 font-medium">Input-shift heuristic</h3>
              <p className="mt-1 text-sm text-ink-2">
                Mahalanobis distance of the image&apos;s features to the training data. It separates clearly non-MRI
                inputs (AUROC {rel.input_shift.auroc_test_vs_synthetic_non_mri.toFixed(2)}) but{" "}
                <strong>not</strong> MRI from a different source (AUROC {rel.input_shift.auroc_test_vs_external_mri.toFixed(2)},
                close to chance) — so it is shown as a heuristic warning, never as out-of-distribution detection.
              </p>
            </Card>
          )}
        </section>
      )}

      {/* 6. robustness & generalisation --------------------------------------------- */}
      {cmp && (
        <section aria-labelledby="robust" className="grid gap-4">
          <h2 id="robust" className="text-2xl font-semibold">6 · Robustness and generalisation</h2>
          <Card>
            <RobustnessPanels
              title="Accuracy under image corruptions (400 test images)"
              caption="Noise and blur augmentation in retraining mainly fixed the noise collapse."
              series={SERIES}
              data={Object.fromEntries(Object.keys(cmp.current.robustness_by_corruption).map((c) => [c, {
                previous: cmp.previous.robustness_by_corruption[c],
                current: cmp.current.robustness_by_corruption[c],
              }]))}
            />
          </Card>
          <div className="grid gap-4 sm:grid-cols-3">
            <StatTile label="External MRI set accuracy (n = 253)" value={pct(cmp.current.external_accuracy)}
                      sub={`previous ${pct(cmp.previous.external_accuracy)} · tumor vs no tumor`} />
            <StatTile label="External tumor recall" value={pct(cmp.current.external_tumor_recall)}
                      sub={`previous ${pct(cmp.previous.external_tumor_recall)}`} />
            <StatTile label="External no-tumor specificity" value={pct(cmp.current.external_notumor_specificity)}
                      sub={`previous ${pct(cmp.previous.external_notumor_specificity)}`} />
          </div>
          <Notice tone="warning">
            On a separate public MRI dataset the model finds only about half of the tumors. In-distribution accuracy
            does not carry over to other sources; this is the clearest reason the platform is for research only.
          </Notice>
        </section>
      )}

      {/* 7. retrieval ----------------------------------------------------------------- */}
      <section aria-labelledby="rag" className="grid gap-4">
        <h2 id="rag" className="text-2xl font-semibold">7 · Literature retrieval evaluation</h2>
        {retr ? (
          <div className="overflow-x-auto rounded-xl border border-line bg-raised">
            <table className="w-full text-sm">
              <caption className="px-4 pt-3 text-left text-sm text-ink-2">
                {retr.n_questions} labelled questions (known-item questions written from article abstracts, plus
                questions about this project). Recall is a lower bound: other articles may also be relevant.
              </caption>
              <thead className="border-b border-line text-left text-muted">
                <tr><th className="px-4 py-2 font-normal">Configuration</th><th className="px-4 py-2 font-normal">Recall@5</th><th className="px-4 py-2 font-normal">Recall@10</th><th className="px-4 py-2 font-normal">MRR@10</th><th className="px-4 py-2 font-normal">nDCG@10</th><th className="px-4 py-2 font-normal">p50 latency</th></tr>
              </thead>
              <tbody>
                {Object.entries(retr.configs as Record<string, { [k: string]: number }>).map(([k, r]) => (
                  <tr key={k} className="border-b border-line last:border-0">
                    <td className="px-4 py-2 font-mono text-xs">{k}</td>
                    <td className="px-4 py-2 tabular-nums">{r["recall@5"].toFixed(3)}</td>
                    <td className="px-4 py-2 tabular-nums">{r["recall@10"].toFixed(3)}</td>
                    <td className="px-4 py-2 tabular-nums">{r["mrr@10"].toFixed(3)}</td>
                    <td className="px-4 py-2 tabular-nums">{r["ndcg@10"].toFixed(3)}</td>
                    <td className="px-4 py-2 tabular-nums">{r.latency_ms_p50.toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Notice>The retrieval evaluation has not been run on this deployment.</Notice>
        )}
      </section>
    </div>
  );
}
