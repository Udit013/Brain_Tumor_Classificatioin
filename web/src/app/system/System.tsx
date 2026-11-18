"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorState, Notice, Pill, Spinner } from "@/components/ui";

type Ready = { status: string; checks: Record<string, Record<string, unknown>> };

export function System() {
  const [ready, setReady] = useState<Ready | null>(null);
  const [manifest, setManifest] = useState<Record<string, any> | null>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.ready().then(setReady).catch(async (e) => {
      setError(e.message);
    });
    api.manifest().then(setManifest).catch(() => setManifest(null));
  }, []);

  const label: Record<string, string> = {
    database: "Database", vision: "Vision model", retrieval: "Literature index", llm: "Language model",
  };

  return (
    <div className="grid gap-6">
      <Card title="Live system status">
        {error && <ErrorState error={error} />}
        {!ready && !error && <Spinner label="Checking subsystems…" />}
        {ready && (
          <ul className="grid gap-2 sm:grid-cols-2">
            {Object.entries(ready.checks).map(([k, v]) => (
              <li key={k} className="rounded-lg border border-line px-3 py-2 text-sm">
                <span className="flex items-center justify-between">
                  <span className="font-medium">{label[k] ?? k}</span>
                  <span className={v.ok ? "text-good-ink" : "text-critical-ink"}>
                    <span aria-hidden>{v.ok ? "● " : "■ "}</span>{v.ok ? "available" : "unavailable"}
                  </span>
                </span>
                <span className="mt-1 block font-mono text-xs text-muted">
                  {flatten(v).join(" · ")}
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className="mt-3 text-sm text-ink-2">
          {ready?.checks.database?.persistent === false && (
            <span className="mb-2 block"><Notice tone="warning">
              This deployment stores history on a temporary disk (free hosting): analyses disappear when the server
              restarts. Configure a free Postgres database to keep them.
            </Notice></span>
          )}
          Subsystems fail independently: without the literature index analyses still run; without a language model,
          explanations are extractive quotations; benchmark pages need only the API.
        </p>
      </Card>

      <Card title="Model" aside={manifest && <Pill>{manifest.model_version}</Pill>}>
        {!manifest && <p className="text-sm text-ink-2">Model manifest unavailable.</p>}
        {manifest && (
          <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[12rem_1fr]">
            <dt className="text-muted">Architecture</dt>
            <dd>EfficientNetB3 (ImageNet-pretrained), max-pool, dense head 512 → 256 → 4; two-stage fine-tune</dd>
            <dt className="text-muted">Classes</dt><dd>{manifest.class_names.join(", ")}</dd>
            <dt className="text-muted">Preprocessing</dt><dd>{manifest.input.preprocessing}</dd>
            <dt className="text-muted">Calibration</dt><dd>temperature {manifest.temperature.toFixed(3)} (fit on the validation split)</dd>
            <dt className="text-muted">Serving</dt><dd>ONNX Runtime on CPU; Grad-CAM computed analytically (no TensorFlow)</dd>
            <dt className="text-muted">Export verification</dt>
            <dd>
              probabilities within {manifest.verification.probs_max_abs_diff_vs_tensorflow.toExponential(1)} of
              TensorFlow on {manifest.verification.n_test_images} test images
              ({manifest.verification.argmax_disagreements_vs_tensorflow} changed predictions); Grad-CAM within{" "}
              {manifest.verification.gradcam_max_abs_diff_vs_tensorflow.toExponential(1)}
            </dd>
            <dt className="text-muted">Input-shift heuristic</dt>
            <dd>{manifest.ood.method}; threshold {manifest.ood.threshold.toFixed(1)} ({manifest.ood.threshold_rule})</dd>
            <dt className="text-muted">Weights SHA-256</dt><dd className="break-all font-mono text-xs">{manifest.source_weights_sha256}</dd>
            <dt className="text-muted">ONNX SHA-256</dt><dd className="break-all font-mono text-xs">{manifest.files["neuroclass_serving.onnx"]}</dd>
            <dt className="text-muted">Versioned on</dt>
            <dd><a href={`https://huggingface.co/${manifest.hub_repo}`}>{manifest.hub_repo}</a> (previous model at tag parity-2024)</dd>
          </dl>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Data and licences">
          <ul className="grid gap-2 text-sm text-ink">
            <li><strong>Images:</strong> Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle), CC BY 4.0 — a compilation of figshare, SARTAJ and Br35H; 5,600 train / 1,600 test images in the current release.</li>
            <li><strong>Split:</strong> 4,759 training / 841 validation images; near-duplicate clusters never straddle the split.</li>
            <li><strong>External check:</strong> a separate public tumor/no-tumor MRI set (n = 253).</li>
            <li><strong>Literature:</strong> 241 PubMed Central open-access articles whose XML declares a Creative Commons licence (no-derivatives licences excluded), plus this project&apos;s documents (MIT).</li>
          </ul>
        </Card>
        <Card title="Privacy and retention">
          <ul className="grid gap-2 text-sm text-ink">
            <li>Uploads are stored privately (owner-only file permissions), never shared publicly and deleted after 24 hours.</li>
            <li>Analyses are visible only to the browser or account that created them, and to reviewers.</li>
            <li>Logs and traces record steps, timings and versions — never image data, file names or free text.</li>
            <li>Do not upload images that identify a person.</li>
          </ul>
        </Card>
      </div>

      <Card title="Intended use">
        <p className="text-sm text-ink">
          NeuroClass is a research and teaching platform for studying a published classifier, its evaluation and its
          failure modes. It is not a medical device, has no regulatory clearance and must not be used for diagnosis,
          triage or any decision about a person&apos;s care. Disclaimers do not by themselves make a product safe or
          compliant; the limitations shown throughout the platform are part of the result.
        </p>
      </Card>
    </div>
  );
}

function flatten(v: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(v).flatMap(([key, val]) => {
    if (key === "ok" && !prefix) return [];
    if (val && typeof val === "object" && !Array.isArray(val)) return flatten(val as Record<string, unknown>, `${prefix}${key}.`);
    return val === null || val === undefined ? [] : [`${prefix}${key}: ${String(val)}`];
  });
}
