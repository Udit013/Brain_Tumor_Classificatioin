"""Reproducible research reports (Markdown and standalone HTML).

Language is fixed: "model-predicted class", "research interpretation", never
"diagnosis". A human review, when present, is recorded as a research review —
explicitly not clinical validation.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from ..schemas import AnalysisOut, ModelInfo

DISCLAIMER = ("Research and education use only. This is an experimental model's output, not a "
              "diagnosis; it has no regulatory clearance and must not inform anyone's care. "
              "Literature excerpts provide research context and do not verify this image.")
DISPLAY = {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no tumor",
           "pituitary": "pituitary tumor"}


def render_markdown(a: AnalysisOut, model: ModelInfo | None) -> str:
    out = [f"# NeuroClass research report — analysis `{a.id}`", "", f"> {DISCLAIMER}", ""]
    out += ["## Provenance", "",
            f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"- Analysis created: {a.created_at:%Y-%m-%d %H:%M UTC}; status: **{a.status.value}**",
            f"- Input: {'example image `' + a.sample_id + '` (Kaggle test split, CC BY 4.0)' if a.sample_id else 'user upload (image not included in this report)'}"]
    if model:
        out += [f"- Model: {model.model_name} `{model.model_version}` "
                f"(ONNX sha256 `{model.onnx_sha256[:16]}…`, weights sha256 "
                f"`{model.source_weights_sha256[:16]}…`, Hub `{model.hub_repo}`)",
                f"- Preprocessing: {model.preprocessing}",
                f"- Calibration temperature: {model.temperature:.3f}"]
    if a.reliability:
        out.append(f"- Reliability rules: `{a.reliability.rules_version}`")
    if a.explanation:
        e = a.explanation
        out.append(f"- Explanation mode: **{e.mode}**" + (f" (`{e.llm_model}`)" if e.llm_model else
                                                         " (no language model; quoted extracts)"))
    out.append("")

    if a.inference:
        inf = a.inference
        out += ["## Model output", "",
                f"**Model-predicted class: {DISPLAY[inf.predicted_class]}** — calibrated confidence "
                f"{inf.calibrated_confidence:.1%} (uncalibrated {inf.raw_confidence:.1%}).", "",
                "| Class | Calibrated | Uncalibrated |", "|---|---|---|"]
        for p in sorted(inf.probabilities, key=lambda p: -p.calibrated):
            out.append(f"| {DISPLAY[p.label]} | {p.calibrated:.1%} | {p.raw:.1%} |")
        out += ["", f"- Stability: {inf.tta_agreement} of {inf.tta_views} augmented views agree "
                f"(predictive entropy {inf.tta_entropy:.3f}).",
                f"- Input-shift heuristic: distance {inf.ood_distance:.1f} vs threshold "
                f"{inf.ood_threshold:.1f} → {'flagged' if inf.ood_flag else 'not flagged'}.",
                f"- Input: {inf.input_quality.width}×{inf.input_quality.height} px, "
                f"{inf.input_quality.format or 'unknown format'}.",
                f"- Latency: {inf.latency_ms.get('total', 0):.0f} ms on CPU.", ""]
    if a.reliability:
        out += [f"## Reliability assessment: {a.reliability.level}", ""]
        for f in a.reliability.findings:
            out.append(f"- **{f.severity.value}** `{f.code}` — {f.message}")
        out.append("")
    if a.explanation:
        e = a.explanation
        out += ["## Research interpretation", "", e.summary, ""]
        for s in e.statements:
            cites = "".join(f" [{c}]" for c in s.citations)
            out.append(f"- ({s.kind.replace('_', ' ')}) {s.text}{cites}")
        if e.removed_statements:
            out += ["", "### Statements removed by verification", ""]
            for s in e.removed_statements:
                out.append(f"- ~~{s.text}~~ — {s.reason}")
        if e.passages:
            out += ["", "## Sources", ""]
            for p in e.passages:
                year = f" ({p.year})" if p.year else ""
                out.append(f"- [{p.chunk_id}] *{p.title}*{year}, section “{p.section}”. "
                           f"{p.source_url} — licence: {p.license}")
        out.append("")
    out += ["## Review", ""]
    if a.reviews:
        for r in a.reviews:
            out.append(f"- {r.created_at:%Y-%m-%d %H:%M UTC} — **{r.decision.value}** by {r.reviewer}"
                       + (f": {r.note}" if r.note else ""))
        out.append("")
        out.append("A review records a researcher's assessment of this report. It is not clinical "
                   "validation and does not make the output a diagnosis.")
    else:
        out.append("Not reviewed." + (" Review was recommended by the reliability rules."
                                      if a.reliability and a.reliability.requires_review else ""))
    out += ["", "## Known limitations of the model", "",
            "- Trained and tested on one public Kaggle compilation with documented train/test "
            "overlap; accuracy on images without a near-duplicate in training is lower (see the "
            "project's model card).",
            "- Glioma is the weakest class; some tumors are predicted as 'no tumor'.",
            "- On a separate public MRI dataset accuracy drops substantially (external validation).",
            "- Results come from a single training seed.", ""]
    if a.trace:
        out += ["## Execution trace", "", "| Step | Status | ms |", "|---|---|---|"]
        for t in a.trace:
            out.append(f"| {t.step} | {t.status} | {'' if t.duration_ms is None else f'{t.duration_ms:.0f}'} |")
        out.append("")
    return "\n".join(out)


def render_html(markdown_text: str, title: str) -> str:
    """Minimal standalone HTML (no external assets) from the Markdown above."""
    import re

    lines, body, in_table, in_list = markdown_text.splitlines(), [], False, False

    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"~~(.+?)~~", r"<del>\1</del>", t)
        t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', t)

    for line in lines:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                body.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        if line.startswith("### "):
            body.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            body.append(f'<p class="notice">{inline(line[2:])}</p>')
        elif line.strip():
            body.append(f"<p>{inline(line)}</p>")
    if in_table:
        body.append("</table>")
    if in_list:
        body.append("</ul>")
    css = ("body{font:15px/1.55 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;"
           "color:#1c2430}table{border-collapse:collapse}td,th{border:1px solid #c9d1db;padding:4px 8px}"
           ".notice{border-left:4px solid #b45309;padding:8px 12px;background:#fdf6ec}code{font-size:90%}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}"
            f"</title><style>{css}</style></head><body>{''.join(body)}</body></html>")
