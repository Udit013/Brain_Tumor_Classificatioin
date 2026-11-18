"""Reproducible corpus ingestion: PMC open-access articles + project documents.

Licensing rule: an article enters the corpus only if its own JATS XML declares
a Creative Commons licence; no-derivatives variants (ND) are excluded to stay
conservative about storing and displaying excerpts. Every document keeps its
licence, source URL, PMCID/PMID/DOI and a SHA-256 of its normalised text.

Retrieved text is untrusted: it is stripped of markup and control characters
here, and passed to any LLM only as quoted data (see rag/prompting.py).

Usage (platform environment; network access to NCBI E-utilities):
    python -m neuroclass.rag.corpus fetch      # -> data/corpus/pmc_raw/*.xml
    python -m neuroclass.rag.corpus build      # -> data/corpus/documents.jsonl
                                               #    results/rag/corpus_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request

from lxml import etree

from ..settings import REPO_ROOT

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
RAW_DIR = CORPUS_DIR / "pmc_raw"
DOCS_PATH = CORPUS_DIR / "documents.jsonl"
MANIFEST_PATH = REPO_ROOT / "results" / "rag" / "corpus_manifest.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA = ' AND "open access"[filter]'

# topic -> PubMed Central query (field-restricted, relevance-sorted)
TOPIC_QUERIES: dict[str, str] = {
    "brain_tumor_classification":
        '("brain tumor"[tiab] OR "brain tumour"[tiab] OR glioma[tiab] OR meningioma[tiab] '
        'OR pituitary[tiab]) AND (MRI[tiab] OR "magnetic resonance"[tiab]) AND '
        '("deep learning"[tiab] OR "convolutional neural network"[tiab] OR CNN[tiab]) '
        'AND classification[tiab]',
    "transfer_learning":
        '"transfer learning"[tiab] AND (EfficientNet[tiab] OR VGG16[tiab] OR InceptionV3[tiab] '
        'OR ImageNet[tiab]) AND ("medical image"[tiab] OR "medical images"[tiab] OR MRI[tiab])',
    "explainability":
        '("Grad-CAM"[tiab] OR "gradient-weighted class activation"[tiab] OR saliency[tiab]) AND '
        '("medical imaging"[tiab] OR radiology[tiab] OR MRI[tiab]) AND '
        '(explainab*[tiab] OR interpretab*[tiab])',
    "calibration_uncertainty":
        '(calibration[tiab] OR "temperature scaling"[tiab] OR "uncertainty estimation"[tiab] OR '
        '"predictive uncertainty"[tiab] OR "uncertainty quantification"[tiab]) AND '
        '"deep learning"[tiab] AND ("medical image"[tiab] OR "medical imaging"[tiab])',
    "data_leakage":
        '("data leakage"[tiab] OR "information leakage"[tiab] OR "patient-level"[tiab] OR '
        '"subject-level"[tiab] OR "duplicate images"[tiab] OR "train-test overlap"[tiab]) AND '
        '("deep learning"[tiab] OR "machine learning"[tiab]) AND '
        '(imaging[tiab] OR MRI[tiab] OR radiology[tiab])',
    "generalization":
        '("domain shift"[tiab] OR "external validation"[tiab] OR "dataset shift"[tiab] OR '
        '"generalizability"[tiab] OR "generalisability"[tiab]) AND "deep learning"[tiab] AND '
        '(MRI[tiab] OR "medical imaging"[tiab])',
    "class_imbalance":
        '("class imbalance"[tiab] OR "imbalanced data"[tiab] OR "imbalanced dataset"[tiab]) AND '
        '"deep learning"[tiab] AND ("medical image"[tiab] OR "medical imaging"[tiab])',
    "robustness":
        '("test-time augmentation"[tiab] OR "image corruption"[tiab] OR "robustness"[tiab] OR '
        '"noise"[tiab] OR "perturbation"[tiab]) AND "deep learning"[tiab] AND MRI[tiab] AND '
        '(classification[tiab] OR segmentation[tiab])',
    "ood_detection":
        '("out-of-distribution"[tiab] OR "novelty detection"[tiab] OR "anomaly detection"[tiab]) '
        'AND ("medical imaging"[tiab] OR MRI[tiab]) AND "deep learning"[tiab]',
    "reporting_guidelines":
        '(CLAIM[tiab] OR TRIPOD[tiab] OR "reporting guideline"[tiab] OR checklist[tiab]) AND '
        '"artificial intelligence"[tiab] AND (radiology[tiab] OR "medical imaging"[tiab])',
}
PER_TOPIC = 30
ALLOWED_LICENSES = ("by", "by-sa", "by-nc", "by-nc-sa", "zero", "publicdomain")


def _get(url: str, params: dict) -> bytes:
    query = urllib.parse.urlencode({**params, "tool": "neuroclass-research-platform"})
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "neuroclass/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"NCBI request failed: {url}")


def fetch(per_topic: int = PER_TOPIC) -> dict[str, list[str]]:
    """Search each topic and cache raw JATS XML per article."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    topics: dict[str, list[str]] = {}
    for topic, query in TOPIC_QUERIES.items():
        data = json.loads(_get(f"{EUTILS}/esearch.fcgi", {
            "db": "pmc", "term": query + OA, "retmode": "json", "retmax": per_topic,
            "sort": "relevance"}))
        topics[topic] = data["esearchresult"]["idlist"]
        time.sleep(0.4)                                  # NCBI: <= 3 requests/s without a key
    ids = sorted({i for v in topics.values() for i in v} -
                 {p.stem for p in RAW_DIR.glob("*.xml")})
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        xml = _get(f"{EUTILS}/efetch.fcgi", {"db": "pmc", "id": ",".join(batch)})
        root = etree.fromstring(xml, parser=etree.XMLParser(huge_tree=True, resolve_entities=False,
                                                            no_network=True))
        for art in root.iter("article"):
            pmc_uid = next((a.text for a in art.iter("article-id")
                            if a.get("pub-id-type") in ("pmcid", "pmc")), None)
            if not pmc_uid:
                continue
            uid = pmc_uid.removeprefix("PMC")
            (RAW_DIR / f"{uid}.xml").write_bytes(etree.tostring(art))
        time.sleep(0.4)
    (CORPUS_DIR / "topics.json").write_text(json.dumps(topics, indent=2))
    return topics


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏ -‮]")
_WS = re.compile(r"\s+")


_EMPTY_REFS = re.compile(r"\s*[\[(]\s*(?:[,;–\-]\s*)*[\])]")


def clean_text(text: str) -> str:
    """NFKC, no control characters, no empty brackets left by removed citation markers."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = _EMPTY_REFS.sub("", text)
    return _WS.sub(" ", text).strip()


def _text(el) -> str:
    if el is None:
        return ""
    # drop citation markers' brackets but keep inline text; skip formulas and tables
    for bad in el.iter("table-wrap", "disp-formula", "fig", "inline-formula", "xref"):
        if bad.tag == "xref":
            bad.text = ""
            continue
        bad.getparent().remove(bad) if bad.getparent() is not None else None
    return clean_text("".join(el.itertext()))


def license_of(art) -> tuple[str | None, str | None]:
    """(normalised CC code, licence URL) or (None, None)."""
    lic = art.find(".//permissions/license")
    if lic is None:
        return None, None
    url = lic.get("{http://www.w3.org/1999/xlink}href") or ""
    for ref in lic.iter("{http://www.niso.org/schemas/ali/1.0/}license_ref"):
        url = url or (ref.text or "")
    url = url.strip()
    m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)", url.lower())
    if not m:
        return None, url or None
    return m.group(2), url


def parse_article(xml: bytes) -> dict | None:
    art = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    code, url = license_of(art)
    if code is None or code not in ALLOWED_LICENSES:
        return None
    ids = {a.get("pub-id-type"): (a.text or "").strip() for a in art.iter("article-id")}
    pmcid = ids.get("pmcid") or ("PMC" + ids.get("pmc", ""))
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid
    meta = art.find(".//article-meta")
    title = _text(meta.find(".//title-group/article-title")) if meta is not None else ""
    journal = _text(art.find(".//journal-meta//journal-title"))
    year = None
    for pd in art.iter("pub-date"):
        y = pd.findtext("year")
        if y and y.isdigit():
            year = int(y) if year is None else min(year, int(y))
    authors = []
    for c in art.iter("contrib"):
        if c.get("contrib-type") == "author":
            sur, given = c.findtext(".//surname"), c.findtext(".//given-names")
            if sur:
                authors.append(f"{sur} {given[0] + '.' if given else ''}".strip())
    sections: list[dict] = []
    abstract = meta.find("abstract") if meta is not None else None
    if abstract is not None:
        paras = [_text(p) for p in abstract.iter("p")]
        sections.append({"title": "Abstract", "paragraphs": [p for p in paras if p]})
    body = art.find("body")
    if body is not None:
        for sec in body.findall("sec"):
            heading = _text(sec.find("title")) or "Body"
            paras = [_text(p) for p in sec.iter("p")]
            paras = [p for p in paras if len(p.split()) >= 8]
            if paras and not re.search(r"(?i)^(acknowledg|funding|conflict|author contrib|"
                                       r"data availability|supplementary|ethic)", heading):
                sections.append({"title": heading[:200], "paragraphs": paras})
    if not title or not sections:
        return None
    full = "\n".join(p for s in sections for p in s["paragraphs"])
    return {
        "id": pmcid, "source_type": "pmc_article", "title": title, "journal": journal or None,
        "year": year, "authors": ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""),
        "pmid": ids.get("pmid"), "doi": ids.get("doi"),
        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "license": f"CC {code.upper()}" if code not in ("zero", "publicdomain") else "CC0",
        "license_url": url, "sections": sections,
        "content_sha256": hashlib.sha256(full.encode()).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Project documents (MIT-licensed, authored in this repository)
# --------------------------------------------------------------------------- #
PROJECT_DOCS = {
    "project:readme": ("README.md", "NeuroClass README — research foundation, audit and results"),
    "project:model_card": ("MODEL_CARD.md", "NeuroClass model card"),
    "project:documentation": ("PROJECT_DOCUMENTATION_NeuroClass.md",
                              "NeuroClass project documentation"),
}
GITHUB_BLOB = "https://github.com/Udit013/Brain_Tumor_Classificatioin/blob/main/"


def parse_markdown(doc_id: str, rel_path: str, title: str) -> dict:
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    sections, current, paras, buf = [], "Introduction", [], []

    def flush_para():
        if buf:
            p = clean_text(" ".join(buf))
            p = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", p)               # images
            p = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", p)           # links -> text
            p = re.sub(r"<!--.*?-->", "", p).replace("**", "").replace("`", "")
            if len(p.split()) >= 6:
                paras.append(p)
            buf.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            flush_para()
            continue
        if in_code:
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush_para()
            if paras:
                sections.append({"title": current, "paragraphs": paras})
            current, paras = clean_text(m.group(2).replace("`", "")), []
        elif not line.strip():
            flush_para()
        else:
            buf.append(line.strip().lstrip("> ").lstrip("- "))
    flush_para()
    if paras:
        sections.append({"title": current, "paragraphs": paras})
    full = "\n".join(p for s in sections for p in s["paragraphs"])
    return {"id": doc_id, "source_type": "project_document", "title": title, "journal": None,
            "year": 2026, "authors": "NeuroClass project", "pmid": None, "doi": None,
            "source_url": GITHUB_BLOB + rel_path, "license": "MIT", "license_url": None,
            "sections": sections, "content_sha256": hashlib.sha256(full.encode()).hexdigest()}


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def build() -> dict:
    """Parse cached XML + project docs into documents.jsonl and a tracked manifest."""
    topics = json.loads((CORPUS_DIR / "topics.json").read_text())
    topic_of: dict[str, list[str]] = {}
    for topic, ids in topics.items():
        for uid in ids:
            topic_of.setdefault(uid, []).append(topic)
    docs, rejected, seen_titles = [], {"license": 0, "unparseable": 0, "duplicate": 0}, set()
    for path in sorted(RAW_DIR.glob("*.xml")):
        doc = parse_article(path.read_bytes())
        if doc is None:
            rejected["license" if license_of(etree.fromstring(path.read_bytes()))[0]
                     not in ALLOWED_LICENSES else "unparseable"] += 1
            continue
        key = _norm_title(doc["title"])
        if key in seen_titles:
            rejected["duplicate"] += 1
            continue
        seen_titles.add(key)
        doc["topics"] = sorted(topic_of.get(path.stem, []))
        docs.append(doc)
    for doc_id, (rel, title) in PROJECT_DOCS.items():
        if (REPO_ROOT / rel).exists():
            d = parse_markdown(doc_id, rel, title)
            d["topics"] = ["project"]
            docs.append(d)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest = {
        "license_rule": "Creative Commons licence declared in the article XML; ND excluded",
        "queries": TOPIC_QUERIES, "per_topic": PER_TOPIC,
        "n_documents": len(docs), "rejected": rejected,
        "documents": [{k: d[k] for k in ("id", "title", "year", "journal", "license",
                                         "source_url", "content_sha256", "topics")} for d in docs],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "build"])
    args = parser.parse_args()
    if args.command == "fetch":
        topics = fetch()
        print({k: len(v) for k, v in topics.items()})
    else:
        m = build()
        print(json.dumps({k: m[k] for k in ("n_documents", "rejected")}, indent=2))


if __name__ == "__main__":
    main()
