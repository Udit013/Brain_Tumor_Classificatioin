# Provenance and limitations

## What came from where

| Layer | Origin | Status |
|---|---|---|
| Four-architecture benchmark (CNN, VGG16, InceptionV3, EfficientNetB3) | IEEE ICC-ROBINS 2024, [DOI 10.1109/ICC-ROBINS60238.2024.10533941](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941); notebooks in `legacy/` | preserved byte-for-byte; its 99.844% is the **paper's reported claim**, scored on 639/640 images (49% of the test set), and is shown separately from every measured number |
| Audit, leakage-safe split, retrained model (2026.10-finetune), calibration, robustness, external validation | this repository (`src/btc`), 2026 | measured; see the [model card](../MODEL_CARD.md) |
| TF-free serving, reliability rules, retrieval, LangGraph workflow, verifier, LLM adapters, API, web app, MCP server | this repository (`src/neuroclass`, `web/`), November 2026 | **not part of the IEEE paper**; evaluated in [EVALUATION.md](EVALUATION.md) |

The LLM, retrieval and agent features were added after publication and are not
claims of the paper.

## Data

| Data | Source | Licence | Use |
|---|---|---|---|
| Brain Tumor MRI Dataset (7,200 images as distributed; the paper used 7,023) | Masoud Nickparvar, Kaggle; compiled from figshare CE-MRI, SARTAJ, Br35H | CC BY 4.0 | training, validation, test; the four example images |
| External tumor/no-tumor set (n=253) | Navoneel Chakrabarty, Kaggle | as published by its author | external evaluation only |
| Literature corpus: 241 articles | PubMed Central open-access subset via NCBI E-utilities | Creative Commons declared in each article's JATS XML: CC BY 220, CC BY-NC 17, CC BY-NC-SA 3, CC0 1; **ND licences excluded** (51 articles rejected for licence, 1 duplicate) | retrieval; excerpts are shown with title, year, licence and a link |
| Project documents (README, model card, documentation) | this repository | MIT | retrieval for questions about NeuroClass |
| Test fixture `tests/fixtures/decoder_parity/Te-no_120.jpg` | Kaggle test split | CC BY 4.0 | decoder-parity golden test |

The corpus manifest (`results/rag/corpus_manifest.json`) records the PubMed
queries (10 topics × 30 articles), the licence rule and the rejections; each
indexed document keeps its PMCID, DOI, licence URL and source URL
(`GET /api/research/documents/{id}`).

## Models

| Model | Licence / source | Role |
|---|---|---|
| EfficientNetB3 fine-tune `2026.10-finetune` | this project; weights on the Hub (`Udit013/brain-tumor-efficientnetb3`), previous model at tag `parity-2024` | classifier |
| `BAAI/bge-small-en-v1.5` | MIT | query and passage embeddings (local ONNX) |
| `Xenova/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | optional reranker (off by default) |
| Gemini free-tier model (optional) | Google, Gemini API terms | drafts explanations; output verified |

## Limitations

**Model**
- Accuracy 93.4% on the 1,600-image test set, but 89.4% on the 886 test images
  with no near-duplicate in training, and 72.7% on an external dataset where it
  finds only 56% of tumours. The dataset has no patient ids, so leakage control
  works on image similarity and is a lower bound.
- Glioma is the weakest class (recall 79.5%); 25 of 400 gliomas are labelled
  "no tumor".
- One training seed; no multi-site or prospective evaluation; T1-weighted 2-D
  slices only. Any other image still receives one of the four labels.
- Grad-CAM shows what influenced the score, not where a tumour is.

**Platform**
- The reliability levels are rules over measured statistics, not guarantees:
  results labelled "standard" are still wrong about 4% of the time on test.
- The input-shift heuristic detects non-MRI inputs well (AUROC 0.95) but not MRI
  from a different source (AUROC 0.52).
- Retrieval covers 241 openly licensed articles. Answers are only as good as
  that subset, and relevance labels in the evaluation are incomplete.
- The verifier is lexical; extractive answers are quotations, not synthesis.
- The LLM-vs-RAG generation comparison has not been run (no free-tier key was
  available); only extractive results are reported.
- The free deployment sleeps when idle, keeps history only until restart unless
  a database is configured, and serves modest traffic.
