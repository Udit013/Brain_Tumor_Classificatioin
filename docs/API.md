# API

Interactive docs: `/docs` locally, [`/api/docs`](https://udit013-brain-tumor-mri-classifier.hf.space/api/docs)
on the Space. The schema is exported to `web/openapi.json` (CI fails if it
drifts: `scripts/export_openapi.py --check`) and turned into TypeScript types
(`npm --prefix web run gen:api`).

All responses carry `X-Request-ID`. Errors are `{"detail": ...}` with a
meaningful status code. Model outputs are research predictions, never
diagnoses.

## Identity

| Caller | How | Sees |
|---|---|---|
| Anonymous browser | `X-Client-Id: <≥16 random chars>` (the web app generates one per browser) | its own analyses; review off |
| Signed-in user | `Authorization: Bearer <JWT>` from `/auth/signup` or `/auth/login` | its own analyses; can request review |
| Reviewer | an email listed in `NEUROCLASS_REVIEWER_EMAILS` **and** the operator's `NEUROCLASS_REVIEWER_CODE`, given at sign-up (`reviewer_code`) or later via `POST /auth/reviewer` | the review queue; can approve / request revision / reject |

Requests without either header get 401 on owned resources. Another owner's
analysis is a 404, not a 403. Reviewers can open analyses that are awaiting
review or have been reviewed, and nothing else.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | per-subsystem status: database (backend, persistent), vision, retrieval (index version), llm (provider, model, budget, pause) — 503 when database or vision is down |
| POST | `/analyses` | start an analysis: multipart `file` (JPEG/PNG ≤ 10 MB) **or** `sample_id`; optional `question` (≤ 1,000 chars), `review_mode` (`auto`/`always`/`never`), `Idempotency-Key` header. 202 with the analysis id |
| GET | `/analyses` | your history |
| GET | `/analyses/{id}` | full result: inference, reliability, explanation (statements with citations, removed statements with reasons, `fallback_reason`), reviews, trace |
| GET | `/analyses/{id}/events` | server-sent events: `step` events, then one `status` event |
| GET | `/analyses/{id}/gradcam.png` | the overlay |
| GET | `/analyses/{id}/report?format=md\|html\|json` | downloadable research report |
| DELETE | `/analyses/{id}` | owner only: deletes the analysis, its reviews, trace events, checkpoints and upload (204); 409 while it is still running |
| GET | `/analyses/review-queue` | reviewer only |
| POST | `/analyses/{id}/reviews` | reviewer only: `{"decision": "approve"\|"request_revision"\|"reject", "note": ...}`; resumes the paused graph |
| POST | `/research/search` | ranked passages (routed hybrid search): `{"question", "top_k" ≤ 12, "filters"}` |
| POST | `/research/ask` | verified, cited answer (LLM or extractive) with retrieval/generation timings |
| GET | `/research/documents/{doc_id}` | provenance of one source (title, authors, journal, year, DOI, PMID, licence, URL) |
| GET | `/metrics/benchmarks` | the paper's reported table and the measured evaluations, kept separate |
| GET | `/models/current`, `/models/manifest` | model version, checksums, export verification |
| GET | `/samples`, `/samples/{id}/image` | example images (test split, CC BY 4.0) |
| POST | `/auth/signup` (`email`, `password`, optional `reviewer_code`), `/auth/login`; GET `/auth/me` | accounts |
| POST | `/auth/reviewer` | `{"code": ...}` — upgrades the signed-in account to reviewer if its email is listed and the code matches; 403 otherwise |

Status codes worth knowing: 413 upload too large; 400 unusable image (with a
machine-readable `code`: `empty_file`, `file_too_large`,
`unsupported_format`, `too_many_pixels`, `corrupt_image`, `too_small`); 422 neither or both of `file`/`sample_id`; 403
review requested without signing in; 409 report or review requested in the
wrong state; 503 `the database is temporarily unavailable` (e.g. Neon suspended at its
free limit — benchmarks and literature search keep working); 429 rate
limited (per client: 10 analyses/min, 60 research calls/min), 503 subsystem
unavailable.

## Example

```bash
curl -s -X POST https://udit013-brain-tumor-mri-classifier.hf.space/api/analyses \
  -H 'X-Client-Id: my-own-random-client-id-123' -F sample_id=glioma
```
```bash
curl -s -X POST https://udit013-brain-tumor-mri-classifier.hf.space/api/research/ask \
  -H 'content-type: application/json' -H 'X-Client-Id: my-own-random-client-id-123' \
  -d '{"question": "Why can Grad-CAM maps be misleading?"}'
```

## MCP server (local)

`python -m neuroclass.mcp_server` (stdio) exposes read-only tools —
`get_benchmarks`, `get_model_info`, `get_reproducibility`, `search_literature`,
`get_analysis` (only for the account in `NEUROCLASS_MCP_TOKEN`) — all annotated
read-only, idempotent and closed-world. No uploads, writes or file access. It
is not deployed on the Space.
