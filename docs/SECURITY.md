# Security, privacy and safety

NeuroClass is a research and education tool. It is **not a medical device**,
has no regulatory clearance, and must not be used for any decision about a
person's care. Do not upload images that identify a patient.

## Threat model (what we defend against)

| Threat | Defence |
|---|---|
| Prompt injection via retrieved papers or user text | Passages, questions and notes are HTML-escaped and wrapped in `<passage>`, `<question>`, `<user_note>` tags; the system prompt says text inside them is data and never instructions; tools are read-only and closed-world (`search_literature`, `get_model_facts`); the model's output is untrusted and verified before display |
| Fabricated citations or numbers | `agent.verify`: literature statements need citations; cited ids must have been retrieved; numbers must appear in the cited passages (literature) or in the inference/reliability data (model statements); low lexical support is rejected; a summary with untraceable numbers is withheld. Removed statements are shown with the reason |
| Presenting predictions as diagnoses | prompts forbid it; UI and reports say "model-predicted class"; a test fails if result pages use "diagnosis" |
| Malicious uploads (bombs, wrong types) | size cap before decoding (10 MB), pixel cap (25 MP), format whitelist by content (JPEG/PNG), minimum size, decoding errors mapped to 400 |
| Reading other people's analyses | every analysis has an owner key (user id or a hash of the anonymous client id); other owners get 404 |
| Account attacks | scrypt password hashing (stdlib), HS256 JWTs with issuer and expiry, rate-limited sign-up/login, production refuses to start without a secret |
| Privilege escalation to reviewer | reviewer role requires a listed email **and** the operator's secret reviewer code; without a configured code nobody becomes a reviewer |
| Abuse / resource exhaustion | per-client sliding-window rate limits, inference and LLM timeouts, bounded retries, bounded agent loops, LLM call budgets |
| Unexpected spend | no paid provider unless `NEUROCLASS_ALLOW_PAID_LLM=true`; free-tier quota guard; see [FREE_TIER.md](FREE_TIER.md) |
| Secrets in the repository | keys are environment variables or Space secrets; `.env*`, `.vercel/`, credentials and local notes are gitignored |
| Vulnerable dependencies | `pip-audit` and `npm audit --omit=dev` in CI; Pillow 13 adopted after measuring that it changes no prediction |

## Privacy and data handling

- **Images.** The API stores an upload (mode 0600) only while its analysis runs
  and deletes it after 24 h (hourly job). Images are never logged and never sent to an LLM. The Grad-CAM
  overlay is stored with the analysis.
- **Logs.** Structured JSON access logs carry request id, method, path, status
  and duration — no bodies, file names or image data.
- **LLM data flow.** Only with a configured key: model numbers, reliability
  findings, public literature excerpts, and the user's question or note go to
  the Gemini API. On the free tier Google may use this content to improve its
  products; the assistant page says so. Without a key nothing leaves the server.
- **Anonymous use.** The browser generates a random client id kept in
  `localStorage`; the server stores only its hash.
- **Deletion.** Operators can delete rows; on the free Space without a database
  everything resets when it restarts.

## Known limitations (stated, not hidden)

- **No email verification.** Emails are not verified (that would need a mail
  provider). The reviewer role therefore needs both a listed email and the
  operator's reviewer code (`NEUROCLASS_REVIEWER_CODE`, compared in constant
  time); a listed email alone grants nothing, so registering someone else's
  address first gains no privilege. Ordinary accounts only see their own data.
- **Rate limits are per process and trust forwarded addresses.** Behind the HF
  proxy the client address comes from `X-Forwarded-For`, which a client can
  influence; limits slow casual abuse but are not a security boundary. LLM
  budgets are global, so spoofing cannot raise LLM use.
- **Tokens live in `localStorage`**, readable by any script on the page. The
  web app has no user-generated HTML and sets `nosniff`, `X-Frame-Options`,
  `Referrer-Policy` and `Permissions-Policy`, but no Content-Security-Policy
  yet (Next.js needs per-request nonces for that, which would make every page
  dynamic).
- **The verifier is lexical.** It catches invented ids and numbers and weakly
  supported sentences; it cannot judge whether a well-supported sentence is
  used in a misleading way.
- **The input-shift heuristic is not OOD detection** (AUROC 0.52 against a
  different MRI dataset; see [EVALUATION.md](EVALUATION.md)).

## Reporting

Open a GitHub issue without sensitive details, or contact the maintainer
through the profile linked in the repository, for anything security-related.
