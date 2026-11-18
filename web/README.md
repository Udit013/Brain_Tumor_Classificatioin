# NeuroClass web app

Next.js 16 (App Router) + TypeScript + Tailwind v4 frontend for the NeuroClass
research platform. Research and education only — not a medical device.

Live: https://neuroclass-gilt.vercel.app (Vercel Hobby), calling the API on the
free Hugging Face Space.

```bash
npm ci
```
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```
```bash
npm run lint && npm run typecheck && npm test
```
```bash
npm run test:e2e        # needs the API on :8000 and this app on :3000
```

- Pages: `/` landing, `/workbench`, `/analyses/[id]`, `/assistant`, `/history`,
  `/benchmarks`, `/system`, `/review`, `/login`.
- `src/lib/api-types.ts` is generated from `openapi.json`
  (`scripts/export_openapi.py`, then `npm run gen:api`); CI fails if they drift.
- Charts (`src/components/charts.tsx`) all have a table view; status is never
  shown by colour alone.
- E2E targets: `NEUROCLASS_E2E_WEB`, `NEUROCLASS_E2E_API` (defaults localhost).
- Deployment: [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).
