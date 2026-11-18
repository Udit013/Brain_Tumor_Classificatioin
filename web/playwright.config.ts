import { defineConfig, devices } from "@playwright/test";

/**
 * Browser-level journeys against a running stack (real model, real index):
 *   API  on NEUROCLASS_E2E_API (default http://localhost:8000)
 *   web  on NEUROCLASS_E2E_WEB (default http://localhost:3000)
 * Start them with `docker compose -f deploy/docker-compose.yml up` or the dev
 * servers, then `npm run test:e2e`.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  reporter: [["list"]],
  use: { baseURL: process.env.NEUROCLASS_E2E_WEB ?? "http://localhost:3000", trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});
