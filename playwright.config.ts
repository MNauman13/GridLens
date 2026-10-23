import { defineConfig, devices } from "playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"], ["html", {outputFolder:"reports/local/playwright-report",open:"never"}]],
  outputDir: "reports/local/browser-results",
  use: {baseURL:"http://127.0.0.1:3000",trace:"retain-on-failure",screenshot:"only-on-failure"},
  projects: [{name:"chromium-desktop",use:{...devices["Desktop Chrome"]}}, {name:"chromium-mobile",use:{...devices["iPhone 13"],browserName:"chromium"}}],
  webServer: [
    {command:"uv run --frozen --extra api --extra ml python -m uvicorn gridlens.api:app --host 127.0.0.1 --port 8000",url:"http://127.0.0.1:8000/v1/health/live",reuseExistingServer:!process.env.CI,timeout:60_000},
    {command:"npm run start",url:"http://127.0.0.1:3000",reuseExistingServer:!process.env.CI,timeout:60_000},
  ],
});
