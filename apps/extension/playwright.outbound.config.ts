import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "outbound-privacy.spec.ts",
  timeout: 90_000,
  retries: 0,
  workers: 1,
  use: {
    trace: "off",
    screenshot: "off",
    video: "off"
  }
})
