import { defineConfig } from "@playwright/test"

import { chromiumFixtureOptions } from "./e2e/utils/outbound-observer"

export default defineConfig({
  testDir: "./e2e",
  testMatch: "outbound-privacy.spec.ts",
  timeout: 90_000,
  retries: 0,
  workers: 1,
  use: {
    browserName: "chromium",
    channel: "chromium",
    launchOptions: chromiumFixtureOptions(),
    headless: true,
    locale: "en-US",
    trace: "off",
    screenshot: "off",
    video: "off"
  }
})
