import { defineConfig, devices } from "@playwright/test"

const baseURL = process.env.TLDW_WEB_URL || "http://localhost:18111"

export default defineConfig({
  testDir: ".",
  timeout: 180_000,
  expect: {
    timeout: 30_000,
  },
  retries: 0,
  workers: 1,
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "uat-desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: "uat-mobile",
      use: {
        ...devices["Pixel 7"],
      },
    },
  ],
})
