import { defineConfig } from "@playwright/test"

const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim()
  if (!value) throw new Error(`${name} is required for Skills WebUI certification`)
  return value
}

const baseURL = requireEnv("TLDW_WEB_URL")
const reportPath = requireEnv("TLDW_SKILLS_CERT_WEB_REPORT")
const outputDir = requireEnv("TLDW_SKILLS_CERT_WEB_OUTPUT")
requireEnv("TLDW_SKILLS_CERT_WEB_RESULT")

export default defineConfig({
  testDir: ".",
  testMatch: "skills.live.spec.ts",
  timeout: 60_000,
  retries: 0,
  workers: 1,
  fullyParallel: false,
  forbidOnly: true,
  reporter: [["line"], ["json", { outputFile: reportPath }]],
  outputDir,
  use: {
    baseURL,
    trace: "off",
    video: "off",
    screenshot: "only-on-failure",
  },
})
