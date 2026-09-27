import { defineConfig } from "@playwright/test"
import path from "node:path"

const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim()
  if (!value)
    throw new Error(`${name} is required for Skills extension certification`)
  return value
}

if (process.env.TLDW_E2E_SKIP_EXTENSION_BUILD !== "1") {
  throw new Error(
    "TLDW_E2E_SKIP_EXTENSION_BUILD must equal 1 for Skills extension certification"
  )
}

const reportPath = requireEnv("TLDW_SKILLS_CERT_EXTENSION_REPORT")
const outputDir = requireEnv("TLDW_SKILLS_CERT_EXTENSION_OUTPUT")
requireEnv("TLDW_SKILLS_CERT_EXTENSION_RESULT")
requireEnv("TLDW_SKILLS_CERT_EXTENSION_LEDGER")
requireEnv("TLDW_SKILLS_CERT_EXTENSION_PROFILE_ROOT")

export default defineConfig({
  testDir: "tests/e2e",
  testMatch: "skills.live-certification.spec.ts",
  timeout: 180_000,
  retries: 0,
  workers: 1,
  fullyParallel: false,
  forbidOnly: true,
  reporter: [["line"], ["json", { outputFile: reportPath }]],
  outputDir,
  use: {
    trace: "off",
    video: "off",
    screenshot: "only-on-failure"
  },
  globalSetup: path.resolve(__dirname, "tests/e2e/setup/build-extension.ts")
})
