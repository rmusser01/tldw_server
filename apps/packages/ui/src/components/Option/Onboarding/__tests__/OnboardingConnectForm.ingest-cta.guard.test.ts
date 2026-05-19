import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readOnboardingSource = () =>
  fs.readFileSync(
    path.resolve(
      __dirname,
      "..",
      "OnboardingConnectForm.tsx"
    ),
    "utf8"
  )

describe("OnboardingConnectForm ingest CTA route guard", () => {
  it("navigates the ingest CTA to a layout route that can host Quick Ingest", () => {
    const source = readOnboardingSource()

    expect(source).toContain('await finishAndNavigate("/media", { openQuickIngestIntro: true })')
    expect(source).not.toContain('await finishAndNavigate("/", { openQuickIngestIntro: true })')
    expect(source).not.toContain("document.querySelector('[data-testid=\"open-quick-ingest\"]')")
  })
})
