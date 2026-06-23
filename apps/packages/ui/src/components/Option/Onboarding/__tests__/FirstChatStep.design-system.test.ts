import { readFile } from "node:fs/promises"
import path from "node:path"
import { describe, expect, it } from "vitest"

const guard = await import(
  "../../../../../scripts/design-system-product-state-rules.mjs"
)

const firstChatStepRelativePath =
  "src/components/Option/Onboarding/steps/FirstChatStep.tsx"
const firstChatStepPath = path.resolve(process.cwd(), firstChatStepRelativePath)

describe("FirstChatStep design-system state labels", () => {
  it("routes the retrying label through the design-system state registry", async () => {
    const source = await readFile(firstChatStepPath, "utf8")
    const findings = guard.analyzeSource({
      relativePath: firstChatStepRelativePath,
      source
    })

    expect(findings).not.toContainEqual(
      expect.objectContaining({
        rule: "canonical-state-label",
        subject: "Retrying"
      })
    )
  })
})
