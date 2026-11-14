import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("PlaygroundMessage error recovery guard", () => {
  it("keeps retry/switch/continue recovery actions in the error bubble", () => {
    const messageSourcePath = path.resolve(__dirname, "../Message.tsx")
    const messageContentSourcePath = path.resolve(__dirname, "../MessageContent.tsx")
    const messageStateSourcePath = path.resolve(__dirname, "../useMessageState.ts")
    const messageSourcesSectionPath = path.resolve(
      __dirname,
      "../MessageSourcesSection.tsx"
    )
    const messageSource = fs.readFileSync(messageSourcePath, "utf8")
    const messageContentSource = fs.readFileSync(messageContentSourcePath, "utf8")
    const messageStateSource = fs.readFileSync(messageStateSourcePath, "utf8")
    const messageSourcesSectionSource = fs.readFileSync(
      messageSourcesSectionPath,
      "utf8"
    )
    expect(messageSource).toContain("playground:errorRecovery.retrySameModel")
    expect(messageSource).toContain("playground:errorRecovery.switchModel")
    expect(messageSource).toContain("playground:errorRecovery.tryProviderFallback")
    expect(messageSource).toContain("playground:errorRecovery.continueFromPartial")
    expect(messageSource).toContain("playground:errorRecovery.interruptedSummary")
    expect(messageSource).toContain("interruptedGeneration")
    expect(messageSource).not.toContain("Recommended next actions:")
    expect(messageContentSource).toContain("role=\"alert\"")
    expect(messageContentSource).toContain("aria-live=\"assertive\"")
    expect(messageSource).toContain("role=\"status\"")
    expect(messageSource).toContain("aria-live=\"polite\"")
    expect(messageSource).toContain("tldw:open-model-settings")
    expect(messageSourcesSectionSource).toContain("playground:sources.askWithSources")
    expect(messageSourcesSectionSource).toContain("pinnedState")
    expect(messageSource).toContain("message-fallback-audit")
    expect(messageStateSource).toContain("playground:routing.policyAuto")
    expect(messageSource).toContain("playground:routing.attempts")
  })
})
