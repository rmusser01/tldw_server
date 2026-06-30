import { describe, expect, it } from "vitest"
import enOption from "@/assets/locale/en/option.json"

const REQUIRED_SKILLS_KEYS = [
  "skills.testRun",
  "skills.testRunTitle",
  "skills.previewArgs",
  "skills.previewArgsPlaceholder",
  "skills.renderOnlyAction",
  "skills.testRunAction",
  "skills.previewRendered",
  "skills.previewForkOutput",
  "skills.testRunError",
  "skills.loadingStatus",
  "skills.renderingPromptStatus",
  "skills.runningTestStatus",
  "skills.renderedPromptReadyStatus",
  "skills.testResultReadyStatus"
] as const

const getPathValue = (source: unknown, key: string): unknown =>
  key.split(".").reduce<unknown>((value, segment) => {
    if (!value || typeof value !== "object") return undefined
    return (value as Record<string, unknown>)[segment]
  }, source)

describe("Skills locale keys", () => {
  it("has required English WebUI option locale keys", () => {
    for (const key of REQUIRED_SKILLS_KEYS) {
      const value = getPathValue(enOption as unknown, key)
      expect(typeof value, `Missing or non-string locale key: ${key}`).toBe("string")
      expect(String(value).trim().length).toBeGreaterThan(0)
    }
  })
})
