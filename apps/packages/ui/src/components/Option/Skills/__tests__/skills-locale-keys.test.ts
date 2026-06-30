import { describe, expect, it } from "vitest"
import enOption from "@/assets/locale/en/option.json"
import publicEnOption from "@/public/_locales/en/option.json"

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

const getPublicLocaleValue = (source: unknown, key: string): unknown => {
  const publicKey = key.replace(".", "_")
  const entry = getPathValue(source, publicKey)
  return entry && typeof entry === "object"
    ? (entry as Record<string, unknown>).message
    : undefined
}

describe("Skills locale keys", () => {
  it("has required English WebUI and extension option locale keys", () => {
    for (const key of REQUIRED_SKILLS_KEYS) {
      const webuiValue = getPathValue(enOption as unknown, key)
      expect(typeof webuiValue, `Missing or non-string WebUI locale key: ${key}`).toBe("string")
      expect(String(webuiValue).trim().length).toBeGreaterThan(0)

      const publicValue = getPublicLocaleValue(publicEnOption as unknown, key)
      expect(typeof publicValue, `Missing or non-string public locale key: ${key}`).toBe("string")
      expect(String(publicValue).trim()).toBe(String(webuiValue).trim())
    }
  })
})
