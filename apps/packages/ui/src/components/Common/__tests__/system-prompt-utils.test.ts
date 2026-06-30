import { describe, expect, it, vi } from "vitest"
import {
  normalizeSystemPromptOverrideValue,
  resolveEffectiveSystemPromptState,
  resolveSelectedSystemPromptContent
} from "../system-prompt-utils"

const makePrompt = (content: string) => ({
  id: "prompt-1",
  title: "Prompt",
  content,
  is_system: true,
  createdAt: Date.parse("2026-01-01T00:00:00Z")
})

describe("system prompt utils", () => {
  it("returns selected template content when systemPrompt override is empty", async () => {
    await expect(
      resolveEffectiveSystemPromptState({
        selectedSystemPrompt: "prompt-1",
        systemPrompt: "",
        getPromptByIdFn: vi.fn(async () => makePrompt("Template body"))
      })
    ).resolves.toMatchObject({
      templateContent: "Template body",
      effectiveContent: "Template body",
      overrideActive: false
    })
  })

  it("treats non-empty systemPrompt as the active override", async () => {
    await expect(
      resolveEffectiveSystemPromptState({
        selectedSystemPrompt: "prompt-1",
        systemPrompt: "Conversation override",
        getPromptByIdFn: vi.fn(async () => makePrompt("Template body"))
      })
    ).resolves.toMatchObject({
      templateContent: "Template body",
      effectiveContent: "Conversation override",
      overrideActive: true
    })
  })

  it("falls back to an empty reset value when template lookup fails", async () => {
    await expect(
      resolveSelectedSystemPromptContent(
        "prompt-1",
        vi.fn(async () => {
          throw new Error("lookup failed")
        })
      )
    ).resolves.toBe("")
  })

  it("clears redundant overrides that match the selected template", () => {
    expect(
      normalizeSystemPromptOverrideValue({
        draft: "Template body",
        templateContent: "Template body"
      })
    ).toBe("")
  })
})
