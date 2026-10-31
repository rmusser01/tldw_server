import { describe, expect, it, vi } from "vitest"
import {
  captureSystemPromptOverrideSnapshot,
  normalizeSystemPromptOverrideValue,
  resolveEffectiveSystemPromptState,
  resolveSelectedSystemPromptContent,
  restoreSystemPromptOverrideSnapshot
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

  it.each([undefined, ""])(
    "resolves raw override %s to the selected template without an active override",
    async (systemPrompt) => {
      await expect(
        resolveEffectiveSystemPromptState({
          selectedSystemPrompt: "prompt-1",
          systemPrompt,
          getPromptByIdFn: vi.fn(async () => makePrompt("Template body"))
        })
      ).resolves.toEqual({
        templateContent: "Template body",
        effectiveContent: "Template body",
        overrideActive: false
      })
    }
  )

  it("treats a raw value matching the selected template as no override", async () => {
    await expect(
      resolveEffectiveSystemPromptState({
        selectedSystemPrompt: "prompt-1",
        systemPrompt: "Template body",
        getPromptByIdFn: vi.fn(async () => makePrompt("Template body"))
      })
    ).resolves.toEqual({
      templateContent: "Template body",
      effectiveContent: "Template body",
      overrideActive: false
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

  it.each([
    [undefined, ""],
    ["", ""],
    ["Template body", ""],
    ["Conversation override", "Conversation override"]
  ])("normalizes draft %s to override %s", (draft, expected) => {
    expect(
      normalizeSystemPromptOverrideValue({
        draft,
        templateContent: "Template body"
      })
    ).toBe(expected)
  })

  it.each([undefined, "", "Conversation override"])(
    "round-trips the exact raw override snapshot %s",
    (rawOverride) => {
      expect(
        restoreSystemPromptOverrideSnapshot(
          captureSystemPromptOverrideSnapshot(rawOverride)
        )
      ).toBe(rawOverride)
    }
  )
})
