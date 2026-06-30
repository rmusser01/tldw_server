import { describe, expect, it } from "vitest"
import {
  createStartupTemplateBundle,
  isRolePlayRelevantBundle,
  parseStartupTemplateBundles,
  serializeStartupTemplateBundles,
  upsertStartupTemplateBundle,
  removeStartupTemplateBundle,
  sanitizeStartupTemplateName
} from "../startup-template-bundles"

describe("startup template bundles integration", () => {
  it("roundtrips saved bundles through serialized storage format", () => {
    const bundle = createStartupTemplateBundle(
      {
        name: "  Research kickoff template  ",
        selectedModel: "openai:gpt-4.1",
        systemPrompt: "You are a rigorous analyst.",
        selectedSystemPromptId: "prompt-1",
        promptStudioPromptId: 88,
        promptTitle: "Research kickoff",
        promptSource: "prompt-studio",
        presetKey: "balanced",
        character: {
          id: 12,
          name: "Archivist"
        } as any,
        ragPinnedResults: [
          {
            id: "source-1",
            snippet: "Pinned evidence",
            title: "Dataset",
            source: "docs"
          }
        ]
      },
      {
        id: "template-1",
        now: 1_700_000_000_000
      }
    )

    const raw = serializeStartupTemplateBundles([bundle])
    const parsed = parseStartupTemplateBundles(raw)

    expect(parsed).toEqual([bundle])
  })

  it("roundtrips saved role-play setup metadata through serialized storage format", () => {
    const bundle = createStartupTemplateBundle(
      {
        name: "Role-play setup",
        selectedModel: "openai:gpt-4.1",
        systemPrompt: "Stay in character.",
        presetKey: "creative",
        source: "role-play-setup",
        character: {
          id: "char-mira",
          name: "Mira"
        } as any,
        rolePlay: {
          source: "role-play-setup",
          identity: {
            kind: "character",
            id: "char-mira",
            name: "Mira"
          },
          behavior: {
            source: "template",
            templateId: "character-actor",
            templateTitle: "Character Actor",
            templateCategory: "roleplay",
            systemPrompt: "Stay in character.",
            modified: false
          },
          scene: {
            version: 3,
            isEnabled: true,
            aspects: [],
            notes: "In the observatory.",
            chatPosition: "before",
            chatDepth: 0,
            chatRole: "system"
          } as any,
          generation: {
            presetKey: "creative",
            settings: {
              temperature: 1.2
            }
          },
          context: {
            ragPinnedCount: 1,
            ragPinnedResultIds: ["source-1"]
          }
        }
      },
      {
        id: "role-play-setup-1",
        now: 1_700_000_000_000
      }
    )

    const parsed = parseStartupTemplateBundles(serializeStartupTemplateBundles([bundle]))

    expect(parsed).toEqual([bundle])
    expect(parsed[0]?.source).toBe("role-play-setup")
    expect(parsed[0]?.rolePlay?.identity?.name).toBe("Mira")
    expect(isRolePlayRelevantBundle(parsed[0]!)).toBe(true)
  })

  it("upserts and removes bundles while preserving newest-first order", () => {
    const older = createStartupTemplateBundle(
      {
        name: "Older",
        selectedModel: "gpt-4.1",
        systemPrompt: "A",
        presetKey: "balanced"
      },
      { id: "older", now: 1 }
    )
    const newer = createStartupTemplateBundle(
      {
        name: "Newer",
        selectedModel: "gpt-4.1-mini",
        systemPrompt: "B",
        presetKey: "precise"
      },
      { id: "newer", now: 2 }
    )

    const upserted = upsertStartupTemplateBundle([older], newer)
    expect(upserted.map((entry) => entry.id)).toEqual(["newer", "older"])

    const removed = removeStartupTemplateBundle(upserted, "newer")
    expect(removed.map((entry) => entry.id)).toEqual(["older"])
  })

  it("sanitizes names and discards malformed stored entries", () => {
    const invalidRaw = JSON.stringify([
      {
        id: "valid",
        name: "   ",
        selectedModel: "",
        systemPrompt: 123,
        presetKey: "unknown",
        ragPinnedResults: [
          { id: "ok", snippet: "keep" },
          { id: "bad" }
        ]
      },
      {
        name: "missing-id"
      }
    ])

    const parsed = parseStartupTemplateBundles(invalidRaw)
    expect(parsed).toHaveLength(1)
    expect(parsed[0]?.name).toBe("New startup template")
    expect(parsed[0]?.selectedModel).toBeNull()
    expect(parsed[0]?.presetKey).toBe("custom")
    expect(parsed[0]?.ragPinnedResults).toEqual([{ id: "ok", snippet: "keep" }])

    expect(
      sanitizeStartupTemplateName(
        "This is a very long startup template name that should be trimmed to fit the max length boundary in one shot"
      ).length
    ).toBeLessThanOrEqual(80)
  })

  it("defensively normalizes malformed role-play metadata", () => {
    const invalidRaw = JSON.stringify([
      {
        id: "role-play-invalid",
        name: "Broken role-play setup",
        source: "role-play-setup",
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: {
            kind: "assistant",
            id: "",
            name: ""
          },
          behavior: {
            source: "template",
            templateTitle: "",
            systemPrompt: 123
          },
          generation: {
            presetKey: "not-real",
            settings: "bad"
          },
          context: {
            ragPinnedCount: "two",
            ragPinnedResultIds: [1, null]
          }
        }
      }
    ])

    const parsed = parseStartupTemplateBundles(invalidRaw)

    expect(parsed).toHaveLength(1)
    expect(parsed[0]?.rolePlay).toBeNull()
    expect(parsed[0]?.source).toBe("startup-template")
    expect(isRolePlayRelevantBundle(parsed[0]!)).toBe(false)
  })
})
