import { describe, expect, it } from "vitest"
import {
  buildImagePromptRefineMessages,
  extractImagePromptRefineCandidate
} from "@/utils/image-prompt-refinement"

describe("image prompt refinement utilities", () => {
  it("builds deterministic refinement messages with context blend entries", () => {
    const messages = buildImagePromptRefineMessages({
      originalPrompt: "Portrait of Lana in neon rain.",
      strategyLabel: "Expression",
      backend: "local-sd",
      contextEntries: [
        {
          id: "character",
          label: "Character",
          text: "Lana Reed",
          weight: 0.3,
          quality: 0.9,
          score: 0.27
        },
        {
          id: "mood",
          label: "Mood",
          text: "focused and intense",
          weight: 0.2,
          quality: 0.8,
          score: 0.16
        }
      ]
    })

    expect(messages).toHaveLength(2)
    expect(messages[0]).toMatchObject({ role: "system" })
    expect(messages[1]).toMatchObject({ role: "user" })
    const userContent = String(messages[1].content || "")
    expect(userContent).toContain("Prompt mode: Expression")
    expect(userContent).toContain("Backend: local-sd")
    expect(userContent).toContain("Character (27%): Lana Reed")
    expect(userContent).toContain("Mood (16%): focused and intense")
  })

  it("replaces only the editable semantics while preserving locked request carriers", () => {
    const messages = buildImagePromptRefineMessages({
      originalPrompt: "Portrait of Lana in neon rain.",
      strategyLabel: "Expression",
      backend: "local-sd",
      contextEntries: [
        {
          id: "character",
          label: "Character",
          text: "Lana Reed",
          weight: 0.3,
          quality: 0.9,
          score: 0.27
        }
      ],
      systemSemantics: "Custom refinement guidance.",
      rewriteSemantics: "Custom rewrite guidance."
    })

    expect(messages[0].content).toBe(
      "Custom refinement guidance. Output only the final refined prompt as plain text. Do not include markdown, labels, bullets, or commentary."
    )
    expect(messages[1].content).toBe(
      "Prompt mode: Expression\n\nBackend: local-sd\n\nOriginal prompt:\nPortrait of Lana in neon rain.\n\n\n\nContext blend cues:\nCharacter (27%): Lana Reed\n\nCustom rewrite guidance."
    )
    expect(messages[0].content).not.toContain(
      "Preserve intent while improving clarity"
    )
    expect(messages[1].content).not.toContain(
      "Rewrite the prompt to be concise"
    )
  })

  it("extracts refined prompt text from fenced completion payloads", () => {
    const candidate = extractImagePromptRefineCandidate({
      choices: [
        {
          message: {
            content:
              "```text\nPrompt: cinematic portrait of Lana, rain-soaked neon alley, shallow depth of field\n```"
          }
        }
      ]
    })

    expect(candidate).toBe(
      "cinematic portrait of Lana, rain-soaked neon alley, shallow depth of field"
    )
  })

  it("returns null when completion payload has no usable text", () => {
    expect(extractImagePromptRefineCandidate({ choices: [] })).toBeNull()
    expect(extractImagePromptRefineCandidate({})).toBeNull()
  })
})
