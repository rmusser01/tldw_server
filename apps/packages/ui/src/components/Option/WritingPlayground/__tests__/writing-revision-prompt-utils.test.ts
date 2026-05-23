import { describe, expect, it } from "vitest"
import type {
  WritingRevisionOperation,
  WritingRevisionTarget
} from "../writing-revision-types"
import {
  buildRevisionUserPrompt,
  parseRevisionModelResponse
} from "../writing-revision-prompt-utils"

const target: WritingRevisionTarget = {
  mode: "paragraph",
  start: 6,
  end: 19,
  beforeText: "the old line.",
  anchor: {
    documentFingerprint: "fingerprint-1",
    prefix: "Once ",
    suffix: "\nThen"
  },
  label: "current paragraph",
  requiresConfirmation: false
}

const buildPrompt = (
  overrides: Partial<Parameters<typeof buildRevisionUserPrompt>[0]> = {}
) =>
  buildRevisionUserPrompt({
    action: "rewrite",
    operation: "replace",
    instruction: "Make the paragraph more vivid.",
    documentText: "Once the old line.\nThen the scene changed.",
    target,
    writingContext: {
      chatMode: false,
      generationSettingsSummary: {}
    },
    ...overrides
  })

const parseResponse = (
  overrides: Partial<Parameters<typeof parseRevisionModelResponse>[0]> = {}
) =>
  parseRevisionModelResponse({
    responseText: '{"title":"Cleaner line","replacement":"the new line.","rationale":"Sharper image.","notes":["Keep tense."]}',
    sessionId: "session-1",
    action: "rewrite",
    operation: "replace",
    instruction: "Make the paragraph more vivid.",
    target,
    createdAt: "2026-05-22T10:00:00.000Z",
    id: "proposal-1",
    ...overrides
  })

describe("writing revision prompt utilities", () => {
  it("creates a text-changing rewrite prompt that asks for JSON only", () => {
    const prompt = buildPrompt()

    expect(prompt).toContain(
      "Return only valid JSON. Do not include markdown fences."
    )
    expect(prompt).toContain("Action: rewrite")
    expect(prompt).toContain("Operation: replace")
    expect(prompt).toContain("Instruction: Make the paragraph more vivid.")
    expect(prompt).toContain('"replacement":"..."')
    expect(prompt).not.toContain('"rawText":"..."')
  })

  it("uses the advisory JSON shape for outline requests", () => {
    const prompt = buildPrompt({
      action: "outline",
      operation: "advisory",
      instruction: "Outline the next scene.",
      target: {
        ...target,
        mode: "document",
        start: 0,
        end: 41,
        beforeText: "Once the old line.\nThen the scene changed.",
        label: "whole document"
      }
    })

    expect(prompt).toContain("Action: outline")
    expect(prompt).toContain("Operation: advisory")
    expect(prompt).toContain('"rawText":"..."')
    expect(prompt).not.toContain('"replacement":"..."')
  })

  it("includes the preset instruction and existing Writing Playground context", () => {
    const prompt = buildPrompt({
      presetInstruction:
        "Improve clarity without changing the author's point of view.",
      writingContext: {
        selectedTemplateName: "Novel chapter",
        selectedThemeName: "Noir",
        chatMode: true,
        contextComposedPrompt: "Earlier context: rain at the train station.",
        memoryBlock: { facts: ["Mara distrusts the conductor."] },
        authorNote: { note: "Keep the voice restrained." },
        worldInfoEntries: [{ key: "city", content: "Glass harbor" }],
        provider: "openai",
        model: "gpt-4.1",
        generationSettingsSummary: {
          temperature: 0.7,
          max_tokens: 400
        }
      }
    })

    expect(prompt).toContain(
      "Workflow preset: Improve clarity without changing the author's point of view."
    )
    expect(prompt).toContain("Template: Novel chapter")
    expect(prompt).toContain("Theme: Noir")
    expect(prompt).toContain("Chat mode: enabled")
    expect(prompt).toContain("Provider: openai")
    expect(prompt).toContain("Model: gpt-4.1")
    expect(prompt).toContain('"temperature":0.7')
    expect(prompt).toContain('"max_tokens":400')
    expect(prompt).toContain("Earlier context: rain at the train station.")
    expect(prompt).toContain("Mara distrusts the conductor.")
    expect(prompt).toContain("Keep the voice restrained.")
    expect(prompt).toContain("Glass harbor")
  })

  it("falls back to context messages when no composed prompt is present", () => {
    const prompt = buildPrompt({
      writingContext: {
        selectedTemplateName: null,
        selectedThemeName: null,
        chatMode: false,
        contextComposedPrompt: "",
        contextMessages: [
          { role: "system", content: "Write in close third person." },
          { role: "user", content: "The last scene ended at the docks." }
        ],
        provider: null,
        model: null,
        generationSettingsSummary: {}
      }
    })

    expect(prompt).toContain("Template: (none)")
    expect(prompt).toContain("Theme: (none)")
    expect(prompt).toContain("Provider: (default)")
    expect(prompt).toContain("Model: (unset)")
    expect(prompt).toContain("Write in close third person.")
    expect(prompt).toContain("The last scene ended at the docks.")
  })

  it.each<WritingRevisionOperation>(["insert", "replace"])(
    "turns valid %s JSON with replacement into a pending proposal",
    (operation) => {
      const proposal = parseResponse({
        operation,
        presetId: "polish_prose",
        presetInstruction: "Improve clarity and rhythm."
      })

      expect(proposal).toEqual({
        id: "proposal-1",
        sessionId: "session-1",
        action: "rewrite",
        operation,
        presetId: "polish_prose",
        presetInstruction: "Improve clarity and rhythm.",
        instruction: "Make the paragraph more vivid.",
        target,
        replacementText: "the new line.",
        rationale: "Sharper image.",
        title: "Cleaner line",
        notes: ["Keep tense."],
        createdAt: "2026-05-22T10:00:00.000Z",
        status: "pending"
      })
    }
  )

  it("turns valid advisory JSON without replacement into an advisory proposal", () => {
    const proposal = parseResponse({
      responseText:
        '{"title":"Next beats","rawText":"1. Raise the stakes.\\n2. Reveal the letter.","rationale":"This keeps the outline non-mutating.","notes":["Use as guidance."]}',
      action: "outline",
      operation: "advisory",
      instruction: "Outline the next scene.",
      presetId: null,
      presetInstruction: null
    })

    expect(proposal).toMatchObject({
      action: "outline",
      operation: "advisory",
      presetId: null,
      presetInstruction: null,
      instruction: "Outline the next scene.",
      rawText: "1. Raise the stakes.\n2. Reveal the letter.",
      rationale: "This keeps the outline non-mutating.",
      title: "Next beats",
      notes: ["Use as guidance."],
      status: "advisory"
    })
    expect(proposal).not.toHaveProperty("replacementText")
  })

  it("keeps text-changing JSON without a string replacement as a raw suggestion", () => {
    const proposal = parseResponse({
      responseText:
        '{"title":"No edit","rawText":"The model only gave advice.","rationale":"Missing replacement."}'
    })

    expect(proposal).toMatchObject({
      title: "No edit",
      rawText: "The model only gave advice.",
      rationale: "Missing replacement.",
      status: "raw_suggestion"
    })
    expect(proposal).not.toHaveProperty("replacementText")
  })

  it("turns malformed JSON into a raw suggestion", () => {
    const proposal = parseResponse({
      responseText: "Use a sharper verb and shorten the sentence."
    })

    expect(proposal).toMatchObject({
      rawText: "Use a sharper verb and shorten the sentence.",
      status: "raw_suggestion"
    })
    expect(proposal).not.toHaveProperty("replacementText")
  })

  it("does not make streamed partial JSON applyable until complete parsing succeeds", () => {
    const partial = parseResponse({
      responseText: '{"title":"Partial","replacement":"the new'
    })

    expect(partial).toMatchObject({
      rawText: '{"title":"Partial","replacement":"the new',
      status: "raw_suggestion"
    })
    expect(partial).not.toHaveProperty("replacementText")

    const complete = parseResponse({
      responseText: '{"title":"Complete","replacement":"the new line."}'
    })

    expect(complete).toMatchObject({
      title: "Complete",
      replacementText: "the new line.",
      status: "pending"
    })
  })
})
