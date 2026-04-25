import { describe, expect, it } from "vitest"

import { materializeFlashcardTemplateDraft } from "../flashcard-template-resolution"
import type { FlashcardTemplate } from "@/services/flashcards"

const template: FlashcardTemplate = {
  id: 17,
  name: "Vocabulary definition",
  model_type: "basic",
  front_template: "What does {{ term }} mean?",
  back_template: "{{definition}}\n\nExample: {{example}}",
  notes_template: "Source: {{source}}",
  extra_template: null,
  placeholder_definitions: [
    {
      key: "term",
      label: "Term",
      help_text: null,
      default_value: null,
      required: true,
      targets: ["front_template"]
    },
    {
      key: "definition",
      label: "Definition",
      help_text: null,
      default_value: null,
      required: true,
      targets: ["back_template"]
    },
    {
      key: "example",
      label: "Example",
      help_text: null,
      default_value: "ATP fuels cellular work.",
      required: false,
      targets: ["back_template"]
    },
    {
      key: "source",
      label: "Source",
      help_text: null,
      default_value: "Biology notes",
      required: false,
      targets: ["notes_template"]
    }
  ],
  created_at: "2026-04-15T00:00:00Z",
  last_modified: "2026-04-15T00:00:00Z",
  deleted: false,
  client_id: "test-client",
  version: 1
}

describe("flashcard-template-resolution", () => {
  it("materializes template defaults and placeholder values into a flashcard draft", () => {
    expect(
      materializeFlashcardTemplateDraft(
        template,
        {
          term: "ATP",
          definition: "The cell's energy currency"
        },
        {
          deck_id: 42,
          tags: ["vocab", "language"]
        }
      )
    ).toEqual(
      expect.objectContaining({
        deck_id: 42,
        tags: ["vocab", "language"],
        model_type: "basic",
        front: "What does ATP mean?",
        back: "The cell's energy currency\n\nExample: ATP fuels cellular work.",
        notes: "Source: Biology notes"
      })
    )
  })

  it("rejects missing required placeholder values", () => {
    expect(() =>
      materializeFlashcardTemplateDraft(template, {
        term: "ATP"
      })
    ).toThrow("Missing required placeholder value: definition")
  })

  it("preserves cloze syntax when materializing cloze templates", () => {
    const clozeTemplate: FlashcardTemplate = {
      ...template,
      id: 18,
      name: "Cloze recall",
      model_type: "cloze",
      front_template: "The {{c1::mitochondria}} generates ATP.",
      back_template: "Mitochondria",
      notes_template: null,
      placeholder_definitions: []
    }

    expect(
      materializeFlashcardTemplateDraft(clozeTemplate, {})
    ).toEqual(
      expect.objectContaining({
        model_type: "cloze",
        front: "The {{c1::mitochondria}} generates ATP.",
        back: "Mitochondria"
      })
    )
  })

  it("rejects undeclared placeholder tokens", () => {
    const invalidTemplate: FlashcardTemplate = {
      ...template,
      id: 19,
      front_template: "What does {{term}} mean in {{context}}?"
    }

    expect(() =>
      materializeFlashcardTemplateDraft(invalidTemplate, {
        term: "ATP",
        definition: "The cell's energy currency"
      })
    ).toThrow("Unknown placeholder token: context")
  })
})
