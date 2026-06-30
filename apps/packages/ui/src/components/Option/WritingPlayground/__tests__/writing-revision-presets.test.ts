import { describe, expect, it } from "vitest"
import type { WritingRevisionPresetId } from "../writing-revision-types"
import {
  getWritingRevisionPreset,
  WRITING_REVISION_PRESETS
} from "../writing-revision-presets"

const EXPECTED_PRESETS: Array<{
  id: WritingRevisionPresetId
  label: string
  instruction: string
}> = [
  {
    id: "draft_freely",
    label: "Draft freely",
    instruction: "Prioritize momentum, vivid continuation, and useful new material."
  },
  {
    id: "polish_prose",
    label: "Polish prose",
    instruction: "Improve clarity, rhythm, word choice, and sentence flow without changing intent."
  },
  {
    id: "developmental_edit",
    label: "Developmental edit",
    instruction: "Focus on structure, stakes, pacing, continuity, and what the passage needs next."
  },
  {
    id: "preserve_voice",
    label: "Preserve voice",
    instruction: "Keep the author's diction, cadence, point of view, and stylistic fingerprints."
  },
  {
    id: "make_concise",
    label: "Make concise",
    instruction: "Reduce redundancy and sharpen phrasing while preserving meaning and voice."
  },
  {
    id: "expand_sensory_detail",
    label: "Expand sensory detail",
    instruction: "Add concrete sensory detail grounded in the existing scene and tone."
  }
]

describe("writing revision presets", () => {
  it("defines the six spec presets with stable ids and visible copy", () => {
    expect(WRITING_REVISION_PRESETS).toEqual(EXPECTED_PRESETS)
  })

  it("keeps preset ids assignable to WritingRevisionPresetId", () => {
    const presetIds: WritingRevisionPresetId[] = WRITING_REVISION_PRESETS.map(
      (preset) => preset.id
    )

    expect(presetIds).toEqual(EXPECTED_PRESETS.map((preset) => preset.id))
  })

  it("exposes non-empty user-facing labels and instructions", () => {
    WRITING_REVISION_PRESETS.forEach((preset) => {
      expect(preset.label.trim()).toBe(preset.label)
      expect(preset.label).not.toBe("")
      expect(preset.instruction.trim()).toBe(preset.instruction)
      expect(preset.instruction).not.toBe("")
    })
  })

  it("resolves presets by id without inventing a custom-instruction default", () => {
    EXPECTED_PRESETS.forEach((preset) => {
      expect(getWritingRevisionPreset(preset.id)).toEqual(preset)
    })

    const customInstruction = "Keep the ending ambiguous."
    expect(getWritingRevisionPreset(null)?.instruction ?? customInstruction).toBe(
      customInstruction
    )
    expect(getWritingRevisionPreset(undefined)).toBeNull()
  })
})
