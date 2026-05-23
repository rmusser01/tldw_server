import type { WritingRevisionPresetId } from "./writing-revision-types"

export type WritingRevisionPreset = {
  id: WritingRevisionPresetId
  label: string
  instruction: string
}

export const WRITING_REVISION_PRESETS: WritingRevisionPreset[] = [
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

export const getWritingRevisionPreset = (
  id?: WritingRevisionPresetId | null
): WritingRevisionPreset | null =>
  WRITING_REVISION_PRESETS.find((preset) => preset.id === id) ?? null
