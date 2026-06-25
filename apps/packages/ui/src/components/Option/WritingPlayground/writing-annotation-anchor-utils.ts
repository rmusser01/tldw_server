import type {
  ManuscriptAnnotationCategory,
  ManuscriptAnnotationCreateInput
} from "@/services/writing-playground"
import type { WritingEditorSelection } from "./writing-editor-adapter"

export const ANNOTATION_CONTEXT_MAX_CHARS = 240

type SelectedRangeValidationInput = {
  documentText: string
  selection: WritingEditorSelection | null
  selectedText?: string | null
}

type ValidSelectedRange = {
  ok: true
  start: number
  end: number
  codePointStart: number
  codePointEnd: number
  selectedText: string
}

type InvalidSelectedRange = {
  ok: false
  reason: "empty" | "stale" | "out_of_bounds"
}

export type SelectedRangeValidationResult =
  | ValidSelectedRange
  | InvalidSelectedRange

const clampUtf16Offset = (value: number, textLength: number): number => {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(textLength, Math.floor(value)))
}

const normalizeSelection = (
  selection: WritingEditorSelection,
  textLength: number
): WritingEditorSelection => {
  const start = clampUtf16Offset(selection.start, textLength)
  const end = clampUtf16Offset(selection.end, textLength)
  return start <= end ? { start, end } : { start: end, end: start }
}

export const utf16OffsetToCodePointOffset = (
  text: string,
  utf16Offset: number
): number => {
  const offset = clampUtf16Offset(utf16Offset, text.length)
  return Array.from(text.slice(0, offset)).length
}

export const captureAnnotationContext = (
  documentText: string,
  selection: WritingEditorSelection
) => {
  const normalized = normalizeSelection(selection, documentText.length)
  const prefix = Array.from(documentText.slice(0, normalized.start))
    .slice(-ANNOTATION_CONTEXT_MAX_CHARS)
    .join("")
  const suffix = Array.from(documentText.slice(normalized.end))
    .slice(0, ANNOTATION_CONTEXT_MAX_CHARS)
    .join("")

  return { prefix, suffix }
}

export const fingerprintSelectedText = (selectedText: string): string => {
  let hash = 2166136261
  for (const char of selectedText) {
    hash ^= char.codePointAt(0) ?? 0
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(16).padStart(8, "0")
}

export const validateSelectedRange = ({
  documentText,
  selection,
  selectedText
}: SelectedRangeValidationInput): SelectedRangeValidationResult => {
  if (!selection) return { ok: false, reason: "empty" }
  if (
    selection.start < 0 ||
    selection.end < 0 ||
    selection.start > documentText.length ||
    selection.end > documentText.length
  ) {
    return { ok: false, reason: "out_of_bounds" }
  }

  const normalized = normalizeSelection(selection, documentText.length)
  if (normalized.start === normalized.end) {
    return { ok: false, reason: "empty" }
  }

  const documentSelectedText = documentText.slice(normalized.start, normalized.end)
  if (!documentSelectedText.trim()) {
    return { ok: false, reason: "empty" }
  }

  if (selectedText != null && selectedText !== documentSelectedText) {
    return { ok: false, reason: "stale" }
  }

  return {
    ok: true,
    start: normalized.start,
    end: normalized.end,
    codePointStart: utf16OffsetToCodePointOffset(documentText, normalized.start),
    codePointEnd: utf16OffsetToCodePointOffset(documentText, normalized.end),
    selectedText: documentSelectedText
  }
}

export const buildSceneRangeAnnotationInput = ({
  canCreateRangeAnnotation,
  sceneId,
  sceneVersion,
  documentText,
  selection,
  category,
  body
}: {
  canCreateRangeAnnotation: boolean
  sceneId: string | null
  sceneVersion: number | null
  documentText: string
  selection: WritingEditorSelection | null
  category: ManuscriptAnnotationCategory
  body: string
}): ManuscriptAnnotationCreateInput => {
  if (!canCreateRangeAnnotation || !sceneId || sceneVersion == null) {
    throw new Error("Create range annotations from a saved scene binding.")
  }

  const validated = validateSelectedRange({ documentText, selection })
  if (!validated.ok) {
    throw new Error("Select current scene text before creating a range annotation.")
  }

  const context = captureAnnotationContext(documentText, {
    start: validated.start,
    end: validated.end
  })

  return {
    target_type: "scene",
    target_id: sceneId,
    category,
    body,
    scene_version: sceneVersion,
    start: validated.codePointStart,
    end: validated.codePointEnd,
    selected_text: validated.selectedText,
    metadata: {
      anchor_prefix: context.prefix,
      anchor_suffix: context.suffix,
      anchor_fingerprint: fingerprintSelectedText(validated.selectedText)
    }
  }
}
