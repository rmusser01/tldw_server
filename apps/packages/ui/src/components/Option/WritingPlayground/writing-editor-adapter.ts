import type { TextAreaRef } from "antd/es/input/TextArea"
import type { Editor } from "@tiptap/react"
import type { RefObject } from "react"

export type WritingEditorSelection = {
  start: number
  end: number
}

export type WritingEditorAdapter = {
  getSelection: () => WritingEditorSelection
  setSelection: (selection: WritingEditorSelection) => void
  getSelectedText: (currentValue: string) => string
  focus: () => void
}

const clampIndex = (value: number, length: number): number => {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(length, Math.floor(value)))
}

const normalizeSelection = (
  selection: WritingEditorSelection,
  length: number,
): WritingEditorSelection => {
  const start = clampIndex(selection.start, length)
  const end = clampIndex(selection.end, length)

  return start <= end ? { start, end } : { start: end, end: start }
}

const normalizeUnboundedSelection = (
  selection: WritingEditorSelection,
): WritingEditorSelection => {
  const start = Math.max(0, Math.floor(selection.start))
  const end = Math.max(0, Math.floor(selection.end))

  return start <= end ? { start, end } : { start: end, end: start }
}

const resolveTextareaNode = (
  textareaRef: RefObject<TextAreaRef | null>,
): HTMLTextAreaElement | null =>
  textareaRef.current?.resizableTextArea?.textArea ?? null

const getTextareaSelection = (
  textareaRef: RefObject<TextAreaRef | null>,
): WritingEditorSelection => {
  const node = resolveTextareaNode(textareaRef)
  const length = node?.value.length ?? 0

  return normalizeSelection(
    {
      start: node?.selectionStart ?? 0,
      end: node?.selectionEnd ?? 0,
    },
    length,
  )
}

export const createTextareaEditorAdapter = (
  textareaRef: RefObject<TextAreaRef | null>,
): WritingEditorAdapter => ({
  getSelection: () => getTextareaSelection(textareaRef),
  setSelection: (selection) => {
    const node = resolveTextareaNode(textareaRef)
    if (!node) return

    const normalized = normalizeSelection(selection, node.value.length)
    node.setSelectionRange(normalized.start, normalized.end)
  },
  getSelectedText: (currentValue) => {
    const { start, end } = getTextareaSelection(textareaRef)
    return currentValue.slice(start, end)
  },
  focus: () => resolveTextareaNode(textareaRef)?.focus(),
})

const getTipTapSelection = (editor: Editor): WritingEditorSelection => {
  const { from, to } = editor.state.selection

  return {
    start: Math.max(0, from - 1),
    end: Math.max(0, to - 1),
  }
}

export const createTipTapEditorAdapter = (
  editor: Editor | null | undefined,
): WritingEditorAdapter | null => {
  if (!editor) return null

  return {
    getSelection: () => getTipTapSelection(editor),
    setSelection: (selection) => {
      const normalized = normalizeUnboundedSelection(selection)

      editor.commands.setTextSelection({
        from: normalized.start + 1,
        to: normalized.end + 1,
      })
    },
    getSelectedText: (currentValue) => {
      const { start, end } = getTipTapSelection(editor)
      return currentValue.slice(start, end)
    },
    focus: () => editor.commands.focus(),
  }
}
