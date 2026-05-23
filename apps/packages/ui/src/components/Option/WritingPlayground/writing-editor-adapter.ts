import type { TextAreaRef } from "antd/es/input/TextArea"
import type { Editor } from "@tiptap/react"
import type { RefObject } from "react"
import { tipTapJsonToPlainText } from "./writing-tiptap-utils"

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

const TIPTAP_SCENE_BREAK_TEXT = "\n***\n"

type TipTapNode = Editor["state"]["doc"]

type TipTapPositionOffsetPoint = {
  position: number
  offset: number
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

const getTipTapPlainText = (editor: Editor): string =>
  tipTapJsonToPlainText(editor.getJSON())

const pushTipTapMappingPoint = (
  points: TipTapPositionOffsetPoint[],
  position: number,
  offset: number,
) => {
  points.push({ position, offset })
}

const appendTipTapNodeMappings = (
  node: TipTapNode,
  position: number,
  offset: number,
  points: TipTapPositionOffsetPoint[],
): number => {
  pushTipTapMappingPoint(points, position, offset)

  if (node.isText) {
    const textLength = node.text?.length ?? 0
    for (let index = 1; index <= textLength; index += 1) {
      pushTipTapMappingPoint(points, position + index, offset + index)
    }
    return offset + textLength
  }

  if (node.type.name === "hardBreak") {
    const nextOffset = offset + 1
    pushTipTapMappingPoint(points, position + node.nodeSize, nextOffset)
    return nextOffset
  }

  if (node.type.name === "sceneBreak") {
    const nextOffset = offset + TIPTAP_SCENE_BREAK_TEXT.length
    pushTipTapMappingPoint(points, position + node.nodeSize, nextOffset)
    return nextOffset
  }

  if (node.isLeaf) {
    pushTipTapMappingPoint(points, position + node.nodeSize, offset)
    return offset
  }

  let nextOffset = offset
  node.forEach((childNode, childOffset) => {
    nextOffset = appendTipTapNodeMappings(
      childNode,
      position + 1 + childOffset,
      nextOffset,
      points,
    )
  })

  pushTipTapMappingPoint(points, position + node.nodeSize - 1, nextOffset)

  if (node.type.name === "paragraph" || node.type.name === "heading") {
    nextOffset += 1
  }

  pushTipTapMappingPoint(points, position + node.nodeSize, nextOffset)
  return nextOffset
}

const getTipTapPositionOffsetPoints = (
  editor: Editor,
): TipTapPositionOffsetPoint[] => {
  const points: TipTapPositionOffsetPoint[] = [{ position: 0, offset: 0 }]
  let offset = 0

  editor.state.doc.forEach((childNode, childOffset) => {
    offset = appendTipTapNodeMappings(childNode, childOffset, offset, points)
  })

  return points.sort((a, b) => a.position - b.position || a.offset - b.offset)
}

const getPlainTextOffsetAtTipTapPosition = (
  editor: Editor,
  position: number,
): number => {
  const doc = editor.state.doc
  const resolvedPosition = clampIndex(position, doc.content.size)
  const plainTextLength = getTipTapPlainText(editor).length
  const points = getTipTapPositionOffsetPoints(editor)
  let offset = 0

  for (const point of points) {
    if (point.position > resolvedPosition) break
    offset = point.offset
  }

  return clampIndex(offset, plainTextLength)
}

const getTipTapPositionAtPlainTextOffset = (
  editor: Editor,
  offset: number,
): number => {
  const doc = editor.state.doc
  const plainTextLength = getTipTapPlainText(editor).length
  const targetOffset = clampIndex(offset, plainTextLength)
  const points = getTipTapPositionOffsetPoints(editor)
  let candidatePosition = points[0]?.position ?? 0

  for (const point of points) {
    const currentOffset = clampIndex(point.offset, plainTextLength)
    if (currentOffset > targetOffset) {
      return candidatePosition
    }
    candidatePosition = point.position
  }

  return clampIndex(candidatePosition, doc.content.size)
}

export const getTipTapSelection = (editor: Editor): WritingEditorSelection => {
  const { from, to } = editor.state.selection

  return {
    start: getPlainTextOffsetAtTipTapPosition(editor, from),
    end: getPlainTextOffsetAtTipTapPosition(editor, to),
  }
}

export const createTipTapEditorAdapter = (
  editor: Editor | null | undefined,
): WritingEditorAdapter | null => {
  if (!editor) return null

  return {
    getSelection: () => getTipTapSelection(editor),
    setSelection: (selection) => {
      const normalized = normalizeSelection(
        normalizeUnboundedSelection(selection),
        getTipTapPlainText(editor).length,
      )

      editor.commands.setTextSelection({
        from: getTipTapPositionAtPlainTextOffset(editor, normalized.start),
        to: getTipTapPositionAtPlainTextOffset(editor, normalized.end),
      })
    },
    getSelectedText: (currentValue) => {
      const { start, end } = getTipTapSelection(editor)
      return currentValue.slice(start, end)
    },
    focus: () => editor.commands.focus(),
  }
}
