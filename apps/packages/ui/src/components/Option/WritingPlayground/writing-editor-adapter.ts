import type { TextAreaRef } from "antd/es/input/TextArea"
import type { Editor } from "@tiptap/react"
import type { RefObject } from "react"
import { tipTapJsonToPlainText } from "./writing-tiptap-utils"

export type WritingEditorSelection = {
  start: number
  end: number
}

export type WritingEditorRangeMeasurement = {
  top: number
  bottom: number
  height: number
}

export type WritingEditorAdapter = {
  getSelection: () => WritingEditorSelection
  setSelection: (selection: WritingEditorSelection) => void
  getSelectedText: (currentValue: string) => string
  focus: () => void
  measureRange?: (selection: WritingEditorSelection) => WritingEditorRangeMeasurement | null
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

const TIPTAP_SCENE_BREAK_TEXT = "***"

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

const TIPTAP_BLOCK_CONTAINER_TYPES = new Set([
  "bulletList",
  "orderedList",
  "listItem",
  "blockquote"
])

const usesTipTapBlockChildSeparators = (node: TipTapNode): boolean =>
  TIPTAP_BLOCK_CONTAINER_TYPES.has(node.type.name)

const getTipTapBlockSeparatorLength = (previousNode: TipTapNode | null): number =>
  previousNode ? 2 : 0

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
  let previousChildNode: TipTapNode | null = null
  const shouldSeparateChildren = usesTipTapBlockChildSeparators(node)
  node.forEach((childNode, childOffset) => {
    if (shouldSeparateChildren) {
      const separatorLength = getTipTapBlockSeparatorLength(previousChildNode)
      const childPosition = position + 1 + childOffset
      for (let index = 1; index <= separatorLength; index += 1) {
        pushTipTapMappingPoint(points, childPosition, nextOffset + index)
      }
      nextOffset += separatorLength
    }
    nextOffset = appendTipTapNodeMappings(
      childNode,
      position + 1 + childOffset,
      nextOffset,
      points,
    )
    previousChildNode = childNode
  })

  pushTipTapMappingPoint(points, position + node.nodeSize - 1, nextOffset)
  pushTipTapMappingPoint(points, position + node.nodeSize, nextOffset)
  return nextOffset
}

const getTipTapPositionOffsetPoints = (
  editor: Editor,
): TipTapPositionOffsetPoint[] => {
  const points: TipTapPositionOffsetPoint[] = [{ position: 0, offset: 0 }]
  let offset = 0
  let previousNode: TipTapNode | null = null

  editor.state.doc.forEach((childNode, childOffset) => {
    const separatorLength = getTipTapBlockSeparatorLength(previousNode)
    for (let index = 1; index <= separatorLength; index += 1) {
      pushTipTapMappingPoint(points, childOffset, offset + index)
    }
    offset += separatorLength
    offset = appendTipTapNodeMappings(childNode, childOffset, offset, points)
    previousNode = childNode
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

const isMeasurableTipTapSelection = (
  selection: WritingEditorSelection,
  plainTextLength: number,
): boolean =>
  Number.isFinite(selection.start) &&
  Number.isFinite(selection.end) &&
  selection.start >= 0 &&
  selection.end > selection.start &&
  selection.end <= plainTextLength

const measureTipTapRange = (
  editor: Editor,
  selection: WritingEditorSelection,
): WritingEditorRangeMeasurement | null => {
  if (editor.isDestroyed) return null

  const plainTextLength = getTipTapPlainText(editor).length
  if (!isMeasurableTipTapSelection(selection, plainTextLength)) return null

  const from = getTipTapPositionAtPlainTextOffset(editor, selection.start)
  const to = getTipTapPositionAtPlainTextOffset(editor, selection.end)
  if (to <= from) return null

  try {
    const startCoords = editor.view.coordsAtPos(from)
    const endCoords = editor.view.coordsAtPos(to)
    const editorTop = editor.view.dom.getBoundingClientRect().top
    const top = Math.min(startCoords.top, endCoords.top) - editorTop
    const bottom = Math.max(startCoords.bottom, endCoords.bottom) - editorTop
    const height = bottom - top

    if (![top, bottom, height].every(Number.isFinite) || height < 0) {
      return null
    }

    return { top, bottom, height }
  } catch {
    return null
  }
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
    measureRange: (selection) => measureTipTapRange(editor, selection),
  }
}
