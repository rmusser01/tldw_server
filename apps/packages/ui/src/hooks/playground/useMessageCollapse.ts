import React from "react"
import { useTranslation } from "react-i18next"
import { PASTED_TEXT_CHAR_LIMIT } from "@/utils/constant"

export type CollapsedRange = {
  start: number
  end: number
}

export function useMessageCollapse() {
  const { t } = useTranslation(["playground"])

  const pendingCaretRef = React.useRef<number | null>(null)
  const lastDisplaySelectionRef = React.useRef<{
    start: number
    end: number
  } | null>(null)
  const pendingCollapsedStateRef = React.useRef<{
    message: string
    range: CollapsedRange
    caret: number
  } | null>(null)
  const pointerDownRef = React.useRef(false)
  const selectionFromPointerRef = React.useRef(false)

  const [isMessageCollapsed, setIsMessageCollapsed] = React.useState(false)
  const [collapsedRange, setCollapsedRange] =
    React.useState<CollapsedRange | null>(null)
  const [hasExpandedLargeText, setHasExpandedLargeText] = React.useState(false)

  const normalizeCollapsedRange = React.useCallback(
    (range: CollapsedRange, messageLength: number): CollapsedRange => {
      const start = Math.max(0, Math.min(range.start, messageLength))
      const end = Math.max(start, Math.min(range.end, messageLength))
      return { start, end }
    },
    []
  )

  const parseCollapsedRange = React.useCallback(
    (value: unknown, messageLength: number): CollapsedRange | null => {
      if (!value || typeof value !== "object") return null
      const start = Number((value as { start?: number }).start)
      const end = Number((value as { end?: number }).end)
      if (!Number.isFinite(start) || !Number.isFinite(end)) return null
      const range = normalizeCollapsedRange({ start, end }, messageLength)
      if (range.end <= range.start) return null
      return range
    },
    [normalizeCollapsedRange]
  )

  const buildCollapsedMessageLabel = React.useCallback(
    (text: string) => {
      const lineCount =
        text ? (text.match(/\r\n|\r|\n/g)?.length ?? 0) + 1 : 0
      return t(
        "playground:composer.collapsedMessageLabel",
        "[{lines, plural, one {# line} other {# lines}}/{chars, plural, one {# char} other {# chars}} in message]",
        { lines: lineCount, chars: text.length }
      )
    },
    [t]
  )

  const getCollapsedDisplayMeta = React.useCallback(
    (text: string, range: CollapsedRange) => {
      const normalizedRange = normalizeCollapsedRange(range, text.length)
      const collapsedText = text.slice(
        normalizedRange.start,
        normalizedRange.end
      )
      const label = buildCollapsedMessageLabel(collapsedText)
      const prefix = text.slice(0, normalizedRange.start)
      const suffix = text.slice(normalizedRange.end)
      const labelStart = prefix.length
      const labelEnd = labelStart + label.length
      const blockLength = normalizedRange.end - normalizedRange.start
      return {
        display: `${prefix}${label}${suffix}`,
        label,
        labelStart,
        labelEnd,
        labelLength: label.length,
        blockLength,
        rangeStart: normalizedRange.start,
        rangeEnd: normalizedRange.end,
        messageLength: text.length
      }
    },
    [buildCollapsedMessageLabel, normalizeCollapsedRange]
  )

  const getDisplayCaretFromMessage = React.useCallback(
    (
      messageCaret: number,
      meta: ReturnType<typeof getCollapsedDisplayMeta>
    ) => {
      if (messageCaret <= meta.rangeStart) return messageCaret
      if (messageCaret >= meta.rangeEnd) {
        return (
          messageCaret -
          meta.blockLength +
          meta.labelLength
        )
      }
      return meta.labelEnd
    },
    []
  )

  const getMessageCaretFromDisplay = React.useCallback(
    (
      displayCaret: number,
      meta: ReturnType<typeof getCollapsedDisplayMeta>,
      options?: { prefer?: "before" | "after" }
    ) => {
      if (displayCaret <= meta.labelStart) return displayCaret
      if (displayCaret >= meta.labelEnd) {
        return (
          displayCaret -
          meta.labelLength +
          meta.blockLength
        )
      }
      return options?.prefer === "before"
        ? meta.rangeStart
        : meta.rangeEnd
    },
    []
  )

  const collapseLargeMessage = React.useCallback(
    (text: string, options?: { force?: boolean; range?: CollapsedRange }) => {
      if (text.length <= PASTED_TEXT_CHAR_LIMIT) {
        setIsMessageCollapsed(false)
        setHasExpandedLargeText(false)
        setCollapsedRange(null)
        return
      }
      if (!options?.force && hasExpandedLargeText) return
      const range =
        options?.range ?? { start: 0, end: text.length }
      const normalizedRange = normalizeCollapsedRange(range, text.length)
      setIsMessageCollapsed(true)
      setHasExpandedLargeText(false)
      setCollapsedRange(normalizedRange)
    },
    [hasExpandedLargeText, normalizeCollapsedRange]
  )

  const expandLargeMessage = React.useCallback(
    (options?: { force?: boolean }) => {
      if (!isMessageCollapsed && !options?.force) return
      setIsMessageCollapsed(false)
      setHasExpandedLargeText(true)
      setCollapsedRange(null)
    },
    [isMessageCollapsed]
  )

  const restoreMessageValue = React.useCallback(
    (
      value: string,
      metadata?: { wasExpanded?: boolean; collapsedRange?: CollapsedRange | null }
    ) => {
      pendingCaretRef.current = null
      lastDisplaySelectionRef.current = null
      pendingCollapsedStateRef.current = null
      if (value.length <= PASTED_TEXT_CHAR_LIMIT) {
        setIsMessageCollapsed(false)
        setHasExpandedLargeText(false)
        setCollapsedRange(null)
        return { collapsed: false }
      }
      const wasExpanded = Boolean(metadata?.wasExpanded)
      if (wasExpanded) {
        setIsMessageCollapsed(false)
        setHasExpandedLargeText(true)
        setCollapsedRange(null)
        return { collapsed: false }
      }
      const range =
        parseCollapsedRange(metadata?.collapsedRange, value.length) ?? {
          start: 0,
          end: value.length
        }
      setIsMessageCollapsed(true)
      setHasExpandedLargeText(false)
      setCollapsedRange(range)
      return { collapsed: true }
    },
    [parseCollapsedRange]
  )

  return {
    isMessageCollapsed,
    setIsMessageCollapsed,
    collapsedRange,
    setCollapsedRange,
    hasExpandedLargeText,
    setHasExpandedLargeText,
    pendingCaretRef,
    lastDisplaySelectionRef,
    pendingCollapsedStateRef,
    pointerDownRef,
    selectionFromPointerRef,
    normalizeCollapsedRange,
    parseCollapsedRange,
    buildCollapsedMessageLabel,
    getCollapsedDisplayMeta,
    getDisplayCaretFromMessage,
    getMessageCaretFromDisplay,
    collapseLargeMessage,
    expandLargeMessage,
    restoreMessageValue
  }
}
