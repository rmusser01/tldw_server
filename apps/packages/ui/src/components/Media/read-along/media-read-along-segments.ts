import {
  hasLeadingTranscriptTimings,
  parseLeadingTranscriptTiming
} from '@/utils/media-transcript-display'

import type {
  BuildReadAlongSegmentsInput,
  ReadAlongSegment,
  ReadAlongSelection,
  ReadAlongTtsRequestSegment,
  ResolveReadAlongScopeInput
} from './types'

const SENTENCE_BOUNDARY_PATTERN = /[^.!?\n]+[.!?]+(?=\s|$)|[^.!?\n]+$/g
const MARKDOWN_HEADING_PATTERN = /^\s{0,3}#{1,6}\s+\S/

const normalizeNewlines = (value: string): string => value.replace(/\r\n/g, '\n')

const buildSegmentId = (
  mediaId: string,
  index: number,
  kind: ReadAlongSegment['kind'],
  sourceStart: number,
  sourceEnd: number
): string => `${mediaId}:${index}:${kind}:${sourceStart}:${sourceEnd}`

const parseTimestampSeconds = (timestamp: string): number | undefined => {
  const parts = timestamp.split(':').map((part) => Number.parseInt(part, 10))
  if (parts.some((part) => Number.isNaN(part))) return undefined

  if (parts.length === 2) {
    const [minutes, seconds] = parts
    return minutes * 60 + seconds
  }

  if (parts.length === 3) {
    const [hours, minutes, seconds] = parts
    return hours * 3600 + minutes * 60 + seconds
  }

  return undefined
}

const findDisplayOffset = (
  displayContent: string | undefined,
  text: string,
  after: number
): number | undefined => {
  if (!displayContent) return undefined
  const offset = displayContent.indexOf(text, Math.max(0, after))
  return offset >= 0 ? offset : undefined
}

const buildTranscriptSegments = ({
  mediaId,
  content,
  displayContent
}: BuildReadAlongSegmentsInput): ReadAlongSegment[] => {
  const normalized = normalizeNewlines(content)
  const segments: ReadAlongSegment[] = []
  let sourceOffset = 0
  let nextDisplayStart = 0

  for (const line of normalized.split('\n')) {
    const parsed = parseLeadingTranscriptTiming(line)
    if (parsed) {
      const prefixLength =
        parsed.leadingWhitespace.length +
        parsed.timestamp.length +
        parsed.separator.length +
        (line.includes(`[${parsed.timestamp}]`) ? 2 : 0)
      const rawTextStart = sourceOffset + prefixLength
      const textLeadingTrim = parsed.text.length - parsed.text.trimStart().length
      const text = parsed.text.trim()
      const sourceStart = rawTextStart + textLeadingTrim
      const sourceEnd = sourceStart + text.length

      if (text.length > 0) {
        const displayStart = findDisplayOffset(displayContent, text, nextDisplayStart)
        if (displayStart != null) {
          nextDisplayStart = displayStart + text.length
        }

        segments.push({
          id: buildSegmentId(
            mediaId,
            segments.length,
            'transcript-line',
            sourceStart,
            sourceEnd
          ),
          index: segments.length,
          kind: 'transcript-line',
          text,
          sourceStart,
          sourceEnd,
          displayStart,
          displayEnd: displayStart == null ? undefined : displayStart + text.length,
          sectionId: 'transcript',
          timestampSeconds: parseTimestampSeconds(parsed.timestamp)
        })
      }
    }

    sourceOffset += line.length + 1
  }

  return segments
}

const splitSentences = (
  text: string,
  sourceOffset: number
): Array<{ text: string; sourceStart: number; sourceEnd: number }> => {
  const sentences: Array<{ text: string; sourceStart: number; sourceEnd: number }> = []

  for (const match of text.matchAll(SENTENCE_BOUNDARY_PATTERN)) {
    const matchedText = match[0]
    const localStart = match.index ?? 0
    const leadingTrim = matchedText.length - matchedText.trimStart().length
    const trimmed = matchedText.trim()
    if (!trimmed) continue

    const sourceStart = sourceOffset + localStart + leadingTrim
    sentences.push({
      text: trimmed,
      sourceStart,
      sourceEnd: sourceStart + trimmed.length
    })
  }

  return sentences
}

const buildProseSegments = ({
  mediaId,
  content,
  displayContent,
  renderMode
}: BuildReadAlongSegmentsInput): ReadAlongSegment[] => {
  const normalized = normalizeNewlines(content)
  const lines = normalized.split('\n')
  const segments: ReadAlongSegment[] = []
  let sourceOffset = 0
  let sectionIndex = 0
  let paragraphIndex = 0
  let nextDisplayStart = 0

  for (const line of lines) {
    const isHeading = renderMode === 'markdown' && MARKDOWN_HEADING_PATTERN.test(line)
    if (isHeading) {
      sectionIndex += segments.length === 0 && sectionIndex === 0 ? 0 : 1
      sourceOffset += line.length + 1
      continue
    }

    if (line.trim().length === 0) {
      paragraphIndex += 1
      sourceOffset += line.length + 1
      continue
    }

    const sectionId =
      renderMode === 'markdown' ? `section-${sectionIndex}` : `paragraph-${paragraphIndex}`

    for (const sentence of splitSentences(line, sourceOffset)) {
      const displayStart = findDisplayOffset(
        displayContent,
        sentence.text,
        nextDisplayStart
      )
      if (displayStart != null) {
        nextDisplayStart = displayStart + sentence.text.length
      }

      segments.push({
        id: buildSegmentId(
          mediaId,
          segments.length,
          'sentence',
          sentence.sourceStart,
          sentence.sourceEnd
        ),
        index: segments.length,
        kind: 'sentence',
        text: sentence.text,
        sourceStart: sentence.sourceStart,
        sourceEnd: sentence.sourceEnd,
        displayStart,
        displayEnd:
          displayStart == null ? undefined : displayStart + sentence.text.length,
        sectionId
      })
    }

    sourceOffset += line.length + 1
  }

  return segments
}

export const buildReadAlongSegments = (
  input: BuildReadAlongSegmentsInput
): ReadAlongSegment[] => {
  if (!input.content.trim()) return []

  if (hasLeadingTranscriptTimings(input.content)) {
    return buildTranscriptSegments(input)
  }

  return buildProseSegments(input)
}

const buildTransientSelectionSegment = (
  selection: ReadAlongSelection
): ReadAlongSegment[] => {
  const text = selection.selectedText.trim()
  if (!text) return []

  return [
    {
      id: `transient-selection:0:${text.length}`,
      index: 0,
      kind: 'transient-selection',
      text,
      sourceStart: 0,
      sourceEnd: text.length
    }
  ]
}

const findSegmentIndexForSelection = (
  segments: ReadAlongSegment[],
  selection: ReadAlongSelection
): number => {
  const idIndex = selection.startSegmentId
    ? segments.findIndex((segment) => segment.id === selection.startSegmentId)
    : -1
  if (idIndex >= 0) return idIndex

  if (selection.sourceStart == null) return -1

  const containingIndex = segments.findIndex(
    (segment) =>
      selection.sourceStart != null &&
      selection.sourceStart >= segment.sourceStart &&
      selection.sourceStart < segment.sourceEnd
  )
  if (containingIndex >= 0) return containingIndex

  return segments.findIndex((segment) => selection.sourceStart! < segment.sourceStart)
}

const findSelectedSegments = (
  segments: ReadAlongSegment[],
  selection: ReadAlongSelection
): ReadAlongSegment[] => {
  if (selection.startSegmentId || selection.endSegmentId) {
    const startIndex = findSegmentIndexForSelection(segments, selection)
    const endIndex = selection.endSegmentId
      ? segments.findIndex((segment) => segment.id === selection.endSegmentId)
      : startIndex
    if (startIndex >= 0 && endIndex >= 0) {
      return segments.slice(
        Math.min(startIndex, endIndex),
        Math.max(startIndex, endIndex) + 1
      )
    }
  }

  if (selection.sourceStart == null || selection.sourceEnd == null) {
    return []
  }

  return segments.filter(
    (segment) =>
      segment.sourceEnd > selection.sourceStart! &&
      segment.sourceStart < selection.sourceEnd!
  )
}

export const resolveReadAlongScope = ({
  scope,
  segments,
  selection
}: ResolveReadAlongScopeInput): ReadAlongSegment[] => {
  if (scope === 'full-item') {
    return segments
  }

  if (scope === 'selection') {
    const selectedSegments = findSelectedSegments(segments, selection)
    return selectedSegments.length > 0
      ? selectedSegments
      : buildTransientSelectionSegment(selection)
  }

  const startIndex = findSegmentIndexForSelection(segments, selection)
  if (startIndex < 0) {
    return buildTransientSelectionSegment(selection)
  }

  if (scope === 'from-here') {
    return segments.slice(startIndex)
  }

  const sectionId = segments[startIndex]?.sectionId
  if (!sectionId) {
    return [segments[startIndex]]
  }

  return segments.filter((segment) => segment.sectionId === sectionId)
}

export const splitSegmentForTtsRequest = (
  segment: ReadAlongSegment,
  maxLength: number
): ReadAlongTtsRequestSegment[] => {
  const normalizedMaxLength = Math.max(1, maxLength)
  if (segment.text.length <= normalizedMaxLength) {
    return [
      {
        id: `${segment.id}:part:0:${segment.sourceStart}:${segment.sourceEnd}`,
        parentSegmentId: segment.id,
        index: 0,
        text: segment.text,
        sourceStart: segment.sourceStart,
        sourceEnd: segment.sourceEnd
      }
    ]
  }

  const parts: ReadAlongTtsRequestSegment[] = []
  let current = ''
  let currentStartOffset = 0
  let cursor = 0

  const flush = () => {
    const text = current.trim()
    if (!text) return
    const sourceStart = segment.sourceStart + currentStartOffset
    const sourceEnd = sourceStart + text.length
    parts.push({
      id: `${segment.id}:part:${parts.length}:${sourceStart}:${sourceEnd}`,
      parentSegmentId: segment.id,
      index: parts.length,
      text,
      sourceStart,
      sourceEnd
    })
  }

  for (const wordMatch of segment.text.matchAll(/\S+\s*/g)) {
    const token = wordMatch[0]
    const tokenStart = wordMatch.index ?? cursor

    if (!current) {
      current = token
      currentStartOffset = tokenStart
    } else if ((current + token).trim().length <= normalizedMaxLength) {
      current += token
    } else {
      flush()
      current = token
      currentStartOffset = tokenStart
    }

    while (current.trim().length > normalizedMaxLength) {
      const text = current.trim().slice(0, normalizedMaxLength)
      const sourceStart = segment.sourceStart + currentStartOffset
      const sourceEnd = sourceStart + text.length
      parts.push({
        id: `${segment.id}:part:${parts.length}:${sourceStart}:${sourceEnd}`,
        parentSegmentId: segment.id,
        index: parts.length,
        text,
        sourceStart,
        sourceEnd
      })
      current = current.trim().slice(normalizedMaxLength)
      currentStartOffset += normalizedMaxLength
    }

    cursor = tokenStart + token.length
  }

  flush()

  return parts
}
