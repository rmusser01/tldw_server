import type {
  WritingRevisionAction,
  WritingRevisionAnchor,
  WritingRevisionApplyPlan,
  WritingRevisionOperation,
  WritingRevisionProposal,
  WritingRevisionTarget
} from "./writing-revision-types"

const DEFAULT_ANCHOR_WINDOW = 80
const DEFAULT_LARGE_TARGET_CHARS = 1800

const clampIndex = (value: number, length: number): number => {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(length, Math.floor(value)))
}

const replaceRange = (
  text: string,
  start: number,
  end: number,
  replacementText: string
): string => text.slice(0, start) + replacementText + text.slice(end)

const findExactMatches = (
  text: string,
  searchText: string
): Array<{ start: number; end: number }> => {
  if (searchText.length === 0) return []

  const matches: Array<{ start: number; end: number }> = []
  let offset = 0
  while (offset <= text.length) {
    const foundAt = text.indexOf(searchText, offset)
    if (foundAt === -1) break
    matches.push({ start: foundAt, end: foundAt + searchText.length })
    offset = foundAt + 1
  }
  return matches
}

const findInsertionAnchorOffsets = (
  text: string,
  anchor: WritingRevisionAnchor
): number[] => {
  const offsets: number[] = []
  for (let offset = 0; offset <= text.length; offset += 1) {
    const prefixStart = offset - anchor.prefix.length
    if (prefixStart < 0) continue
    if (text.slice(prefixStart, offset) !== anchor.prefix) continue
    if (text.slice(offset, offset + anchor.suffix.length) !== anchor.suffix) {
      continue
    }
    offsets.push(offset)
  }
  return offsets
}

export const countWords = (text: string): number => {
  const matches = text.trim().match(/\S+/g)
  return matches ? matches.length : 0
}

export const createDocumentFingerprint = (text: string): string => {
  let hash = 2166136261
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(16)
}

export const buildInsertionAnchor = (
  text: string,
  offset: number,
  windowSize = DEFAULT_ANCHOR_WINDOW
): WritingRevisionAnchor => {
  const safeOffset = clampIndex(offset, text.length)
  const safeWindowSize = Math.max(0, Math.floor(windowSize))
  return {
    documentFingerprint: createDocumentFingerprint(text),
    prefix: text.slice(Math.max(0, safeOffset - safeWindowSize), safeOffset),
    suffix: text.slice(
      safeOffset,
      Math.min(text.length, safeOffset + safeWindowSize)
    )
  }
}

export const findParagraphRange = (
  text: string,
  cursor: number
): { start: number; end: number } => {
  const safeCursor = clampIndex(cursor, text.length)
  const before = text.lastIndexOf("\n\n", safeCursor - 1)
  const after = text.indexOf("\n\n", safeCursor)
  return {
    start: before === -1 ? 0 : before + 2,
    end: after === -1 ? text.length : after
  }
}

const makeRevisionTarget = (input: {
  text: string
  mode: WritingRevisionTarget["mode"]
  start: number
  end: number
  operation: WritingRevisionOperation
  label: string
  confirmationReason?: string
}): WritingRevisionTarget => {
  const normalizedStart = clampIndex(input.start, input.text.length)
  const normalizedEnd = clampIndex(input.end, input.text.length)
  const start = Math.min(normalizedStart, normalizedEnd)
  const end = Math.max(normalizedStart, normalizedEnd)
  const isTextChanging = input.operation !== "advisory"
  return {
    mode: input.mode,
    start,
    end,
    beforeText: input.text.slice(start, end),
    anchor: buildInsertionAnchor(input.text, start),
    label: input.label,
    requiresConfirmation: Boolean(isTextChanging && input.confirmationReason),
    confirmationReason: isTextChanging ? input.confirmationReason : undefined
  }
}

export const confirmRevisionTarget = (
  target: WritingRevisionTarget
): WritingRevisionTarget => ({
  ...target,
  requiresConfirmation: false,
  confirmationReason: undefined
})

export const resolveRevisionTarget = (input: {
  text: string
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  selection?: { start: number; end: number } | null
  cursor?: number | null
  preferredTargetMode?: WritingRevisionTarget["mode"] | null
  maxAutomaticTargetCharacters?: number
}): WritingRevisionTarget => {
  const { text, action, operation, selection } = input
  const maxAutomaticTargetCharacters =
    input.maxAutomaticTargetCharacters ?? DEFAULT_LARGE_TARGET_CHARS
  if (selection && selection.start !== selection.end) {
    return makeRevisionTarget({
      text,
      mode: "selection",
      start: selection.start,
      end: selection.end,
      operation,
      label: "selection"
    })
  }

  const cursor = clampIndex(input.cursor ?? text.length, text.length)
  if (action === "continue") {
    return makeRevisionTarget({
      text,
      mode: "cursor",
      start: cursor,
      end: cursor,
      operation: "insert",
      label: cursor === text.length ? "document end" : "cursor"
    })
  }

  if (action === "outline" || operation === "advisory") {
    return makeRevisionTarget({
      text,
      mode: "document",
      start: 0,
      end: text.length,
      operation: "advisory",
      label: "whole document"
    })
  }

  if (input.preferredTargetMode === "document") {
    return makeRevisionTarget({
      text,
      mode: "document",
      start: 0,
      end: text.length,
      operation,
      label: "whole document",
      confirmationReason:
        "Confirm before applying a whole-document text-changing request."
    })
  }

  const paragraph = findParagraphRange(text, cursor)
  const paragraphLength = paragraph.end - paragraph.start
  if (paragraphLength > 0 && paragraphLength <= maxAutomaticTargetCharacters) {
    return makeRevisionTarget({
      text,
      mode: "paragraph",
      start: paragraph.start,
      end: paragraph.end,
      operation,
      label: "current paragraph"
    })
  }

  return makeRevisionTarget({
    text,
    mode: "document",
    start: 0,
    end: text.length,
    operation,
    label: "whole document",
    confirmationReason: "The current paragraph could not be resolved safely."
  })
}

export const planRevisionApply = (
  text: string,
  proposal: WritingRevisionProposal
): WritingRevisionApplyPlan => {
  if (proposal.operation === "advisory") {
    return { type: "noop", reason: "Advisory revisions do not mutate text." }
  }

  if (proposal.target.requiresConfirmation) {
    return {
      type: "conflict",
      reason: "Confirm the revision target before applying text changes."
    }
  }

  const replacementText = proposal.replacementText
  if (typeof replacementText !== "string") {
    return {
      type: "noop",
      reason: "Text-changing revisions require replacement text."
    }
  }

  const start = clampIndex(proposal.target.start, text.length)
  const end = clampIndex(proposal.target.end, text.length)
  const normalizedStart = Math.min(start, end)
  const normalizedEnd = Math.max(start, end)
  const beforeText = proposal.target.beforeText

  if (
    proposal.operation === "insert" &&
    normalizedStart === normalizedEnd &&
    beforeText === ""
  ) {
    if (
      createDocumentFingerprint(text) === proposal.target.anchor.documentFingerprint
    ) {
      return {
        type: "apply",
        start: normalizedStart,
        end: normalizedEnd,
        nextText: replaceRange(
          text,
          normalizedStart,
          normalizedEnd,
          replacementText
        )
      }
    }

    const offsets = findInsertionAnchorOffsets(text, proposal.target.anchor)
    if (offsets.length === 1) {
      const nextStart = offsets[0]
      return {
        type: "retarget",
        start: nextStart,
        end: nextStart,
        nextText: replaceRange(text, nextStart, nextStart, replacementText)
      }
    }

    return {
      type: "conflict",
      reason: "The insertion anchor no longer identifies a unique target."
    }
  }

  if (beforeText.length === 0) {
    return {
      type: "conflict",
      reason: "Empty target text cannot validate a replacement."
    }
  }

  if (text.slice(normalizedStart, normalizedEnd) === beforeText) {
    return {
      type: "apply",
      start: normalizedStart,
      end: normalizedEnd,
      nextText: replaceRange(text, normalizedStart, normalizedEnd, replacementText)
    }
  }

  const matches = findExactMatches(text, beforeText)
  if (matches.length === 1) {
    const [match] = matches
    return {
      type: "retarget",
      start: match.start,
      end: match.end,
      nextText: replaceRange(text, match.start, match.end, replacementText)
    }
  }

  return {
    type: "conflict",
    reason: "The revision target no longer matches uniquely."
  }
}
