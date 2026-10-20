import { toPhraseHighlightQuery } from "@/utils/text-highlight"

export type ThreadSearchDirection = 1 | -1

export type ThreadSearchMessage = {
  message?: string | null
}

const collapseWhitespace = (value: string): string => value.replace(/\s+/g, " ")

export const normalizeThreadSearchQuery = (value: string): string =>
  collapseWhitespace(value.trim().toLowerCase())

export const collectThreadSearchMatches = (
  messages: ThreadSearchMessage[],
  query: string
): number[] => {
  const needle = normalizeThreadSearchQuery(query)
  if (!needle) return []

  const matches: number[] = []
  messages.forEach((entry, index) => {
    const content =
      typeof entry?.message === "string"
        ? collapseWhitespace(entry.message.toLowerCase())
        : ""
    if (content.includes(needle)) {
      matches.push(index)
    }
  })
  return matches
}

/**
 * The query the transcript highlights: the whole typed phrase, and nothing
 * once the search bar is closed (CS-11).
 */
export const threadSearchHighlightQuery = (
  open: boolean,
  query: string
): string => (open ? toPhraseHighlightQuery(query) : "")

/**
 * When the active match sits in a message taller than the transcript,
 * centring the message can leave the match off screen, so bring its first
 * highlight into view instead. Returns whether it scrolled.
 */
export const revealThreadSearchMark = (
  container: HTMLElement | null,
  messageIndex: number,
  behavior: ScrollBehavior
): boolean => {
  if (!container) return false
  const message = container.querySelector<HTMLElement>(
    `[data-testid="chat-message"][data-index="${messageIndex}"]`
  )
  if (!message) return false
  if (message.getBoundingClientRect().height <= container.clientHeight) {
    return false
  }
  const mark = message.querySelector<HTMLElement>("mark")
  if (!mark) return false
  mark.scrollIntoView({ block: "center", behavior })
  return true
}

export const getWrappedMatchIndex = (
  currentIndex: number,
  totalMatches: number,
  direction: ThreadSearchDirection
): number => {
  if (totalMatches <= 0) return 0
  const normalizedCurrent =
    currentIndex >= 0 && currentIndex < totalMatches ? currentIndex : 0
  return (normalizedCurrent + direction + totalMatches) % totalMatches
}
