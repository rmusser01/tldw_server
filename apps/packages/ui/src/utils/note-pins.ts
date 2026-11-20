/**
 * Notes attached to a chat message as pinned context (decision [4]). The
 * composer's note picker (components/Option/Playground/note-mentions.ts)
 * makes them; the send paths and New chat read them here.
 *
 * An attached note belongs to the next message only: it is sent with that
 * message, even with file retrieval on (it is an explicit attachment), and
 * its chip clears once the message is sent or queued. Other pins keep their
 * own behaviour.
 */
import { formatPinnedResults, type RagPinnedResult } from "@/utils/rag-format"

export const NOTE_PINNED_ID_PREFIX = "note:"
export const NOTE_PINNED_TYPE = "note"

/** The note id of a pinned result attached from a note, or null. */
export const noteIdFromPinnedResult = (result: RagPinnedResult): string | null =>
  result.type === NOTE_PINNED_TYPE && result.id.startsWith(NOTE_PINNED_ID_PREFIX)
    ? result.id.slice(NOTE_PINNED_ID_PREFIX.length) || null
    : null

export const isNotePin = (result: RagPinnedResult): boolean =>
  noteIdFromPinnedResult(result) !== null

/** The pins without the attached notes (a new chat starts without them). */
export const withoutNotePins = (pins: RagPinnedResult[] | null | undefined): RagPinnedResult[] =>
  (pins ?? []).filter((result) => !isNotePin(result))

/**
 * The pins that go with the next message. Attached notes always go; other
 * pins go unless file retrieval replaces them or the caller leaves them out.
 */
export const sendablePins = (
  pins: RagPinnedResult[] | null | undefined,
  options: { fileRetrievalEnabled?: boolean; ignoreOtherPins?: boolean }
): RagPinnedResult[] =>
  (pins ?? []).filter(
    (result) =>
      isNotePin(result) || (!options.fileRetrievalEnabled && !options.ignoreOtherPins)
  )

type Translate = (key: string, fallback: string, options?: Record<string, unknown>) => string
const englishFallback: Translate = (_key, fallback, options) =>
  fallback.replace(/\{\{(\w+)\}\}/g, (_match, name) => String(options?.[name] ?? ""))

/**
 * An attached note as the model sees it: a label with its title, then its
 * content in a fence longer than any backtick run inside it, so the note
 * can't end the block early.
 */
export const formatAttachedNote = (note: RagPinnedResult, t: Translate = englishFallback): string => {
  const longestRun = Math.max(0, ...(note.snippet.match(/`+/g) ?? []).map((run) => run.length))
  const fence = "`".repeat(Math.max(3, longestRun + 1))
  const title = note.title?.trim() || t("playground:noteMentions.untitled", "Untitled note")
  const lines = [
    t("playground:noteMentions.attachedNoteLabel", "Attached note: {{title}}", { title }),
    fence,
    note.snippet,
    fence,
    ...(note.truncated ? [t("playground:noteMentions.shortened", "[Note shortened]")] : [])
  ]
  return lines.join("\n")
}

/** The message with the sendable pins appended, as the model will see it. */
export const appendPinnedContext = (
  message: string,
  pins: RagPinnedResult[] | null | undefined,
  options: { fileRetrievalEnabled?: boolean; ignoreOtherPins?: boolean; t?: Translate }
): string => {
  const sendable = sendablePins(pins, options)
  if (sendable.length === 0) return message
  const others = sendable.filter((result) => !isNotePin(result))
  const notes = sendable.filter(isNotePin)
  return [
    message,
    others.length ? formatPinnedResults(others, "markdown") : "",
    ...notes.map((note) => formatAttachedNote(note, options.t))
  ]
    .filter(Boolean)
    .join("\n\n")
}

/** Removes the notes that went with a sent or queued message. */
export const clearSentNotePins = (
  sent: RagPinnedResult[],
  store: {
    getState: () => {
      ragPinnedResults: RagPinnedResult[]
      setRagPinnedResults: (results: RagPinnedResult[]) => void
    }
  }
) => {
  const sentIds = new Set(sent.filter(isNotePin).map((result) => result.id))
  if (sentIds.size === 0) return
  const { ragPinnedResults, setRagPinnedResults } = store.getState()
  setRagPinnedResults((ragPinnedResults ?? []).filter((result) => !sentIds.has(result.id)))
}
