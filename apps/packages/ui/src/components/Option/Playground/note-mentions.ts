/**
 * Notes as message context in the chat composer (decision [4], CC-11+).
 *
 * Two triggers open the same note picker:
 * - "@query" at the caret, on the web. The extension keeps "@" for its tab
 *   mentions, so there `atEnabled` is false.
 * - A draft that is "/note" plus a search, on every surface. The slash menu's
 *   /note leaves "/note " in the composer for this.
 *
 * A picked note becomes one pinned result (`type: "note"`) that belongs to
 * the next message only: it is sent with that message (even with file
 * retrieval on, as an explicit attachment), and its chip clears once the
 * message is sent or queued. Other pins keep their own behaviour.
 */
import type { RagPinnedResult } from "@/utils/rag-format"
import { sliceCodeUnitsSafely } from "@/utils/safe-slice"
import { NOTE_PINNED_ID_PREFIX, NOTE_PINNED_TYPE } from "@/utils/note-pins"

export {
  appendPinnedContext,
  clearSentNotePins,
  isNotePin,
  noteIdFromPinnedResult,
  sendablePins,
  withoutNotePins
} from "@/utils/note-pins"

/** A note longer than this is shortened before it is attached. */
export const NOTE_CONTENT_MAX_CHARS = 12_000

export type NoteTrigger = {
  kind: "at" | "command"
  /** The trigger text's range in the draft, removed when a note is picked. */
  start: number
  end: number
  query: string
}

const COMMAND_PATTERN = /^\s*\/note\s+([\s\S]*)$/

export const detectNoteTrigger = (
  text: string,
  caret: number,
  options: { atEnabled: boolean }
): NoteTrigger | null => {
  const command = text.match(COMMAND_PATTERN)
  if (command) {
    return { kind: "command", start: 0, end: text.length, query: command[1].trim() }
  }
  if (!options.atEnabled) return null
  const beforeCaret = text.slice(0, Math.max(0, Math.min(caret, text.length)))
  const at = beforeCaret.lastIndexOf("@")
  if (at < 0) return null
  if (at > 0 && !/\s/.test(beforeCaret[at - 1])) return null
  const query = beforeCaret.slice(at + 1)
  if (/\s/.test(query)) return null
  return { kind: "at", start: at, end: beforeCaret.length, query }
}

/** The draft without the trigger text, and where the caret goes. */
export const removeNoteTrigger = (
  text: string,
  trigger: NoteTrigger
): { text: string; caret: number } => {
  if (trigger.kind === "command") return { text: "", caret: 0 }
  const before = text.slice(0, trigger.start)
  let after = text.slice(trigger.end)
  // "a @query b" leaves "a b", not "a  b".
  if ((!before || /\s$/.test(before)) && after.startsWith(" ")) after = after.slice(1)
  return { text: before + after, caret: before.length }
}

export type NoteRow = {
  id: string | number
  title?: string | null
  content?: string | null
}

export const noteToPinnedResult = (note: NoteRow): RagPinnedResult => {
  const content = (note.content ?? "").trim()
  const truncated = content.length > NOTE_CONTENT_MAX_CHARS
  return {
    id: `${NOTE_PINNED_ID_PREFIX}${note.id}`,
    type: NOTE_PINNED_TYPE,
    title: note.title?.trim() || undefined,
    source: "Notes",
    // The "[Note shortened]" marker is added, translated, when it is sent.
    snippet: truncated ? sliceCodeUnitsSafely(content, NOTE_CONTENT_MAX_CHARS).trimEnd() : content,
    ...(truncated ? { truncated: true } : {})
  }
}

/** Rows from a notes list or search response. */
export const readNoteRows = (response: unknown): NoteRow[] => {
  const rows = Array.isArray(response)
    ? response
    : ["items", "notes", "results"]
        .map((key) => (response as Record<string, unknown> | null)?.[key])
        .find(Array.isArray) ?? []
  return (rows as Array<Record<string, unknown>>)
    .filter((row) => row && row.id != null && String(row.id).trim())
    .map((row) => ({
      id: row.id as string | number,
      title: typeof row.title === "string" ? row.title : null,
      content: typeof row.content === "string" ? row.content : null
    }))
}

