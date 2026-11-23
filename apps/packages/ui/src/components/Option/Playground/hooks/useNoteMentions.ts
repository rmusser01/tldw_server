/**
 * The composer's note picker (decision [4], CC-11+): "@query" on the web and
 * "/note query" everywhere search the user's notes; picking one attaches its
 * content as pinned context and removes the trigger text. See
 * note-mentions.ts for the triggers and utils/note-pins.ts for what is sent.
 *
 * The picker is a textbox autocomplete (AX-12, useComposerCombobox): focus
 * stays in the textarea and the keys below drive the list. It opens only
 * from the user's own typing (a caret they placed), never for a draft set
 * from code, and closes when the textarea loses focus. Results answer one
 * query on one account: a newer query, or another account, hides them until
 * its own search is back, so a fast Enter can't pick a stale note.
 */
import React from "react"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import {
  detectNoteTrigger,
  noteToPinnedResult,
  readNoteRows,
  removeNoteTrigger,
  type NoteRow
} from "../note-mentions"

const SEARCH_LIMIT = 8
const SEARCH_DEBOUNCE_MS = 200

export type NoteMentionStatus = "loading" | "ready" | "error"

type SearchState = { key: string; status: NoteMentionStatus; rows: NoteRow[] }

/** The note itself: search rows may be trimmed, so the attached content is read by id. */
const fetchNote = async (id: NoteRow["id"], signal: AbortSignal): Promise<NoteRow> => {
  const note = await tldwClient.getNote(id, { signal })
  const [row] = readNoteRows([note])
  if (!row || String(row.id) !== String(id)) throw new Error("note_changed")
  return row
}

export const useNoteMentions = ({
  value,
  textareaRef,
  onSetMessage,
  enabled = true
}: {
  value: string
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
  /** Replaces the draft; without it the picker is off. */
  onSetMessage?: (value: string) => void
  /** False while Notes is unavailable or the draft shows a collapsed paste. */
  enabled?: boolean
}) => {
  // Set only from the user's input while the textarea has focus.
  const [caret, setCaret] = React.useState<number | null>(null)
  const [dismissedAt, setDismissedAt] = React.useState<string | null>(null)
  const [accountEpoch, setAccountEpoch] = React.useState(0)
  const [search, setSearch] = React.useState<SearchState | null>(null)
  const [activeIndex, setActiveIndex] = React.useState(0)
  const pickRef = React.useRef<AbortController | null>(null)

  React.useEffect(
    () =>
      watchChatAccountChanges((invalidated) => {
        if (!invalidated) return
        pickRef.current?.abort()
        setAccountEpoch((epoch) => epoch + 1)
      }),
    []
  )
  React.useEffect(() => () => pickRef.current?.abort(), [])

  const syncCaret = React.useCallback(() => {
    const textarea = textareaRef.current
    setCaret(textarea && textarea === document.activeElement ? textarea.selectionStart : null)
  }, [textareaRef])
  const clearCaret = React.useCallback(() => setCaret(null), [])

  const trigger =
    onSetMessage && enabled && caret !== null
      ? detectNoteTrigger(value, caret, {
          // The extension keeps "@" for tab mentions; /note works everywhere.
          atEnabled: !isExtensionRuntime()
        })
      : null
  const open = trigger !== null && dismissedAt !== value
  const query = trigger?.query ?? ""
  const key = JSON.stringify([accountEpoch, query])
  const current = open && search?.key === key ? search : null
  const results = current?.rows ?? []
  const status: NoteMentionStatus = current?.status ?? "loading"

  React.useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setSearch({ key, status: "loading", rows: [] })
    setActiveIndex(0)
    const timer = setTimeout(() => {
      tldwClient
        .searchNotes(
          query,
          { limit: SEARCH_LIMIT, legacy_aliases: false, include_snippets: false },
          { signal: controller.signal }
        )
        .then((response) => {
          if (!controller.signal.aborted)
            setSearch({ key, status: "ready", rows: readNoteRows(response) })
        })
        .catch(() => {
          if (!controller.signal.aborted) setSearch({ key, status: "error", rows: [] })
        })
    }, query ? SEARCH_DEBOUNCE_MS : 0)
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [key, open, query])

  const select = React.useCallback(
    async (row: NoteRow) => {
      if (!trigger || !onSetMessage || status !== "ready") return
      pickRef.current?.abort()
      const controller = new AbortController()
      pickRef.current = controller
      const pickedFrom = { value, key }
      let note: NoteRow
      try {
        note = await fetchNote(row.id, controller.signal)
      } catch {
        if (!controller.signal.aborted) setSearch({ key, status: "error", rows: [] })
        return
      }
      // A newer query, draft or account makes this pick stale.
      if (controller.signal.aborted || pickRef.current !== controller || pickedFrom.key !== key) return
      pickRef.current = null
      const pinned = noteToPinnedResult(note)
      const { ragPinnedResults, setRagPinnedResults } = useStoreMessageOption.getState()
      const pins = ragPinnedResults ?? []
      if (!pins.some((result) => result.id === pinned.id)) setRagPinnedResults([...pins, pinned])
      const next = removeNoteTrigger(pickedFrom.value, trigger)
      onSetMessage(next.text)
      requestAnimationFrame(() => {
        const textarea = textareaRef.current
        if (!textarea) return
        textarea.focus()
        textarea.setSelectionRange(next.caret, next.caret)
      })
    },
    [key, onSetMessage, status, textareaRef, trigger, value]
  )

  /** True when the key belonged to the picker. */
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!open) return false
    // An IME commit, or Ctrl/Cmd+Enter (send), belongs to the composer.
    if (event.nativeEvent.isComposing || event.keyCode === 229) return false
    if (event.ctrlKey || event.metaKey || event.altKey) return false
    const count = results.length
    if ((event.key === "ArrowDown" || event.key === "ArrowUp") && count > 0) {
      event.preventDefault()
      setActiveIndex((index) =>
        event.key === "ArrowDown" ? (index + 1) % count : (index - 1 + count) % count
      )
      return true
    }
    if (event.key === "Escape") {
      event.preventDefault()
      setDismissedAt(value)
      return true
    }
    if (event.shiftKey) return false
    if (event.key === "Enter" || event.key === "Tab") {
      const row = status === "ready" ? results[activeIndex] : undefined
      if (row) {
        event.preventDefault()
        void select(row)
        return true
      }
      // "/note" with nothing to pick yet is not a message to send.
      if (trigger?.kind === "command" && event.key === "Enter") {
        event.preventDefault()
        return true
      }
    }
    return false
  }

  return {
    open,
    /** "/note" owns the composer while it is open (no tab mentions beside it). */
    commandOpen: open && trigger?.kind === "command",
    results,
    status,
    activeIndex,
    setActiveIndex,
    select,
    handleKeyDown,
    syncCaret,
    clearCaret
  }
}
