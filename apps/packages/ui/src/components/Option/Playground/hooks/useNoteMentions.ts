/**
 * The composer's note picker (decision [4], CC-11+): "@query" on the web and
 * "/note query" everywhere search the user's notes; picking one attaches its
 * content as pinned context and removes the trigger text. See
 * note-mentions.ts for the triggers and the attached shape.
 *
 * The picker is a textbox autocomplete (AX-12, useComposerCombobox): focus
 * stays in the textarea and the keys below drive the list.
 */
import React from "react"
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

export const useNoteMentions = ({
  value,
  textareaRef,
  onSetMessage
}: {
  value: string
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
  /** Replaces the draft; without it the picker is off. */
  onSetMessage?: (value: string) => void
}) => {
  const [caret, setCaret] = React.useState<number | null>(null)
  const [dismissedAt, setDismissedAt] = React.useState<string | null>(null)
  const [results, setResults] = React.useState<NoteRow[]>([])
  const [status, setStatus] = React.useState<NoteMentionStatus>("loading")
  const [activeIndex, setActiveIndex] = React.useState(0)

  // The caret matters only while the user is typing in the textarea; a draft
  // set from elsewhere is read up to its end.
  const syncCaret = React.useCallback(() => {
    const textarea = textareaRef.current
    setCaret(textarea && textarea === document.activeElement ? textarea.selectionStart : null)
  }, [textareaRef])
  React.useLayoutEffect(syncCaret, [syncCaret, value])

  const trigger = onSetMessage
    ? detectNoteTrigger(value, caret ?? value.length, {
        // The extension keeps "@" for tab mentions; /note works everywhere.
        atEnabled: !isExtensionRuntime()
      })
    : null
  const open = trigger !== null && dismissedAt !== value
  const query = trigger?.query ?? ""

  React.useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setStatus("loading")
    const timer = setTimeout(() => {
      tldwClient
        .searchNotes(
          query,
          { limit: SEARCH_LIMIT, legacy_aliases: false },
          { signal: controller.signal }
        )
        .then((response) => {
          if (controller.signal.aborted) return
          setResults(readNoteRows(response))
          setActiveIndex(0)
          setStatus("ready")
        })
        .catch(() => {
          if (controller.signal.aborted) return
          setResults([])
          setStatus("error")
        })
    }, query ? SEARCH_DEBOUNCE_MS : 0)
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [open, query])

  const select = React.useCallback(
    (note: NoteRow) => {
      if (!trigger || !onSetMessage) return
      const pinned = noteToPinnedResult(note)
      const { ragPinnedResults, setRagPinnedResults } = useStoreMessageOption.getState()
      const current = ragPinnedResults ?? []
      if (!current.some((result) => result.id === pinned.id))
        setRagPinnedResults([...current, pinned])
      const next = removeNoteTrigger(value, trigger)
      onSetMessage(next.text)
      requestAnimationFrame(() => {
        const textarea = textareaRef.current
        if (!textarea) return
        textarea.focus()
        textarea.setSelectionRange(next.caret, next.caret)
      })
    },
    [onSetMessage, textareaRef, trigger, value]
  )

  /** True when the key belonged to the picker. */
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!open || event.nativeEvent.isComposing) return false
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
    if ((event.key === "Enter" && !event.shiftKey) || event.key === "Tab") {
      const note = results[activeIndex]
      if (note) {
        event.preventDefault()
        select(note)
        return true
      }
      // "/note" with nothing to pick is not a message to send.
      if (trigger?.kind === "command" && event.key === "Enter") {
        event.preventDefault()
        return true
      }
    }
    return false
  }

  return {
    open,
    results: open ? results : [],
    status,
    activeIndex,
    setActiveIndex,
    select,
    handleKeyDown,
    syncCaret
  }
}
