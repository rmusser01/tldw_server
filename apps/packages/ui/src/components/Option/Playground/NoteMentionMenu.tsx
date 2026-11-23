/**
 * The note picker's list and the attached-note chips (decision [4]).
 *
 * The list follows the slash menu's listbox pattern (AX-12): the textarea
 * points into it with aria-activedescendant and keeps focus. An attached note
 * shows as a chip with its title, which links to the note
 * (`/notes?note=<id>`), and a remove button.
 */
import { StickyNote, X } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import { SuggestionCountStatus } from "@/components/Common/SuggestionCountStatus"
import { comboboxOptionId } from "@/hooks/useComposerCombobox"
import { COMPOSER_TOOLBAR_TOKENS } from "@/components/Chat/composer/composer-toolbar-tokens"
import { cn } from "@/libs/utils"
import { buildNotePath } from "@/routes/hooks/useNotesRouteAddress"
import { sliceCodeUnitsSafely } from "@/utils/safe-slice"
import { useStoreMessageOption } from "@/store/option"
import type { NoteMentionStatus } from "./hooks/useNoteMentions"
import { noteIdFromPinnedResult, type NoteRow } from "./note-mentions"

type NoteMentionMenuProps = {
  open: boolean
  notes: NoteRow[]
  status: NoteMentionStatus
  activeIndex: number
  onActiveIndexChange: (index: number) => void
  onSelect: (note: NoteRow) => void
  listboxId: string
}

export const NoteMentionMenu: React.FC<NoteMentionMenuProps> = ({
  open,
  notes,
  status,
  activeIndex,
  onActiveIndexChange,
  onSelect,
  listboxId
}) => {
  const { t } = useTranslation(["playground"])
  const activeOptionRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    if (open) activeOptionRef.current?.scrollIntoView?.({ block: "nearest" })
  }, [open, activeIndex, notes])
  const untitled = t("playground:noteMentions.untitled", "Untitled note")
  const emptyText =
    status === "loading"
      ? t("playground:noteMentions.searching", "Searching notes…")
      : status === "error"
        ? t("playground:noteMentions.error", "Couldn't search your notes.")
        : t("playground:noteMentions.empty", "No notes found")
  const countText =
    notes.length === 0
      ? emptyText
      : String(
          t("playground:noteMentions.count", {
            count: notes.length,
            defaultValue: "{{count}} notes",
            defaultValue_one: "{{count}} note"
          })
        )
  return (
    <>
      <SuggestionCountStatus open={open} text={countText} />
      {open && (
        <div className="relative z-20 mx-3 mb-2">
          <div className="max-h-48 overflow-auto rounded-md border border-border bg-elevated shadow-lg">
            {notes.length === 0 ? (
              <div className="px-3 py-2 text-xs text-text-muted">{emptyText}</div>
            ) : (
              <div
                role="listbox"
                id={listboxId}
                aria-label={String(t("playground:noteMentions.label", "Notes"))}
              >
                {notes.map((note, index) => (
                  <div
                    key={String(note.id)}
                    ref={index === activeIndex ? activeOptionRef : undefined}
                    id={comboboxOptionId(listboxId, index)}
                    role="option"
                    aria-selected={index === activeIndex}
                    // Focus stays in the composer; the option is chosen by click.
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => onActiveIndexChange(index)}
                    onClick={() => onSelect(note)}
                    className={cn(
                      "flex w-full cursor-pointer items-start gap-2 px-3 py-2 text-left text-sm text-text transition-colors",
                      index === activeIndex && "bg-surface2"
                    )}
                  >
                    <StickyNote className="mt-0.5 size-3.5 shrink-0 text-text-muted" aria-hidden="true" />
                    <span className="flex min-w-0 flex-col">
                      <span className="truncate">{note.title?.trim() || untitled}</span>
                      {note.content?.trim() && (
                        <span className="truncate text-xs text-text-subtle">
                          {sliceCodeUnitsSafely(note.content.trim().replace(/\s+/g, " "), 120)}
                        </span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

/** The notes attached to the next message, each removable. */
export const AttachedNoteChips: React.FC<{ onRemoved?: () => void }> = ({ onRemoved }) => {
  const { t } = useTranslation(["playground"])
  const pinned = useStoreMessageOption((state) => state.ragPinnedResults)
  const setPinned = useStoreMessageOption((state) => state.setRagPinnedResults)
  const notes = (pinned ?? []).flatMap((result) => {
    const noteId = noteIdFromPinnedResult(result)
    return noteId ? [{ result, noteId }] : []
  })
  if (notes.length === 0) return null
  return (
    <div
      role="group"
      aria-label={t("playground:noteMentions.attached", "Attached notes")}
      className="flex flex-wrap gap-1 px-3 pb-1"
    >
      {notes.map(({ result, noteId }) => {
        const title = result.title || t("playground:noteMentions.untitled", "Untitled note")
        return (
          <span
            key={result.id}
            data-testid="attached-note-chip"
            className="inline-flex min-h-11 max-w-full items-center gap-1 rounded-full border border-border bg-surface2 pl-2 text-xs text-text sm:min-h-8 [@media(pointer:coarse)]:min-h-11"
          >
            <StickyNote className="size-3 shrink-0 text-text-muted" aria-hidden="true" />
            <Link
              to={buildNotePath(noteId)}
              title={t("playground:noteMentions.open", "Open this note")}
              className="min-w-0 truncate hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              {title}
            </Link>
            <button
              type="button"
              aria-label={t("playground:noteMentions.remove", "Remove note {{title}}", { title })}
              onClick={() => {
                setPinned((pinned ?? []).filter((item) => item.id !== result.id))
                // The chip and its button are gone; typing carries on.
                onRemoved?.()
              }}
              // 32px on desktop, 44px on touch (decision 3).
              className={cn(
                "flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-text-muted hover:bg-surface hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                COMPOSER_TOOLBAR_TOKENS.iconControl
              )}
            >
              <X className="size-3" aria-hidden="true" />
            </button>
          </span>
        )
      })}
    </div>
  )
}
