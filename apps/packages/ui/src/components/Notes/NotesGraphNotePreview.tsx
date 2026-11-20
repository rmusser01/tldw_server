import { stripKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"
import { toPlainPreview } from "@/utils/plain-text-preview"
import React from "react"
import { useTranslation } from "react-i18next"

const PREVIEW_MAX_LENGTH = 280

export type NotesGraphNotePreviewData = {
  noteId: string
  title: string
  content: string
  loading?: boolean
}

type NotesGraphNotePreviewProps = {
  preview: NotesGraphNotePreviewData
  onOpenInEditor?: (noteId: string) => void
}

/**
 * The picked note in Graph mode (NL-11): its title, the first lines as plain
 * text, and "Open in editor", which leaves Graph for the editor. Picking a
 * note no longer switches views by itself.
 */
const NotesGraphNotePreview: React.FC<NotesGraphNotePreviewProps> = ({
  preview,
  onOpenInEditor
}) => {
  const { t } = useTranslation(["option", "common"])
  const headingId = React.useId()
  const excerpt = preview.loading
    ? ""
    : toPlainPreview(stripKnowledgeNoteProvenance(preview.content), {
        skipLine: preview.title,
        maxLength: PREVIEW_MAX_LENGTH
      })

  return (
    <section
      aria-labelledby={headingId}
      className="border-b border-border p-3"
      data-testid="notes-graph-note-preview">
      <h2
        id={headingId}
        className="text-xs font-semibold uppercase text-text-muted">
        {t("option:notesSearch.graphPreviewHeading", {
          defaultValue: "Note preview"
        })}
      </h2>
      <p className="mt-1 break-words text-base font-semibold text-text">
        {preview.title ||
          t("option:notesSearch.graphPreviewUntitled", {
            defaultValue: "Untitled note"
          })}
      </p>
      <p className="mt-1 line-clamp-4 break-words text-sm text-text-muted">
        {preview.loading
          ? t("common:loading.title", { defaultValue: "Loading..." })
          : excerpt ||
            t("option:notesSearch.graphPreviewEmpty", {
              defaultValue: "This note has no text yet."
            })}
      </p>
      {onOpenInEditor ? (
        <button
          type="button"
          className="mt-2 min-h-9 border border-border bg-surface px-3 text-sm text-text hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          onClick={() => onOpenInEditor(preview.noteId)}>
          {t("option:notesSearch.graphOpenInEditor", {
            defaultValue: "Open in editor"
          })}
        </button>
      ) : null}
    </section>
  )
}

export default NotesGraphNotePreview
