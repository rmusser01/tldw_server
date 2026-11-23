/**
 * The "Saved to Notes" toast for an answer saved as a plain note (XP-01),
 * with an "Open note" link to the note the server created. In the
 * extension the link opens the full page's Notes in a new tab.
 */
import React from "react"
import { buildNotePath } from "@/routes/hooks/useNotesRouteAddress"
import { resolveOptionsHref } from "@/utils/browser-runtime"

type Translate = (key: string, fallback: string) => string

export const savedAnswerNoteToast = (t: Translate, created: unknown) => {
  const rawId = (created as { id?: unknown } | null)?.id
  const noteId = rawId == null ? "" : String(rawId).trim()
  const saved = t("savedToNotes", "Saved to Notes")
  if (!noteId) return { content: saved }
  const href = resolveOptionsHref(buildNotePath(noteId))
  return {
    content: (
      <span>
        {saved}{" "}
        <a
          href={href}
          target={href.startsWith("/") ? undefined : "_blank"}
          rel="noreferrer"
          className="font-medium text-primary underline"
        >
          {t("openNote", "Open note")}
        </a>
      </span>
    ),
    // Long enough to reach the link.
    duration: 6
  }
}
