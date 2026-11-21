import { MessageSquare, StickyNote } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"

import {
  buildServerChatPath,
  planServerChatNavigation
} from "@/hooks/chat/server-chat-route"
import { buildNotePath } from "@/routes/hooks/useNotesRouteAddress"
import { tldwClient } from "@/services/tldw/TldwApiClient"

import type { CommandItem } from "./CommandPalette"
import {
  COMMAND_PALETTE_CONTENT_LIMIT,
  type CommandPaletteContentProvider
} from "./command-palette-content"

const readNoteRows = (response: unknown): Array<Record<string, unknown>> => {
  if (Array.isArray(response)) return response
  const record = (response ?? {}) as Record<string, unknown>
  for (const key of ["items", "notes", "results"]) {
    if (Array.isArray(record[key])) return record[key] as Array<Record<string, unknown>>
  }
  return []
}

const text = (value: unknown): string => (typeof value === "string" ? value.trim() : "")

/** A search snippet's segments as one line of plain text (NL-07). */
const snippetText = (snippet: unknown): string =>
  Array.isArray(snippet)
    ? snippet
        .map((segment) => {
          const value = segment && typeof segment === "object" ? (segment as { text?: unknown }).text : null
          return typeof value === "string" ? value : ""
        })
        .join("")
        .replace(/\s+/g, " ")
        .trim()
    : ""

/**
 * The web palette's content providers (XP-06 slice 1): saved chats and notes
 * matching the query, searched on the server by title and body. Enter opens
 * a chat at /chat?chat=<id> and a note at /notes?note=<id>.
 */
export const useCommandPaletteContentProviders = (): CommandPaletteContentProvider[] => {
  const navigate = useNavigate()
  const { t } = useTranslation(["common"])
  const navigateRef = React.useRef(navigate)
  const translateRef = React.useRef(t)
  React.useEffect(() => {
    navigateRef.current = navigate
    translateRef.current = t
  }, [navigate, t])

  return React.useMemo<CommandPaletteContentProvider[]>(
    () => [
      {
        id: "chats",
        search: async (query, signal) => {
          await tldwClient.initialize().catch(() => null)
          const { chats } = await tldwClient.searchConversationsWithMeta(
            {
              query,
              search_in: "title,content",
              limit: COMMAND_PALETTE_CONTENT_LIMIT,
              order_by: "hybrid"
            },
            { signal }
          )
          return chats
            .filter((chat) => text(chat.id))
            .map(
              (chat): CommandItem => ({
                id: `chat-${chat.id}`,
                label:
                  text(chat.title) ||
                  translateRef.current("common:commandPalette.untitledChat", "Untitled chat"),
                description: text(chat.match_snippet) || undefined,
                icon: <MessageSquare className="size-4" />,
                targetPath: buildServerChatPath(chat.id),
                category: "chat",
                action: () => {
                  const target = planServerChatNavigation(chat.id)
                  navigateRef.current(target.to, { replace: target.replace })
                }
              })
            )
        }
      },
      {
        id: "notes",
        search: async (query, signal) => {
          const response = await tldwClient.searchNotes(
            query,
            { limit: COMMAND_PALETTE_CONTENT_LIMIT, legacy_aliases: false },
            { signal }
          )
          return readNoteRows(response)
            .filter((note) => note?.id != null && String(note.id).trim())
            .map((note): CommandItem => {
              const noteId = String(note.id)
              const path = buildNotePath(noteId)
              return {
                id: `note-${noteId}`,
                label:
                  text(note.title) ||
                  translateRef.current("common:commandPalette.untitledNote", "Untitled note"),
                description: snippetText(note.snippet) || undefined,
                icon: <StickyNote className="size-4" />,
                targetPath: path,
                category: "note",
                action: () => navigateRef.current(path)
              }
            })
        }
      }
    ],
    []
  )
}
