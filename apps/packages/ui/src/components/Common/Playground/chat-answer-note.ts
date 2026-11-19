/**
 * Save to Notes for an answer the server can't link to a saved message
 * (XP-01, decision [5]): a temporary chat, a chat kept only on this device,
 * or a turn whose first send was not saved. The answer becomes a plain note
 * through the Notes API, titled from the question when there is one, with a
 * one-line provenance footer and the "from-chat" tag. It carries no link to a
 * server conversation; saved chats keep the server path, which does.
 */
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ChatScope } from "@/types/chat-scope"

export const FROM_CHAT_NOTE_KEYWORD = "from-chat"
const TITLE_MAX_CHARS = 60

type Translate = (
  key: string,
  fallback: string,
  options?: Record<string, unknown>
) => string

/** Offered for an assistant answer with text, outside a workspace (which has its own Save to workspace notes). */
export const canSaveAnswerAsNote = (input: {
  hasNotes: boolean
  isBot: boolean
  text: string | null | undefined
  errored: boolean
  scope?: ChatScope
}) =>
  input.hasNotes &&
  input.isBot &&
  Boolean(input.text?.trim()) &&
  !input.errored &&
  (!input.scope || input.scope.type === "global")

export const buildAnswerNoteTitle = (
  question: string | null | undefined,
  savedAt: Date,
  t: Translate
) => {
  const firstLine = (question ?? "").trim().split(/\r?\n/)[0]?.trim() ?? ""
  if (!firstLine)
    return t("playground:answerNote.untitled", "Chat answer, {{date}}", {
      date: savedAt.toLocaleDateString()
    })
  return firstLine.length > TITLE_MAX_CHARS
    ? `${firstLine.slice(0, TITLE_MAX_CHARS - 1).trimEnd()}…`
    : firstLine
}

export const buildAnswerNoteContent = (
  answer: string,
  input: { temporaryChat: boolean; modelName?: string | null; savedAt: Date },
  t: Translate
) => {
  const date = input.savedAt.toLocaleString()
  const model = input.modelName?.trim()
  const footer = input.temporaryChat
    ? model
      ? t("playground:answerNote.fromTemporaryChatModel", "Saved from a temporary chat ({{model}}) on {{date}}.", { model, date })
      : t("playground:answerNote.fromTemporaryChat", "Saved from a temporary chat on {{date}}.", { date })
    : model
      ? t("playground:answerNote.fromChatModel", "Saved from a chat ({{model}}) on {{date}}.", { model, date })
      : t("playground:answerNote.fromChat", "Saved from a chat on {{date}}.", { date })
  return `${answer.trim()}\n\n---\n${footer}`
}

export const saveAnswerAsNote = async (input: {
  answer: string
  question?: string | null
  temporaryChat: boolean
  modelName?: string | null
  t: Translate
  savedAt?: Date
}) => {
  const savedAt = input.savedAt ?? new Date()
  await tldwClient.initialize().catch(() => null)
  return tldwClient.createNote(
    buildAnswerNoteContent(input.answer, { ...input, savedAt }, input.t),
    {
      title: buildAnswerNoteTitle(input.question, savedAt, input.t),
      keywords: [FROM_CHAT_NOTE_KEYWORD]
    }
  )
}
