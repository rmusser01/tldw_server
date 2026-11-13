/**
 * One transcript format for every per-chat export and copy (CS-09): the chat
 * page's row and header menus, the side panel's tab menu, and the chat's
 * copy-as-Markdown. File contents stay in English, as before.
 *
 * A transcript holds every saved message, alternative replies included, so the
 * JSON export is lossless. The Markdown export is the readable newest path of
 * the chat (CM-N2) and says when alternatives were left out.
 */
import { newestLeafPath } from "@/utils/newest-leaf-path"

export type TranscriptRole = "user" | "assistant" | "system"

export type TranscriptMessage = {
  /** With `parentId`, lets the Markdown export follow the chat's newest path. */
  id?: string
  parentId?: string | null
  /** "user", "assistant" or "system"; another role the server records (for example "tool") is kept as is. */
  role: TranscriptRole | (string & {})
  /** The server's sender name, when it records one. */
  sender?: string | null
  content: string
  /** Shown instead of "You", "Assistant" or "System", for example a model name. */
  speaker?: string | null
  timestamp?: string | null
  version?: number
  model?: string | null
  provider?: string | null
  /** The message's stored metadata, as the server returns it. */
  metadata?: Record<string, unknown>
  images?: readonly string[]
}

export type ChatTranscript = {
  id: string
  title: string
  createdAt?: string | null
  updatedAt?: string | null
  model?: string | null
  messages: TranscriptMessage[]
  /** The chat has images this export doesn't carry; JSON says so as `images_omitted`. */
  imagesOmitted?: boolean
}

const ROLE_LABEL: Record<string, string> = {
  user: "You",
  assistant: "Assistant",
  system: "System"
}

/** "You", "Assistant", "System", or another role's own name, capitalized. */
const roleLabel = (role: string) =>
  ROLE_LABEL[role] ?? (role ? role.charAt(0).toUpperCase() + role.slice(1) : "Message")

/** One "## Speaker" section per message, for a copy or the body of an export. */
export const formatTranscriptMessagesMarkdown = (messages: readonly TranscriptMessage[]): string =>
  messages
    .map((message) => {
      const lines = [`## ${message.speaker?.trim() || roleLabel(message.role)}`]
      if (message.timestamp) lines.push(`*${new Date(message.timestamp).toLocaleString()}*`)
      lines.push("", message.content)
      for (const image of message.images ?? []) if (image) lines.push("", `![Image](${image})`)
      return lines.join("\n")
    })
    .join("\n\n")

/**
 * The messages a reader follows: the newest path when every message carries
 * its id (alternative replies left out), otherwise all of them in order.
 */
const readableMessages = (messages: readonly TranscriptMessage[]) =>
  messages.length && messages.every((message) => message.id)
    ? newestLeafPath(messages, (message) => ({ id: message.id!, parentId: message.parentId ?? null }))
    : { path: [...messages], hasAlternatives: false }

/** A whole chat as Markdown: its title, date and model, then its newest path. */
export const formatTranscriptMarkdown = (transcript: ChatTranscript): string => {
  const { path, hasAlternatives } = readableMessages(transcript.messages)
  const header = [`# ${transcript.title}`, ""]
  if (hasAlternatives)
    header.push("> Alternative replies are omitted here; the JSON export includes them.", "")
  if (transcript.createdAt) header.push(`**Date:** ${new Date(transcript.createdAt).toLocaleDateString()}`)
  if (transcript.model) header.push(`**Model:** ${transcript.model}`)
  header.push("", "---", "")
  return `${header.join("\n")}\n${formatTranscriptMessagesMarkdown(path)}\n`
}

/** The JSON export's own name and version, so a reader can recognize it. */
export const TRANSCRIPT_EXPORT_FORMAT = "tldw-chat-export"
export const TRANSCRIPT_EXPORT_VERSION = 1

export const formatTranscriptJson = ({ imagesOmitted, ...transcript }: ChatTranscript): string =>
  JSON.stringify(
    {
      format: TRANSCRIPT_EXPORT_FORMAT,
      version: TRANSCRIPT_EXPORT_VERSION,
      ...transcript,
      ...(imagesOmitted ? { images_omitted: true } : {})
    },
    null,
    2
  )

/** "<title>_<time>.<ext>", without characters a file name can't hold. */
export const transcriptFileName = (title: string, extension: "md" | "json", time = Date.now()): string => {
  const base = (title.trim() || "chat")
    .replace(/[/\\?%*:|"<>]/g, "-")
    .replace(/\s+/g, "_")
    .substring(0, 100)
  return `${base}_${time}.${extension}`
}

/** Save text as a file through the browser's download. */
export const downloadTextFile = (content: string, filename: string, mimeType: string) => {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }))
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export type TranscriptFormat = "markdown" | "json"

/** Download a transcript as Markdown or JSON. */
export const downloadTranscript = (transcript: ChatTranscript, format: TranscriptFormat) => {
  if (format === "json")
    downloadTextFile(formatTranscriptJson(transcript), transcriptFileName(transcript.title, "json"), "application/json")
  else
    downloadTextFile(formatTranscriptMarkdown(transcript), transcriptFileName(transcript.title, "md"), "text/markdown")
}
