/**
 * One transcript format for every per-chat export and copy (CS-09): the chat
 * page's row and header menus, the side panel's tab menu, and the chat's
 * copy-as-Markdown. File contents stay in English, as before.
 */

export type TranscriptRole = "user" | "assistant" | "system"

export type TranscriptMessage = {
  role: TranscriptRole
  content: string
  /** Shown instead of "You", "Assistant" or "System", for example a model name. */
  speaker?: string | null
  timestamp?: string | null
  model?: string | null
  images?: readonly string[]
}

export type ChatTranscript = {
  id: string
  title: string
  createdAt?: string | null
  updatedAt?: string | null
  model?: string | null
  messages: TranscriptMessage[]
}

const ROLE_LABEL: Record<TranscriptRole, string> = {
  user: "You",
  assistant: "Assistant",
  system: "System"
}

/** One "## Speaker" section per message, for a copy or the body of an export. */
export const formatTranscriptMessagesMarkdown = (messages: readonly TranscriptMessage[]): string =>
  messages
    .map((message) => {
      const lines = [`## ${message.speaker?.trim() || ROLE_LABEL[message.role]}`]
      if (message.timestamp) lines.push(`*${new Date(message.timestamp).toLocaleString()}*`)
      lines.push("", message.content)
      for (const image of message.images ?? []) if (image) lines.push("", `![Image](${image})`)
      return lines.join("\n")
    })
    .join("\n\n")

/** A whole chat as Markdown: its title, date and model, then its messages. */
export const formatTranscriptMarkdown = (transcript: ChatTranscript): string => {
  const header = [`# ${transcript.title}`, ""]
  if (transcript.createdAt) header.push(`**Date:** ${new Date(transcript.createdAt).toLocaleDateString()}`)
  if (transcript.model) header.push(`**Model:** ${transcript.model}`)
  header.push("", "---", "")
  return `${header.join("\n")}\n${formatTranscriptMessagesMarkdown(transcript.messages)}\n`
}

export const formatTranscriptJson = (transcript: ChatTranscript): string =>
  JSON.stringify(transcript, null, 2)

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
