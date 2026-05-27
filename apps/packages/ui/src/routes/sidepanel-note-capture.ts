import { CAPTURED_NOTE_KEYWORD } from "@/services/note-capture"

type BuildSidepanelCapturedNotePayloadInput = {
  content: string
  title: string
  sourceUrl?: string
}

export const appendSourceUrlToCapturedNoteContent = (
  content: string,
  sourceUrl?: string
): string => {
  const trimmedContent = content.trim()
  const trimmedSourceUrl = String(sourceUrl || "").trim()

  if (!trimmedSourceUrl || trimmedContent.includes(trimmedSourceUrl)) {
    return trimmedContent
  }

  return `${trimmedContent}\n\nSource: ${trimmedSourceUrl}`
}

export const buildSidepanelCapturedNotePayload = ({
  content,
  title,
  sourceUrl
}: BuildSidepanelCapturedNotePayloadInput) => ({
  content: appendSourceUrlToCapturedNoteContent(content, sourceUrl),
  noteFields: {
    title,
    keywords: [CAPTURED_NOTE_KEYWORD]
  }
})
