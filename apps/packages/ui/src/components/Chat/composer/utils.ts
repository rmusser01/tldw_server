export const appendDictationTranscript = (
  currentMessage: string | null | undefined,
  transcript: string
): string => {
  const text = transcript.trim()
  if (!text) return currentMessage || ""
  const current = String(currentMessage || "")
  if (!current.trim()) return text
  return `${current.trimEnd()} ${text}`
}
