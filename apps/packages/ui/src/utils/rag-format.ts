export type RagPinnedResult = {
  id: string
  title?: string
  source?: string
  url?: string
  snippet: string
  type?: string
  mediaId?: number
  contextOrigin?: "media-library"
}

export type RagCopyFormat = "markdown" | "text"

const formatSourceLine = (url?: string, source?: string) => {
  if (url) return `Source: ${url}`
  if (source) return `Source: ${source}`
  return ""
}

export const formatRagResult = (
  result: RagPinnedResult,
  format: RagCopyFormat
) => {
  const title = result.title?.trim()
  const sourceLine = formatSourceLine(result.url, result.source)
  if (format === "markdown") {
    const parts = [
      title ? `**${title}**` : null,
      result.snippet,
      sourceLine
    ].filter(Boolean)
    return parts.join("\n\n")
  }
  const parts = [title, result.snippet, sourceLine].filter(Boolean)
  return parts.join("\n\n")
}

export const formatPinnedResults = (
  results: RagPinnedResult[],
  format: RagCopyFormat
) =>
  results
    .map((result) => formatRagResult(result, format))
    .join("\n\n")
