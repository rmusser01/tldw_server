/** Hide only a valid leading ingestion envelope; retain malformed source text. */
export const stripMediaMetadata = (content: string): string => {
  const text = content.trimStart()
  if (!text.startsWith("[METADATA]")) return content
  const json = text.slice("[METADATA]".length).trimStart()
  if (!json.startsWith("{")) return content

  let depth = 0
  let quoted = false
  let escaped = false
  for (let i = 0; i < json.length; i += 1) {
    const char = json[i]
    if (quoted) {
      if (escaped) escaped = false
      else if (char === "\\") escaped = true
      else if (char === '"') quoted = false
    } else if (char === '"') {
      quoted = true
    } else if (char === "{") {
      depth += 1
    } else if (char === "}" && --depth === 0) {
      const remaining = json.slice(i + 1).trimStart()
      if (!remaining.startsWith("[/METADATA]")) return content
      try {
        JSON.parse(json.slice(0, i + 1))
        return remaining.slice("[/METADATA]".length).trim()
      } catch {
        return content
      }
    }
  }
  return content
}
