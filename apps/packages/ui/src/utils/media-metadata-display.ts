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
        const body = remaining.slice("[/METADATA]".length)
        // Only the exact historical writer layout owns this eight-space prefix.
        if (
          text.startsWith("[METADATA]\n        {") &&
          json.slice(i + 1).startsWith("\n        [/METADATA]\n\n        ")
        ) return body.slice("\n\n        ".length)
        return body.replace(/^(?:\r?\n){1,2}|^ (?![ \t])/, "")
      } catch {
        return content
      }
    }
  }
  return content
}
