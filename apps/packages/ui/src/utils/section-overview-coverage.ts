const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const citations = /\[[^\]\n]+\](?:[ \t]*(?:\r?\n[ \t]*)?(?:[,;][ \t]*(?:\r?\n[ \t]*)?)?\[[^\]\n]+\])*/g
const brackets = /\[([^\]\n]+)\]/g
const fencePattern = /^ {0,3}(`{3,}|~{3,})(.*)$/
const headingPattern = /^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$/
const setext = /^ {0,3}(=+|-+)[ \t]*$/
const list = /^\s*(?:[-*+]\s+|\d+[.)]\s+)/
const reference = /^ {0,3}\[([^\]\n]+)\]:/
const inventories = new Set([
  "references", "sources", "citations", "evidence", "bibliography", "works cited",
  "source inventory", "citation inventory", "chapter coverage", "evidence ids", "citation ids"
])

export type SectionOverviewCoverage = {
  selected_sections: number
  completed_sections: number
  cited_sections: number
  missing_sections: Array<{ index: number; title: string }>
  complete: boolean
}

const label = (text: string) => text.replace(brackets, "").replace(list, "")
  .replace(/^\s*[*_]*(?:Source section|Chapter|Section)(?:\s+\d+)?[*_]*\s*:\s*[*_]*\s*/i, "")
  .replace(list, "").toLowerCase()
  .replace(/[^\p{L}\p{N}\s]/gu, " ").trim().split(/\s+/).join(" ")

// Match the backend's section_overview_coverage.py grammar; this is structural, not a support verdict.
function* proseBlocks(text: string, titles: Set<string>): Generator<string> {
  const lines = text.replace(/<!--[\s\S]*?(?:-->|$)/g, "").split(/\r\n?|\n/)
  let fence: { marker: string; quoteDepth: number; listIndent: number } | undefined
  let inventoryLevel: number | undefined
  let current: string[] = []
  for (const [index, original] of lines.entries()) {
    let line = original
    const quotePrefix = /^(?:[ \t]{0,3}>[ \t]?)+/.exec(line)?.[0] || ""
    const quoteDepth = quotePrefix.split(">").length - 1
    const unquoted = line.slice(quotePrefix.length)
    if (fence && (quoteDepth < fence.quoteDepth || (fence.listIndent > 0 && unquoted.trim() &&
        !unquoted.startsWith(" ".repeat(fence.listIndent))))) fence = undefined
    if (fence) {
      const closing = fencePattern.exec(unquoted.slice(fence.listIndent))
      if (closing && closing[1][0] === fence.marker[0] && closing[1].length >= fence.marker.length && !closing[2].trim()) fence = undefined
      continue
    }
    const listIndent = list.exec(unquoted)?.[0].length || 0
    const fenceMatch = fencePattern.exec(unquoted.slice(listIndent))
    if (fenceMatch) { fence = { marker: fenceMatch[1], quoteDepth, listIndent }; line = "" }
    const heading = headingPattern.exec(line)
    const following = index + 1 < lines.length ? setext.exec(lines[index + 1]) : null
    if (heading || (line.trim() && following)) {
      const level = heading ? heading[1].length : following![1][0] === "=" ? 1 : 2
      const title = label(heading ? heading[2] : line)
      if (inventoryLevel !== undefined && level <= inventoryLevel) inventoryLevel = undefined
      if (inventories.has(title)) inventoryLevel = level
      line = ""
    }
    if (inventoryLevel !== undefined || line.startsWith("    ") || line.startsWith("\t") || reference.test(line) || setext.test(line)) line = ""
    line = line.replace(/(`+).*?\1/g, "").replace(/!?\[[^\]\n]*\]\([^\n)]*\)/g, "")
    if (titles.has(label(line)) || inventories.has(label(line))) line = ""
    if ((!line.trim() || list.test(line)) && current.length) {
      yield current.join("\n")
      current = []
    }
    if (line.trim()) current.push(line)
  }
  if (current.length) yield current.join("\n")
}

/** Recompute representation from the exact overview and saved evidence; establish no new source authority. */
export const getSectionOverviewCoverage = (text: unknown, sections: unknown): SectionOverviewCoverage => {
  const selected: unknown[] = Array.isArray(sections) ? sections : []
  const overview = typeof text === "string" ? text : ""
  const completed = new Map<number, string>()
  const owners = new Map<string, number>()
  const ambiguous = new Set<string>()
  for (const [index, section] of selected.entries()) {
    if (!record(section) || section.status !== "completed") continue
    completed.set(index, String(section.title || `Section ${index + 1}`))
    for (const batch of Array.isArray(section.batches) ? section.batches : []) {
      for (const item of record(batch) && Array.isArray(batch.evidence) ? batch.evidence : []) {
        if (!record(item) || typeof item.id !== "string" || !item.id) continue
        if (owners.has(item.id) && owners.get(item.id) !== index) ambiguous.add(item.id)
        owners.set(item.id, index)
      }
    }
  }
  for (const id of ambiguous) owners.delete(id)
  const references = new Set(overview.split(/\r\n?|\n/).flatMap(line => {
    const match = reference.exec(line)
    return match ? [match[1].trim()] : []
  }))
  const titles = new Set(Array.from(completed.values(), label))
  const covered = new Set<number>()
  for (const block of proseBlocks(overview, titles)) {
    let cursor = 0
    for (const match of block.matchAll(citations)) {
      const prose = block.slice(cursor, match.index)
      const ids = Array.from(match[0].matchAll(brackets)).flatMap(group => group[1].split(",").map(id => id.trim().replace(/^Source /, "")))
      if (/\p{L}/u.test(prose) && !titles.has(label(prose)) && !inventories.has(label(prose)) && ids.length &&
          ids.every(id => owners.has(id) && !references.has(id))) {
        ids.forEach(id => covered.add(owners.get(id)!))
      }
      cursor = match.index! + match[0].length
    }
  }
  const missing = Array.from(completed, ([index, title]) => ({ index, title })).filter(item => !covered.has(item.index))
  return { selected_sections: selected.length, completed_sections: completed.size, cited_sections: covered.size,
    missing_sections: missing, complete: completed.size > 0 && missing.length === 0 }
}
