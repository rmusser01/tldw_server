import type { NoteTaskStatus } from "@/services/notes-tasks"

export type ParsedChecklistItem = {
  lineIndex: number
  lineNumber: number
  startOffset: number
  endOffset: number
  indent: string
  bullet: string
  marker: "[ ]" | "[x]" | "[X]"
  checked: boolean
  text: string
  rawLine: string
  hasChildContent: boolean
}

const CHECKLIST_LINE_RE = /^(\s*)([-*+]\s+)(\[(?: |x|X)\])(\s+)(.*)$/

const getLineStartOffsets = (lines: string[]): number[] => {
  const offsets: number[] = []
  let offset = 0
  for (const line of lines) {
    offsets.push(offset)
    offset += line.length + 1
  }
  return offsets
}

const leadingWhitespaceLength = (line: string): number => {
  const match = line.match(/^\s*/)
  return match ? match[0].length : 0
}

const hasIndentedChildContent = (
  lines: string[],
  itemLineIndex: number,
  itemIndentLength: number
): boolean => {
  for (let index = itemLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index]
    if (!line.trim()) continue
    const nextChecklist = line.match(CHECKLIST_LINE_RE)
    const indentLength = leadingWhitespaceLength(line)
    if (nextChecklist && indentLength <= itemIndentLength) return false
    if (indentLength > itemIndentLength) return true
    return false
  }
  return false
}

export const parseChecklistItems = (markdown: string): ParsedChecklistItem[] => {
  const lines = markdown.split("\n")
  const offsets = getLineStartOffsets(lines)

  return lines.flatMap((line, index) => {
    const match = line.match(CHECKLIST_LINE_RE)
    if (!match) return []
    const [, indent, bullet, marker, , text] = match
    const itemIndentLength = indent.length
    return [
      {
        lineIndex: index,
        lineNumber: index + 1,
        startOffset: offsets[index],
        endOffset: offsets[index] + line.length,
        indent,
        bullet,
        marker: marker as ParsedChecklistItem["marker"],
        checked: marker.toLowerCase() === "[x]",
        text: text.trimEnd(),
        rawLine: line,
        hasChildContent: hasIndentedChildContent(lines, index, itemIndentLength)
      }
    ]
  })
}

export const toggleChecklistItemMarker = (
  markdown: string,
  lineNumber: number,
  checked: boolean
): string => {
  if (!Number.isFinite(lineNumber) || lineNumber < 1) return markdown
  const lines = markdown.split("\n")
  const lineIndex = Math.trunc(lineNumber) - 1
  const line = lines[lineIndex]
  if (line === undefined) return markdown
  if (!CHECKLIST_LINE_RE.test(line)) return markdown
  const nextMarker = checked ? "[x]" : "[ ]"
  lines[lineIndex] = line.replace(CHECKLIST_LINE_RE, `$1$2${nextMarker}$4$5`)
  return lines.join("\n")
}

export const getNextTaskStatus = (checked: boolean): NoteTaskStatus =>
  checked ? "open" : "done"

export const stripChecklistMetadataForLabel = (text: string): string =>
  text
    .replace(/<!--.*?-->/g, "")
    .replace(/\s+\{#[^}]+\}/g, "")
    .replace(/\s+/g, " ")
    .trim()
