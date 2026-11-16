import type { Document } from "@/types/document"

const pageLocation = (span: unknown): string | null => {
  if (!span || typeof span !== "object") return null
  const page = span as Record<string, unknown>
  if (typeof page.page_number !== "number" || !Number.isSafeInteger(page.page_number) || page.page_number < 1) return null
  const label = typeof page.printed_page_label === "string" && /^[\w.-]{1,24}$/.test(page.printed_page_label)
    ? ` (printed page ${page.printed_page_label})` : ""
  return `PDF page ${page.page_number}${label}`
}

const pageContent = (doc: Document): string => {
  const content = doc.pageContent
  const spans = doc.metadata?.page_spans
  const start = doc.metadata?.start_char
  const end = doc.metadata?.end_char
  const offset = (value: unknown): value is number =>
    typeof value === "number" && Number.isSafeInteger(value) && value >= 0
  if (!Array.isArray(spans) || !spans.length || spans.length > 20 || !offset(start) || !offset(end)) return content
  // Backend character offsets count Unicode code points, not UTF-16 code units.
  const characters = Array.from(content)
  if (end - start !== characters.length) return content
  const parts: string[] = []
  let previousEnd = start
  let previousPage = 0
  for (const span of spans) {
    const location = pageLocation(span)
    if (!location || !offset(span.start_char) || !offset(span.end_char) ||
        span.page_number <= previousPage || span.start_char < previousEnd ||
        span.start_char >= span.end_char || span.end_char > end ||
        (previousPage === 0 && span.start_char !== start)) return content
    const gap = characters.slice(previousEnd - start, span.start_char - start).join("")
    if (!/^[ \t\r\n\f\v]*$/.test(gap)) return content
    parts.push(gap, `\n[${location} begins]\n`, characters.slice(span.start_char - start, span.end_char - start).join(""))
    previousEnd = span.end_char
    previousPage = span.page_number
  }
  return previousEnd === end ? parts.join("") : content
}

/**
 * Format documents for inclusion in a prompt.
 * Keep source indices stable and include only bounded, validated page labels.
 */
export const formatDocs = (docs: Document[]) => {
  const escape = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  return docs
    .map((doc, i) => {
      const spans = Array.isArray(doc.metadata?.page_spans)
        ? doc.metadata.page_spans.slice(0, 20)
        : [{ page_number: doc.metadata?.page_number }]
      const locations = spans.flatMap((span) => pageLocation(span) ?? [])
      return `<doc id='${i}'>Source [${i + 1}]. ${locations.join("; ") || "Page location unavailable"}\n${escape(pageContent(doc))}</doc>`
    })
    .join("\n")
}
