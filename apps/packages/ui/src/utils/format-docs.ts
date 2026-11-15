import type { Document } from "@/types/document"

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
      const locations = spans.flatMap((span) => {
        if (!span || !Number.isSafeInteger(span.page_number) || span.page_number < 1) return []
        const label = typeof span.printed_page_label === "string" && /^[\w.-]{1,24}$/.test(span.printed_page_label)
          ? ` (printed page ${span.printed_page_label})` : ""
        return [`PDF page ${span.page_number}${label}`]
      })
      return `<doc id='${i}'>Source [${i + 1}]. ${locations.join("; ") || "Page location unavailable"}\n${escape(doc.pageContent)}</doc>`
    })
    .join("\n")
}
