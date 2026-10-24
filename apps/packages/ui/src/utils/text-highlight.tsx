import React from "react"

/**
 * Split a query into the terms to highlight: each word, except that text in
 * double quotes stays together as one phrase (whitespace inside collapses).
 */
export function parseHighlightTerms(query: string): string[] {
  const terms: string[] = []
  for (const match of query.matchAll(/"([^"]*)"|(\S+)/g)) {
    const term = (match[1] ?? match[2] ?? "").replace(/"/g, "").trim().replace(/\s+/g, " ")
    if (term) terms.push(term)
  }
  return terms
}

/** Highlight `phrase` as a whole rather than word by word. */
export function toPhraseHighlightQuery(phrase: string): string {
  const cleaned = phrase.replace(/"/g, " ").trim().replace(/\s+/g, " ")
  return cleaned ? `"${cleaned}"` : ""
}

/**
 * Highlights all occurrences of a search query in the given text.
 * Returns a React node with matched text wrapped in <mark> elements.
 *
 * @param text - The text to search within
 * @param query - The search query to highlight
 * @param options - Optional configuration
 * @returns React node with highlighted matches, or the original text if no query
 */
export function highlightText(
  text: string,
  query: string,
  options: {
    caseSensitive?: boolean
    highlightClassName?: string
  } = {}
): React.ReactNode {
  if (!query || !text) {
    return text
  }

  const {
    caseSensitive = false,
    highlightClassName = "bg-warn/20 rounded px-0.5"
  } = options

  // Words match one by one; a "quoted phrase" matches as a whole (CS-11).
  const terms = parseHighlightTerms(query)
  if (terms.length === 0) {
    return text
  }

  // Longer terms first, so a phrase wins over the words inside it.
  const pattern = [...terms]
    .sort((a, b) => b.length - a.length)
    .map((term) =>
      term
        .split(" ")
        .map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
        .join("\\s+")
    )
    .join("|")
  const flags = caseSensitive ? "g" : "gi"
  const regex = new RegExp(`(${pattern})`, flags)

  const parts = text.split(regex)

  if (parts.length === 1) {
    // No matches found
    return text
  }

  const normalize = (value: string) => {
    const collapsed = value.replace(/\s+/g, " ")
    return caseSensitive ? collapsed : collapsed.toLowerCase()
  }
  const normalizedTerms = new Set(terms.map(normalize))

  return (
    <>
      {parts.map((part, index) => {
        // Check if this part matches any of the search terms
        const isMatch = normalizedTerms.has(normalize(part))

        if (isMatch) {
          return (
            <mark key={index} className={highlightClassName}>
              {part}
            </mark>
          )
        }
        return part
      })}
    </>
  )
}

/**
 * Checks if a text contains the search query.
 *
 * @param text - The text to search within
 * @param query - The search query
 * @param caseSensitive - Whether the search should be case-sensitive
 * @returns true if the text contains the query
 */
export function textContainsQuery(
  text: string,
  query: string,
  caseSensitive = false
): boolean {
  if (!query || !text) {
    return false
  }

  const terms = query.split(/\s+/).filter(Boolean)
  if (terms.length === 0) {
    return false
  }

  const searchText = caseSensitive ? text : text.toLowerCase()

  return terms.some((term) => {
    const searchTerm = caseSensitive ? term : term.toLowerCase()
    return searchText.includes(searchTerm)
  })
}

/**
 * Counts the number of matches for a query in the given text.
 *
 * @param text - The text to search within
 * @param query - The search query
 * @param caseSensitive - Whether the search should be case-sensitive
 * @returns The number of matches
 */
export function countMatches(
  text: string,
  query: string,
  caseSensitive = false
): number {
  if (!query || !text) {
    return 0
  }

  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  const terms = escapedQuery.split(/\s+/).filter(Boolean)
  if (terms.length === 0) {
    return 0
  }

  const pattern = terms.join("|")
  const flags = caseSensitive ? "g" : "gi"
  const regex = new RegExp(pattern, flags)

  const matches = text.match(regex)
  return matches ? matches.length : 0
}
