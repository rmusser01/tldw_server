import React from 'react'

/** One run of a search excerpt; `match` marks a matched term (NL-07). */
export type NoteSearchSnippetSegment = { text: string; match: boolean }

/**
 * Validate the `snippet` the notes search endpoint returns: a list of
 * `{ text, match }` runs with at least one match. Anything else is null, so
 * the row falls back to the note's first line.
 */
export const normalizeNoteSearchSnippet = (value: unknown): NoteSearchSnippetSegment[] | null => {
  if (!Array.isArray(value) || value.length === 0) return null
  const segments: NoteSearchSnippetSegment[] = []
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') return null
    const { text, match } = entry as { text?: unknown; match?: unknown }
    if (typeof text !== 'string' || typeof match !== 'boolean') return null
    segments.push({ text, match })
  }
  return segments.some((segment) => segment.match) ? segments : null
}

type NoteSearchSnippetProps = {
  id?: string
  segments: NoteSearchSnippetSegment[]
}

/**
 * Where a note matched the search: the excerpt as plain text, each matched
 * term in a <mark>. Text renders as React text nodes, never as markup.
 */
export const NoteSearchSnippet = ({ id, segments }: NoteSearchSnippetProps) => (
  <div
    id={id}
    className="mt-1 line-clamp-2 break-words text-xs text-text-muted"
    data-testid="notes-search-snippet"
  >
    {segments.map((segment, index) =>
      segment.match ? (
        <mark key={index} className="rounded-sm bg-primary/20 px-0.5 text-text">
          {segment.text}
        </mark>
      ) : (
        <React.Fragment key={index}>{segment.text}</React.Fragment>
      )
    )}
  </div>
)
