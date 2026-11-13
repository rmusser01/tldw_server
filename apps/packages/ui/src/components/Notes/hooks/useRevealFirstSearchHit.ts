import React from 'react'
import { revealTextareaOffset } from '../notes-manager-utils'

const SEARCH_TERM_PATTERN = /"([^"]+)"|(\S+)/g

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Offset of `term` in `text`, ignoring case; a regex keeps offsets exact. */
const indexOfIgnoringCase = (text: string, term: string): number => {
  if (!term) return -1
  const match = new RegExp(escapeRegExp(term), 'i').exec(text)
  return match ? match.index : -1
}

/**
 * Where a search query first matches a note body: the whole query (quotes
 * dropped), else the earliest of its terms or quoted phrases. -1 when nothing
 * matches.
 */
export const findFirstSearchHit = (content: string, query: string): number => {
  const whole = query.replace(/"/g, ' ').replace(/\s+/g, ' ').trim()
  const wholeIndex = indexOfIgnoringCase(content, whole)
  if (wholeIndex >= 0) return wholeIndex
  let first = -1
  for (const match of query.matchAll(SEARCH_TERM_PATTERN)) {
    const index = indexOfIgnoringCase(content, (match[1] || match[2] || '').trim())
    if (index >= 0 && (first < 0 || index < first)) first = index
  }
  return first
}

type RevealFirstSearchHitOptions = {
  selectedId: string | number | null
  content: string
  query: string
  textareaRef: React.MutableRefObject<HTMLTextAreaElement | null>
  /** The Markdown textarea is showing (not the rich editor or preview). */
  enabled: boolean
}

/**
 * NL-07: when a note opens while a search is active, scroll its Markdown
 * editor to the first hit. Focus and the caret stay where they are, and a
 * note the user is typing in (a new draft getting its id) is left alone.
 */
export const useRevealFirstSearchHit = ({
  selectedId,
  content,
  query,
  textareaRef,
  enabled
}: RevealFirstSearchHitOptions) => {
  // The note the reveal last ran for: it runs once per opened note, not per edit.
  const handledIdRef = React.useRef<string | number | null | undefined>(undefined)

  React.useEffect(() => {
    if (handledIdRef.current === selectedId) return
    handledIdRef.current = selectedId
    if (selectedId == null || !enabled || !query.trim()) return
    const textarea = textareaRef.current
    if (!textarea || textarea.ownerDocument.activeElement === textarea) return
    const offset = findFirstSearchHit(content, query)
    if (offset < 0) return
    window.requestAnimationFrame(() => {
      if (textareaRef.current === textarea) revealTextareaOffset(textarea, offset)
    })
  }, [content, enabled, query, selectedId, textareaRef])
}
