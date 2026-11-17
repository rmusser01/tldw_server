import { NOTE_ASSIST_STOP_WORDS } from './notes-manager-utils'

/**
 * Words that say how a note is organised rather than what it is about. They
 * top a word count in almost any note, so they never make useful tags.
 */
const NOTE_KEYWORD_DISCOURSE_WORDS = new Set([
  'all', 'another', 'any', 'been', 'being', 'both', 'did', 'does', 'doing', 'done', 'each',
  'even', 'etc', 'example', 'examples', 'few', 'first', 'second', 'third', 'last', 'next',
  'get', 'gets', 'got', 'had', 'her', 'here', 'him', 'his', 'how', 'idea', 'ideas',
  'important', 'include', 'includes', 'just', 'key', 'like', 'lot', 'lots', 'main', 'make',
  'makes', 'many', 'may', 'might', 'more', 'most', 'much', 'must', 'need', 'needs', 'new',
  'note', 'notes', 'now', 'one', 'only', 'other', 'others', 'own', 'paragraph',
  'paragraphs', 'part', 'parts', 'point', 'points', 'really', 'same', 'section',
  'sections', 'see', 'she', 'some', 'something', 'still', 'such', 'take', 'takes',
  'thing', 'things', 'these', 'those', 'them', 'two', 'three', 'very', 'want', 'way',
  'ways', 'well', 'who', 'why', 'yet'
])

/** At most this many suggestions are offered. */
const MAX_SUGGESTIONS = 5
/** A word that is not already a tag must appear at least this often. */
const MIN_NEW_TAG_COUNT = 2

export type NoteKeywordSuggestions = {
  /** Tags to offer, existing tags first. */
  suggestions: string[]
  /** The suggestions to tick by default: only those that are already tags. */
  preselected: string[]
}

const isCandidateWord = (token: string): boolean =>
  !NOTE_ASSIST_STOP_WORDS.has(token) &&
  !NOTE_KEYWORD_DISCOURSE_WORDS.has(token) &&
  !/^\d+$/.test(token)

/**
 * Suggest tags for a note from word frequency (NL-17).
 *
 * Plurals are folded into their singular when both appear, words that are
 * already tags elsewhere are offered first (in their existing spelling), and
 * a new word has to appear at least twice. Only existing tags are pre-ticked,
 * so the writer opts in to anything new.
 */
export const suggestNoteKeywords = (
  rawContent: string,
  noteKeywords: readonly string[],
  knownKeywords: readonly string[] = []
): NoteKeywordSuggestions => {
  const onNote = new Set(
    noteKeywords.map((keyword) => String(keyword || '').trim().toLowerCase()).filter(Boolean)
  )
  const known = new Map<string, string>()
  for (const keyword of knownKeywords) {
    const text = String(keyword || '').trim()
    if (text && !known.has(text.toLowerCase())) known.set(text.toLowerCase(), text)
  }

  const counts = new Map<string, number>()
  const tokens = String(rawContent || '').toLowerCase().match(/[a-z0-9][a-z0-9-]{2,}/g) || []
  for (const token of tokens) {
    if (!isCandidateWord(token)) continue
    counts.set(token, (counts.get(token) || 0) + 1)
  }
  // Fold "groups" into "group" when the singular is used too.
  for (const [token, count] of Array.from(counts.entries())) {
    if (token.length <= 3 || !token.endsWith('s') || token.endsWith('ss')) continue
    const singular = token.slice(0, -1)
    if (!counts.has(singular) || known.has(token)) continue
    counts.set(singular, (counts.get(singular) || 0) + count)
    counts.delete(token)
  }

  const ranked = Array.from(counts.entries())
    .filter(([token, count]) => !onNote.has(token) && (known.has(token) || count >= MIN_NEW_TAG_COUNT))
    .sort((a, b) => {
      const knownA = known.has(a[0]) ? 1 : 0
      const knownB = known.has(b[0]) ? 1 : 0
      if (knownA !== knownB) return knownB - knownA
      if (b[1] !== a[1]) return b[1] - a[1]
      return a[0].localeCompare(b[0])
    })
    .slice(0, MAX_SUGGESTIONS)

  const suggestions = ranked.map(([token]) => known.get(token) ?? token)
  const preselected = ranked.filter(([token]) => known.has(token)).map(([token]) => known.get(token) as string)
  return { suggestions, preselected }
}
