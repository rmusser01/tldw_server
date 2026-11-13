/**
 * One-line plain-text previews of Markdown (NL-04, #3101).
 *
 * Lists show a note's first meaningful line. Raw Markdown there reads as noise
 * (`**Mood:**`, ```` ```sql ````, `[[Title]]`), so this strips the common
 * markup and keeps the words. It is a display helper, not a Markdown parser:
 * anything it does not recognise stays as written.
 *
 * Only the start of a line is read: about four times the preview length.
 * That keeps the emphasis patterns linear in what a preview can show, however
 * long the line is.
 */

const DEFAULT_MAX_LENGTH = 140
/** How much of a line is read, as a multiple of the preview length. */
const SCAN_FACTOR = 4
/** The most of one line markdownLineToPlainText reads when called on its own. */
const MAX_LINE_SCAN = 4000
const ELLIPSIS = '…'

const FENCE_LINE = /^(`{3,}|~{3,})/
const RULE_LINE = /^([-*_])(\s*\1){2,}\s*$/
const SETEXT_UNDERLINE = /^=+\s*$/
const TABLE_SEPARATOR_LINE = /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/
const FRONT_MATTER_FENCE = /^---\s*$/
const FRONT_MATTER_END = /^(---|\.\.\.)\s*$/
const FRONT_MATTER_KEY = /^[\w-]+\s*:/
/** Front matter longer than this is not front matter. */
const MAX_FRONT_MATTER_LINES = 50

/** A link or image target: one level of parentheses in the URL, an optional title. */
const LINK_TARGET = String.raw`\((?:[^()\s]|\([^()\s]*\))*(?:\s+"[^"]*")?\)`
const IMAGE = new RegExp(String.raw`!\[([^\]]*)\]` + LINK_TARGET, 'g')
const LINK = new RegExp(String.raw`\[([^\]]+)\]` + LINK_TARGET, 'g')
const AUTOLINK = /<((?:https?|mailto):[^<>\s]+)>/g
const HTML_TAG = /<\/?[a-zA-Z][^<>]*>/g
const CALLOUT_MARKER = /^\[![\w-]+\][+-]?\s*/

/** A backslash-escaped character is held in the private use area until the markup is gone. */
const ESCAPED_CHARACTER = /\\([\\`*_{}[\]()#+\-.!|~<>])/g
const ESCAPE_BASE = 0xe000
const HELD_CHARACTER = /[-]/g

/** `[[id:UUID]]` links name a note by id; the raw id is not preview text (XP-14). */
const ID_WIKILINK_PREFIX = /^id:/i

const replaceWikilink = (_match: string, inner: string): string => {
  const [target, alias] = inner.split('|')
  if (alias?.trim()) return alias.trim()
  const text = (target || '').trim()
  return ID_WIKILINK_PREFIX.test(text) ? '' : text
}

/** `| a | b |` reads as "a · b". */
const tableRowToText = (line: string): string =>
  line.startsWith('|')
    ? line
        .replace(/^\|/, '')
        .replace(/\|\s*$/, '')
        .split('|')
        .map((cell) => cell.trim())
        .filter(Boolean)
        .join(' · ')
    : line

/** Strips one line's Markdown markup: headings, quotes, callouts, lists, tables, HTML, emphasis, code, links. */
export const markdownLineToPlainText = (line: string): string => {
  const held = line
    .slice(0, MAX_LINE_SCAN)
    .trim()
    .replace(ESCAPED_CHARACTER, (_match, character: string) =>
      String.fromCharCode(ESCAPE_BASE + character.charCodeAt(0))
    )
  const blocks = held
    .replace(/^#{1,6}\s+/, '')
    .replace(/\s+#+\s*$/, '')
    .replace(/^(>\s?)+/, '')
    .replace(CALLOUT_MARKER, '')
    .replace(/^([-*+]|\d+[.)])\s+/, '')
    .replace(/^\[[ xX]\]\s+/, '')
  return tableRowToText(blocks)
    .replace(AUTOLINK, '$1')
    .replace(HTML_TAG, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[\[([^\]]+)\]\]/g, replaceWikilink)
    .replace(IMAGE, '$1')
    .replace(LINK, '$1')
    .replace(/(\*\*|__)(?=\S)(.+?)(?<=\S)\1/g, '$2')
    .replace(/(^|[^\w*])\*(?=\S)([^*]+?)(?<=\S)\*(?![\w*])/g, '$1$2')
    .replace(/(^|[^\w_])_(?=\S)([^_]+?)(?<=\S)_(?![\w_])/g, '$1$2')
    .replace(/~~(?=\S)(.+?)(?<=\S)~~/g, '$1')
    .replace(HELD_CHARACTER, (character) => String.fromCharCode(character.charCodeAt(0) - ESCAPE_BASE))
    .replace(/\s+/g, ' ')
    .trim()
}

const isMarkupOnlyLine = (line: string): boolean =>
  FENCE_LINE.test(line) ||
  RULE_LINE.test(line) ||
  SETEXT_UNDERLINE.test(line) ||
  TABLE_SEPARATOR_LINE.test(line)

const comparable = (value: string): string => markdownLineToPlainText(value).toLowerCase()

/** The index of the first line after YAML front matter, or 0 when there is none. */
const bodyStart = (lines: readonly string[]): number => {
  if (!FRONT_MATTER_FENCE.test(lines[0] ?? '') || !FRONT_MATTER_KEY.test(lines[1] ?? '')) return 0
  const limit = Math.min(lines.length, MAX_FRONT_MATTER_LINES)
  for (let index = 2; index < limit; index += 1) {
    if (FRONT_MATTER_END.test(lines[index])) return index + 1
  }
  return 0
}

/** Cuts by character (never through a surrogate pair) and marks the cut. */
const clip = (text: string, maxLength: number, wasCut: boolean): string => {
  const characters = Array.from(text)
  if (characters.length > maxLength) return `${characters.slice(0, maxLength).join('').trimEnd()}${ELLIPSIS}`
  return wasCut ? `${text}${ELLIPSIS}` : text
}

type PlainPreviewOptions = {
  /** A line whose text matches this one is skipped (a note body that repeats its title). */
  skipLine?: string | null
  maxLength?: number
}

/**
 * The first meaningful line of `markdown` as plain text, cut to `maxLength`
 * characters with "…". Returns "" when nothing but markup (or the skipped line) is left.
 */
export const toPlainPreview = (markdown: string, options: PlainPreviewOptions = {}): string => {
  const maxLength = options.maxLength ?? DEFAULT_MAX_LENGTH
  const scanLimit = Math.max(1, maxLength) * SCAN_FACTOR
  const skip = options.skipLine ? comparable(options.skipLine) : ''
  const lines = String(markdown || '').split(/\r?\n/)
  for (let index = bodyStart(lines); index < lines.length; index += 1) {
    const line = lines[index].trim()
    if (!line || isMarkupOnlyLine(line)) continue
    const text = markdownLineToPlainText(line.slice(0, scanLimit))
    if (!text) continue
    if (skip && text.toLowerCase() === skip) continue
    return clip(text, maxLength, line.length > scanLimit)
  }
  return ''
}
