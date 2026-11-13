/**
 * One-line plain-text previews of Markdown (NL-04, #3101).
 *
 * Lists show a note's first meaningful line. Raw Markdown there reads as noise
 * (`**Mood:**`, ```` ```sql ````, `[[Title]]`), so this strips the common
 * markup and keeps the words. It is a display helper, not a Markdown parser:
 * anything it does not recognise stays as written.
 */

const DEFAULT_MAX_LENGTH = 140

const FENCE_LINE = /^(`{3,}|~{3,})/
const RULE_LINE = /^([-*_])(\s*\1){2,}\s*$/
const TABLE_SEPARATOR_LINE = /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/

/** `[[id:UUID]]` links name a note by id; the raw id is not preview text (XP-14). */
const ID_WIKILINK_PREFIX = /^id:/i

const replaceWikilink = (_match: string, inner: string): string => {
  const [target, alias] = inner.split("|")
  if (alias?.trim()) return alias.trim()
  const text = (target || "").trim()
  return ID_WIKILINK_PREFIX.test(text) ? "" : text
}

/** Strips one line's Markdown markup: headings, quotes, list markers, emphasis, code, links. */
export const markdownLineToPlainText = (line: string): string =>
  line
    .trim()
    .replace(/^#{1,6}\s+/, "")
    .replace(/\s+#+\s*$/, "")
    .replace(/^(>\s?)+/, "")
    .replace(/^([-*+]|\d+[.)])\s+/, "")
    .replace(/^\[[ xX]\]\s+/, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[\[([^\]]+)\]\]/g, replaceWikilink)
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/(\*\*|__)(?=\S)(.+?)(?<=\S)\1/g, "$2")
    .replace(/(^|[^\w*])\*(?=\S)([^*]+?)(?<=\S)\*(?![\w*])/g, "$1$2")
    .replace(/(^|[^\w_])_(?=\S)([^_]+?)(?<=\S)_(?![\w_])/g, "$1$2")
    .replace(/~~(?=\S)(.+?)(?<=\S)~~/g, "$1")
    .replace(/\s+/g, " ")
    .trim()

const isMarkupOnlyLine = (line: string): boolean =>
  FENCE_LINE.test(line) || RULE_LINE.test(line) || TABLE_SEPARATOR_LINE.test(line)

const comparable = (value: string): string => markdownLineToPlainText(value).toLowerCase()

type PlainPreviewOptions = {
  /** A line whose text matches this one is skipped (a note body that repeats its title). */
  skipLine?: string | null
  maxLength?: number
}

/**
 * The first meaningful line of `markdown` as plain text, truncated to `maxLength`.
 * Returns "" when nothing but markup (or the skipped line) is left.
 */
export const toPlainPreview = (markdown: string, options: PlainPreviewOptions = {}): string => {
  const maxLength = options.maxLength ?? DEFAULT_MAX_LENGTH
  const skip = options.skipLine ? comparable(options.skipLine) : ""
  for (const rawLine of String(markdown || "").split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || isMarkupOnlyLine(line)) continue
    const text = markdownLineToPlainText(line)
    if (!text) continue
    if (skip && text.toLowerCase() === skip) continue
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text
  }
  return ""
}
