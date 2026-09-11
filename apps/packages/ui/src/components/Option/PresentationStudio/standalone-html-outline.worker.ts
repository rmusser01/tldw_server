import { hasDirectUrlLikeText } from "./standalone-html-outline-text-policy"

const COMPLEXITY_FAILURE = {
  ok: false as const,
  code: "document_too_complex" as const,
  message: "Outline unavailable: document too complex."
}

const MAX_POTENTIAL_TOKENS = 50_000
const MAX_START_TAGS = 10_000
const MAX_ATTRIBUTES = 20_000
const MAX_COMMENTS_DECLARATIONS = 10_000
const MAX_TEXT_RUNS = 20_000
const MAX_DEPTH = 128
const MAX_RUN_BYTES = 65_536
const MAX_PARSED_NODES = 50_000
const MAX_SLIDES = 30
const MAX_BLOCK_SCALARS = 4_096
const MAX_SLIDE_SCALARS = 20_000
const MAX_TOTAL_SCALARS = 100_000
const TRUNCATION_MARKER = "... [truncated]"
const TRUNCATION_MARKER_SCALARS = Array.from(TRUNCATION_MARKER).length

const VOID_TAGS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
  "source", "track", "wbr"
])

const FORBIDDEN_SUBTREES = new Set([
  "a", "applet", "audio", "button", "canvas", "embed", "fencedframe", "font", "form",
  "frame", "frameset", "iframe", "img", "input", "link", "math", "noscript", "object",
  "nav", "picture", "portal", "script", "select", "source", "style", "svg", "template", "textarea",
  "track", "video"
])

const CHROME_CLASSES = new Set([
  "deck-header", "deck-footer", "slide-number", "progress", "navigation", "nav"
])

const TRUSTED_BLOCKS: Record<string, OutlineBlockKind> = {
  h1: "heading",
  h2: "heading",
  h3: "heading",
  h4: "heading",
  h5: "heading",
  h6: "heading",
  p: "paragraph",
  li: "list_item",
  dt: "definition",
  dd: "definition",
  caption: "caption",
  th: "table_cell",
  td: "table_cell",
  figcaption: "caption",
  pre: "code",
  code: "code",
  blockquote: "quote"
}

export type OutlineBlockKind =
  | "heading"
  | "paragraph"
  | "list_item"
  | "definition"
  | "caption"
  | "table_cell"
  | "code"
  | "quote"

export type StandaloneHtmlOutlineBlock = {
  kind: OutlineBlockKind
  text: string
  truncated: boolean
}

export type StandaloneHtmlOutlineSlide = {
  index: number
  blocks: StandaloneHtmlOutlineBlock[]
  notes: StandaloneHtmlOutlineBlock[]
  truncated: boolean
}

export type StandaloneHtmlOutline = {
  digest: string
  slides: StandaloneHtmlOutlineSlide[]
  truncated: boolean
}

const utf8Length = (value: string): number => {
  let bytes = 0
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index)
    if (unit <= 0x7f) bytes += 1
    else if (unit <= 0x7ff) bytes += 2
    else if (unit >= 0xd800 && unit <= 0xdbff) {
      bytes += 4
      index += 1
    } else bytes += 3
  }
  return bytes
}

const countAttributes = (tag: string): number => {
  let index = 0
  while (index < tag.length && !/\s/.test(tag[index]) && tag[index] !== "/") index += 1
  let count = 0
  while (index < tag.length) {
    while (index < tag.length && (/\s/.test(tag[index]) || tag[index] === "/")) index += 1
    if (index >= tag.length) break
    count += 1
    while (index < tag.length && !/[\s=]/.test(tag[index])) index += 1
    while (index < tag.length && /\s/.test(tag[index])) index += 1
    if (tag[index] !== "=") continue
    index += 1
    while (index < tag.length && /\s/.test(tag[index])) index += 1
    const quote = tag[index] === '"' || tag[index] === "'" ? tag[index++] : null
    if (quote) {
      while (index < tag.length && tag[index] !== quote) index += 1
      if (index < tag.length) index += 1
    } else {
      while (index < tag.length && !/\s/.test(tag[index])) index += 1
    }
  }
  return count
}

const findTagEnd = (source: string, start: number): number => {
  let quote: string | null = null
  for (let index = start; index < source.length; index += 1) {
    const character = source[index]
    if (quote) {
      if (character === quote) quote = null
    } else if (character === '"' || character === "'") {
      quote = character
    } else if (character === ">") {
      return index
    }
  }
  return -1
}

export const preflightStandaloneHtmlOutline = (source: string) => {
  let potentialTokens = 0
  let startTags = 0
  let attributes = 0
  let commentsDeclarations = 0
  let textRuns = 0
  const openTags: string[] = []
  let index = 0

  const overBudget = () =>
    potentialTokens > MAX_POTENTIAL_TOKENS ||
    startTags > MAX_START_TAGS ||
    attributes > MAX_ATTRIBUTES ||
    commentsDeclarations > MAX_COMMENTS_DECLARATIONS ||
    textRuns > MAX_TEXT_RUNS ||
    openTags.length > MAX_DEPTH

  while (index < source.length) {
    if (source[index] !== "<") {
      const end = source.indexOf("<", index)
      const next = end === -1 ? source.length : end
      const text = source.slice(index, next)
      if (text.length > 0) {
        textRuns += 1
        potentialTokens += 1
        if (utf8Length(text) > MAX_RUN_BYTES || overBudget()) return COMPLEXITY_FAILURE
      }
      index = next
      continue
    }

    if (source.startsWith("<!--", index)) {
      const end = source.indexOf("-->", index + 4)
      if (end === -1) return COMPLEXITY_FAILURE
      const token = source.slice(index, end + 3)
      commentsDeclarations += 1
      potentialTokens += 1
      if (utf8Length(token) > MAX_RUN_BYTES || overBudget()) return COMPLEXITY_FAILURE
      index = end + 3
      continue
    }

    const end = findTagEnd(source, index + 1)
    if (end === -1) return COMPLEXITY_FAILURE
    const token = source.slice(index, end + 1)
    if (utf8Length(token) > MAX_RUN_BYTES) return COMPLEXITY_FAILURE
    potentialTokens += 1

    const inner = token.slice(1, -1).trim()
    if (inner.startsWith("!") || inner.startsWith("?")) {
      commentsDeclarations += 1
    } else if (inner.startsWith("/")) {
      const tagName = inner.slice(1).trim().split(/[\s>]/, 1)[0].toLowerCase()
      if (openTags.at(-1) === tagName) openTags.pop()
    } else if (inner.length > 0) {
      startTags += 1
      attributes += countAttributes(inner)
      const tagName = inner.split(/[\s/]/, 1)[0].toLowerCase()
      if (!VOID_TAGS.has(tagName)) openTags.push(tagName)
    }
    if (overBudget()) return COMPLEXITY_FAILURE
    index = end + 1
  }

  return { ok: true as const }
}

const complexityError = () =>
  Object.assign(new Error(COMPLEXITY_FAILURE.message), {
    code: COMPLEXITY_FAILURE.code
  })

type ParserNode = {
  type?: string
  name?: string
  data?: string
  attribs?: Record<string, string>
  children?: ParserNode[]
}

const elementName = (node: ParserNode): string =>
  typeof node.name === "string" ? node.name.toLowerCase() : ""

const hasClass = (node: ParserNode, expected: string): boolean =>
  String(node.attribs?.class ?? "")
    .split(/\s+/)
    .some((value) => value === expected)

const isBlockedSubtree = (node: ParserNode): boolean =>
  FORBIDDEN_SUBTREES.has(elementName(node)) ||
  String(node.attribs?.class ?? "")
    .split(/\s+/)
    .some((value) => CHROME_CLASSES.has(value))

const cleanText = (value: string): string =>
  value
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, "")
    .replace(/[\u202a-\u202e\u2066-\u206f\u200e\u200f\u061c]/g, "")
    .replace(/[\t\n\r ]+/g, " ")
    .trim()

const collectTrustedText = (root: ParserNode): string => {
  const pieces: string[] = []
  const stack = [...(root.children ?? [])].reverse()
  while (stack.length > 0) {
    const node = stack.pop()!
    if (node.type === "text") {
      if (typeof node.data === "string") pieces.push(node.data)
      continue
    }
    if (isBlockedSubtree(node) || hasClass(node, "notes") || hasClass(node, "slide")) continue
    const children = node.children ?? []
    for (let index = children.length - 1; index >= 0; index -= 1) stack.push(children[index])
  }
  return cleanText(pieces.join(""))
}

const takeScalars = (value: string, maximum: number) => {
  const scalars = Array.from(value)
  if (scalars.length <= maximum) {
    return {
      text: value,
      count: scalars.length,
      truncated: false
    }
  }
  const capacity = Math.max(0, maximum)
  if (capacity < TRUNCATION_MARKER_SCALARS) {
    return { text: "", count: 0, truncated: true }
  }
  const prefixLength = capacity - TRUNCATION_MARKER_SCALARS
  let prefix = scalars.slice(0, prefixLength).join("")
  while (prefix.endsWith(TRUNCATION_MARKER)) {
    prefix = prefix.slice(0, -TRUNCATION_MARKER.length)
  }
  const text = `${prefix}${TRUNCATION_MARKER}`
  return {
    text,
    count: Array.from(text).length,
    truncated: true
  }
}

export const validateStandaloneHtmlOutlineTree = (root: ParserNode): ParserNode[] => {
  const slides: ParserNode[] = []
  const stack: Array<{
    node: ParserNode
    depth: number
    blocked: boolean
    inNotes: boolean
    slideAncestor: boolean
  }> = [
    { node: root, depth: 0, blocked: false, inNotes: false, slideAncestor: false }
  ]
  let nodes = 0
  while (stack.length > 0) {
    const current = stack.pop()!
    nodes += 1
    if (nodes > MAX_PARSED_NODES || current.depth > MAX_DEPTH) throw complexityError()
    const blocked = current.blocked || isBlockedSubtree(current.node)
    const inNotes = current.inNotes || hasClass(current.node, "notes")
    const isSlide = hasClass(current.node, "slide")
    if (!blocked && !inNotes && !current.slideAncestor && isSlide) slides.push(current.node)
    const children = current.node.children ?? []
    for (let index = children.length - 1; index >= 0; index -= 1) {
      stack.push({
        node: children[index],
        depth: current.depth + 1,
        blocked,
        inNotes,
        slideAncestor: current.slideAncestor || isSlide
      })
    }
  }
  return slides
}

export const extractStandaloneHtmlOutline = async (
  source: string,
  digest: string
): Promise<StandaloneHtmlOutline> => {
  if (!preflightStandaloneHtmlOutline(source).ok) throw complexityError()
  const cheerio = (await import("cheerio/slim")) as any
  const $ = cheerio.load(source)
  const roots = $.root().toArray() as ParserNode[]
  const slideNodes = roots.flatMap(validateStandaloneHtmlOutlineTree)
  const outline: StandaloneHtmlOutline = {
    digest,
    slides: [],
    truncated: slideNodes.length > MAX_SLIDES
  }
  let totalScalars = 0

  for (const [slideOffset, slideNode] of slideNodes.slice(0, MAX_SLIDES).entries()) {
    const card: StandaloneHtmlOutlineSlide = {
      index: slideOffset + 1,
      blocks: [],
      notes: [],
      truncated: false
    }
    let slideScalars = 0
    const stack: Array<{ node: ParserNode; notes: boolean }> = []
    const children = slideNode.children ?? []
    for (let index = children.length - 1; index >= 0; index -= 1) {
      stack.push({ node: children[index], notes: false })
    }

    while (stack.length > 0) {
      const current = stack.pop()!
      const name = elementName(current.node)
      if (isBlockedSubtree(current.node) || hasClass(current.node, "slide")) continue
      const inNotes = current.notes || hasClass(current.node, "notes")
      const kind = TRUSTED_BLOCKS[name]
      if (kind) {
        const cleaned = collectTrustedText(current.node)
        if (!cleaned || hasDirectUrlLikeText(cleaned)) continue
        const absoluteRemaining = Math.min(
          MAX_BLOCK_SCALARS,
          MAX_SLIDE_SCALARS - slideScalars,
          MAX_TOTAL_SCALARS - totalScalars
        )
        if (absoluteRemaining <= 0) {
          card.truncated = true
          outline.truncated = true
          continue
        }
        const capped = takeScalars(cleaned, absoluteRemaining)
        if (!capped.text) {
          card.truncated = true
          outline.truncated = true
          continue
        }
        const block = { kind, text: capped.text, truncated: capped.truncated }
        ;(inNotes ? card.notes : card.blocks).push(block)
        slideScalars += capped.count
        totalScalars += capped.count
        if (capped.truncated) {
          card.truncated = true
          outline.truncated = true
        }
        continue
      }
      const nextChildren = current.node.children ?? []
      for (let index = nextChildren.length - 1; index >= 0; index -= 1) {
        stack.push({ node: nextChildren[index], notes: inNotes })
      }
    }
    outline.slides.push(card)
  }

  return outline
}

type WorkerRequest = {
  type: "extract"
  requestId: number
  digest: string
  source: string
}

const isWorkerRequest = (value: unknown): value is WorkerRequest => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return (
    Object.keys(record).length === 4 &&
    record.type === "extract" &&
    typeof record.requestId === "number" &&
    Number.isInteger(record.requestId) &&
    typeof record.digest === "string" &&
    /^[0-9a-f]{64}$/.test(record.digest) &&
    typeof record.source === "string"
  )
}

const workerScope = globalThis as unknown as {
  addEventListener?: (type: string, listener: (event: MessageEvent<unknown>) => void) => void
  postMessage?: (message: unknown) => void
  document?: unknown
}

if (typeof workerScope.document === "undefined" && workerScope.addEventListener && workerScope.postMessage) {
  workerScope.addEventListener("message", (event) => {
    if (!isWorkerRequest(event.data)) return
    const request = event.data
    void extractStandaloneHtmlOutline(request.source, request.digest)
      .then((outline) => {
        workerScope.postMessage?.({
          type: "result",
          requestId: request.requestId,
          digest: request.digest,
          outline
        })
      })
      .catch(() => {
        workerScope.postMessage?.({
          type: "failed",
          requestId: request.requestId,
          digest: request.digest,
          code: COMPLEXITY_FAILURE.code,
          message: COMPLEXITY_FAILURE.message
        })
      })
  })
}
