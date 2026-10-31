import type {
  StandaloneHtmlOutline,
  StandaloneHtmlOutlineBlock,
  StandaloneHtmlOutlineSlide
} from "./standalone-html-outline.worker"
import { hasDirectUrlLikeText } from "./standalone-html-outline-text-policy"
import { preflightStandaloneHtmlSource } from "./standalone-html-source"

type OutlineStatus = "current" | "stale" | "failed"

type OutlineState = {
  status: OutlineStatus
  digest: string | null
  message?: string
}

type OutlineWorker = Pick<Worker, "postMessage" | "terminate" | "onmessage" | "onerror">

type OutlineInput = { source: string; digest: string }
type ActiveOutlineInput = OutlineInput & { requestId: number }

const TRUNCATION_MARKER = "... [truncated]"
const TRUNCATION_MARKER_SCALARS = Array.from(TRUNCATION_MARKER).length
const MAX_OUTLINE_BLOCKS = 50_000
const MAX_OUTLINE_SCALARS = 100_000

export const createStandaloneHtmlOutlineWorker = (_ignoredSource?: string): Worker =>
  new Worker(new URL("./standalone-html-outline.worker.ts", import.meta.url), {
    type: "module",
    name: "StandaloneHtmlOutlineWorker"
  })

const hasExactKeys = (value: Record<string, unknown>, keys: string[]): boolean => {
  const actual = Object.keys(value)
  return actual.length === keys.length && keys.every((key) => actual.includes(key))
}

const containsUnsafeOutlineControls = (value: string): boolean =>
  /[\u0000-\u001f\u007f-\u009f\u061c\u200e\u200f\u202a-\u202e\u2066-\u206f]/.test(value)

const isDenseArray = (value: unknown): value is unknown[] => {
  if (!Array.isArray(value) || Object.keys(value).length !== value.length) return false
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.prototype.hasOwnProperty.call(value, index)) return false
  }
  return true
}

const validBlock = (value: unknown): value is StandaloneHtmlOutlineBlock => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const block = value as Record<string, unknown>
  const textPreflight = preflightStandaloneHtmlSource(block.text, { allowEmpty: false })
  return (
    hasExactKeys(block, ["kind", "text", "truncated"]) &&
    typeof block.kind === "string" &&
    ["heading", "paragraph", "list_item", "definition", "caption", "table_cell", "code", "quote"].includes(block.kind) &&
    typeof block.text === "string" &&
    textPreflight.ok &&
    Array.from(block.text).length <= 4_096 &&
    !containsUnsafeOutlineControls(block.text) &&
    !hasDirectUrlLikeText(block.text) &&
    typeof block.truncated === "boolean" &&
    (!block.truncated ||
      (block.text.endsWith(TRUNCATION_MARKER) &&
        !block.text.slice(0, -TRUNCATION_MARKER.length).endsWith(TRUNCATION_MARKER)))
  )
}

const validBlockArray = (value: unknown): value is StandaloneHtmlOutlineBlock[] =>
  isDenseArray(value) && value.every(validBlock)

const validSlide = (value: unknown): value is StandaloneHtmlOutlineSlide => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const slide = value as Record<string, unknown>
  const structurallyValid =
    hasExactKeys(slide, ["index", "blocks", "notes", "truncated"]) &&
    typeof slide.index === "number" &&
    Number.isInteger(slide.index) &&
    slide.index > 0 &&
    validBlockArray(slide.blocks) &&
    validBlockArray(slide.notes) &&
    typeof slide.truncated === "boolean"
  if (!structurallyValid) return false
  const blocks = [
    ...(slide.blocks as StandaloneHtmlOutlineBlock[]),
    ...(slide.notes as StandaloneHtmlOutlineBlock[])
  ]
  if (blocks.some((block) => block.truncated) && !slide.truncated) return false
  return blocks.reduce((total, block) => total + Array.from(block.text).length, 0) <= 20_000
}

const validOutline = (value: unknown, digest: string): value is StandaloneHtmlOutline => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const outline = value as Record<string, unknown>
  const structurallyValid =
    hasExactKeys(outline, ["digest", "slides", "truncated"]) &&
    outline.digest === digest &&
    isDenseArray(outline.slides) &&
    outline.slides.length <= 30 &&
    typeof outline.truncated === "boolean"
  if (!structurallyValid) return false
  let blockCount = 0
  for (const value of outline.slides as unknown[]) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return false
    const slide = value as Record<string, unknown>
    if (!Array.isArray(slide.blocks) || !Array.isArray(slide.notes)) return false
    blockCount += slide.blocks.length + slide.notes.length
    if (blockCount > MAX_OUTLINE_BLOCKS || !validSlide(value)) return false
  }
  const slides = outline.slides as StandaloneHtmlOutlineSlide[]
  if (!slides.every((slide, index) => slide.index === index + 1)) return false
  if (slides.some((slide) => slide.truncated) && !outline.truncated) return false
  const renderMarkerCount = slides.reduce((total, slide) => {
    const inlineMarkers = slide.blocks
      .concat(slide.notes)
      .filter((block) => block.truncated).length
    return total + inlineMarkers + (slide.truncated && inlineMarkers === 0 ? 1 : 0)
  }, outline.truncated && !slides.some((slide) => slide.truncated) ? 1 : 0)
  if (renderMarkerCount * TRUNCATION_MARKER_SCALARS > MAX_OUTLINE_SCALARS) return false
  return (
    slides.reduce(
      (total, slide) =>
        total +
        slide.blocks
          .concat(slide.notes)
          .reduce((slideTotal, block) => slideTotal + Array.from(block.text).length, 0),
      0
    ) <= MAX_OUTLINE_SCALARS
  )
}

export class StandaloneHtmlOutlineController {
  private readonly workerFactory: () => OutlineWorker
  private readonly onState: (state: OutlineState) => void
  private readonly onOutline: (outline: StandaloneHtmlOutline) => void
  private readonly watchdogMs: number
  private worker: OutlineWorker | null = null
  private active: ActiveOutlineInput | null = null
  private pending: OutlineInput | null = null
  private latestDigest: string | null = null
  private requestId = 0
  private watchdog: ReturnType<typeof setTimeout> | null = null
  private disposed = false

  constructor(options: {
    workerFactory?: () => OutlineWorker
    onState: (state: OutlineState) => void
    onOutline: (outline: StandaloneHtmlOutline) => void
    watchdogMs?: number
  }) {
    this.workerFactory = options.workerFactory ?? createStandaloneHtmlOutlineWorker
    this.onState = options.onState
    this.onOutline = options.onOutline
    this.watchdogMs = options.watchdogMs ?? 10_000
  }

  request(input: OutlineInput): void {
    if (this.disposed) return
    const preflight = preflightStandaloneHtmlSource(input.source)
    if (!preflight.ok || !/^[0-9a-f]{64}$/.test(input.digest)) {
      this.onState({ status: "failed", digest: null, message: "Outline unavailable" })
      return
    }
    this.latestDigest = input.digest
    this.onState({ status: "stale", digest: input.digest })
    if (this.active) {
      this.pending = input
      return
    }
    this.dispatch(input)
  }

  private ensureWorker(): OutlineWorker {
    if (this.worker) return this.worker
    const worker = this.workerFactory()
    try {
      worker.onmessage = (event) => this.handleMessage(event.data)
      worker.onerror = () => {
        if (this.worker !== worker) return
        this.failActive(true)
      }
    } catch (error) {
      try {
        worker.terminate()
      } catch {
        // The failed worker is discarded even if its own cleanup is unavailable.
      }
      throw error
    }
    this.worker = worker
    return worker
  }

  private dispatch(input: OutlineInput): void {
    if (this.disposed) return
    const active = { ...input, requestId: ++this.requestId }
    this.active = active
    try {
      this.ensureWorker().postMessage({ type: "extract", ...active })
      this.watchdog = setTimeout(() => this.handleTimeout(), this.watchdogMs)
    } catch {
      this.clearWatchdog()
      this.active = null
      this.replaceWorker()
      this.onState({ status: "failed", digest: active.digest, message: "Outline unavailable" })
    }
  }

  private clearWatchdog(): void {
    if (this.watchdog !== null) {
      clearTimeout(this.watchdog)
      this.watchdog = null
    }
  }

  private replaceWorker(): void {
    const worker = this.worker
    this.worker = null
    try {
      worker?.terminate()
    } catch {
      // Cleanup failure must not escape the source-free controller boundary.
    }
  }

  private handleTimeout(): void {
    const failedDigest = this.active?.digest ?? this.latestDigest
    const retry = this.pending
    this.pending = null
    this.active = null
    this.clearWatchdog()
    this.replaceWorker()
    if (retry) {
      this.dispatch(retry)
    } else {
      this.onState({ status: "failed", digest: failedDigest, message: "Outline unavailable" })
    }
  }

  private failActive(replaceWorker = false): void {
    const active = this.active
    if (!active) return
    const pending = this.pending
    this.pending = null
    this.clearWatchdog()
    this.active = null
    if (replaceWorker) this.replaceWorker()
    if (pending) {
      this.dispatch(pending)
    } else {
      this.onState({ status: "failed", digest: active.digest, message: "Outline unavailable" })
    }
  }

  private handleMessage(value: unknown): void {
    const active = this.active
    if (!active || !value || typeof value !== "object" || Array.isArray(value)) return
    const message = value as Record<string, unknown>
    if (message.requestId !== active.requestId) return
    const validEnvelope =
      hasExactKeys(message, ["type", "requestId", "digest", "outline"]) &&
      message.type === "result" &&
      message.requestId === active.requestId &&
      message.digest === active.digest &&
      validOutline(message.outline, active.digest)

    this.clearWatchdog()
    this.active = null
    const pending = this.pending
    this.pending = null
    if (!validEnvelope) {
      if (!pending) {
        this.onState({ status: "failed", digest: active.digest, message: "Outline unavailable" })
      }
    } else if (active.digest === this.latestDigest && !pending) {
      this.onOutline(message.outline as StandaloneHtmlOutline)
      this.onState({ status: "current", digest: active.digest })
    }

    if (pending) this.dispatch(pending)
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.clearWatchdog()
    this.replaceWorker()
    this.active = null
    this.pending = null
    this.latestDigest = null
  }
}
