import type {
  StandaloneHtmlOutline,
  StandaloneHtmlOutlineBlock,
  StandaloneHtmlOutlineSlide
} from "./standalone-html-outline.worker"
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

export const createStandaloneHtmlOutlineWorker = (_ignoredSource?: string): Worker =>
  new Worker(new URL("./standalone-html-outline.worker.ts", import.meta.url), { type: "module" })

const hasExactKeys = (value: Record<string, unknown>, keys: string[]): boolean => {
  const actual = Object.keys(value)
  return actual.length === keys.length && keys.every((key) => actual.includes(key))
}

const validBlock = (value: unknown): value is StandaloneHtmlOutlineBlock => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const block = value as Record<string, unknown>
  return (
    hasExactKeys(block, ["kind", "text", "truncated"]) &&
    typeof block.kind === "string" &&
    ["heading", "paragraph", "list_item", "definition", "caption", "table_cell", "code", "quote"].includes(block.kind) &&
    typeof block.text === "string" &&
    Array.from(block.text).length <= 4_096 &&
    typeof block.truncated === "boolean"
  )
}

const validSlide = (value: unknown): value is StandaloneHtmlOutlineSlide => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const slide = value as Record<string, unknown>
  const structurallyValid =
    hasExactKeys(slide, ["index", "blocks", "notes", "truncated"]) &&
    typeof slide.index === "number" &&
    Number.isInteger(slide.index) &&
    slide.index > 0 &&
    Array.isArray(slide.blocks) &&
    slide.blocks.every(validBlock) &&
    Array.isArray(slide.notes) &&
    slide.notes.every(validBlock) &&
    typeof slide.truncated === "boolean"
  if (!structurallyValid) return false
  const blocks = [
    ...(slide.blocks as StandaloneHtmlOutlineBlock[]),
    ...(slide.notes as StandaloneHtmlOutlineBlock[])
  ]
  return blocks.reduce((total, block) => total + Array.from(block.text).length, 0) <= 20_000
}

const validOutline = (value: unknown, digest: string): value is StandaloneHtmlOutline => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const outline = value as Record<string, unknown>
  const structurallyValid =
    hasExactKeys(outline, ["digest", "slides", "truncated"]) &&
    outline.digest === digest &&
    Array.isArray(outline.slides) &&
    outline.slides.length <= 30 &&
    outline.slides.every(validSlide) &&
    typeof outline.truncated === "boolean"
  if (!structurallyValid) return false
  const slides = outline.slides as StandaloneHtmlOutlineSlide[]
  if (!slides.every((slide, index) => slide.index === index + 1)) return false
  return (
    slides.reduce(
      (total, slide) =>
        total +
        slide.blocks
          .concat(slide.notes)
          .reduce((slideTotal, block) => slideTotal + Array.from(block.text).length, 0),
      0
    ) <= 100_000
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
    worker.onmessage = (event) => this.handleMessage(event.data)
    worker.onerror = () => this.failActive(true)
    this.worker = worker
    return worker
  }

  private dispatch(input: OutlineInput): void {
    if (this.disposed) return
    const active = { ...input, requestId: ++this.requestId }
    this.active = active
    this.ensureWorker().postMessage({ type: "extract", ...active })
    this.watchdog = setTimeout(() => this.handleTimeout(), this.watchdogMs)
  }

  private clearWatchdog(): void {
    if (this.watchdog !== null) {
      clearTimeout(this.watchdog)
      this.watchdog = null
    }
  }

  private replaceWorker(): void {
    this.worker?.terminate()
    this.worker = null
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
