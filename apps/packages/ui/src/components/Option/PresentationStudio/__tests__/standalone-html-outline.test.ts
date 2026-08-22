import React from "react"
import { act, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const loadWorker = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-outline.worker"].join("/"))
const loadClient = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-outline-client"].join("/"))
const loadView = () =>
  vi.importActual<Record<string, any>>(["..", "StandaloneHtmlSafeOutline"].join("/"))

class FakeWorker {
  onmessage: ((event: MessageEvent<any>) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  posted: any[] = []
  terminate = vi.fn()

  postMessage(payload: any) {
    this.posted.push(payload)
  }

  respond(payload: any) {
    this.onmessage?.({ data: payload } as MessageEvent<any>)
  }
}

const resultFor = (request: any, title: string) => ({
  type: "result",
  requestId: request.requestId,
  digest: request.digest,
  outline: {
    digest: request.digest,
    slides: [
      {
        index: 1,
        blocks: [{ kind: "heading", text: title, truncated: false }],
        notes: [],
        truncated: false
      }
    ],
    truncated: false
  }
})

describe("safe outline lexical and parser boundary", () => {
  it.each([
    ["more than 50,000 potential tokens", "</x>".repeat(50_001)],
    ["more than 10,000 start tags", "<x/>".repeat(10_001)],
    [
      "more than 20,000 attributes",
      `${'<x a="" b=""/>'.repeat(9_999)}<x a="" b="" c=""/>`
    ],
    ["more than 10,000 comments/declarations", "<!---->".repeat(10_001)],
    ["more than 20,000 text-run transitions", `x${"</x>x".repeat(20_000)}`],
    ["apparent depth 129", `${"<x>".repeat(129)}${"</x>".repeat(129)}`],
    ["a text run above 65,536 UTF-8 bytes", "x".repeat(65_537)],
    ["an unterminated quoted tag", '<section class="slide>'],
    ["an unterminated comment", "<!-- private source"]
  ])("refuses %s with a fixed source-free result before parsing", async (_case, source) => {
    const subject = await loadWorker()

    const result = subject.preflightStandaloneHtmlOutline(source)

    expect(result).toEqual({
      ok: false,
      code: "document_too_complex",
      message: "Outline unavailable — document too complex"
    })
    expect(JSON.stringify(result)).not.toContain("private source")
  })

  it("extracts only capped semantic text from slides and separately labelled notes", async () => {
    const subject = await loadWorker()
    const fetchSpy = vi.spyOn(globalThis, "fetch")
    const DomParser = globalThis.DOMParser
    Object.defineProperty(globalThis, "DOMParser", {
      configurable: true,
      value: class {
        constructor() {
          throw new Error("browser DOMParser must never run")
        }
      }
    })
    const source = `<!doctype html><html><head><style>.secret{background:url(https://css.invalid)}</style></head><body>
      <header class="deck-header">Deck chrome</header>
      <section class="slide" data-secret="discard-me">
        <h1>First\u202e title</h1><p>Plain <a href="https://link.invalid">linked secret</a> text\u0007.</p>
        <ul><li>One</li><li>Two</li></ul><table><caption>Metrics</caption><tr><th>A</th><td>42</td></tr></table>
        <figure><figcaption>Trusted caption</figcaption><img src="https://asset.invalid/private.png"></figure>
        <script>globalThis.OUTLINE_SENTINEL = true</script><svg><text>SVG secret</text></svg>
        <form><p>Form secret</p></form><div class="notes"><p>Speaker-only note</p></div>
      </section>
      <section class="slide"><blockquote>Second slide</blockquote><pre><code>const safe = "text"</code></pre></section>
      <script>globalThis.SECOND_SENTINEL = true</script></body></html>`

    try {
      const result = await subject.extractStandaloneHtmlOutline(source, "a".repeat(64))

      expect(result.digest).toBe("a".repeat(64))
      expect(result.slides).toHaveLength(2)
      expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual([
        "First title",
        "Plain text.",
        "One",
        "Two",
        "Metrics",
        "A",
        "42",
        "Trusted caption"
      ])
      expect(result.slides[0].notes.map((block: any) => block.text)).toEqual([
        "Speaker-only note"
      ])
      const serialized = JSON.stringify(result)
      for (const forbidden of [
        "https://",
        "discard-me",
        "linked secret",
        "Deck chrome",
        "globalThis",
        "SVG secret",
        "Form secret",
        "<script",
        "style="
      ]) {
        expect(serialized).not.toContain(forbidden)
      }
      expect(fetchSpy).not.toHaveBeenCalled()
      expect((globalThis as any).OUTLINE_SENTINEL).not.toBe(true)
    } finally {
      Object.defineProperty(globalThis, "DOMParser", { configurable: true, value: DomParser })
    }
  })

  it("enforces card, block, slide, total-scalar, node, and actual-depth output ceilings", async () => {
    const subject = await loadWorker()
    const hugeBlock = "x".repeat(5_000)
    const slide = `<section class="slide">${`<p>${hugeBlock}</p>`.repeat(5)}</section>`
    const source = `${slide.repeat(31)}`

    const result = await subject.extractStandaloneHtmlOutline(source, "b".repeat(64))

    expect(result.slides.length).toBeLessThanOrEqual(30)
    expect(Array.from(result.slides[0].blocks[0].text).length).toBeLessThanOrEqual(4_096)
    for (const card of result.slides) {
      const scalars = card.blocks.concat(card.notes).reduce(
        (total: number, block: any) => total + Array.from(block.text).length,
        0
      )
      expect(scalars).toBeLessThanOrEqual(20_000)
    }
    const total = result.slides.reduce(
      (sum: number, card: any) =>
        sum + card.blocks.concat(card.notes).reduce(
          (inner: number, block: any) => inner + Array.from(block.text).length,
          0
        ),
      0
    )
    expect(total).toBeLessThanOrEqual(100_000)

    await expect(
      subject.extractStandaloneHtmlOutline(
        `<section class="slide">${"<div>".repeat(129)}deep${"</div>".repeat(129)}</section>`,
        "c".repeat(64)
      )
    ).rejects.toMatchObject({ code: "document_too_complex" })
  })
})

describe("StandaloneHtmlOutlineController", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("uses one static application-owned worker URL", async () => {
    const subject = await loadClient()
    const WorkerConstructor = vi.fn(function (this: any, url: URL, options: any) {
      this.url = url
      this.options = options
    })
    Object.defineProperty(globalThis, "Worker", {
      configurable: true,
      writable: true,
      value: WorkerConstructor
    })

    subject.createStandaloneHtmlOutlineWorker("source-must-not-in-url")

    expect(WorkerConstructor).toHaveBeenCalledTimes(1)
    const [url, options] = WorkerConstructor.mock.calls[0]
    expect(String(url)).toMatch(/standalone-html-outline\.worker/)
    expect(String(url)).not.toContain("source-must-not-in-url")
    expect(options).toEqual({ type: "module" })
  })

  it("keeps one active parse and coalesces only the latest pending source", async () => {
    const subject = await loadClient()
    const workers: FakeWorker[] = []
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => {
        const worker = new FakeWorker()
        workers.push(worker)
        return worker
      },
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })

    controller.request({ source: "source A", digest: "a".repeat(64) })
    controller.request({ source: "source B", digest: "b".repeat(64) })
    controller.request({ source: "source C", digest: "c".repeat(64) })

    expect(workers).toHaveLength(1)
    expect(workers[0].posted).toHaveLength(1)
    expect(workers[0].posted[0]).toEqual(
      expect.objectContaining({ source: "source A", digest: "a".repeat(64) })
    )
    const first = workers[0].posted[0]
    workers[0].respond(resultFor(first, "stale A"))

    expect(workers[0].posted).toHaveLength(2)
    expect(workers[0].posted[1]).toEqual(
      expect.objectContaining({ source: "source C", digest: "c".repeat(64) })
    )
    expect(JSON.stringify(workers[0].posted)).not.toContain("source B")
    expect(outlines).toEqual([])

    const latest = workers[0].posted[1]
    workers[0].respond(resultFor(latest, "current C"))

    expect(outlines).toHaveLength(1)
    expect(outlines[0].digest).toBe("c".repeat(64))
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "current", digest: "c".repeat(64) }))
  })

  it("ignores a duplicate stale response without cancelling the newer active parse", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: () => undefined,
      onOutline: (outline: any) => outlines.push(outline)
    })

    controller.request({ source: "source A", digest: "a".repeat(64) })
    controller.request({ source: "source B", digest: "b".repeat(64) })
    const first = worker.posted[0]
    worker.respond(resultFor(first, "A"))
    const latest = worker.posted[1]

    worker.respond(resultFor(first, "duplicate A"))
    worker.respond(resultFor(latest, "B"))

    expect(outlines).toHaveLength(1)
    expect(outlines[0].digest).toBe("b".repeat(64))
    controller.dispose()
  })

  it("rejects invalid Unicode before worker construction or encoding", async () => {
    const subject = await loadClient()
    const workerFactory = vi.fn(() => new FakeWorker())
    const states: any[] = []
    const RealEncoder = globalThis.TextEncoder
    const encoder = vi.fn(() => {
      throw new Error("invalid source must not be encoded")
    })
    Object.defineProperty(globalThis, "TextEncoder", {
      configurable: true,
      writable: true,
      value: encoder
    })
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory,
      onState: (state: any) => states.push(state),
      onOutline: () => undefined
    })

    try {
      controller.request({ source: "private\ud800source", digest: "a".repeat(64) })
      expect(workerFactory).not.toHaveBeenCalled()
      expect(encoder).not.toHaveBeenCalled()
      expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
      expect(JSON.stringify(states)).not.toContain("private")
    } finally {
      Object.defineProperty(globalThis, "TextEncoder", {
        configurable: true,
        writable: true,
        value: RealEncoder
      })
      controller.dispose()
    }
  })

  it("rejects a closed-looking DTO whose per-slide scalar total exceeds the cap", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "1".repeat(64) })
    const request = worker.posted[0]
    worker.respond({
      type: "result",
      requestId: request.requestId,
      digest: request.digest,
      outline: {
        digest: request.digest,
        slides: [
          {
            index: 1,
            blocks: Array.from({ length: 5 }, () => ({
              kind: "paragraph",
              text: "x".repeat(4_096),
              truncated: false
            })),
            notes: [],
            truncated: false
          }
        ],
        truncated: false
      }
    })

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it("terminates and replaces a hung worker after ten seconds, then runs only the latest pending source", async () => {
    const subject = await loadClient()
    const workers: FakeWorker[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => {
        const worker = new FakeWorker()
        workers.push(worker)
        return worker
      },
      onState: () => undefined,
      onOutline: () => undefined,
      watchdogMs: 10_000
    })

    controller.request({ source: "hung", digest: "d".repeat(64) })
    controller.request({ source: "latest", digest: "e".repeat(64) })
    await vi.advanceTimersByTimeAsync(10_000)

    expect(workers[0].terminate).toHaveBeenCalledTimes(1)
    expect(workers).toHaveLength(2)
    expect(workers[1].posted).toEqual([
      expect.objectContaining({ source: "latest", digest: "e".repeat(64) })
    ])
  })

  it("terminates a lone hung parse once and reports bounded failure without retrying forever", async () => {
    const subject = await loadClient()
    const workers: FakeWorker[] = []
    const states: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => {
        const worker = new FakeWorker()
        workers.push(worker)
        return worker
      },
      onState: (state: any) => states.push(state),
      onOutline: () => undefined,
      watchdogMs: 10_000
    })

    controller.request({ source: "hung", digest: "d".repeat(64) })
    await vi.advanceTimersByTimeAsync(10_000)

    expect(workers).toHaveLength(1)
    expect(workers[0].terminate).toHaveBeenCalledTimes(1)
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    expect(vi.getTimerCount()).toBe(0)
    controller.dispose()
  })

  it("terminates an errored worker before dispatching the latest pending source", async () => {
    const subject = await loadClient()
    const workers: FakeWorker[] = []
    const states: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => {
        const worker = new FakeWorker()
        workers.push(worker)
        return worker
      },
      onState: (state: any) => states.push(state),
      onOutline: () => undefined
    })

    controller.request({ source: "first", digest: "a".repeat(64) })
    controller.request({ source: "latest", digest: "b".repeat(64) })
    workers[0].onerror?.(new Event("error"))

    expect(workers[0].terminate).toHaveBeenCalledTimes(1)
    expect(workers).toHaveLength(2)
    expect(workers[1].posted).toEqual([
      expect.objectContaining({ source: "latest", digest: "b".repeat(64) })
    ])
    expect(states.at(-1)).toEqual(
      expect.objectContaining({ status: "stale", digest: "b".repeat(64) })
    )
    controller.dispose()
  })

  it("ignores a late worker error after the latest outline is already current", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: () => undefined
    })

    controller.request({ source: "current", digest: "c".repeat(64) })
    worker.respond(resultFor(worker.posted[0], "Current"))
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "current" }))

    worker.onerror?.(new Event("error"))

    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "current" }))
    controller.dispose()
  })

  it("rejects stale, malformed, and failed DTOs and disposes all retained work", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })

    controller.request({ source: "latest", digest: "f".repeat(64) })
    const request = worker.posted[0]
    worker.respond({ ...resultFor(request, "wrong digest"), digest: "0".repeat(64), source: "leak" })

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    expect(JSON.stringify(states)).not.toContain("latest")

    controller.dispose()
    expect(worker.terminate).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe("StandaloneHtmlSafeOutline rendering", () => {
  it("renders only React text nodes with automatic direction and bidi isolation", async () => {
    const { StandaloneHtmlSafeOutline } = await loadView()
    const outline = {
      digest: "a".repeat(64),
      slides: [
        {
          index: 1,
          blocks: [
            { kind: "heading", text: '<img src="https://never.invalid"> שלום', truncated: false },
            { kind: "paragraph", text: "Plain outline", truncated: false }
          ],
          notes: [{ kind: "paragraph", text: "Speaker notes", truncated: false }],
          truncated: false
        }
      ],
      truncated: false
    }

    render(React.createElement(StandaloneHtmlSafeOutline, { status: "stale", outline }))

    expect(screen.getByText("Safe outline — text only; code never runs in Studio")).toBeVisible()
    expect(screen.getByText("Stale")).toBeVisible()
    const card = screen.getByRole("article", { name: "Slide 1" })
    expect(within(card).queryByRole("img")).not.toBeInTheDocument()
    expect(within(card).queryByRole("link")).not.toBeInTheDocument()
    const literal = within(card).getByText('<img src="https://never.invalid"> שלום')
    expect(literal).toHaveAttribute("dir", "auto")
    expect(literal).toHaveClass("[unicode-bidi:isolate]")
    expect(within(card).getByText("Speaker notes")).toHaveAttribute("dir", "auto")
  })

  it("keeps the prior outline visible while stale and shows bounded failed state without source", async () => {
    const { StandaloneHtmlSafeOutline } = await loadView()
    const outline = {
      digest: "b".repeat(64),
      slides: [{ index: 1, blocks: [{ kind: "heading", text: "Last safe outline", truncated: false }], notes: [], truncated: false }],
      truncated: false
    }
    const view = render(
      React.createElement(StandaloneHtmlSafeOutline, { status: "stale", outline })
    )
    expect(screen.getByText("Last safe outline")).toBeVisible()

    view.rerender(
      React.createElement(StandaloneHtmlSafeOutline, { status: "failed", outline })
    )
    expect(screen.getByText("Last safe outline")).toBeVisible()
    expect(screen.getByText("Outline unavailable")).toBeVisible()
    expect(document.body.textContent).not.toContain("private source")
  })
})
