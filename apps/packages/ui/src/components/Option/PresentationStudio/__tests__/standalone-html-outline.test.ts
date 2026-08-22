import React from "react"
import { act, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const loadWorker = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-outline.worker"].join("/"))
const loadClient = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-outline-client"].join("/"))
const loadTextPolicy = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-outline-text-policy"].join("/"))
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

const URL_LIKE_TEXT_CASES = [
  ["www host", "www.example.com/path"],
  ["unlisted payment scheme", "bitcoin:1abc"],
  ["unlisted editor scheme", "vscode:workspace"],
  ["unlisted intent scheme", "intent:launch"],
  ["generic RFC scheme", "custom+extension.1:opaque"],
  ["bare domain", "example.com/path"],
  ["domain with port and suffix", "sub.example.co.uk:8080/path?q=1#frag"],
  ["IPv4 host", "127.0.0.1:8080/path"],
  ["IPv6 host", "[2001:db8::1]:8443/path"],
  ["localhost host", "localhost:3000/path"],
  ["Unicode U-label host", "例え.テスト/path"],
  ["mixed U-label host", "example.テスト/path"],
  ["ideographic-full-stop U-label host", "例え。テスト/path"],
  ["fullwidth-full-stop U-label host", "例え．テスト/path"],
  ["halfwidth-full-stop U-label host", "例え｡テスト/path"],
  ["Unicode U-label host with port", "例え.テスト:8443/path"],
  ["Unicode U-label host with query", "例え.テスト?deck=1"],
  ["Unicode U-label host with fragment", "例え.テスト#slide"],
  ["absolute path", "/absolute/path"],
  ["dot-relative path", "./relative/path"],
  ["parent-relative path", "../parent/path"]
] as const

const ORDINARY_NON_URL_TEXT_CASES = [
  "Title: text follows.",
  "Ratios 3:2 and decimals 3.14 stay prose.",
  "Versions 1.2.3 and v2.10.4 remain visible.",
  "Natural punctuation: commas, parentheses (draft), and periods.",
  "Date 2026-08-22, time 10:30.",
  "Chapter/section labels stay ordinary prose.",
  "これは通常の日本語です。",
  "章/節 の説明",
  "版本 2.1：説明",
  "普通的国际化文本。"
] as const

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
    ["bogus closing tags cannot conceal apparent depth", "<x></z>".repeat(129)],
    ["a text run above 65,536 UTF-8 bytes", "x".repeat(65_537)],
    ["an unterminated quoted tag", '<section class="slide>'],
    ["an unterminated comment", "<!-- private source"]
  ])("refuses %s with a fixed source-free result before parsing", async (_case, source) => {
    const subject = await loadWorker()

    const result = subject.preflightStandaloneHtmlOutline(source)

    expect(result).toEqual({
      ok: false,
      code: "document_too_complex",
      message: "Outline unavailable: document too complex."
    })
    expect(JSON.stringify(result)).not.toContain("private source")
  })

  it("treats syntactic self-closing non-void HTML elements as open for the depth preflight", async () => {
    vi.resetModules()
    const parserLoaded = vi.fn()
    vi.doMock("cheerio/slim", async () => {
      parserLoaded()
      return vi.importActual("cheerio/slim")
    })

    try {
      const subject = await loadWorker()
      expect(subject.preflightStandaloneHtmlOutline("<div/>".repeat(128))).toEqual({ ok: true })
      expect(subject.preflightStandaloneHtmlOutline("<div/>".repeat(129))).toEqual({
        ok: false,
        code: "document_too_complex",
        message: "Outline unavailable: document too complex."
      })
      await expect(
        subject.extractStandaloneHtmlOutline("<div/>".repeat(129), "a".repeat(64))
      ).rejects.toMatchObject({ code: "document_too_complex" })
      expect(parserLoaded).not.toHaveBeenCalled()
    } finally {
      vi.doUnmock("cheerio/slim")
      vi.resetModules()
    }
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
        <h1>First\u202e title</h1><p>Plain <a href="https://link.invalid">linked secret</a> text\u0007.</p><p>Deprecated\u206a control</p>
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
        "Deprecated control",
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

  it("discards whole semantic blocks containing direct URL-like text without changing source", async () => {
    const subject = await loadWorker()
    const schemes = [
      "about", "blob", "cid", "data", "file", "ftp", "geo", "git", "http", "https",
      "irc", "ircs", "javascript", "mailto", "news", "nntp", "sftp", "sms", "ssh",
      "tel", "urn", "webcal", "ws", "wss"
    ]
    const source = `
      <section class="slide">
        <p>Ordinary prose with a 3:2 ratio and version 2.1.</p>
        <p>HTTP://example.invalid/case</p>
        <p>https&#58;//entity.invalid/decoded</p>
        <p>//protocol-relative.invalid/resource</p>
        ${schemes.map((scheme) => `<p>Unsafe ${scheme}:resource</p>`).join("")}
      </section>
    `

    const result = await subject.extractStandaloneHtmlOutline(source, "f".repeat(64))

    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual([
      "Ordinary prose with a 3:2 ratio and version 2.1."
    ])
    expect(source).toContain("https&#58;//entity.invalid/decoded")
  })

  it("drops domain, host, generic-scheme, and path tokens while preserving ordinary prose", async () => {
    const subject = await loadWorker()
    const source = `<section class="slide">
      ${ORDINARY_NON_URL_TEXT_CASES.map((text) => `<p>${text}</p>`).join("")}
      ${URL_LIKE_TEXT_CASES.map(([, text]) => `<p>Unsafe ${text}</p>`).join("")}
    </section>`

    const result = await subject.extractStandaloneHtmlOutline(source, "e".repeat(64))

    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual(
      ORDINARY_NON_URL_TEXT_CASES
    )
    for (const [, text] of URL_LIKE_TEXT_CASES) {
      expect(JSON.stringify(result)).not.toContain(text)
      expect(source).toContain(text)
    }
  })

  it("classifies every authoritative URL marker with a bounded text-only policy", async () => {
    const policy = await loadTextPolicy()
    const schemes = [
      "about", "blob", "cid", "data", "file", "ftp", "geo", "git", "http", "https",
      "irc", "ircs", "javascript", "mailto", "news", "nntp", "sftp", "sms", "ssh",
      "tel", "urn", "webcal", "ws", "wss"
    ]

    for (const value of [
      "HTTP://example.invalid/case",
      "//protocol-relative.invalid/resource",
      ...URL_LIKE_TEXT_CASES.map(([, text]) => text),
      ...schemes.map((scheme) => `prefix ${scheme}:resource`)
    ]) {
      expect(policy.hasDirectUrlLikeText(value), value).toBe(true)
    }
    for (const value of [
      ...ORDINARY_NON_URL_TEXT_CASES,
      "The word https without punctuation"
    ]) {
      expect(policy.hasDirectUrlLikeText(value), value).toBe(false)
    }
  })

  it.each([
    "deck-header",
    "deck-footer",
    "slide-number",
    "progress",
    "navigation",
    "nav"
  ])("suppresses nested .%s chrome from trusted slide text", async (className) => {
    const subject = await loadWorker()
    const source = `<section class="slide">
      <p>Visible <span class="${className}">hidden inline</span> text</p>
      <div class="${className}"><p>hidden subtree</p></div>
    </section>`

    const result = await subject.extractStandaloneHtmlOutline(source, "1".repeat(64))

    expect(result.slides).toHaveLength(1)
    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual(["Visible text"])
    expect(JSON.stringify(result)).not.toContain("hidden")
  })

  it("does not discover a slide card whose root is also deck chrome", async () => {
    const subject = await loadWorker()
    const result = await subject.extractStandaloneHtmlOutline(
      `<section class="slide deck-header"><h1>Hidden root</h1></section>
       <section class="slide"><h1>Visible slide</h1></section>`,
      "2".repeat(64)
    )

    expect(result.slides).toHaveLength(1)
    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual(["Visible slide"])
    expect(JSON.stringify(result)).not.toContain("Hidden root")
  })

  it("blocks semantic nav chrome text and nested slide discovery inside nav", async () => {
    const subject = await loadWorker()
    const result = await subject.extractStandaloneHtmlOutline(
      `<section class="slide">
        <h1>Visible slide</h1>
        <nav>
          <p>Generated navigation chrome</p>
          <section class="slide"><p>Nested navigation slide</p></section>
        </nav>
      </section>`,
      "3".repeat(64)
    )

    expect(result.slides).toHaveLength(1)
    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual(["Visible slide"])
    expect(JSON.stringify(result)).not.toContain("navigation")
  })

  it("admits only the outer card and excludes a plain nested slide subtree from its text", async () => {
    const subject = await loadWorker()
    const result = await subject.extractStandaloneHtmlOutline(
      `<section class="slide">
        <h1>Outer slide</h1>
        <section class="slide"><h2>Nested duplicate</h2></section>
      </section>`,
      "4".repeat(64)
    )

    expect(result.slides).toHaveLength(1)
    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual(["Outer slide"])
    expect(JSON.stringify(result)).not.toContain("Nested duplicate")
  })

  it("does not discover or duplicate a slide nested below speaker notes", async () => {
    const subject = await loadWorker()
    const result = await subject.extractStandaloneHtmlOutline(
      `<section class="slide">
        <h1>Outer slide</h1>
        <div class="notes">
          <p>Speaker-only note</p>
          <section class="slide"><h2>Nested note slide</h2></section>
        </div>
      </section>`,
      "5".repeat(64)
    )

    expect(result.slides).toHaveLength(1)
    expect(result.slides[0].blocks.map((block: any) => block.text)).toEqual(["Outer slide"])
    expect(result.slides[0].notes.map((block: any) => block.text)).toEqual(["Speaker-only note"])
    expect(JSON.stringify(result)).not.toContain("Nested note slide")
  })

  it.each(["applet", "fencedframe", "frame", "frameset", "portal", "track"])(
    "blocks slide discovery below forbidden <%s> subtrees",
    async (tagName) => {
      const subject = await loadWorker()
      const root = {
        type: "root",
        children: [
          {
            type: "tag",
            name: tagName,
            children: [
              {
                type: "tag",
                name: "section",
                attribs: { class: "slide" },
                children: [{ type: "tag", name: "h1", children: [{ type: "text", data: "Hidden" }] }]
              }
            ]
          }
        ]
      }

      const slides = subject.validateStandaloneHtmlOutlineTree(root)

      expect(slides).toEqual([])
    }
  )

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

  it("counts every node and depth inside forbidden subtrees while suppressing their output", async () => {
    const subject = await loadWorker()
    expect(typeof subject.validateStandaloneHtmlOutlineTree).toBe("function")
    const nodeStorm = {
      type: "root",
      children: [
        {
          type: "tag",
          name: "script",
          children: Array.from({ length: 50_001 }, () => ({ type: "tag", name: "x" }))
        }
      ]
    }
    const deepRoot: any = { type: "root", children: [] }
    const forbidden = { type: "tag", name: "template", children: [] as any[] }
    deepRoot.children.push(forbidden)
    let cursor = forbidden
    for (let depth = 0; depth < 129; depth += 1) {
      const child = { type: "tag", name: "x", children: [] as any[] }
      cursor.children.push(child)
      cursor = child
    }

    expect(() => subject.validateStandaloneHtmlOutlineTree(nodeStorm)).toThrowError(
      expect.objectContaining({ code: "document_too_complex" })
    )
    expect(() => subject.validateStandaloneHtmlOutlineTree(deepRoot)).toThrowError(
      expect.objectContaining({ code: "document_too_complex" })
    )
  })

  it("reserves room for and appends the exact application-owned truncation marker within the block cap", async () => {
    const subject = await loadWorker()

    const result = await subject.extractStandaloneHtmlOutline(
      `<section class="slide"><p>${"x".repeat(5_000)}</p></section>`,
      "d".repeat(64)
    )

    expect(result.slides[0].blocks[0].text.endsWith("... [truncated]")).toBe(true)
    expect(Array.from(result.slides[0].blocks[0].text)).toHaveLength(4_096)
    expect(result.slides[0].blocks[0].truncated).toBe(true)
  })

  it("emits exactly one terminal marker when a truncated prefix naturally ends with the marker", async () => {
    const subject = await loadWorker()
    const marker = "... [truncated]"
    const naturalMarkerBoundary = `${"x".repeat(4_081 - marker.length)}${marker}`

    const result = await subject.extractStandaloneHtmlOutline(
      `<section class="slide"><p>${naturalMarkerBoundary}${"y".repeat(100)}</p></section>`,
      "9".repeat(64)
    )
    const block = result.slides[0].blocks[0]

    expect(block.truncated).toBe(true)
    expect(block.text.endsWith(marker)).toBe(true)
    expect(block.text.endsWith(`${marker}${marker}`)).toBe(false)
    expect(block.text.split(marker)).toHaveLength(2)
    expect(Array.from(block.text).length).toBeLessThanOrEqual(4_096)
  })

  it("keeps under-cap block text exact and never expands a near-budget block into a marker", async () => {
    const subject = await loadWorker()
    const underCap = "u".repeat(4_090)
    const nearSlideBudget = [
      ...Array.from({ length: 4 }, () => `<p>${"x".repeat(4_096)}</p>`),
      `<p>${"y".repeat(3_599)}</p>`,
      "<p>abc</p>"
    ].join("")

    const underCapResult = await subject.extractStandaloneHtmlOutline(
      `<section class="slide"><p>${underCap}</p></section>`,
      "e".repeat(64)
    )
    const nearBudgetResult = await subject.extractStandaloneHtmlOutline(
      `<section class="slide">${nearSlideBudget}</section>`,
      "f".repeat(64)
    )

    expect(underCapResult.slides[0].blocks[0]).toEqual({
      kind: "paragraph",
      text: underCap,
      truncated: false
    })
    expect(underCapResult.slides[0].truncated).toBe(false)
    expect(underCapResult.truncated).toBe(false)
    expect(nearBudgetResult.slides[0].blocks.at(-1)).toEqual({
      kind: "paragraph",
      text: "abc",
      truncated: false
    })
    expect(nearBudgetResult.slides[0].blocks.every((block: any) => !block.truncated)).toBe(true)
    expect(nearBudgetResult.slides[0].truncated).toBe(false)
    expect(nearBudgetResult.truncated).toBe(false)
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

  it("contains a synchronous worker-factory failure and accepts the next request", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    let attempts = 0
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => {
        attempts += 1
        if (attempts === 1) throw new DOMException("Worker blocked", "SecurityError")
        return worker
      },
      onState: (state: any) => states.push(state),
      onOutline: () => undefined
    })

    expect(() => controller.request({ source: "private first", digest: "1".repeat(64) })).not.toThrow()
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    expect(JSON.stringify(states)).not.toContain("private first")
    expect(vi.getTimerCount()).toBe(0)

    expect(() => controller.request({ source: "safe retry", digest: "2".repeat(64) })).not.toThrow()
    expect(worker.posted).toEqual([
      expect.objectContaining({ source: "safe retry", digest: "2".repeat(64) })
    ])
    controller.dispose()
  })

  it("contains synchronous postMessage failure, terminates that worker, and accepts the next request", async () => {
    const subject = await loadClient()
    const failedWorker = new FakeWorker()
    vi.spyOn(failedWorker, "postMessage").mockImplementation(() => {
      throw new DOMException("Posting blocked", "DataCloneError")
    })
    const retryWorker = new FakeWorker()
    const workers = [failedWorker, retryWorker]
    const states: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => workers.shift()!,
      onState: (state: any) => states.push(state),
      onOutline: () => undefined
    })

    expect(() => controller.request({ source: "private first", digest: "3".repeat(64) })).not.toThrow()
    expect(failedWorker.terminate).toHaveBeenCalledTimes(1)
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    expect(JSON.stringify(states)).not.toContain("private first")
    expect(vi.getTimerCount()).toBe(0)

    expect(() => controller.request({ source: "safe retry", digest: "4".repeat(64) })).not.toThrow()
    expect(retryWorker.posted).toEqual([
      expect.objectContaining({ source: "safe retry", digest: "4".repeat(64) })
    ])
    controller.dispose()
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

  it("rejects a forged DTO with more than the parsed-node block ceiling", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "5".repeat(64) })
    const request = worker.posted[0]
    const blocks = (count: number) => Array.from({ length: count }, () => ({
      kind: "paragraph",
      text: "x",
      truncated: false
    }))

    worker.respond({
      type: "result",
      requestId: request.requestId,
      digest: request.digest,
      outline: {
        digest: request.digest,
        slides: [
          { index: 1, blocks: blocks(20_000), notes: [], truncated: false },
          { index: 2, blocks: blocks(20_000), notes: [], truncated: false },
          { index: 3, blocks: blocks(10_001), notes: [], truncated: false }
        ],
        truncated: false
      }
    })

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it("rejects a DTO whose reserved application markers would exceed the rendered total cap", async () => {
    const subject = await loadClient()
    const { StandaloneHtmlSafeOutline } = await loadView()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "8".repeat(64) })
    const request = worker.posted[0]
    const marker = "... [truncated]"
    const slides = [
      ...Array.from({ length: 5 }, (_, index) => ({
        index: index + 1,
        blocks: Array.from({ length: 1_333 }, () => ({
          kind: "paragraph",
          text: marker,
          truncated: true
        })),
        notes: [],
        truncated: true
      })),
      ...Array.from({ length: 25 }, (_, index) => ({
        index: index + 6,
        blocks: [{ kind: "paragraph", text: "x", truncated: false }],
        notes: [],
        truncated: true
      }))
    ]

    worker.respond({
      type: "result",
      requestId: request.requestId,
      digest: request.digest,
      outline: { digest: request.digest, slides, truncated: true }
    })
    const view = render(
      React.createElement(StandaloneHtmlSafeOutline, {
        status: states.at(-1)?.status ?? "stale",
        outline: outlines[0] ?? null
      })
    )
    const renderedScalars = Array.from(view.container.querySelectorAll("p")).reduce(
      (total, node) => total + Array.from(node.textContent ?? "").length,
      0
    )

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    expect(renderedScalars).toBeLessThanOrEqual(100_000)
    controller.dispose()
  })

  it.each(["blocks", "notes"] as const)(
    "contains and rejects a forged DTO with a sparse %s array",
    async (collection) => {
      const subject = await loadClient()
      const worker = new FakeWorker()
      const states: any[] = []
      const outlines: any[] = []
      const controller = new subject.StandaloneHtmlOutlineController({
        workerFactory: () => worker,
        onState: (state: any) => states.push(state),
        onOutline: (outline: any) => outlines.push(outline)
      })
      controller.request({ source: "latest", digest: "6".repeat(64) })
      const request = worker.posted[0]
      const response = resultFor(request, "safe")
      response.outline.slides[0][collection] = Array(1) as any

      expect(() => worker.respond(response)).not.toThrow()
      expect(outlines).toEqual([])
      expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
      controller.dispose()
    }
  )

  it.each([
    ["sparse", (slides: any[]) => Array(slides.length)],
    ["enumerable-expando", (slides: any[]) => {
      const forged = [...slides] as any
      forged.attacker = true
      return forged
    }]
  ])("rejects a forged DTO with a %s top-level slides array", async (_case, forge) => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "9".repeat(64) })
    const request = worker.posted[0]
    const response = resultFor(request, "safe")
    response.outline.slides = forge(response.outline.slides) as any

    expect(() => worker.respond(response)).not.toThrow()
    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it.each([
    ["C0", "safe\u0007text"],
    ["C1", "safe\u0085text"],
    ["bidi formatting", "safe\u202etext"],
    ["deprecated bidi formatting", "safe\u206atext"],
    ["a lone UTF-16 surrogate", "safe\ud800text"],
    ["empty block text", ""]
  ])("rejects a forged DTO containing %s", async (_case, forgedText) => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "2".repeat(64) })
    const request = worker.posted[0]

    worker.respond(resultFor(request, forgedText))

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it.each([
    ["known scheme", "Contact mailto:owner@example.invalid"],
    ...URL_LIKE_TEXT_CASES
  ])("rejects a forged DTO containing %s URL-like text", async (_case, forgedText) => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "4".repeat(64) })
    const request = worker.posted[0]

    worker.respond(resultFor(request, forgedText))

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it.each([
    [
      "a truncated block without the exact marker",
      (response: any) => {
        response.outline.slides[0].blocks[0].truncated = true
        response.outline.slides[0].truncated = true
        response.outline.truncated = true
      }
    ],
    [
      "block truncation without slide truncation",
      (response: any) => {
        response.outline.slides[0].blocks[0].text = "safe... [truncated]"
        response.outline.slides[0].blocks[0].truncated = true
        response.outline.truncated = true
      }
    ],
    [
      "block and slide truncation without outline truncation",
      (response: any) => {
        response.outline.slides[0].blocks[0].text = "safe... [truncated]"
        response.outline.slides[0].blocks[0].truncated = true
        response.outline.slides[0].truncated = true
      }
    ],
    [
      "slide truncation without outline truncation",
      (response: any) => {
        response.outline.slides[0].truncated = true
      }
    ]
  ])("rejects forged DTO truncation metadata with %s", async (_case, mutate) => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "3".repeat(64) })
    const request = worker.posted[0]
    const response = resultFor(request, "safe block")
    mutate(response)

    worker.respond(response)

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it("rejects a forged truncated block with an adjacent application-marker chain", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const states: any[] = []
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: (state: any) => states.push(state),
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "7".repeat(64) })
    const request = worker.posted[0]
    const response = resultFor(request, "safe... [truncated]... [truncated]")
    response.outline.slides[0].blocks[0].truncated = true
    response.outline.slides[0].truncated = true
    response.outline.truncated = true

    worker.respond(response)

    expect(outlines).toEqual([])
    expect(states.at(-1)).toEqual(expect.objectContaining({ status: "failed" }))
    controller.dispose()
  })

  it("accepts an untruncated natural application-marker suffix as ordinary text", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: () => undefined,
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "8".repeat(64) })
    const request = worker.posted[0]
    const response = resultFor(request, "natural... [truncated]")

    worker.respond(response)

    expect(outlines).toEqual([response.outline])
    controller.dispose()
  })

  it("accepts outline-only structural truncation for omitted slides", async () => {
    const subject = await loadClient()
    const worker = new FakeWorker()
    const outlines: any[] = []
    const controller = new subject.StandaloneHtmlOutlineController({
      workerFactory: () => worker,
      onState: () => undefined,
      onOutline: (outline: any) => outlines.push(outline)
    })
    controller.request({ source: "latest", digest: "4".repeat(64) })
    const request = worker.posted[0]
    const response = resultFor(request, "safe block")
    response.outline.truncated = true

    worker.respond(response)

    expect(outlines).toEqual([response.outline])
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

  it("ignores a retired worker error after dispatching pending work to its replacement", async () => {
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

    controller.request({ source: "first", digest: "a".repeat(64) })
    controller.request({ source: "latest", digest: "b".repeat(64) })
    const retiredWorkerError = workers[0].onerror
    retiredWorkerError?.(new Event("error"))
    const latestRequest = workers[1].posted[0]

    retiredWorkerError?.(new Event("error"))
    workers[1].respond(resultFor(latestRequest, "Latest"))

    expect(workers[1].terminate).not.toHaveBeenCalled()
    expect(outlines).toEqual([
      expect.objectContaining({ digest: "b".repeat(64) })
    ])
    expect(states.at(-1)).toEqual(
      expect.objectContaining({ status: "current", digest: "b".repeat(64) })
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
  it("retains every inline and structural marker at the exact total scalar boundary", async () => {
    const { StandaloneHtmlSafeOutline } = await loadView()
    const marker = "... [truncated]"
    const exactSlideBlocks = () => [
      ...Array.from({ length: 4 }, () => ({
        kind: "paragraph",
        text: "x".repeat(4_096),
        truncated: false
      })),
      { kind: "paragraph", text: "y".repeat(3_616), truncated: false }
    ]
    const inlineBlocks = exactSlideBlocks()
    inlineBlocks[4] = {
      kind: "paragraph",
      text: `${"y".repeat(3_616 - marker.length)}${marker}`,
      truncated: true
    }
    const outline = {
      digest: "5".repeat(64),
      slides: [
        ...Array.from({ length: 4 }, (_, index) => ({
          index: index + 1,
          blocks: exactSlideBlocks(),
          notes: [],
          truncated: false
        })),
        { index: 5, blocks: inlineBlocks, notes: [], truncated: true },
        { index: 6, blocks: [], notes: [], truncated: true }
      ],
      truncated: true
    }

    const view = render(
      React.createElement(StandaloneHtmlSafeOutline, { status: "current", outline })
    )
    const slideFive = screen.getByRole("article", { name: "Slide 5" })
    const slideSix = screen.getByRole("article", { name: "Slide 6" })
    const renderedScalars = Array.from(view.container.querySelectorAll("p")).reduce(
      (total, node) => total + Array.from(node.textContent ?? "").length,
      0
    )

    expect(slideFive.textContent).toContain(marker)
    expect(slideSix.textContent).toContain(marker)
    expect(renderedScalars).toBeLessThanOrEqual(100_000)
  })

  it("keeps an application-owned card marker inside the exact slide scalar cap", async () => {
    const worker = await loadWorker()
    const { StandaloneHtmlSafeOutline } = await loadView()
    const exactSlide = [
      ...Array.from({ length: 4 }, () => `<p>${"x".repeat(4_096)}</p>`),
      `<p>${"y".repeat(3_616)}</p>`,
      "<p>one omitted block</p>"
    ].join("")
    const outline = await worker.extractStandaloneHtmlOutline(
      `<section class="slide">${exactSlide}</section>`,
      "6".repeat(64)
    )

    const view = render(
      React.createElement(StandaloneHtmlSafeOutline, { status: "current", outline })
    )
    const card = screen.getByRole("article", { name: "Slide 1" })
    const renderedScalars = Array.from(card.querySelectorAll("p")).reduce(
      (total, node) => total + Array.from(node.textContent ?? "").length,
      0
    )

    expect(renderedScalars).toBeLessThanOrEqual(20_000)
    expect(card.textContent).toContain("... [truncated]")
    view.unmount()
  })

  it("keeps the max-card structural marker inside the total scalar cap", async () => {
    const worker = await loadWorker()
    const { StandaloneHtmlSafeOutline } = await loadView()
    const exactSlide = `<section class="slide">${[
      ...Array.from({ length: 4 }, () => `<p>${"x".repeat(4_096)}</p>`),
      `<p>${"y".repeat(3_616)}</p>`
    ].join("")}</section>`
    const source = `${exactSlide.repeat(5)}${"<section class=\"slide\"></section>".repeat(26)}`
    const outline = await worker.extractStandaloneHtmlOutline(source, "7".repeat(64))

    const view = render(
      React.createElement(StandaloneHtmlSafeOutline, { status: "current", outline })
    )
    const renderedScalars = Array.from(view.container.querySelectorAll("p")).reduce(
      (total, node) => total + Array.from(node.textContent ?? "").length,
      0
    )

    expect(outline.slides).toHaveLength(30)
    expect(renderedScalars).toBeLessThanOrEqual(100_000)
    expect(view.container.textContent).toContain("... [truncated]")
    view.unmount()
  })

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

    expect(screen.getByText("Safe outline: text only; code never runs in Studio")).toBeVisible()
    expect(screen.getByText("Stale")).toBeVisible()
    const card = screen.getByRole("article", { name: "Slide 1" })
    expect(within(card).queryByRole("img")).not.toBeInTheDocument()
    expect(within(card).queryByRole("link")).not.toBeInTheDocument()
    const literal = within(card).getByText('<img src="https://never.invalid"> שלום')
    expect(literal).toHaveAttribute("dir", "auto")
    expect(literal).toHaveClass("[unicode-bidi:isolate]")
    const disclosure = within(card).getByText("Speaker notes", { selector: "summary" })
    expect(disclosure).toBeVisible()
    expect(disclosure.closest("details")).toContainElement(
      within(card).getByText("Speaker notes", { selector: "p" })
    )
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

  it("renders exactly one application-owned marker at the card or outline structural boundary", async () => {
    const { StandaloneHtmlSafeOutline } = await loadView()
    const marker = "... [truncated]"
    const cardTruncated = {
      digest: "c".repeat(64),
      slides: [{
        index: 1,
        blocks: [{
          kind: "paragraph",
          text: `Natural sentence ${marker}`,
          truncated: false
        }],
        notes: [],
        truncated: true
      }],
      truncated: true
    }
    const view = render(
      React.createElement(StandaloneHtmlSafeOutline, {
        status: "current",
        outline: cardTruncated
      })
    )

    expect(screen.getByText(`Natural sentence ${marker}`)).toBeVisible()
    expect(within(screen.getByRole("article", { name: "Slide 1" })).getByText(marker)).toBeVisible()
    expect(screen.getAllByText(marker)).toHaveLength(1)

    view.rerender(
      React.createElement(StandaloneHtmlSafeOutline, {
        status: "current",
        outline: {
          ...cardTruncated,
          slides: [{ ...cardTruncated.slides[0], truncated: false }]
        }
      })
    )

    expect(within(screen.getByRole("article", { name: "Slide 1" })).queryByText(marker)).toBeNull()
    expect(screen.getByText(marker)).toBeVisible()
  })
})
