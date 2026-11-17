import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

/**
 * Drives the REAL popup render target (the shadow-DOM response element
 * created by the entry) with a mocked chat stream, so the streaming
 * append path (W5 Stage 2, WP-58) is proven at the DOM level:
 *   - final textContent is byte-identical to the chunk concatenation,
 *   - chunks are appended as text nodes (MutationObserver: no childList
 *     removals = no full-text rewrites),
 *   - per-chunk work performs zero synchronous layout reads and all
 *     position updates coalesce into a single rAF flush.
 */

const chatState = vi.hoisted(() => ({
  chunks: [] as string[],
  cancelStreamCalls: 0,
  gate: null as { promise: Promise<void>; resolve: () => void } | null
}))

const runtimeListeners = vi.hoisted(() => ({
  onMessage: undefined as ((message: unknown) => unknown) | undefined
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineContentScript", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {}
})

vi.mock("@/services/tldw/TldwChat", () => ({
  TldwChatService: class MockTldwChatService {
    async *streamMessage(): AsyncGenerator<string, void, unknown> {
      let index = 0
      for (const chunk of chatState.chunks) {
        // The stop-path test parks the stream after the first chunk.
        if (index > 0 && chatState.gate) await chatState.gate.promise
        index += 1
        yield chunk
      }
    }
    cancelStream(): void {
      chatState.cancelStreamCalls += 1
    }
  }
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      sendMessage: vi.fn(async () => ({ ok: true })),
      onMessage: {
        addListener: (listener: (message: unknown) => unknown) => {
          runtimeListeners.onMessage = listener
        }
      }
    },
    i18n: {
      getMessage: () => ""
    },
    storage: {
      sync: {
        get: vi.fn(async () => ({ selectedModel: "popup-test-model" }))
      },
      local: {
        get: vi.fn(async () => ({}))
      }
    }
  }
}))

import copilotPopupEntry, {
  getCopilotPopupHandler,
  registerCopilotPopupHandler
} from "@/entries/copilot-popup.content"

const POPUP_HOST_ID = "tldw-copilot-popup-host"

const getHost = () => document.getElementById(POPUP_HOST_ID)

const getShadowRoot = () => {
  const host = getHost()
  const root = host?.shadowRoot
  expect(root).toBeTruthy()
  return root as ShadowRoot
}

const getResponseEl = () =>
  getShadowRoot().querySelector("[data-role='response']") as HTMLDivElement

const buildChunks = (count: number) => {
  const chunks: string[] = []
  for (let i = 0; i < count; i++) {
    if (i % 97 === 0) chunks.push("")
    chunks.push(i % 3 === 0 ? `chunk-${i}\n` : `chunk-${i} `)
  }
  return chunks
}

const flushAnimationFrame = () =>
  new Promise<void>((resolve) =>
    window.requestAnimationFrame(() => resolve())
  )

const flushAnimationFrames = async () => {
  await flushAnimationFrame()
  await flushAnimationFrame()
}

const flushMutationObserver = async (observer: MutationObserver) => {
  await new Promise((resolve) => setTimeout(resolve, 0))
  return observer.takeRecords()
}

const openPopup = (pending: Promise<void>) => {
  // handlePopupOpen runs synchronously up to its first await, so the
  // popup (and its response element) already exist when this returns.
  expect(getResponseEl()).toBeTruthy()
  return pending
}

const startStream = (chunks: string[]) => {
  // Note: chatState.gate is intentionally NOT reset here so the
  // stop-path test can arm it before starting the stream.
  chatState.chunks = chunks
  const handler = getCopilotPopupHandler()
  expect(typeof handler).toBe("function")
  return openPopup(handler!({
    selectionText: "selected words for the copilot",
    pageUrl: "https://example.com/post"
  }))
}

// jsdom does not implement every geometry constructor; updatePosition
// falls back to `new DOMRect(...)` when there is no anchor range.
if (typeof (globalThis as { DOMRect?: unknown }).DOMRect === "undefined") {
  ;(globalThis as unknown as Record<string, unknown>).DOMRect = class {
    left: number
    top: number
    width: number
    height: number
    right: number
    bottom: number
    constructor(x: number, y: number, width: number, height: number) {
      this.left = x
      this.top = y
      this.width = width
      this.height = height
      this.right = x + width
      this.bottom = y + height
    }
  }
}

describe("copilot popup content script entry", () => {
  beforeEach(() => {
    chatState.chunks = []
    chatState.cancelStreamCalls = 0
    chatState.gate = null
    registerCopilotPopupHandler()
  })

  afterEach(() => {
    getHost()?.remove()
  })

  it("registers the every-page message contract", async () => {
    const contentScript = copilotPopupEntry as {
      matches?: string[]
      main?: () => void
    }

    expect(contentScript.matches).toEqual(["http://*/*", "https://*/*"])

    contentScript.main?.()

    expect(typeof runtimeListeners.onMessage).toBe("function")
    await expect(
      runtimeListeners.onMessage?.({ type: "tldw:popup:open", payload: {} })
    ).resolves.toEqual({ ok: true })
    expect(runtimeListeners.onMessage?.({ type: "unrelated" })).toBeUndefined()
  })

  it("appends 1000 streamed chunks as text nodes with byte-identical final text and no full-text rewrites", async () => {
    const chunks = buildChunks(1000)
    const pending = startStream(chunks)

    const responseEl = getResponseEl()
    expect(responseEl.textContent).toBe("")

    const records: MutationRecord[] = []
    const observer = new MutationObserver((batch) => records.push(...batch))
    observer.observe(responseEl, {
      childList: true,
      characterData: true,
      subtree: true
    })

    await pending
    records.push(...(await flushMutationObserver(observer)))
    observer.disconnect()

    // Byte-identical final render.
    expect(responseEl.textContent).toBe(chunks.join(""))

    // Appends, not replaces: a textContent assignment removes the old
    // child(ren); pure text-node appends never remove anything.
    const rewrites = records.filter((r) => r.removedNodes.length > 0)
    expect(rewrites).toEqual([])

    // One appended text node per non-empty chunk, nothing else.
    const addedNodes = records.flatMap((r) => Array.from(r.addedNodes))
    expect(addedNodes).toHaveLength(chunks.filter((c) => c !== "").length)
    expect(
      addedNodes.every((node) => node.nodeType === Node.TEXT_NODE)
    ).toBe(true)
  })

  it("performs zero synchronous layout reads while streaming and coalesces position updates into one frame", async () => {
    const original = Element.prototype.getBoundingClientRect
    const rectSpy = vi
      .spyOn(Element.prototype, "getBoundingClientRect")
      .mockImplementation(function (this: Element) {
        return original.call(this)
      })

    const pending = startStream(buildChunks(1000))
    await pending

    // Every chunk only *schedules* a rAF; no synchronous layout reads may
    // happen on the host page's main thread during the stream.
    expect(rectSpy).toHaveBeenCalledTimes(0)

    await flushAnimationFrames()

    // 1000 schedulePositionUpdate calls collapsed into exactly one
    // updatePosition pass (one popup rect read) after the first frame.
    expect(rectSpy).toHaveBeenCalledTimes(1)
  })

  it("keeps the appended prefix when stopping mid-stream instead of rewriting it", async () => {
    chatState.chunks = ["first-part ", "second-part ", "third-part "]
    let releaseGate!: () => void
    chatState.gate = {
      promise: new Promise<void>((resolve) => {
        releaseGate = resolve
      }),
      resolve: () => releaseGate()
    }

    const pending = startStream(chatState.chunks)
    const responseEl = getResponseEl()

    const records: MutationRecord[] = []
    const observer = new MutationObserver((batch) => records.push(...batch))
    observer.observe(responseEl, {
      childList: true,
      characterData: true,
      subtree: true
    })

    await vi.waitFor(() =>
      expect(responseEl.textContent).toBe("first-part ")
    )

    getShadowRoot()
      .querySelector("[data-action='stop']")!
      .dispatchEvent(new MouseEvent("click", { bubbles: true }))

    releaseGate()
    await pending
    records.push(...(await flushMutationObserver(observer)))
    observer.disconnect()

    // The stop path (setState("done") + setResponseText(accumulated)) must
    // preserve the appended prefix byte-for-byte, not rewrite it.
    expect(responseEl.textContent).toBe("first-part ")
    expect(records.filter((r) => r.removedNodes.length > 0)).toEqual([])

    const previewEl = getShadowRoot().querySelector(
      "[data-role='preview']"
    ) as HTMLTextAreaElement
    expect(previewEl.value).toBe("first-part ")
    expect(chatState.cancelStreamCalls).toBe(1)
  })
})
