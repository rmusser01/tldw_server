import { afterEach, describe, expect, it, vi } from "vitest"
import { TokenizerWorker } from "../TokenizerWorker"

describe("TokenizerWorker", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it("rejects pending tokenize request when worker emits error", async () => {
    class MockWorker {
      onmessage: ((event: MessageEvent<unknown>) => void) | null = null
      onerror: ((event: Event) => void) | null = null
      onmessageerror: ((event: MessageEvent<unknown>) => void) | null = null

      postMessage() {
        Promise.resolve().then(() => {
          this.onerror?.(new Event("error"))
        })
      }

      terminate() {
        // no-op
      }
    }

    vi.stubGlobal("Worker", MockWorker as unknown as typeof Worker)

    const tokenizer = new TokenizerWorker()
    await expect(tokenizer.tokenize("hello world")).rejects.toThrow(
      "Tokenizer worker error"
    )
    tokenizer.terminate()
  })

  it("rejects pending work on explicit termination", async () => {
    class MockWorker {
      postMessage() {}
      terminate() {}
    }
    vi.stubGlobal("Worker", MockWorker as unknown as typeof Worker)
    const tokenizer = new TokenizerWorker()
    const pending = tokenizer.tokenize("pending")
    const assertion = expect(pending).rejects.toMatchObject({
      name: "AbortError"
    })
    tokenizer.terminate()
    await assertion
  })

  it("terminates a timed-out worker and allows a fresh request", async () => {
    vi.useFakeTimers()
    const terminate = vi.fn()
    let instances = 0
    class MockWorker {
      onmessage: ((event: MessageEvent<unknown>) => void) | null = null
      readonly generation = ++instances
      postMessage(request: { id: string }) {
        if (this.generation > 1)
          queueMicrotask(() =>
            this.onmessage?.({
              data: { id: request.id, tokenCount: 2 }
            } as MessageEvent)
          )
      }
      terminate() {
        terminate()
      }
    }
    vi.stubGlobal("Worker", MockWorker as unknown as typeof Worker)
    const tokenizer = new TokenizerWorker()
    const assertion = expect(tokenizer.tokenize("timeout")).rejects.toThrow(
      "timed out"
    )
    await vi.advanceTimersByTimeAsync(10001)
    await assertion
    expect(terminate).toHaveBeenCalledOnce()
    const recovered = tokenizer.tokenize("working")
    await vi.runAllTicks()
    expect(await recovered).toBe(2)
    tokenizer.terminate()
  })

  it("times out pending tokenize request when worker does not respond", async () => {
    vi.useFakeTimers()

    class MockWorker {
      onmessage: ((event: MessageEvent<unknown>) => void) | null = null
      onerror: ((event: Event) => void) | null = null
      onmessageerror: ((event: MessageEvent<unknown>) => void) | null = null

      postMessage() {
        // intentionally never responds
      }

      terminate() {
        // no-op
      }
    }

    vi.stubGlobal("Worker", MockWorker as unknown as typeof Worker)

    const tokenizer = new TokenizerWorker()
    const pending = tokenizer.tokenize("will timeout")
    const assertion = expect(pending).rejects.toThrow("timed out")
    await vi.advanceTimersByTimeAsync(10_001)
    await assertion
    tokenizer.terminate()
  })
})
