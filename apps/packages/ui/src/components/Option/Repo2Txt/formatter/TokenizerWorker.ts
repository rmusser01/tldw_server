import { encode } from "gpt-tokenizer"
import type {
  ProgressResponse,
  TokenizeFileRequest,
  TokenizeRequest,
  TokenizeResponse
} from "../workers/tokenizer.worker"

type TokenizerMessage = TokenizeResponse | ProgressResponse
type MessageHandler = (payload: TokenizerMessage) => void
type RejectHandler = (error: Error) => void
type PendingRequest = {
  handle: MessageHandler
  reject: RejectHandler
  timeoutId: ReturnType<typeof setTimeout>
}

const REQUEST_TIMEOUT_MS = 10_000

const estimateTokens = (text: string): number => {
  const normalized = String(text || "")
  const trimmed = normalized.trim()
  if (!trimmed) return 0
  try {
    return encode(normalized).length
  } catch {
    return trimmed.split(/\s+/).length
  }
}

export class TokenizerWorker {
  private worker: Worker | null = null
  private handlers = new Map<string, PendingRequest>()
  private requestId = 0

  constructor() {
    this.initWorker()
  }

  private initWorker() {
    if (typeof Worker === "undefined") {
      this.worker = null
      return
    }

    try {
      this.worker = new Worker(
        new URL("../workers/tokenizer.worker.ts", import.meta.url),
        { type: "module" }
      )

      this.worker.onmessage = (event: MessageEvent<TokenizerMessage>) => {
        const payload = event.data
        const id = payload?.id
        if (!id) return
        const pending = this.handlers.get(id)
        if (!pending) return
        pending.handle(payload)
        if ("tokenCount" in payload) {
          clearTimeout(pending.timeoutId)
          this.handlers.delete(id)
        }
      }
      this.worker.onerror = () => {
        this.failPendingRequests("Tokenizer worker error")
      }
      this.worker.onmessageerror = () => {
        this.failPendingRequests("Tokenizer worker message error")
      }
    } catch {
      this.worker = null
    }
  }

  private failPendingRequests(reason: string) {
    for (const [, pending] of this.handlers) {
      clearTimeout(pending.timeoutId)
      pending.reject(new Error(reason))
    }
    this.handlers.clear()
    this.worker?.terminate()
    this.worker = null
    this.initWorker()
  }

  private registerPendingRequest(
    id: string,
    handle: MessageHandler,
    reject: RejectHandler
  ) {
    const timeoutId = setTimeout(() => {
      this.failPendingRequests(
        `Tokenizer worker request timed out after ${REQUEST_TIMEOUT_MS}ms`
      )
    }, REQUEST_TIMEOUT_MS)
    this.handlers.set(id, {
      handle,
      reject,
      timeoutId
    })
  }

  async tokenize(text: string): Promise<number> {
    if (!this.worker) {
      return estimateTokens(text)
    }

    return await new Promise<number>((resolve, reject) => {
      const id = `single-${++this.requestId}`
      this.registerPendingRequest(
        id,
        (payload) => {
          if ("tokenCount" in payload) {
            if (payload.error) {
              reject(new Error(payload.error))
              return
            }
            resolve(payload.tokenCount)
          }
        },
        reject
      )

      const request: TokenizeRequest = {
        id,
        type: "single",
        text
      }
      try {
        this.worker!.postMessage(request)
      } catch (error) {
        const pending = this.handlers.get(id)
        if (pending) {
          clearTimeout(pending.timeoutId)
          this.handlers.delete(id)
        }
        reject(
          error instanceof Error
            ? error
            : new Error("Failed to post tokenize request")
        )
      }
    })
  }

  async tokenizeBatch(
    files: TokenizeFileRequest[],
    onProgress?: (progress: number, current: number, total: number) => void
  ): Promise<{
    totalTokens: number
    files: Array<{ path: string; tokenCount: number; lineCount: number }>
  }> {
    if (!this.worker) {
      const results = files.map((file) => ({
        path: file.path,
        tokenCount: estimateTokens(file.content),
        lineCount: String(file.content || "").split("\n").length
      }))
      return {
        totalTokens: results.reduce((sum, item) => sum + item.tokenCount, 0),
        files: results
      }
    }

    return await new Promise((resolve, reject) => {
      const id = `batch-${++this.requestId}`
      this.registerPendingRequest(
        id,
        (payload) => {
          if ("progress" in payload) {
            onProgress?.(payload.progress, payload.current, payload.total)
            return
          }

          if (payload.error) {
            reject(new Error(payload.error))
            return
          }

          resolve({
            totalTokens: payload.tokenCount,
            files: payload.files ?? []
          })
        },
        reject
      )

      const request: TokenizeRequest = {
        id,
        type: "batch",
        files
      }
      try {
        this.worker!.postMessage(request)
      } catch (error) {
        const pending = this.handlers.get(id)
        if (pending) {
          clearTimeout(pending.timeoutId)
          this.handlers.delete(id)
        }
        reject(
          error instanceof Error
            ? error
            : new Error("Failed to post tokenize batch request")
        )
      }
    })
  }

  terminate(): void {
    for (const [, pending] of this.handlers) {
      clearTimeout(pending.timeoutId)
      pending.reject(new DOMException("Cancelled", "AbortError"))
    }
    this.worker?.terminate()
    this.worker = null
    this.handlers.clear()
  }
}

let tokenizerWorkerInstance: TokenizerWorker | null = null

export const getTokenizerWorker = (): TokenizerWorker => {
  if (!tokenizerWorkerInstance) {
    tokenizerWorkerInstance = new TokenizerWorker()
  }
  return tokenizerWorkerInstance
}

export const terminateTokenizerWorker = (): void => {
  tokenizerWorkerInstance?.terminate()
  tokenizerWorkerInstance = null
}
