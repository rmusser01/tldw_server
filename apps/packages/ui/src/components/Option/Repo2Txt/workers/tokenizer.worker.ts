import { encode } from "gpt-tokenizer"

export type TokenizeFileRequest = {
  path: string
  content: string
}

export type TokenizeRequest = {
  id: string
  type: "single" | "batch"
  text?: string
  files?: TokenizeFileRequest[]
}

export type ProgressResponse = {
  id: string
  progress: number
  current: number
  total: number
}

export type TokenizeResponse = {
  id: string
  tokenCount: number
  files?: Array<{ path: string; tokenCount: number; lineCount: number }>
  error?: string
}

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

const countLines = (text: string): number => {
  const normalized = String(text || "")
  if (normalized.length === 0) return 0
  return normalized.split("\n").length
}

self.onmessage = (event: MessageEvent<TokenizeRequest>) => {
  const request = event.data
  try {
    if (request.type === "single") {
      const payload: TokenizeResponse = {
        id: request.id,
        tokenCount: estimateTokens(request.text ?? "")
      }
      self.postMessage(payload)
      return
    }

    const files = request.files ?? []
    const total = files.length
    const results: Array<{
      path: string
      tokenCount: number
      lineCount: number
    }> = []
    let tokenCount = 0

    for (let index = 0; index < files.length; index++) {
      const item = files[index]
      const fileTokens = estimateTokens(item.content)
      tokenCount += fileTokens
      results.push({
        path: item.path,
        tokenCount: fileTokens,
        lineCount: countLines(item.content)
      })

      const progressPayload: ProgressResponse = {
        id: request.id,
        progress: total === 0 ? 100 : Math.round(((index + 1) / total) * 100),
        current: index + 1,
        total
      }
      self.postMessage(progressPayload)
    }

    const payload: TokenizeResponse = {
      id: request.id,
      tokenCount,
      files: results
    }
    self.postMessage(payload)
  } catch (error) {
    const payload: TokenizeResponse = {
      id: request.id,
      tokenCount: 0,
      error: error instanceof Error ? error.message : "Tokenizer worker failed"
    }
    self.postMessage(payload)
  }
}

export {}
