/**
 * Shared loopback mock tldw_server for extension sidepanel e2e specs.
 *
 * Serves the common route scaffolding the sidepanel chat needs before it can
 * stream: CORS preflight, `/api/v1/health`, `/api/v1/llm/models[/metadata]`,
 * `/openapi.json`, and a JSON 404 catch-all. The one route whose payload
 * genuinely differs between specs — `POST /api/v1/chat/completions` (paced
 * canned SSE vs. the smoke suite's stream/non-stream reply) — is supplied by
 * each consumer via `handleChatCompletions`.
 *
 * Lifecycle (extracted verbatim from the former per-spec implementations):
 * ephemeral port on 127.0.0.1, socket-error tolerance after the browser
 * context closes, and a stop path of `close()` + `closeAllConnections()` with
 * a 1s unref'd fallback.
 *
 * Extracted from performance-streaming.spec.ts and sidepanel-chat-smoke.spec.ts
 * (TASK-13521, WebUI perf batch W1 Task 0).
 */

import http from "node:http"
import { AddressInfo } from "node:net"

/**
 * Context handed to the per-spec `POST /api/v1/chat/completions` handler.
 * The request body has already been fully read by the time it runs.
 */
export interface SidepanelMockChatRequest {
  req: http.IncomingMessage
  res: http.ServerResponse
  /** Fully-read request body (empty string when the request had none). */
  body: string
  /** Sends a JSON response with the mock's CORS headers. */
  sendJson: (code: number, payload: unknown) => void
}

export interface SidepanelMockServerOptions {
  /** Model id served by `/api/v1/llm/models` and `/api/v1/llm/models/metadata`. */
  modelId: string
  /** Display name in the `/api/v1/llm/models/metadata` payload. */
  modelName: string
  /**
   * Handles `POST /api/v1/chat/completions`. Owns the response end-to-end:
   * either `sendJson(...)` for non-streaming replies or write SSE frames and
   * `res.end()` for streaming ones. The return value is ignored, so handlers
   * may `return res.end()` / `return sendJson(...)` for control flow.
   */
  handleChatCompletions: (request: SidepanelMockChatRequest) => void | Promise<unknown>
}

export interface SidepanelMockServer {
  server: http.Server
  baseUrl: string
}

const readBody = (req: http.IncomingMessage) =>
  new Promise<string>((resolve) => {
    let body = ""
    req.on("data", (chunk) => {
      body += chunk
    })
    req.on("end", () => resolve(body))
  })

export const startSidepanelMockServer = async (
  options: SidepanelMockServerOptions
): Promise<SidepanelMockServer> => {
  const server = http.createServer(async (req, res) => {
    const method = (req.method || "GET").toUpperCase()
    const url = req.url || "/"

    const sendJson = (code: number, payload: unknown) => {
      res.writeHead(code, {
        "content-type": "application/json",
        "access-control-allow-origin": "http://127.0.0.1",
        "access-control-allow-credentials": "true"
      })
      res.end(JSON.stringify(payload))
    }

    if (method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "http://127.0.0.1",
        "access-control-allow-credentials": "true",
        "access-control-allow-headers": "content-type, x-api-key, authorization"
      })
      return res.end()
    }

    if (url === "/api/v1/health" && method === "GET") {
      return sendJson(200, { status: "ok" })
    }

    if (url === "/api/v1/llm/models/metadata" && method === "GET") {
      return sendJson(200, [
        {
          id: options.modelId,
          name: options.modelName,
          provider: "mock",
          context_length: 4096,
          capabilities: ["chat"]
        }
      ])
    }

    if (url === "/api/v1/llm/models" && method === "GET") {
      return sendJson(200, [options.modelId])
    }

    if (url === "/openapi.json" && method === "GET") {
      return sendJson(200, {
        openapi: "3.0.0",
        info: { version: "mock" },
        paths: {
          "/api/v1/health": {},
          "/api/v1/chat/completions": {},
          "/api/v1/llm/models": {},
          "/api/v1/llm/models/metadata": {}
        }
      })
    }

    if (url === "/api/v1/chat/completions" && method === "POST") {
      const body = await readBody(req)
      return options.handleChatCompletions({ req, res, body, sendJson })
    }

    return sendJson(404, { detail: "not found" })
  })
  server.on("error", () => {
    // Ignore socket errors after the test closes the browser context.
  })

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const addr = server.address() as AddressInfo
  return { server, baseUrl: `http://127.0.0.1:${addr.port}` }
}

export const stopSidepanelMockServer = async (server: http.Server) => {
  await new Promise<void>((resolve) => {
    let settled = false
    const done = () => {
      if (settled) return
      settled = true
      resolve()
    }

    server.close(done)
    server.closeAllConnections?.()
    const fallback = setTimeout(done, 1000)
    fallback.unref?.()
  })
}
