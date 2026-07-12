import { createServer, type Server } from "node:http"

export const MANUAL_API_KEY = "manual-persistence-e2e-key"

export type ManualApiKeyRequest = {
  method: string
  path: string
  authenticated: boolean
}

export type ManualApiKeyFixture = {
  url: string
  requests: () => ManualApiKeyRequest[]
  close: () => Promise<void>
}

export const startManualApiKeyFixture = async (
  port: number
): Promise<ManualApiKeyFixture> => {
  const requests: ManualApiKeyRequest[] = []
  const server: Server = createServer((request, response) => {
    const origin = String(request.headers.origin || "*")
    response.setHeader("Access-Control-Allow-Origin", origin)
    response.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS")
    response.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, X-API-KEY, Authorization"
    )
    response.setHeader("Vary", "Origin")

    if (request.method === "OPTIONS") {
      response.writeHead(204)
      response.end()
      return
    }

    const pathname = new URL(request.url || "/", `http://127.0.0.1:${port}`)
      .pathname
    const authenticated = request.headers["x-api-key"] === MANUAL_API_KEY
    requests.push({
      method: request.method || "GET",
      path: pathname,
      authenticated
    })

    if (!authenticated) {
      response.writeHead(401, { "Content-Type": "application/json" })
      response.end(JSON.stringify({ detail: "Invalid API key" }))
      return
    }

    const body =
      pathname === "/api/v1/users/me/profile"
        ? { id: 1, username: "manual-persistence-e2e" }
        : pathname === "/api/v1/health"
          ? { status: "ok" }
          : pathname === "/api/v1/rag/health"
            ? { status: "healthy" }
            : {}

    response.writeHead(200, { "Content-Type": "application/json" })
    response.end(JSON.stringify(body))
  })

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(port, "127.0.0.1", () => resolve())
  })

  return {
    url: `http://127.0.0.1:${port}`,
    requests: () => requests.map((request) => ({ ...request })),
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()))
      })
  }
}
