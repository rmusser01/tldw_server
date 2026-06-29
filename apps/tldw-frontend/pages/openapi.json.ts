import type { NextApiRequest, NextApiResponse } from "next"

const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8000"

const normalizeBackendOrigin = (value?: string): string | null => {
  const raw = String(value || "").trim()
  if (!raw) return null
  try {
    const url = new URL(raw)
    if (url.protocol !== "http:" && url.protocol !== "https:") return null
    return url.origin
  } catch {
    return null
  }
}

const resolveBackendOrigin = (): string =>
  normalizeBackendOrigin(process.env.TLDW_INTERNAL_API_ORIGIN)
  || normalizeBackendOrigin(process.env.TLDW_SERVER_URL)
  || normalizeBackendOrigin(process.env.NEXT_PUBLIC_API_URL)
  || DEFAULT_BACKEND_ORIGIN

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  res.setHeader("Cache-Control", "no-store, max-age=0")

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET")
    res.status(405).json({ error: "Method not allowed" })
    return
  }

  const openApiUrl = `${resolveBackendOrigin()}/openapi.json`

  try {
    const upstream = await fetch(openApiUrl, {
      method: "GET",
      headers: {
        accept: "application/json"
      }
    })
    const contentType = upstream.headers.get("content-type") || "application/json"
    res.setHeader("Content-Type", contentType)
    res.status(upstream.status)

    if (contentType.toLowerCase().includes("application/json")) {
      res.json(await upstream.json())
      return
    }

    res.send(await upstream.text())
  } catch {
    res.status(502).json({ error: "Failed to fetch backend OpenAPI document" })
  }
}
