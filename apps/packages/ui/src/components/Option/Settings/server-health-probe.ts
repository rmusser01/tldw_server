type ProbeServerHealthParams = {
  serverUrl?: string
  authMode?: "single-user" | "multi-user"
  apiKey?: string
  timeoutMs?: number
  fetchFn?: typeof fetch
}

type ProbeServerHealthResult = {
  ok: boolean
  status: number
  error?: string
}

const AUTHENTICATED_READINESS_PATH = "/api/v1/users/storage"

const parseErrorBody = async (response: Response): Promise<string | undefined> => {
  try {
    const contentType = response.headers.get("content-type") || ""
    if (contentType.includes("application/json")) {
      const json = await response.json()
      const jsonError = (json as { error?: unknown; detail?: unknown } | null)?.error
      const jsonDetail = (json as { error?: unknown; detail?: unknown } | null)?.detail
      if (typeof jsonError === "string" && jsonError.trim()) return jsonError
      if (typeof jsonDetail === "string" && jsonDetail.trim()) return jsonDetail
    }
    const text = await response.text()
    if (text.trim()) return text.trim()
  } catch {
    // best effort; fall back to status text
  }
  return response.statusText || undefined
}

export const probeServerHealth = async (
  params: ProbeServerHealthParams
): Promise<ProbeServerHealthResult> => {
  const baseUrl = String(params.serverUrl || "").trim().replace(/\/$/, "")
  if (!baseUrl) {
    return {
      ok: false,
      status: 400,
      error: "tldw server not configured"
    }
  }

  const endpoint = `${baseUrl}/api/v1/health`
  const headers: Record<string, string> = {}
  const key = String(params.apiKey || "").trim()
  if (params.authMode === "single-user" && key.length > 0) {
    headers["X-API-KEY"] = key
  }

  const controller = typeof AbortController !== "undefined" ? new AbortController() : null
  const timeoutMs = Number(params.timeoutMs) > 0 ? Number(params.timeoutMs) : 10000
  const timeoutHandle =
    controller && typeof setTimeout === "function"
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null

  try {
    const response = await (params.fetchFn || fetch)(endpoint, {
      method: "GET",
      headers: Object.keys(headers).length ? headers : undefined,
      signal: controller?.signal
    })
    if (response.ok) {
      if (params.authMode === "single-user" && key.length > 0) {
        const authResponse = await (params.fetchFn || fetch)(
          `${baseUrl}${AUTHENTICATED_READINESS_PATH}`,
          {
            method: "GET",
            headers,
            signal: controller?.signal
          }
        )
        if (!authResponse.ok) {
          return {
            ok: false,
            status: authResponse.status,
            error: await parseErrorBody(authResponse)
          }
        }
      }
      return { ok: true, status: response.status }
    }
    return {
      ok: false,
      status: response.status,
      error: await parseErrorBody(response)
    }
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : String(error || "Request failed")
    const isAbort =
      (typeof DOMException !== "undefined" && error instanceof DOMException)
        ? error.name === "AbortError"
        : typeof message === "string" && /abort/i.test(message)
    return {
      ok: false,
      status: 0,
      error: isAbort ? "Request timed out" : message
    }
  } finally {
    if (timeoutHandle) clearTimeout(timeoutHandle)
  }
}

export type { ProbeServerHealthParams, ProbeServerHealthResult }
