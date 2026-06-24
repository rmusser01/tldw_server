import type { NextApiRequest, NextApiResponse } from "next"

type RuntimeConfigResponse = {
  runtimeAuth: {
    available: boolean
    authMode?: "single-user"
    apiKey?: string
    reason?: string
  }
  networking: {
    deploymentMode: string
    serverUrl: string
  }
}

const FORWARDED_HEADER_NAMES = [
  "forwarded",
  "x-forwarded-for",
  "x-forwarded-host",
  "x-real-ip"
]

const PLACEHOLDER_KEYS = new Set([
  "change-me",
  "changeme",
  "change_me",
  "your-api-key",
  "your_api_key",
  "placeholder",
  "replace-me",
  "replace_me"
])

const normalizeEnvValue = (value?: string): string => String(value || "").trim()

const isSingleUserMode = (value?: string): boolean => value === "single_user"

const isEnabled = (value?: string): boolean => value === "1"

const isUsableApiKey = (value?: string): value is string => {
  if (!value) return false
  if (/\s/.test(value)) return false
  const normalized = value.toLowerCase()
  if (normalized.startsWith("change_me")) return false
  return !PLACEHOLDER_KEYS.has(normalized)
}

const extractHostname = (hostHeader?: string | string[]): string => {
  const host = Array.isArray(hostHeader) ? hostHeader[0] : hostHeader
  const normalized = normalizeEnvValue(host).toLowerCase()
  if (!normalized) return ""
  if (normalized.startsWith("[") && normalized.includes("]")) {
    return normalized.slice(1, normalized.indexOf("]"))
  }
  if (normalized === "::1") return normalized
  const colonCount = (normalized.match(/:/g) || []).length
  if (colonCount > 1) return normalized
  return normalized.split(":")[0] || ""
}

const isLoopbackHost = (hostHeader?: string | string[]): boolean => {
  const hostname = extractHostname(hostHeader)
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1"
}

const hasForwardingHeaders = (req: NextApiRequest): boolean =>
  FORWARDED_HEADER_NAMES.some((name) =>
    Object.prototype.hasOwnProperty.call(req.headers, name)
  )

const unavailable = (reason: string): RuntimeConfigResponse => ({
  runtimeAuth: {
    available: false,
    reason
  },
  networking: {
    deploymentMode: normalizeEnvValue(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE) || "quickstart",
    serverUrl: ""
  }
})

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<RuntimeConfigResponse | { error: string }>
) {
  res.setHeader("Cache-Control", "no-store, max-age=0")

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET")
    res.status(405).json({ error: "Method not allowed" })
    return
  }

  if (!isSingleUserMode(process.env.AUTH_MODE)) {
    res.status(200).json(unavailable("auth-mode"))
    return
  }

  if (!isEnabled(process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH)) {
    res.status(200).json(unavailable("disabled"))
    return
  }

  if (!isLoopbackHost(req.headers.host)) {
    res.status(200).json(unavailable("host"))
    return
  }

  if (hasForwardingHeaders(req)) {
    res.status(200).json(unavailable("forwarded"))
    return
  }

  const apiKey = process.env.SINGLE_USER_API_KEY
  if (!isUsableApiKey(apiKey)) {
    res.status(200).json(unavailable("api-key"))
    return
  }

  res.status(200).json({
    runtimeAuth: {
      available: true,
      authMode: "single-user",
      apiKey
    },
    networking: {
      deploymentMode: normalizeEnvValue(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE) || "quickstart",
      serverUrl: ""
    }
  })
}
