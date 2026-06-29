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

const UNTRUSTED_FORWARDING_HEADER_NAMES = ["x-real-ip"]

const PLACEHOLDER_KEYS = new Set([
  "change-me",
  "changeme",
  "change_me",
  "default",
  "test-key",
  "your-api-key",
  "your-api-key-here",
  "your_api_key",
  "your_api_key_here",
  "placeholder",
  "replace-me",
  "replace_me"
])

const MIN_API_KEY_LENGTH = 16

const LOOPBACK_PEER_ADDRESSES = new Set([
  "127.0.0.1",
  "::1",
  "::ffff:127.0.0.1"
])

const normalizeEnvValue = (value?: string): string => String(value || "").trim()

const getDeploymentMode = (): string =>
  normalizeEnvValue(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE) || "quickstart"

const isSingleUserMode = (value?: string): boolean => value === "single_user"

const isEnabled = (value?: string): boolean => value === "1"

const isUsableApiKey = (value?: string): value is string => {
  if (!value) return false
  if (/\s/.test(value)) return false
  if (value.length < MIN_API_KEY_LENGTH) return false
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

const isLoopbackPeerAddress = (remoteAddress?: string): boolean =>
  LOOPBACK_PEER_ADDRESSES.has(normalizeEnvValue(remoteAddress).toLowerCase())

const normalizeForwardedValue = (value?: string): string => {
  const normalized = normalizeEnvValue(value)
  if (normalized.length >= 2 && normalized.startsWith("\"") && normalized.endsWith("\"")) {
    return normalized.slice(1, -1)
  }
  return normalized
}

const isLoopbackForwardedFor = (value?: string): boolean => {
  const normalized = normalizeForwardedValue(value)
  if (!normalized || normalized.toLowerCase() === "unknown") return false
  return isLoopbackHost(normalized) || isLoopbackPeerAddress(normalized)
}

const parseForwardedEntry = (entry: string): Record<string, string> => {
  const params: Record<string, string> = {}
  for (const part of entry.split(";")) {
    const [rawKey, ...rawValueParts] = part.split("=")
    const key = normalizeEnvValue(rawKey).toLowerCase()
    const value = normalizeForwardedValue(rawValueParts.join("="))
    if (key && value) {
      params[key] = value
    }
  }
  return params
}

const isTrustedForwardedHeader = (headerValue?: string | string[]): boolean => {
  const values = Array.isArray(headerValue) ? headerValue : [headerValue]
  return values.every((value) => {
    const normalized = normalizeEnvValue(value)
    if (!normalized) return false

    return normalized.split(",").every((entry) => {
      const params = parseForwardedEntry(entry)
      return isLoopbackForwardedFor(params.for)
    })
  })
}

const isTrustedForwardedForHeader = (headerValue?: string | string[]): boolean => {
  const values = Array.isArray(headerValue) ? headerValue : [headerValue]
  return values.every((value) => {
    const normalized = normalizeEnvValue(value)
    if (!normalized) return false
    return normalized.split(",").every((entry) => isLoopbackForwardedFor(entry))
  })
}

const extractIPv4Address = (remoteAddress?: string): number[] | null => {
  const normalized = normalizeEnvValue(remoteAddress).toLowerCase()
  const address = normalized.startsWith("::ffff:")
    ? normalized.slice("::ffff:".length)
    : normalized
  const octets = address.split(".")

  if (octets.length !== 4) return null

  const parts = octets.map((octet) => {
    if (!/^\d+$/.test(octet)) return Number.NaN
    const value = Number(octet)
    return value >= 0 && value <= 255 ? value : Number.NaN
  })

  return parts.every((part) => Number.isInteger(part)) ? parts : null
}

const isDockerGatewayPeerAddress = (remoteAddress?: string): boolean => {
  const parts = extractIPv4Address(remoteAddress)
  if (!parts) return false

  const [first, second, third, fourth] = parts
  const isLinuxBridgeGateway =
    first === 172 && second >= 16 && second <= 31 && third === 0 && fourth === 1
  const isDockerDesktopGateway =
    first === 192 && second === 168 && third === 65 && fourth === 1

  return isLinuxBridgeGateway || isDockerDesktopGateway
}

const isTrustedLocalPeerAddress = (
  remoteAddress: string | undefined,
  deploymentMode: string,
  runtimeAuthEnabled: boolean
): boolean => {
  if (isLoopbackPeerAddress(remoteAddress)) return true
  return runtimeAuthEnabled && deploymentMode === "quickstart" && isDockerGatewayPeerAddress(remoteAddress)
}

const hasUntrustedForwardingHeaders = (req: NextApiRequest): boolean => {
  if (
    UNTRUSTED_FORWARDING_HEADER_NAMES.some((name) =>
      Object.prototype.hasOwnProperty.call(req.headers, name)
    )
  ) {
    return true
  }

  if (Object.prototype.hasOwnProperty.call(req.headers, "x-forwarded-for")) {
    return !isTrustedForwardedForHeader(req.headers["x-forwarded-for"])
  }

  if (Object.prototype.hasOwnProperty.call(req.headers, "forwarded")) {
    return !isTrustedForwardedHeader(req.headers.forwarded)
  }

  return false
}

const unavailable = (reason: string): RuntimeConfigResponse => ({
  runtimeAuth: {
    available: false,
    reason
  },
  networking: {
    deploymentMode: getDeploymentMode(),
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

  const runtimeAuthEnabled = isEnabled(process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH)
  if (!runtimeAuthEnabled) {
    res.status(200).json(unavailable("disabled"))
    return
  }

  const deploymentMode = getDeploymentMode()

  if (deploymentMode !== "quickstart") {
    res.status(200).json(unavailable("deployment-mode"))
    return
  }

  if (!isLoopbackHost(req.headers.host)) {
    res.status(200).json(unavailable("host"))
    return
  }

  if (!isTrustedLocalPeerAddress(req.socket?.remoteAddress, deploymentMode, runtimeAuthEnabled)) {
    res.status(200).json(unavailable("peer"))
    return
  }

  if (hasUntrustedForwardingHeaders(req)) {
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
      deploymentMode,
      serverUrl: ""
    }
  })
}
