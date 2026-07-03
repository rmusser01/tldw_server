import type { FirstRunMetadata, FirstRunState } from "@/types/setup-onboarding"

export type ApiSetupUrlResolution = {
  href: string
  source: "metadata" | "configured_server"
}

export const mutableWebUiSetupStatuses = new Set<FirstRunState["status"]>([
  "not_started",
  "in_progress",
  "first_chat_complete",
])

export const setupEntryChoiceStatuses = new Set<FirstRunState["status"]>([
  ...mutableWebUiSetupStatuses,
  "blocked",
])

export const isBlockedSetupState = (state: FirstRunState | null): boolean =>
  state?.status === "blocked"

export const isMutableWebUiSetupState = (
  state: FirstRunState | null
): boolean => Boolean(state && mutableWebUiSetupStatuses.has(state.status))

export const shouldShowSetupEntryChoice = (
  state: FirstRunState | null,
  metadata: FirstRunMetadata | null
): boolean => {
  if (!state || !metadata) {
    return false
  }

  return (
    metadata.setup_required === true &&
    metadata.setup_completed === false &&
    setupEntryChoiceStatuses.has(state.status)
  )
}

export function resolveApiSetupUrl(input: {
  metadata: FirstRunMetadata | null
  configuredServerUrl?: string | null
  currentOrigin?: string | null
}): ApiSetupUrlResolution | null {
  const currentOriginUrl = parseUrl(input.currentOrigin)
  const frontendOriginUrl = parseUrl(input.metadata?.connection.frontend_origin)
  const candidates: Array<{
    value: string | null | undefined
    source: ApiSetupUrlResolution["source"]
  }> = [
    {
      value: input.metadata?.connection.api_origin,
      source: "metadata",
    },
    {
      value: input.configuredServerUrl,
      source: "configured_server",
    },
  ]

  for (const candidate of candidates) {
    const url = parseUrl(candidate.value)

    if (!url || !isHttpUrl(url)) {
      continue
    }

    if (
      url.origin === currentOriginUrl?.origin ||
      url.origin === frontendOriginUrl?.origin
    ) {
      continue
    }

    if (!isBrowserOpenableApiOrigin(url, currentOriginUrl)) {
      continue
    }

    url.pathname = "/setup"
    url.search = ""
    url.hash = ""

    return {
      href: url.href,
      source: candidate.source,
    }
  }

  return null
}

const parseUrl = (value: string | null | undefined): URL | null => {
  if (!value) {
    return null
  }

  try {
    return new URL(value)
  } catch {
    return null
  }
}

const isHttpUrl = (url: URL): boolean =>
  url.protocol === "http:" || url.protocol === "https:"

const isBrowserOpenableApiOrigin = (
  url: URL,
  currentOriginUrl: URL | null
): boolean => {
  const hostname = normalizeHostname(url.hostname)
  const currentHostname = currentOriginUrl
    ? normalizeHostname(currentOriginUrl.hostname)
    : null
  const ipv4 = parseIpv4(hostname)

  if (isLoopbackHostname(hostname, ipv4)) {
    return true
  }

  if (
    currentOriginUrl &&
    currentHostname === hostname &&
    currentOriginUrl.port !== url.port
  ) {
    return true
  }

  if (ipv4) {
    return isPrivateIpv4(ipv4)
  }

  return hostname.includes(".")
}

const normalizeHostname = (hostname: string): string =>
  hostname.toLowerCase().replace(/^\[|\]$/g, "")

const parseIpv4 = (hostname: string): [number, number, number, number] | null => {
  const parts = hostname.split(".")

  if (parts.length !== 4) {
    return null
  }

  const octets = parts.map((part) => {
    if (!/^\d+$/.test(part)) {
      return Number.NaN
    }

    return Number(part)
  })

  if (
    octets.some(
      (octet) => !Number.isInteger(octet) || octet < 0 || octet > 255
    )
  ) {
    return null
  }

  return octets as [number, number, number, number]
}

const isLoopbackHostname = (
  hostname: string,
  ipv4: [number, number, number, number] | null
): boolean =>
  hostname === "localhost" ||
  hostname === "::1" ||
  hostname === "127.0.0.1" ||
  ipv4?.[0] === 127

const isPrivateIpv4 = (ipv4: [number, number, number, number]): boolean =>
  ipv4[0] === 10 ||
  (ipv4[0] === 172 && ipv4[1] >= 16 && ipv4[1] <= 31) ||
  (ipv4[0] === 192 && ipv4[1] === 168)
