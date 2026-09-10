import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import {
  deriveScopedUserId,
  deriveServerFingerprint
} from "@/utils/media-navigation-scope"

export type ChatSurfaceScopeInput = {
  serverUrl: string | null
  authMode: string | null
  orgId: string | number | null
  userId: string | number | null
  accessToken?: string | null
  apiKey?: string | null
}

const normalizeAuthMode = (authMode: string | null | undefined): string => {
  const normalized = String(authMode || "").trim().toLowerCase()
  return normalized || "unknown"
}

const normalizeOrgScope = (orgId: string | number | null | undefined): string => {
  if (orgId === null || typeof orgId === "undefined") {
    return "org:none"
  }
  const normalized = String(orgId).trim()
  return normalized ? `org:${normalized}` : "org:none"
}

const sha256CredentialScope = (kind: "key" | "token", value: string): string => {
  const digest = sha256(
    utf8ToBytes(`tldw:chat-surface-${kind}:v1\0${value}`)
  )
  return `${kind}:sha256:${bytesToHex(digest)}`
}

export const deriveSingleUserApiKeyCredentialScope = (
  authMode: string | null | undefined,
  apiKey: string | null | undefined
): string | null => {
  if (normalizeAuthMode(authMode) !== "single-user") {
    return null
  }

  const normalizedKey = String(apiKey || "").trim()
  if (!normalizedKey) return "key:none"

  return sha256CredentialScope("key", normalizedKey)
}

const deriveMultiUserTokenCredentialScope = (
  authMode: string | null | undefined,
  accessToken: string | null | undefined
): string | null => {
  if (normalizeAuthMode(authMode) !== "multi-user") return null
  const normalizedToken = String(accessToken || "").trim()
  return normalizedToken
    ? sha256CredentialScope("token", normalizedToken)
    : "token:none"
}

export const buildChatSurfaceScopeKey = (
  input: ChatSurfaceScopeInput
): string => {
  const serverFingerprint = deriveServerFingerprint(input.serverUrl)
  const authScope = `auth:${normalizeAuthMode(input.authMode)}`
  const orgScope = normalizeOrgScope(input.orgId)
  const userScope = deriveScopedUserId({
    userId: input.userId,
    authMode: input.authMode,
    accessToken: input.accessToken ?? null
  })
  const credentialScope =
    deriveSingleUserApiKeyCredentialScope(input.authMode, input.apiKey ?? null) ??
    deriveMultiUserTokenCredentialScope(
      input.authMode,
      input.accessToken ?? null
    )

  return credentialScope
    ? `${serverFingerprint}:${authScope}:${orgScope}:${userScope}:${credentialScope}`
    : `${serverFingerprint}:${authScope}:${orgScope}:${userScope}`
}

export const buildChatSurfaceScopeKeyFromConfig = (
  config: Pick<TldwConfig, "serverUrl" | "authMode" | "orgId" | "accessToken" | "apiKey"> | null | undefined,
  options?: {
    userId?: string | number | null
  }
): string =>
  buildChatSurfaceScopeKey({
    serverUrl: config?.serverUrl ?? null,
    authMode: config?.authMode ?? null,
    orgId: config?.orgId ?? null,
    userId: options?.userId ?? null,
    accessToken: config?.accessToken ?? null,
    apiKey: config?.apiKey ?? null
  })
