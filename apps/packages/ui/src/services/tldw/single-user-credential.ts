import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { isExactOriginCookieSessionConfig } from "@/services/tldw/browser-networking"

export type ApiKeyPersistence = "device" | "session"
export type CredentialSource = "manual" | "cookie-session"

export interface ManualCredentialMetadata {
  credentialSource: "manual"
  apiKeyPersistence: ApiKeyPersistence
  apiKeyServerOrigin: string
}

export interface ManualSessionCredential extends ManualCredentialMetadata {
  apiKeyPersistence: "session"
  apiKey: string
}

export interface CredentialStorage {
  get<T>(key: string): Promise<T | undefined | null>
  set<T>(key: string, value: T): Promise<void>
  remove(key: string): Promise<void>
}

export const MANUAL_SESSION_KEY = "tldwManualSessionApiKey"

export const normalizeServerOrigin = (value: string): string | null => {
  try {
    const url = new URL(String(value || "").trim())
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.origin
      : null
  } catch {
    return null
  }
}

const nonEmptySecret = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.trim()
  return normalized || null
}

export const isCompleteDeviceCredential = (
  config: TldwConfig | null | undefined
): config is TldwConfig &
  ManualCredentialMetadata & { apiKeyPersistence: "device"; apiKey: string } => {
  if (
    config?.authMode !== "single-user" ||
    config.credentialSource !== "manual" ||
    config.apiKeyPersistence !== "device" ||
    !nonEmptySecret(config.apiKey)
  ) {
    return false
  }
  const origin = normalizeServerOrigin(config.serverUrl)
  return Boolean(origin && origin === config.apiKeyServerOrigin)
}

const isCompleteSessionRecord = (
  value: unknown,
  expectedOrigin: string
): value is ManualSessionCredential => {
  if (!value || typeof value !== "object") return false
  const record = value as Partial<ManualSessionCredential>
  return (
    record.credentialSource === "manual" &&
    record.apiKeyPersistence === "session" &&
    record.apiKeyServerOrigin === expectedOrigin &&
    Boolean(nonEmptySecret(record.apiKey))
  )
}

export const resolveManualCredential = async (
  config: TldwConfig | null | undefined,
  stores: { session: CredentialStorage }
): Promise<string | null> => {
  if (isCompleteDeviceCredential(config)) {
    return nonEmptySecret(config.apiKey)
  }
  if (
    config?.authMode !== "single-user" ||
    config.credentialSource !== "manual" ||
    config.apiKeyPersistence !== "session"
  ) {
    return null
  }
  const origin = normalizeServerOrigin(config.serverUrl)
  if (!origin || origin !== config.apiKeyServerOrigin) return null

  const record = await stores.session
    .get<ManualSessionCredential>(MANUAL_SESSION_KEY)
    .catch(() => null)
  return isCompleteSessionRecord(record, origin)
    ? nonEmptySecret(record.apiKey)
    : null
}

export const resolveEffectiveTldwConfig = async (
  stores: {
    persistent: CredentialStorage
    session: CredentialStorage
  },
  cookie?: {
    cookieSession?: TldwConfig | null
    expectedCookieOrigin?: string | null
  }
): Promise<TldwConfig | null> => {
  const stored = await stores.persistent
    .get<TldwConfig>("tldwConfig")
    .catch(() => null)
  const cookieSession = cookie?.cookieSession

  if (
    isExactOriginCookieSessionConfig(
      cookieSession,
      cookie?.expectedCookieOrigin
    ) &&
    cookieSession
  ) {
    const {
      apiKey: _apiKey,
      apiBearer: _apiBearer,
      accessToken: _accessToken,
      refreshToken: _refreshToken,
      credentialSource: _credentialSource,
      apiKeyPersistence: _apiKeyPersistence,
      apiKeyServerOrigin: _apiKeyServerOrigin,
      ...safe
    } = cookieSession
    return safe
  }

  if (!stored || typeof stored !== "object") return null

  const effective = { ...stored }
  const apiKey = await resolveManualCredential(effective, {
    session: stores.session
  })
  if (apiKey) {
    effective.apiKey = apiKey
  } else if (!isCompleteDeviceCredential(effective)) {
    delete effective.apiKey
  }
  return effective
}

export const toPersistedTldwConfig = (config: TldwConfig): TldwConfig => {
  if (isCompleteDeviceCredential(config)) return { ...config }
  const { apiKey: _apiKey, ...safe } = config
  return safe
}

export const clearManualCredentials = async (
  persistent: CredentialStorage,
  session: CredentialStorage
): Promise<void> => {
  let stored: TldwConfig | null | undefined
  const failures: unknown[] = []
  try {
    stored = await persistent.get<TldwConfig>("tldwConfig")
  } catch (error) {
    failures.push(error)
  }
  try {
    await session.remove(MANUAL_SESSION_KEY)
  } catch (error) {
    failures.push(error)
  }

  if (stored) {
    const {
      apiKey: _apiKey,
      credentialSource: _credentialSource,
      apiKeyPersistence: _apiKeyPersistence,
      apiKeyServerOrigin: _apiKeyServerOrigin,
      ...connection
    } = stored
    try {
      await persistent.set("tldwConfig", connection as TldwConfig)
    } catch (error) {
      failures.push(error)
    }
  }
  if (failures.length) throw failures[0]
}
