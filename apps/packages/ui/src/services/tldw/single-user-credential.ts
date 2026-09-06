import type {
  ServicePromptTargetConfig,
  TldwConfig
} from "@/services/tldw/TldwApiClient"
import { isExactOriginCookieSessionConfig } from "@/services/tldw/browser-networking"
import { servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"

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
export const REFRESH_ROTATION_KEY = "tldwRefreshRotation"

/**
 * A possibly-untrusted request-scope target (e.g. parsed from a runtime
 * message). Fields are `unknown` on purpose: the helpers below only ever
 * compare them against the stored config via `servicePromptTargetsMatch`
 * and fail safe on any mismatch, so no field type may be assumed.
 */
type ServicePromptTargetLock = Readonly<{
  serverUrl?: unknown
  authMode?: unknown
  authSource?: unknown
  orgId?: unknown
}>

type RefreshRotationRecord = Readonly<{
  version: 1
  serverUrl?: string
  authMode: "multi-user"
  authSource?: TldwConfig["authSource"]
  orgId?: number
  sourceAccessToken: string
  sourceRefreshToken: string
  accessToken: string
  refreshToken: string
}>

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

const asRefreshRotationRecord = (
  value: unknown
): RefreshRotationRecord | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const record = value as Partial<RefreshRotationRecord>
  const sourceAccessToken = nonEmptySecret(record.sourceAccessToken)
  const sourceRefreshToken = nonEmptySecret(record.sourceRefreshToken)
  const accessToken = nonEmptySecret(record.accessToken)
  const refreshToken = nonEmptySecret(record.refreshToken)
  const serverUrl = nonEmptySecret(record.serverUrl)
  if (
    record.version !== 1 ||
    record.authMode !== "multi-user" ||
    (record.serverUrl != null && !serverUrl) ||
    !sourceAccessToken ||
    !sourceRefreshToken ||
    !accessToken ||
    !refreshToken
  ) {
    return null
  }
  return {
    version: 1,
    ...(serverUrl ? { serverUrl } : {}),
    authMode: record.authMode,
    authSource: record.authSource,
    orgId: record.orgId,
    sourceAccessToken,
    sourceRefreshToken,
    accessToken,
    refreshToken
  }
}

const applicableRefreshRotation = (
  stored: TldwConfig,
  value: unknown
): RefreshRotationRecord | null => {
  const record = asRefreshRotationRecord(value)
  if (
    !record ||
    stored.authMode !== "multi-user" ||
    !servicePromptTargetsMatch(stored, record) ||
    nonEmptySecret(stored.accessToken) !== record.sourceAccessToken ||
    nonEmptySecret(stored.refreshToken) !== record.sourceRefreshToken
  ) {
    return null
  }
  return record
}

const applyRefreshRotation = (
  stored: TldwConfig,
  value: unknown
): TldwConfig => {
  const record = applicableRefreshRotation(stored, value)
  return record
    ? {
        ...stored,
        accessToken: record.accessToken,
        refreshToken: record.refreshToken
      }
    : stored
}

export const hasNewerCurrentRefreshRotation = async (
  persistent: CredentialStorage,
  checked: ServicePromptTargetConfig,
  capturedAccessToken: string
): Promise<boolean> => {
  const captured = nonEmptySecret(capturedAccessToken)
  if (!captured) return false
  const stored = await persistent.get<TldwConfig>("tldwConfig")
  if (
    !stored ||
    typeof stored !== "object" ||
    !servicePromptTargetsMatch(stored, checked)
  ) {
    return false
  }
  const record = applicableRefreshRotation(
    stored,
    await persistent.get<unknown>(REFRESH_ROTATION_KEY)
  )
  return Boolean(record && record.accessToken !== captured)
}

export const hasNewerCurrentAccessToken = async (
  persistent: CredentialStorage,
  checked: ServicePromptTargetLock,
  capturedAccessToken: string
): Promise<boolean> => {
  const captured = nonEmptySecret(capturedAccessToken)
  if (!captured) return false
  const stored = await persistent.get<TldwConfig>("tldwConfig")
  if (
    !stored ||
    typeof stored !== "object" ||
    !servicePromptTargetsMatch(stored, checked)
  ) {
    return false
  }
  const record = applicableRefreshRotation(
    stored,
    await persistent.get<unknown>(REFRESH_ROTATION_KEY)
  )
  if (record && record.accessToken !== captured) {
    return true
  }
  const current = nonEmptySecret(stored.accessToken)
  if (!current || current === captured) return false
  const unknownPrincipal = deriveScopedUserId({
    userId: null,
    authMode: "multi-user",
    accessToken: null
  })
  const capturedPrincipal = deriveScopedUserId({
    userId: null,
    authMode: "multi-user",
    accessToken: captured
  })
  return capturedPrincipal !== unknownPrincipal &&
    deriveScopedUserId({
      userId: null,
      authMode: "multi-user",
      accessToken: current
    }) === capturedPrincipal
}

export const waitForNewerCurrentAccessToken = async (
  persistent: CredentialStorage,
  checked: ServicePromptTargetLock,
  capturedAccessToken: string,
  options: Readonly<{
    timeoutMs?: number
    pollIntervalMs?: number
  }> = {}
): Promise<boolean> => {
  const timeoutMs = Math.max(0, options.timeoutMs ?? 1_000)
  const pollIntervalMs = Math.max(1, options.pollIntervalMs ?? 25)
  const deadline = Date.now() + timeoutMs
  do {
    if (await hasNewerCurrentAccessToken(
      persistent,
      checked,
      capturedAccessToken
    )) {
      return true
    }
    if (Date.now() >= deadline) return false
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs))
  } while (Date.now() <= deadline)
  return false
}

/**
 * Store rotated tokens without rewriting the shared connection config.
 *
 * The record remains effective only while the raw target and its original
 * refresh token still match. A target/account change racing the final write
 * therefore makes the record inert instead of being overwritten.
 */
export const storeRefreshRotationIfCurrent = async (
  persistent: CredentialStorage,
  checked: ServicePromptTargetLock & Readonly<{ accessToken?: unknown }>,
  expectedRefreshToken: string,
  tokens: Readonly<{ accessToken: string; refreshToken: string }>
): Promise<boolean> => {
  const expected = nonEmptySecret(expectedRefreshToken)
  const accessToken = nonEmptySecret(tokens.accessToken)
  const refreshToken = nonEmptySecret(tokens.refreshToken)
  if (!expected || !accessToken || !refreshToken) return false

  const stored = await persistent.get<TldwConfig>("tldwConfig")
  if (
    !stored ||
    typeof stored !== "object" ||
    stored.authMode !== "multi-user" ||
    !servicePromptTargetsMatch(stored, checked)
  ) {
    return false
  }

  const previousValue = await persistent.get<unknown>(REFRESH_ROTATION_KEY)
  const previous = applicableRefreshRotation(stored, previousValue)
  const expectedAccessToken = nonEmptySecret(checked.accessToken)
  const effectiveAccessToken = previous?.accessToken ??
    nonEmptySecret(stored.accessToken)
  const effectiveRefreshToken = previous?.refreshToken ??
    nonEmptySecret(stored.refreshToken)
  const sourceAccessToken = previous?.sourceAccessToken ??
    nonEmptySecret(stored.accessToken)
  const sourceRefreshToken = previous?.sourceRefreshToken ??
    nonEmptySecret(stored.refreshToken)
  if (
    (expectedAccessToken && effectiveAccessToken !== expectedAccessToken) ||
    effectiveRefreshToken !== expected ||
    !sourceAccessToken ||
    !sourceRefreshToken
  ) return false

  await persistent.set<RefreshRotationRecord>(REFRESH_ROTATION_KEY, {
    version: 1,
    ...(nonEmptySecret(stored.serverUrl)
      ? { serverUrl: nonEmptySecret(stored.serverUrl)! }
      : {}),
    authMode: stored.authMode,
    authSource: stored.authSource,
    orgId: stored.orgId,
    sourceAccessToken,
    sourceRefreshToken,
    accessToken,
    refreshToken
  })
  return true
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

  const refreshRotation = await stores.persistent
    .get<unknown>(REFRESH_ROTATION_KEY)
    .catch(() => null)
  const effective = { ...applyRefreshRotation(stored, refreshRotation) }
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
  try {
    await persistent.remove(REFRESH_ROTATION_KEY)
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
