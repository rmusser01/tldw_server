import {
  MAX_STANDALONE_HTML_SOURCE_BYTES,
  preflightStandaloneHtmlSource,
  validateStandaloneHtmlSource,
  type AcceptedStandaloneHtmlSource
} from "./standalone-html-source"

const RECOVERY_PREFIX = "tldw:presentation-studio:html:draft:v1:workspace:"
const RECOVERY_TTL_MS = 24 * 60 * 60 * 1_000
const MAX_RECOVERY_SERIALIZED_CODE_UNITS = MAX_STANDALONE_HTML_SOURCE_BYTES * 6 + 8_192
const RECOVERY_KEYS = [
  "schemaVersion",
  "principalScope",
  "presentationId",
  "baseEtag",
  "baseDigest",
  "source",
  "updatedAt"
] as const

export type PresentationPrincipalScope = {
  serverOrigin: string
  principalId: string
  principalScope: string
}

export type StandaloneHtmlRecoveryRecord = {
  schemaVersion: 1
  principalScope: string
  presentationId: string
  baseEtag: string
  baseDigest: string
  source: string
  updatedAt: number
}

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">

const recoveryFailure = {
  ok: false as const,
  code: "recovery_unavailable" as const,
  message: "Recovery unavailable. Keep this tab open or download your draft."
}

const recoveryUnavailable = {
  kind: "unavailable" as const,
  code: recoveryFailure.code,
  message: recoveryFailure.message
}

export const acquireStandaloneHtmlRecoveryStorage = () => {
  try {
    if (typeof window === "undefined") return recoveryFailure
    return { ok: true as const, storage: window.sessionStorage as StorageLike }
  } catch {
    return recoveryFailure
  }
}

export const createPresentationPrincipalScope = (
  serverUrl: string,
  principalId: string | number
): PresentationPrincipalScope => {
  const fallback = typeof window !== "undefined" ? window.location.origin : undefined
  const serverOrigin = new URL(String(serverUrl).trim(), fallback).origin.toLowerCase()
  const normalizedPrincipal = String(principalId).trim()
  if (!normalizedPrincipal) throw new Error("A trusted principal is required")
  return {
    serverOrigin,
    principalId: normalizedPrincipal,
    principalScope: `${serverOrigin}|${encodeURIComponent(normalizedPrincipal)}`
  }
}

const recoveryKey = (scope: PresentationPrincipalScope, presentationId: string): string =>
  `${RECOVERY_PREFIX}${encodeURIComponent(scope.serverOrigin)}:${encodeURIComponent(
    scope.principalId
  )}:${encodeURIComponent(presentationId)}`

const removeRecoveryRecord = (storage: StorageLike, key: string): boolean => {
  try {
    storage.removeItem(key)
    return true
  } catch {
    return false
  }
}

const isClosedRecoveryRecord = (
  value: unknown,
  scope: PresentationPrincipalScope,
  presentationId: string
): value is StandaloneHtmlRecoveryRecord => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  return (
    keys.length === RECOVERY_KEYS.length &&
    RECOVERY_KEYS.every((key) => Object.prototype.hasOwnProperty.call(record, key)) &&
    record.schemaVersion === 1 &&
    record.principalScope === scope.principalScope &&
    record.presentationId === presentationId &&
    typeof record.baseEtag === "string" &&
    typeof record.baseDigest === "string" &&
    /^[0-9a-f]{64}$/.test(record.baseDigest) &&
    typeof record.source === "string" &&
    typeof record.updatedAt === "number" &&
    Number.isFinite(record.updatedAt)
  )
}

export const writeStandaloneHtmlRecovery = (
  storage: StorageLike,
  scope: PresentationPrincipalScope,
  input: {
    presentationId: string
    baseEtag: string
    baseDigest: string
    acceptedSource: Pick<AcceptedStandaloneHtmlSource, "source" | "digest"> | unknown
    updatedAt: number
  }
) => {
  const accepted = input.acceptedSource as Partial<AcceptedStandaloneHtmlSource> | null
  const source = accepted?.source
  const preflight = preflightStandaloneHtmlSource(source)
  if (
    !preflight.ok ||
    (accepted?.digest !== undefined &&
      (typeof accepted.digest !== "string" || !/^[0-9a-f]{64}$/.test(accepted.digest))) ||
    typeof input.presentationId !== "string" ||
    !input.presentationId ||
    typeof input.baseEtag !== "string" ||
    typeof input.baseDigest !== "string" ||
    !/^[0-9a-f]{64}$/.test(input.baseDigest) ||
    !Number.isFinite(input.updatedAt)
  ) {
    return recoveryFailure
  }

  const record: StandaloneHtmlRecoveryRecord = {
    schemaVersion: 1,
    principalScope: scope.principalScope,
    presentationId: input.presentationId,
    baseEtag: input.baseEtag,
    baseDigest: input.baseDigest,
    source: source as string,
    updatedAt: input.updatedAt
  }
  try {
    storage.setItem(recoveryKey(scope, input.presentationId), JSON.stringify(record))
    return { ok: true as const, record }
  } catch {
    return recoveryFailure
  }
}

export const readStandaloneHtmlRecovery = async (
  storage: StorageLike,
  scope: PresentationPrincipalScope,
  presentationId: string,
  now = Date.now()
) => {
  const key = recoveryKey(scope, presentationId)
  let raw: string | null
  try {
    raw = storage.getItem(key)
  } catch {
    return recoveryUnavailable
  }
  if (!raw) return { kind: "none" as const }
  if (raw.length > MAX_RECOVERY_SERIALIZED_CODE_UNITS) {
    return removeRecoveryRecord(storage, key)
      ? { kind: "none" as const }
      : recoveryUnavailable
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return removeRecoveryRecord(storage, key)
      ? { kind: "none" as const }
      : recoveryUnavailable
  }
  if (!isClosedRecoveryRecord(parsed, scope, presentationId)) {
    return removeRecoveryRecord(storage, key)
      ? { kind: "none" as const }
      : recoveryUnavailable
  }
  if (parsed.updatedAt > now || now - parsed.updatedAt > RECOVERY_TTL_MS) {
    return removeRecoveryRecord(storage, key)
      ? { kind: "none" as const }
      : recoveryUnavailable
  }

  try {
    const acceptedSource = await validateStandaloneHtmlSource(parsed.source)
    if (acceptedSource.ok === false) {
      return removeRecoveryRecord(storage, key)
        ? { kind: "none" as const }
        : recoveryUnavailable
    }
    return { kind: "available" as const, record: parsed, acceptedSource }
  } catch {
    return removeRecoveryRecord(storage, key)
      ? { kind: "none" as const }
      : recoveryUnavailable
  }
}

export const clearStandaloneHtmlRecovery = (
  storage: StorageLike,
  scope: PresentationPrincipalScope,
  presentationId: string
): boolean => removeRecoveryRecord(storage, recoveryKey(scope, presentationId))
