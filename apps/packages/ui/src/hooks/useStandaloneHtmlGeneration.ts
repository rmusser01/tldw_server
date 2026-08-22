import React from "react"

import {
  tldwClient,
  type PresentationGenerationReceipt,
  type PresentationGenerationRequest,
  type SlidesCapabilities
} from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"

const RECORD_SCHEMA_VERSION = 1
const RECORD_TTL_MS = 24 * 60 * 60 * 1000
const MAX_DRAFT_RECORD_BYTES = 1_500_000
const MAX_RESUME_RECORD_BYTES = 2_048
const MAX_POLL_DELAY_MS = 10_000
const MIN_POLL_DELAY_MS = 1_000
const IDEMPOTENCY_KEY_LENGTH = 32
const IDEMPOTENCY_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-"
const MAX_PROGRESS_TEXT_CHARS = 500

const PRESENTATION_TYPES = new Set([
  "pitch-deck", "tech-sharing", "product-launch", "weekly-report", "course-module",
  "keynote", "data-report", "training", "social-media", "case-study", "comparison", "roadmap"
])
const VISUAL_DIRECTIONS = new Set([
  "auto", "dark-technical", "minimal-light", "editorial", "corporate", "soft-pastel",
  "bold-creative", "neo-brutalist"
])
const DELIVERY_STYLES = new Set(["speaker-led", "self-guided"])

export type StandaloneHtmlFormDraft = {
  source: string
  presentationType: PresentationGenerationRequest["html_options"]["presentation_type"]
  audience: string
  slideCount: number
  visualDirection: PresentationGenerationRequest["html_options"]["visual_direction"]
  deliveryStyle: PresentationGenerationRequest["html_options"]["delivery_style"]
}

export type StandaloneHtmlGenerationPhase =
  | "idle"
  | "submitting"
  | "polling"
  | "ambiguous"
  | "rejected"
  | "stopped"
  | "failed"
  | "cancelled"
  | "completed"
  | "completed_missing_binding"
  | "auth_lost"
  | "missing"
  | "throttled"
  | "outage"
  | "configuration_changed"

type EnabledGenerationCapability = Extract<
  SlidesCapabilities["generation_modes"]["standalone_html"],
  { enabled: true }
>

type Scope = { serverOrigin: string; principalId: string }
type ScopeCapture = { scope: Scope; epoch: number }

type ResumeRecord = {
  generationId: string | null
  idempotencyKey: string
  requestDigest: string
  timestamp: number
}

type DraftRecord = {
  schemaVersion: 1
  timestamp: number
  values: StandaloneHtmlFormDraft
  generationConfigRevision: string
}

type PendingAttempt = ScopeCapture & {
  id: number
  request: PresentationGenerationRequest
  key: string
}

export const DEFAULT_STANDALONE_HTML_FORM_DRAFT: StandaloneHtmlFormDraft = {
  source: "",
  presentationType: "tech-sharing",
  audience: "",
  slideCount: 10,
  visualDirection: "auto",
  deliveryStyle: "speaker-led"
}

export const buildStandaloneHtmlStorageKeys = (scope: Scope) => {
  const namespace = `${encodeURIComponent(scope.serverOrigin)}:${encodeURIComponent(scope.principalId)}`
  return {
    draft: `tldw:presentation-studio:html:draft:v1:${namespace}`,
    resume: `tldw:presentation-studio:html:resume:v1:${namespace}`
  }
}

const containsInvalidScalar = (value: string): boolean => {
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index)
    if (unit === 0) return true
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = value.charCodeAt(index + 1)
      if (next < 0xdc00 || next > 0xdfff) return true
      index += 1
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      return true
    }
  }
  return false
}

const cloneDraft = (draft: StandaloneHtmlFormDraft): StandaloneHtmlFormDraft => ({ ...draft })

const utf8Bytes = (value: string): number => new TextEncoder().encode(value).byteLength
const scalarLength = (value: string): number => Array.from(value).length

const effectiveMaxSlides = (contentMaxSlides: number): number =>
  Math.max(1, Math.min(30, Number.isFinite(contentMaxSlides) ? Math.floor(contentMaxSlides) : 30))

const buildRequest = (
  draft: StandaloneHtmlFormDraft,
  revision: string
): PresentationGenerationRequest => ({
  generation_mode: "standalone_html",
  generation_config_revision: revision,
  source: { kind: "prompt", prompt: draft.source },
  html_options: {
    presentation_type: draft.presentationType,
    audience: draft.audience,
    slide_count: draft.slideCount,
    visual_direction: draft.visualDirection,
    delivery_style: draft.deliveryStyle
  }
})

const freezeRequest = (request: PresentationGenerationRequest): PresentationGenerationRequest => {
  Object.freeze(request.source)
  Object.freeze(request.html_options)
  return Object.freeze(request)
}

const digestRequest = (request: PresentationGenerationRequest): string => {
  const value = JSON.stringify(request)
  let high = 0x811c9dc5
  let low = 0x01000193
  for (let index = 0; index < value.length; index += 1) {
    high ^= value.charCodeAt(index)
    high = Math.imul(high, 0x01000193)
    low ^= value.charCodeAt(index)
    low = Math.imul(low, 0x5bd1e995)
  }
  return `v1:${(high >>> 0).toString(16).padStart(8, "0")}${(low >>> 0).toString(16).padStart(8, "0")}`
}

const createIdempotencyKey = (): string => {
  const bytes = new Uint8Array(IDEMPOTENCY_KEY_LENGTH)
  globalThis.crypto.getRandomValues(bytes)
  let key = ""
  for (const byte of bytes) key += IDEMPOTENCY_ALPHABET[byte % IDEMPOTENCY_ALPHABET.length]
  return key
}

const isFreshTimestamp = (value: unknown, now = Date.now()): value is number =>
  typeof value === "number" && Number.isFinite(value) && value <= now && now - value <= RECORD_TTL_MS

const validateDraft = (
  value: unknown,
  capability: EnabledGenerationCapability | null,
  contentMaxSlides = 30
): value is StandaloneHtmlFormDraft => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if (keys.length !== 6 || !["source", "presentationType", "audience", "slideCount", "visualDirection", "deliveryStyle"].every((key) => Object.prototype.hasOwnProperty.call(record, key))) return false
  return (
    typeof record.source === "string" &&
    scalarLength(record.source) <= Math.min(200_000, capability?.input_limits.max_source_chars ?? 200_000) &&
    !containsInvalidScalar(record.source) &&
    typeof record.audience === "string" &&
    scalarLength(record.audience) <= Math.min(500, capability?.input_limits.max_audience_chars ?? 500) &&
    !containsInvalidScalar(record.audience) &&
    typeof record.presentationType === "string" && PRESENTATION_TYPES.has(record.presentationType) &&
    typeof record.visualDirection === "string" && VISUAL_DIRECTIONS.has(record.visualDirection) &&
    typeof record.deliveryStyle === "string" && DELIVERY_STYLES.has(record.deliveryStyle) &&
    typeof record.slideCount === "number" && Number.isInteger(record.slideCount) &&
    record.slideCount >= 1 && record.slideCount <= effectiveMaxSlides(contentMaxSlides)
  )
}

const parseDraftRecord = (
  raw: string | null
): DraftRecord | null => {
  if (!raw || utf8Bytes(raw) > MAX_DRAFT_RECORD_BYTES) return null
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null
    const record = parsed as Record<string, unknown>
    if (record.schemaVersion !== RECORD_SCHEMA_VERSION || !isFreshTimestamp(record.timestamp)) return null
    if (typeof record.generationConfigRevision !== "string" || !/^sha256:[0-9a-f]{64}$/.test(record.generationConfigRevision)) return null
    if (!validateDraft(record.values, null, 30)) return null
    return record as DraftRecord
  } catch {
    return null
  }
}

const parseResumeRecord = (raw: string | null): ResumeRecord | null => {
  if (!raw || utf8Bytes(raw) > MAX_RESUME_RECORD_BYTES) return null
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null
    const record = parsed as Record<string, unknown>
    if (
      Object.keys(record).length !== 4 ||
      !["generationId", "idempotencyKey", "requestDigest", "timestamp"].every((key) => Object.prototype.hasOwnProperty.call(record, key)) ||
      !(record.generationId === null || (typeof record.generationId === "string" && record.generationId.length > 0 && record.generationId.length <= 200)) ||
      typeof record.idempotencyKey !== "string" ||
      !/^[A-Za-z0-9._~-]{16,200}$/.test(record.idempotencyKey) ||
      typeof record.requestDigest !== "string" || record.requestDigest.length > 100 ||
      !isFreshTimestamp(record.timestamp)
    ) return null
    return record as ResumeRecord
  } catch {
    return null
  }
}

const safeErrorCode = (error: unknown, fallback: string): string => {
  if (!error || typeof error !== "object") return fallback
  const details = (error as { details?: unknown }).details
  if (details && typeof details === "object" && !Array.isArray(details)) {
    const record = details as Record<string, unknown>
    for (const candidate of [record.error_code, record.code, record.detail]) {
      if (typeof candidate === "string" && /^[a-z0-9_]{1,100}$/.test(candidate)) return candidate
    }
  }
  return fallback
}

const errorStatus = (error: unknown): number | null => {
  const status = error && typeof error === "object" ? (error as { status?: unknown }).status : null
  return typeof status === "number" && Number.isFinite(status) ? status : null
}

const resolveScope = async (): Promise<Scope | null> => {
  try {
    const [config, user] = await Promise.all([tldwClient.getConfig(), tldwAuth.getCurrentUser()])
    const configured = typeof config?.serverUrl === "string" ? config.serverUrl.trim() : ""
    const fallback = typeof window !== "undefined" ? window.location.origin : ""
    const serverOrigin = new URL(configured || fallback, fallback || undefined).origin.toLowerCase()
    const principalId = String(user?.id ?? "").trim()
    if (!serverOrigin || !principalId) return null
    return { serverOrigin, principalId }
  } catch {
    return null
  }
}

const sameScope = (left: Scope | null, right: Scope | null): boolean =>
  Boolean(left && right && left.serverOrigin === right.serverOrigin && left.principalId === right.principalId)

const retryAfterMs = (error: unknown): number | null => {
  const value = error && typeof error === "object" ? (error as { retryAfterMs?: unknown }).retryAfterMs : null
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.min(MAX_POLL_DELAY_MS, Math.floor(value))
    : null
}

export const probeStandaloneHtmlRecovery = async (): Promise<boolean | null> => {
  const scope = await resolveScope()
  if (!scope) return null
  const key = buildStandaloneHtmlStorageKeys(scope).resume
  try {
    const record = parseResumeRecord(window.sessionStorage.getItem(key))
    if (!record) {
      window.sessionStorage.removeItem(key)
      return false
    }
    return true
  } catch {
    return null
  }
}

export type StandaloneHtmlRecoveryProbeResult = {
  status: "checking" | "available" | "none" | "unavailable"
  retry: () => Promise<void>
}

export const useStandaloneHtmlRecoveryProbe = (): StandaloneHtmlRecoveryProbeResult => {
  const [status, setStatus] = React.useState<StandaloneHtmlRecoveryProbeResult["status"]>("checking")
  const requestIdRef = React.useRef(0)

  const check = React.useCallback(async () => {
    const requestId = ++requestIdRef.current
    setStatus("checking")
    const result = await probeStandaloneHtmlRecovery()
    if (requestId !== requestIdRef.current) return
    setStatus(result === true ? "available" : result === false ? "none" : "unavailable")
  }, [])

  React.useEffect(() => {
    void check()
    const restore = () => { void check() }
    const invalidate = () => {
      requestIdRef.current += 1
      setStatus("checking")
    }
    window.addEventListener("tldw:config-updated", restore)
    window.addEventListener("tldw:auth-principal-changed", restore)
    window.addEventListener("pagehide", invalidate)
    window.addEventListener("pageshow", restore)
    window.addEventListener("focus", restore)
    return () => {
      requestIdRef.current += 1
      window.removeEventListener("tldw:config-updated", restore)
      window.removeEventListener("tldw:auth-principal-changed", restore)
      window.removeEventListener("pagehide", invalidate)
      window.removeEventListener("pageshow", restore)
      window.removeEventListener("focus", restore)
    }
  }, [check])

  return { status, retry: check }
}

export type UseStandaloneHtmlGenerationOptions = {
  capability: EnabledGenerationCapability | null
  contentMaxSlides?: number
  onCapabilitiesChanged?: () => Promise<unknown> | unknown
  onCompleted: (presentationId: string) => void
  onStopWaiting: () => void
}

export const useStandaloneHtmlGeneration = ({
  capability,
  contentMaxSlides = 30,
  onCapabilitiesChanged = () => undefined,
  onCompleted,
  onStopWaiting
}: UseStandaloneHtmlGenerationOptions) => {
  const maxSlides = effectiveMaxSlides(contentMaxSlides)
  const initialDraft = React.useMemo(
    () => ({ ...DEFAULT_STANDALONE_HTML_FORM_DRAFT, slideCount: Math.min(10, maxSlides) }),
    [maxSlides]
  )
  const [scopeReady, setScopeReady] = React.useState(false)
  const [scopeError, setScopeError] = React.useState<string | null>(null)
  const [draft, setDraft] = React.useState<StandaloneHtmlFormDraft>(initialDraft)
  const [fieldErrors, setFieldErrors] = React.useState<Partial<Record<keyof StandaloneHtmlFormDraft, string>>>({})
  const [editError, setEditError] = React.useState<string | null>(null)
  const [phase, setPhase] = React.useState<StandaloneHtmlGenerationPhase>("idle")
  const [snapshot, setSnapshot] = React.useState<PresentationGenerationRequest | null>(null)
  const [backendStatus, setBackendStatus] = React.useState<PresentationGenerationReceipt["status"] | null>(null)
  const [progressText, setProgressText] = React.useState<string | null>(null)
  const [safeError, setSafeError] = React.useState<string | null>(null)
  const [recoveryAvailable, setRecoveryAvailable] = React.useState(false)
  const [storageWarning, setStorageWarning] = React.useState<string | null>(null)
  const [pendingAttempt, setPendingAttempt] = React.useState<PendingAttempt | null>(null)

  const scopeRef = React.useRef<Scope | null>(null)
  const draftRef = React.useRef(draft)
  const snapshotRef = React.useRef<PresentationGenerationRequest | null>(null)
  const resumeRef = React.useRef<ResumeRecord | null>(null)
  const draftRevisionRef = React.useRef(capability?.generation_config_revision ?? "")
  const pollTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollAttemptRef = React.useRef(0)
  const waitingRef = React.useRef(false)
  const mountedRef = React.useRef(true)
  const scopeValidationIdRef = React.useRef(0)
  const scopeEpochRef = React.useRef(0)
  const submitAbortRef = React.useRef<AbortController | null>(null)
  const pollAbortRef = React.useRef<AbortController | null>(null)
  const pendingAttemptIdRef = React.useRef(0)
  const startedAttemptIdRef = React.useRef(0)
  const initialDraftRef = React.useRef(initialDraft)
  initialDraftRef.current = initialDraft

  const locked = snapshot !== null && !["rejected", "failed", "cancelled", "completed_missing_binding"].includes(phase)

  const stopPollTimer = React.useCallback(() => {
    waitingRef.current = false
    pollAbortRef.current?.abort()
    pollAbortRef.current = null
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
    pollTimerRef.current = null
  }, [])

  const isCaptureCurrent = React.useCallback((capture: ScopeCapture): boolean =>
    mountedRef.current &&
    scopeEpochRef.current === capture.epoch &&
    sameScope(scopeRef.current, capture.scope), [])

  const removeResumeRecord = React.useCallback((scope = scopeRef.current) => {
    if (!scope) return
    const keys = buildStandaloneHtmlStorageKeys(scope)
    try {
      window.sessionStorage.removeItem(keys.resume)
    } catch {
      setStorageWarning("Reload recovery is unavailable.")
    }
    resumeRef.current = null
    setRecoveryAvailable(false)
  }, [])

  const removeRecords = React.useCallback((scope = scopeRef.current) => {
    if (!scope) return
    const keys = buildStandaloneHtmlStorageKeys(scope)
    try {
      window.sessionStorage.removeItem(keys.draft)
      window.sessionStorage.removeItem(keys.resume)
    } catch {
      setStorageWarning("Reload recovery is unavailable.")
    }
    if (sameScope(scopeRef.current, scope)) {
      resumeRef.current = null
      setRecoveryAvailable(false)
    }
  }, [])

  const persistDraft = React.useCallback((
    values: StandaloneHtmlFormDraft,
    revision: string,
    capture?: ScopeCapture
  ) => {
    const scope = capture?.scope ?? scopeRef.current
    if (capture && !isCaptureCurrent(capture)) return false
    if (!scope || !revision || !validateDraft(values, capability, maxSlides)) return false
    const record: DraftRecord = {
      schemaVersion: RECORD_SCHEMA_VERSION,
      timestamp: Date.now(),
      values: cloneDraft(values),
      generationConfigRevision: revision
    }
    const serialized = JSON.stringify(record)
    if (utf8Bytes(serialized) > MAX_DRAFT_RECORD_BYTES) {
      setStorageWarning("Reload recovery is unavailable.")
      return false
    }
    try {
      window.sessionStorage.setItem(buildStandaloneHtmlStorageKeys(scope).draft, serialized)
      return true
    } catch {
      setStorageWarning("Reload recovery is unavailable.")
      return false
    }
  }, [capability, isCaptureCurrent, maxSlides])

  const persistResume = React.useCallback((record: ResumeRecord, capture?: ScopeCapture) => {
    const scope = capture?.scope ?? scopeRef.current
    if (capture && !isCaptureCurrent(capture)) return false
    if (!scope) return false
    resumeRef.current = record
    setRecoveryAvailable(true)
    const serialized = JSON.stringify(record)
    if (utf8Bytes(serialized) > MAX_RESUME_RECORD_BYTES) {
      setStorageWarning("Reload recovery is unavailable.")
      return false
    }
    try {
      window.sessionStorage.setItem(buildStandaloneHtmlStorageKeys(scope).resume, serialized)
      return true
    } catch {
      setStorageWarning("Reload recovery is unavailable.")
      return false
    }
  }, [isCaptureCurrent])

  const clearSensitiveMemory = React.useCallback(() => {
    stopPollTimer()
    submitAbortRef.current?.abort()
    submitAbortRef.current = null
    draftRef.current = initialDraft
    snapshotRef.current = null
    resumeRef.current = null
    setPendingAttempt(null)
    setRecoveryAvailable(false)
    setDraft(initialDraft)
    setSnapshot(null)
    setBackendStatus(null)
    setProgressText(null)
    setSafeError(null)
    setFieldErrors({})
    setEditError(null)
    setPhase("idle")
  }, [initialDraft, stopPollTimer])

  const invalidateScopeBoundary = React.useCallback(() => {
    scopeEpochRef.current += 1
    scopeValidationIdRef.current += 1
    scopeRef.current = null
    clearSensitiveMemory()
    setScopeReady(false)
  }, [clearSensitiveMemory])

  const hydrateForScope = React.useCallback((scope: Scope) => {
    const keys = buildStandaloneHtmlStorageKeys(scope)
    let rawDraft: string | null = null
    let rawResume: string | null = null
    try {
      rawDraft = window.sessionStorage.getItem(keys.draft)
      rawResume = window.sessionStorage.getItem(keys.resume)
    } catch {
      setStorageWarning("Reload recovery is unavailable.")
      return
    }
    const storedDraft = parseDraftRecord(rawDraft)
    const storedResume = parseResumeRecord(rawResume)
    if (!storedDraft) {
      try {
        window.sessionStorage.removeItem(keys.draft)
        window.sessionStorage.removeItem(keys.resume)
      } catch {}
    }
    if (!storedResume) {
      try { window.sessionStorage.removeItem(keys.resume) } catch {}
    }
    if (!storedDraft) {
      resumeRef.current = null
      setRecoveryAvailable(false)
      return
    }
    draftRevisionRef.current = storedDraft.generationConfigRevision
    draftRef.current = cloneDraft(storedDraft.values)
    setDraft(cloneDraft(storedDraft.values))
    const request = freezeRequest(buildRequest(storedDraft.values, storedDraft.generationConfigRevision))
    if (storedResume && digestRequest(request) !== storedResume.requestDigest) {
      try { window.sessionStorage.removeItem(keys.resume) } catch {}
      resumeRef.current = null
      setRecoveryAvailable(false)
      return
    }
    if (storedResume) {
      resumeRef.current = storedResume
      snapshotRef.current = request
      setSnapshot(request)
      setPhase("stopped")
      setRecoveryAvailable(true)
    }
  }, [])

  const revalidateScope = React.useCallback(async (clearFirst: boolean, definitiveLogout = false) => {
    const previous = scopeRef.current
    if (clearFirst) {
      invalidateScopeBoundary()
    }
    if (definitiveLogout) removeRecords(previous)
    const validationId = ++scopeValidationIdRef.current
    setScopeError(null)
    const next = await resolveScope()
    if (!mountedRef.current || validationId !== scopeValidationIdRef.current) return
    if (!next) {
      setScopeReady(false)
      setScopeError("Current server and account could not be confirmed.")
      return
    }
    if (previous && (previous.serverOrigin !== next.serverOrigin || previous.principalId !== next.principalId)) {
      removeRecords(previous)
    }
    scopeRef.current = next
    setScopeReady(true)
    hydrateForScope(next)
  }, [hydrateForScope, invalidateScopeBoundary, removeRecords])

  const revalidateScopeRef = React.useRef(revalidateScope)
  revalidateScopeRef.current = revalidateScope

  React.useEffect(() => {
    mountedRef.current = true
    void revalidateScopeRef.current(false)
    return () => {
      mountedRef.current = false
      scopeEpochRef.current += 1
      scopeValidationIdRef.current += 1
      scopeRef.current = null
      submitAbortRef.current?.abort()
      submitAbortRef.current = null
      pollAbortRef.current?.abort()
      pollAbortRef.current = null
      waitingRef.current = false
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
      pollTimerRef.current = null
      draftRef.current = initialDraftRef.current
      snapshotRef.current = null
      resumeRef.current = null
    }
  }, [])

  React.useEffect(() => {
    const restore = () => { void revalidateScope(true) }
    const authBoundary = (event: Event) => {
      const definitive = (event as CustomEvent<{ kind?: string }>).detail?.kind === "logout"
      void revalidateScope(true, definitive)
    }
    const pagehide = () => {
      const values = draftRef.current
      const request = snapshotRef.current
      const scope = scopeRef.current
      if (scope) {
        const capture = { scope, epoch: scopeEpochRef.current }
        persistDraft(values, request?.generation_config_revision ?? draftRevisionRef.current, capture)
      }
      invalidateScopeBoundary()
    }
    const visibility = () => {
      if (document.visibilityState === "visible") restore()
    }
    window.addEventListener("tldw:config-updated", restore)
    window.addEventListener("tldw:auth-principal-changed", authBoundary)
    window.addEventListener("pagehide", pagehide)
    window.addEventListener("pageshow", restore)
    window.addEventListener("focus", restore)
    document.addEventListener("visibilitychange", visibility)
    return () => {
      window.removeEventListener("tldw:config-updated", restore)
      window.removeEventListener("tldw:auth-principal-changed", authBoundary)
      window.removeEventListener("pagehide", pagehide)
      window.removeEventListener("pageshow", restore)
      window.removeEventListener("focus", restore)
      document.removeEventListener("visibilitychange", visibility)
    }
  }, [invalidateScopeBoundary, persistDraft, revalidateScope])

  const replaceDraft = React.useCallback((next: StandaloneHtmlFormDraft) => {
    if (!validateDraft(next, capability, maxSlides)) return false
    const accepted = cloneDraft(next)
    draftRef.current = accepted
    setDraft(accepted)
    setFieldErrors({})
    setEditError(null)
    persistDraft(accepted, capability?.generation_config_revision ?? draftRevisionRef.current)
    return true
  }, [capability, maxSlides, persistDraft])

  const updateField = React.useCallback(<K extends keyof StandaloneHtmlFormDraft>(field: K, value: StandaloneHtmlFormDraft[K]): boolean => {
    const next = { ...draftRef.current, [field]: value } as StandaloneHtmlFormDraft
    let error: string | null = null
    if ((field === "source" || field === "audience") && typeof value === "string") {
      if (containsInvalidScalar(value)) error = `${field === "source" ? "Subject and material" : "Audience"} must contain valid text without NUL characters.`
      const limit = field === "source"
        ? Math.min(200_000, capability?.input_limits.max_source_chars ?? 200_000)
        : Math.min(500, capability?.input_limits.max_audience_chars ?? 500)
      if (!error && scalarLength(value) > limit) error = `${field === "source" ? "Subject and material" : "Audience"} exceeds the ${limit.toLocaleString()} character limit.`
    }
    if (field === "slideCount" && (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > maxSlides)) error = `Slide count must be an integer from 1 to ${maxSlides}.`
    if (!error && !validateDraft(next, capability, maxSlides)) error = "That value is not supported."
    if (error) {
      setEditError(null)
      setFieldErrors((current) => ({ ...current, [field]: error! }))
      return false
    }
    draftRef.current = next
    setDraft(next)
    setEditError(null)
    setFieldErrors((current) => {
      const copy = { ...current }
      delete copy[field]
      return copy
    })
    persistDraft(next, capability?.generation_config_revision ?? draftRevisionRef.current)
    return true
  }, [capability, maxSlides, persistDraft])

  const validateForSubmit = React.useCallback((): boolean => {
    const errors: Partial<Record<keyof StandaloneHtmlFormDraft, string>> = {}
    const current = draftRef.current
    const sourceLimit = Math.min(200_000, capability?.input_limits.max_source_chars ?? 200_000)
    const audienceLimit = Math.min(500, capability?.input_limits.max_audience_chars ?? 500)
    if (!current.source.trim()) errors.source = "Subject and material is required."
    if (!current.audience.trim()) errors.audience = "Audience is required."
    if (!Number.isInteger(current.slideCount) || current.slideCount < 1 || current.slideCount > maxSlides) errors.slideCount = `Slide count must be an integer from 1 to ${maxSlides}.`
    if (scalarLength(current.source) > sourceLimit) errors.source = "Subject and material exceeds the current character limit."
    if (scalarLength(current.audience) > audienceLimit) errors.audience = "Audience exceeds the current character limit."
    const draftIsValid = validateDraft(current, capability, maxSlides)
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0 || !draftIsValid || !capability) return false
    const request = buildRequest(current, capability.generation_config_revision)
    if (utf8Bytes(JSON.stringify(request)) > capability.input_limits.max_request_bytes) {
      setEditError(`Request exceeds the ${capability.input_limits.max_request_bytes.toLocaleString("en-US", { useGrouping: false })} byte limit.`)
      return false
    }
    setEditError(null)
    return true
  }, [capability, maxSlides])

  const finishReceipt = React.useCallback((receipt: PresentationGenerationReceipt, capture: ScopeCapture): boolean => {
    if (!isCaptureCurrent(capture)) return true
    setBackendStatus(receipt.status)
    if ("progress_text" in receipt && receipt.progress_text) {
      setProgressText(receipt.progress_text.slice(0, MAX_PROGRESS_TEXT_CHARS))
    }
    if (receipt.status === "completed") {
      if (!receipt.presentation_id) {
        setPhase("completed_missing_binding")
        setSafeError("generation_completed_without_presentation")
        stopPollTimer()
        return true
      }
      setPhase("completed")
      stopPollTimer()
      removeRecords(capture.scope)
      draftRef.current = initialDraft
      snapshotRef.current = null
      resumeRef.current = null
      setDraft(initialDraft)
      setSnapshot(null)
      setProgressText(null)
      setSafeError(null)
      onCompleted(receipt.presentation_id)
      return true
    }
    if (receipt.status === "failed") {
      setPhase("failed")
      setSafeError(/^[a-z0-9_]{1,100}$/.test(receipt.error_code) ? receipt.error_code : "generation_failed")
      stopPollTimer()
      return true
    }
    if (receipt.status === "cancelled") {
      setPhase("cancelled")
      setSafeError("generation_cancelled")
      stopPollTimer()
      return true
    }
    setPhase("polling")
    return false
  }, [initialDraft, isCaptureCurrent, onCompleted, removeRecords, stopPollTimer])

  const pollGenerationRef = React.useRef<(generationId: string, capture: ScopeCapture) => Promise<void>>(async () => undefined)
  const schedulePoll = React.useCallback((generationId: string, delayHint: number | null, capture: ScopeCapture) => {
    if (!waitingRef.current || !isCaptureCurrent(capture)) return
    const exponential = Math.min(MAX_POLL_DELAY_MS, MIN_POLL_DELAY_MS * (2 ** pollAttemptRef.current))
    pollAttemptRef.current += 1
    const delay = Math.min(MAX_POLL_DELAY_MS, Math.max(MIN_POLL_DELAY_MS, delayHint ?? exponential))
    pollTimerRef.current = setTimeout(() => { void pollGenerationRef.current(generationId, capture) }, delay)
  }, [isCaptureCurrent])

  const pollGeneration = React.useCallback(async (generationId: string, capture: ScopeCapture) => {
    if (!waitingRef.current || !isCaptureCurrent(capture)) return
    const controller = new AbortController()
    pollAbortRef.current?.abort()
    pollAbortRef.current = controller
    try {
      const result = await tldwClient.getPresentationGenerationStatus(generationId, { abortSignal: controller.signal })
      if (!waitingRef.current || controller.signal.aborted || !isCaptureCurrent(capture)) return
      if (!finishReceipt(result.receipt, capture)) schedulePoll(generationId, result.retryAfterMs, capture)
    } catch (error) {
      if (!waitingRef.current || controller.signal.aborted || !isCaptureCurrent(capture)) return
      const status = errorStatus(error)
      if (status === 401 || status === 403) {
        stopPollTimer()
        setPhase("auth_lost")
      } else if (status === 404) {
        stopPollTimer()
        setPhase("missing")
      } else {
        setPhase(status === 429 ? "throttled" : "outage")
        schedulePoll(generationId, retryAfterMs(error), capture)
      }
      setSafeError(status === 404 ? "generation_not_found" : status === 401 || status === 403 ? "authentication_required" : status === 429 ? "generation_status_throttled" : "generation_status_unavailable")
    } finally {
      if (pollAbortRef.current === controller) pollAbortRef.current = null
    }
  }, [finishReceipt, isCaptureCurrent, schedulePoll, stopPollTimer])
  pollGenerationRef.current = pollGeneration

  const acceptReceipt = React.useCallback((receipt: PresentationGenerationReceipt, key: string, request: PresentationGenerationRequest, capture: ScopeCapture) => {
    if (!isCaptureCurrent(capture) || finishReceipt(receipt, capture)) return
    const updated: ResumeRecord = {
      generationId: receipt.generation_id,
      idempotencyKey: key,
      requestDigest: digestRequest(request),
      timestamp: Date.now()
    }
    persistResume(updated, capture)
    waitingRef.current = true
    pollAttemptRef.current = 0
    void pollGeneration(receipt.generation_id, capture)
  }, [finishReceipt, isCaptureCurrent, persistResume, pollGeneration])

  const dispatch = React.useCallback(async (attempt: PendingAttempt) => {
    const controller = new AbortController()
    submitAbortRef.current?.abort()
    submitAbortRef.current = controller
    try {
      const receipt = await tldwClient.submitPresentationGeneration(attempt.request, {
        idempotencyKey: attempt.key,
        abortSignal: controller.signal
      })
      if (controller.signal.aborted || !isCaptureCurrent(attempt)) return
      acceptReceipt(receipt, attempt.key, attempt.request, attempt)
    } catch (error) {
      if (controller.signal.aborted || !isCaptureCurrent(attempt)) return
      const status = errorStatus(error)
      if (status !== null && [400, 409, 413, 415, 422].includes(status)) {
        const code = safeErrorCode(error, "generation_request_rejected")
        removeResumeRecord(attempt.scope)
        snapshotRef.current = null
        setSnapshot(null)
        if (status === 409 && code === "generation_configuration_changed") {
          setPhase("configuration_changed")
          setSafeError(code)
          await onCapabilitiesChanged()
        } else {
          setPhase("rejected")
          setSafeError(code)
        }
      } else {
        setPhase(status === 429 ? "throttled" : "ambiguous")
        setSafeError(status === 429 ? "generation_submission_throttled" : "generation_submission_unknown")
        setRecoveryAvailable(true)
      }
    } finally {
      if (submitAbortRef.current === controller) submitAbortRef.current = null
      if (isCaptureCurrent(attempt)) {
        setPendingAttempt((current) => current?.id === attempt.id ? null : current)
      }
    }
  }, [acceptReceipt, isCaptureCurrent, onCapabilitiesChanged, removeResumeRecord])

  React.useEffect(() => {
    if (!pendingAttempt || startedAttemptIdRef.current >= pendingAttempt.id) return
    startedAttemptIdRef.current = pendingAttempt.id
    void dispatch(pendingAttempt)
  }, [dispatch, pendingAttempt])

  const queueAttempt = React.useCallback((
    request: PresentationGenerationRequest,
    key: string,
    capture: ScopeCapture
  ): Promise<void> => {
    setPhase("submitting")
    setSafeError(null)
    const id = ++pendingAttemptIdRef.current
    setPendingAttempt({ id, request, key, scope: capture.scope, epoch: capture.epoch })
    return Promise.resolve()
  }, [])

  const submit = React.useCallback(async () => {
    const scope = scopeRef.current
    if (!scope || !capability || snapshotRef.current || !validateForSubmit()) return
    const capture = { scope, epoch: scopeEpochRef.current }
    const request = freezeRequest(buildRequest(cloneDraft(draftRef.current), capability.generation_config_revision))
    const key = createIdempotencyKey()
    const resume: ResumeRecord = { generationId: null, idempotencyKey: key, requestDigest: digestRequest(request), timestamp: Date.now() }
    snapshotRef.current = request
    draftRevisionRef.current = request.generation_config_revision
    setSnapshot(request)
    persistDraft(draftRef.current, request.generation_config_revision, capture)
    persistResume(resume, capture)
    await queueAttempt(request, key, capture)
  }, [capability, persistDraft, persistResume, queueAttempt, validateForSubmit])

  const resume = React.useCallback(async () => {
    const request = snapshotRef.current
    const recovery = resumeRef.current
    const scope = scopeRef.current
    if (!request || !recovery || digestRequest(request) !== recovery.requestDigest) {
      removeResumeRecord()
      return
    }
    if (!scope) return
    const capture = { scope, epoch: scopeEpochRef.current }
    if (recovery.generationId) {
      setPhase("polling")
      waitingRef.current = true
      pollAttemptRef.current = 0
      await pollGeneration(recovery.generationId, capture)
    } else {
      await queueAttempt(request, recovery.idempotencyKey, capture)
    }
  }, [pollGeneration, queueAttempt, removeResumeRecord])

  const stopWaiting = React.useCallback(() => {
    stopPollTimer()
    setPhase("stopped")
    onStopWaiting()
  }, [onStopWaiting, stopPollTimer])

  const forget = React.useCallback(() => {
    removeRecords()
    clearSensitiveMemory()
  }, [clearSensitiveMemory, removeRecords])

  const startDifferent = React.useCallback(() => {
    removeResumeRecord()
    snapshotRef.current = null
    setSnapshot(null)
    setBackendStatus(null)
    setProgressText(null)
    setSafeError(null)
    setPhase("idle")
  }, [removeResumeRecord])

  const tryAgain = React.useCallback(async () => {
    const scope = scopeRef.current
    if (!scope || !capability || !validateForSubmit()) return
    const capture = { scope, epoch: scopeEpochRef.current }
    removeResumeRecord(scope)
    snapshotRef.current = null
    setSnapshot(null)
    setPhase("idle")
    const request = freezeRequest(buildRequest(cloneDraft(draftRef.current), capability.generation_config_revision))
    const key = createIdempotencyKey()
    const recovery: ResumeRecord = { generationId: null, idempotencyKey: key, requestDigest: digestRequest(request), timestamp: Date.now() }
    snapshotRef.current = request
    draftRevisionRef.current = request.generation_config_revision
    setSnapshot(request)
    persistDraft(draftRef.current, request.generation_config_revision, capture)
    persistResume(recovery, capture)
    await queueAttempt(request, key, capture)
  }, [capability, persistDraft, persistResume, queueAttempt, removeResumeRecord, validateForSubmit])

  return {
    scopeReady,
    scopeError,
    draft,
    fieldErrors,
    editError,
    phase,
    locked,
    snapshot,
    backendStatus,
    progressText,
    safeError,
    recoveryAvailable,
    storageWarning,
    updateField,
    replaceDraft,
    submit,
    resume,
    stopWaiting,
    forget,
    startDifferent,
    tryAgain
  }
}
