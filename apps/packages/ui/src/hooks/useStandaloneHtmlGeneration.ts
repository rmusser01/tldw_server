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

type EnabledGenerationCapability = Extract<
  SlidesCapabilities["generation_modes"]["standalone_html"],
  { enabled: true }
>

type Scope = { serverOrigin: string; principalId: string }

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
  capability: EnabledGenerationCapability
): value is StandaloneHtmlFormDraft => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if (keys.length !== 6 || !["source", "presentationType", "audience", "slideCount", "visualDirection", "deliveryStyle"].every((key) => Object.prototype.hasOwnProperty.call(record, key))) return false
  return (
    typeof record.source === "string" &&
    record.source.length <= Math.min(200_000, capability.input_limits.max_source_chars) &&
    !containsInvalidScalar(record.source) &&
    typeof record.audience === "string" &&
    record.audience.length <= Math.min(500, capability.input_limits.max_audience_chars) &&
    !containsInvalidScalar(record.audience) &&
    typeof record.presentationType === "string" && PRESENTATION_TYPES.has(record.presentationType) &&
    typeof record.visualDirection === "string" && VISUAL_DIRECTIONS.has(record.visualDirection) &&
    typeof record.deliveryStyle === "string" && DELIVERY_STYLES.has(record.deliveryStyle) &&
    typeof record.slideCount === "number" && Number.isInteger(record.slideCount) &&
    record.slideCount >= 1 && record.slideCount <= 30
  )
}

const parseDraftRecord = (
  raw: string | null,
  capability: EnabledGenerationCapability
): DraftRecord | null => {
  if (!raw || utf8Bytes(raw) > MAX_DRAFT_RECORD_BYTES) return null
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null
    const record = parsed as Record<string, unknown>
    if (record.schemaVersion !== RECORD_SCHEMA_VERSION || !isFreshTimestamp(record.timestamp)) return null
    if (typeof record.generationConfigRevision !== "string" || !/^sha256:[0-9a-f]{64}$/.test(record.generationConfigRevision)) return null
    if (!validateDraft(record.values, capability)) return null
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

export type UseStandaloneHtmlGenerationOptions = {
  capability: EnabledGenerationCapability
  onCompleted: (presentationId: string) => void
  onStopWaiting: () => void
}

export const useStandaloneHtmlGeneration = ({
  capability,
  onCompleted,
  onStopWaiting
}: UseStandaloneHtmlGenerationOptions) => {
  const [scopeReady, setScopeReady] = React.useState(false)
  const [scopeError, setScopeError] = React.useState<string | null>(null)
  const [draft, setDraft] = React.useState<StandaloneHtmlFormDraft>(DEFAULT_STANDALONE_HTML_FORM_DRAFT)
  const [fieldErrors, setFieldErrors] = React.useState<Partial<Record<keyof StandaloneHtmlFormDraft, string>>>({})
  const [editError, setEditError] = React.useState<string | null>(null)
  const [phase, setPhase] = React.useState<StandaloneHtmlGenerationPhase>("idle")
  const [snapshot, setSnapshot] = React.useState<PresentationGenerationRequest | null>(null)
  const [backendStatus, setBackendStatus] = React.useState<PresentationGenerationReceipt["status"] | null>(null)
  const [progressText, setProgressText] = React.useState<string | null>(null)
  const [safeError, setSafeError] = React.useState<string | null>(null)
  const [recoveryAvailable, setRecoveryAvailable] = React.useState(false)
  const [storageWarning, setStorageWarning] = React.useState<string | null>(null)

  const scopeRef = React.useRef<Scope | null>(null)
  const draftRef = React.useRef(draft)
  const snapshotRef = React.useRef<PresentationGenerationRequest | null>(null)
  const resumeRef = React.useRef<ResumeRecord | null>(null)
  const draftRevisionRef = React.useRef(capability.generation_config_revision)
  const pollTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollAttemptRef = React.useRef(0)
  const waitingRef = React.useRef(false)
  const mountedRef = React.useRef(true)
  const scopeValidationIdRef = React.useRef(0)

  const locked = snapshot !== null && !["rejected", "failed", "cancelled", "completed_missing_binding"].includes(phase)

  const stopPollTimer = React.useCallback(() => {
    waitingRef.current = false
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
    pollTimerRef.current = null
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
    resumeRef.current = null
    setRecoveryAvailable(false)
  }, [])

  const persistDraft = React.useCallback((values: StandaloneHtmlFormDraft, revision: string) => {
    const scope = scopeRef.current
    if (!scope || !validateDraft(values, capability)) return false
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
  }, [capability])

  const persistResume = React.useCallback((record: ResumeRecord) => {
    const scope = scopeRef.current
    if (!scope) return false
    const serialized = JSON.stringify(record)
    if (utf8Bytes(serialized) > MAX_RESUME_RECORD_BYTES) {
      setStorageWarning("Reload recovery is unavailable.")
      return false
    }
    try {
      window.sessionStorage.setItem(buildStandaloneHtmlStorageKeys(scope).resume, serialized)
      resumeRef.current = record
      setRecoveryAvailable(true)
      return true
    } catch {
      setStorageWarning("Reload recovery is unavailable.")
      return false
    }
  }, [])

  const clearSensitiveMemory = React.useCallback(() => {
    stopPollTimer()
    draftRef.current = DEFAULT_STANDALONE_HTML_FORM_DRAFT
    snapshotRef.current = null
    setDraft(DEFAULT_STANDALONE_HTML_FORM_DRAFT)
    setSnapshot(null)
    setBackendStatus(null)
    setProgressText(null)
    setSafeError(null)
    setFieldErrors({})
    setEditError(null)
    setPhase("idle")
  }, [stopPollTimer])

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
    const storedDraft = parseDraftRecord(rawDraft, capability)
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
    const request = freezeRequest(buildRequest(storedDraft.values, storedDraft.generationConfigRevision))
    if (storedResume && digestRequest(request) !== storedResume.requestDigest) {
      try {
        window.sessionStorage.removeItem(keys.draft)
        window.sessionStorage.removeItem(keys.resume)
      } catch {}
      resumeRef.current = null
      setRecoveryAvailable(false)
      return
    }
    draftRevisionRef.current = storedDraft.generationConfigRevision
    draftRef.current = cloneDraft(storedDraft.values)
    setDraft(cloneDraft(storedDraft.values))
    if (storedResume) {
      resumeRef.current = storedResume
      snapshotRef.current = request
      setSnapshot(request)
      setPhase("stopped")
      setRecoveryAvailable(true)
    }
  }, [capability])

  const revalidateScope = React.useCallback(async (clearFirst: boolean) => {
    const validationId = ++scopeValidationIdRef.current
    const previous = scopeRef.current
    if (clearFirst) {
      clearSensitiveMemory()
      setScopeReady(false)
    }
    setScopeError(null)
    const next = await resolveScope()
    if (!mountedRef.current || validationId !== scopeValidationIdRef.current) return
    if (!next) {
      removeRecords(previous)
      scopeRef.current = null
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
  }, [clearSensitiveMemory, hydrateForScope, removeRecords])

  React.useEffect(() => {
    mountedRef.current = true
    void revalidateScope(false)
    return () => {
      mountedRef.current = false
      stopPollTimer()
    }
  }, [revalidateScope, stopPollTimer])

  React.useEffect(() => {
    const restore = () => { void revalidateScope(true) }
    const pagehide = () => {
      const values = draftRef.current
      const request = snapshotRef.current
      persistDraft(values, request?.generation_config_revision ?? draftRevisionRef.current)
      clearSensitiveMemory()
    }
    const visibility = () => {
      if (document.visibilityState === "visible") restore()
    }
    window.addEventListener("tldw:config-updated", restore)
    window.addEventListener("pagehide", pagehide)
    window.addEventListener("pageshow", restore)
    window.addEventListener("focus", restore)
    document.addEventListener("visibilitychange", visibility)
    return () => {
      window.removeEventListener("tldw:config-updated", restore)
      window.removeEventListener("pagehide", pagehide)
      window.removeEventListener("pageshow", restore)
      window.removeEventListener("focus", restore)
      document.removeEventListener("visibilitychange", visibility)
    }
  }, [clearSensitiveMemory, persistDraft, revalidateScope])

  const replaceDraft = React.useCallback((next: StandaloneHtmlFormDraft) => {
    if (!validateDraft(next, capability)) return false
    const accepted = cloneDraft(next)
    draftRef.current = accepted
    setDraft(accepted)
    setFieldErrors({})
    setEditError(null)
    persistDraft(accepted, capability.generation_config_revision)
    return true
  }, [capability, persistDraft])

  const updateField = React.useCallback(<K extends keyof StandaloneHtmlFormDraft>(field: K, value: StandaloneHtmlFormDraft[K]): boolean => {
    const next = { ...draftRef.current, [field]: value } as StandaloneHtmlFormDraft
    let error: string | null = null
    if ((field === "source" || field === "audience") && typeof value === "string") {
      if (containsInvalidScalar(value)) error = `${field === "source" ? "Subject and material" : "Audience"} must contain valid text without NUL characters.`
      const limit = field === "source" ? Math.min(200_000, capability.input_limits.max_source_chars) : Math.min(500, capability.input_limits.max_audience_chars)
      if (!error && value.length > limit) error = `${field === "source" ? "Subject and material" : "Audience"} exceeds the ${limit.toLocaleString()} character limit.`
    }
    if (field === "slideCount" && (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > 30)) error = "Slide count must be an integer from 1 to 30."
    if (!error && !validateDraft(next, capability)) error = "That value is not supported."
    if (error) {
      setEditError(error)
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
    persistDraft(next, capability.generation_config_revision)
    return true
  }, [capability, persistDraft])

  const validateForSubmit = React.useCallback((): boolean => {
    const errors: Partial<Record<keyof StandaloneHtmlFormDraft, string>> = {}
    const current = draftRef.current
    if (!current.source.trim()) errors.source = "Subject and material is required."
    if (!current.audience.trim()) errors.audience = "Audience is required."
    if (!Number.isInteger(current.slideCount) || current.slideCount < 1 || current.slideCount > 30) errors.slideCount = "Slide count must be an integer from 1 to 30."
    setFieldErrors(errors)
    return Object.keys(errors).length === 0
  }, [])

  const finishReceipt = React.useCallback((receipt: PresentationGenerationReceipt): boolean => {
    setBackendStatus(receipt.status)
    setProgressText(
      "progress_text" in receipt && receipt.progress_text
        ? receipt.progress_text.slice(0, MAX_PROGRESS_TEXT_CHARS)
        : null
    )
    if (receipt.status === "completed") {
      if (!receipt.presentation_id) {
        setPhase("completed_missing_binding")
        setSafeError("generation_completed_without_presentation")
        stopPollTimer()
        return true
      }
      setPhase("completed")
      stopPollTimer()
      removeRecords()
      draftRef.current = DEFAULT_STANDALONE_HTML_FORM_DRAFT
      snapshotRef.current = null
      setDraft(DEFAULT_STANDALONE_HTML_FORM_DRAFT)
      setSnapshot(null)
      setProgressText(null)
      setSafeError(null)
      onCompleted(receipt.presentation_id)
      return true
    }
    if (receipt.status === "failed") {
      setPhase("failed")
      setSafeError(receipt.error_code)
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
  }, [onCompleted, removeRecords, stopPollTimer])

  const pollGenerationRef = React.useRef<(generationId: string) => Promise<void>>(async () => undefined)
  const schedulePoll = React.useCallback((generationId: string, retryAfterMs: number | null) => {
    if (!waitingRef.current) return
    const exponential = Math.min(MAX_POLL_DELAY_MS, MIN_POLL_DELAY_MS * (2 ** pollAttemptRef.current))
    pollAttemptRef.current += 1
    const delay = Math.min(MAX_POLL_DELAY_MS, Math.max(MIN_POLL_DELAY_MS, retryAfterMs ?? exponential))
    pollTimerRef.current = setTimeout(() => { void pollGenerationRef.current(generationId) }, delay)
  }, [])

  const pollGeneration = React.useCallback(async (generationId: string) => {
    if (!waitingRef.current) return
    try {
      const result = await tldwClient.getPresentationGenerationStatus(generationId)
      if (!mountedRef.current || !waitingRef.current) return
      if (!finishReceipt(result.receipt)) schedulePoll(generationId, result.retryAfterMs)
    } catch (error) {
      if (!mountedRef.current || !waitingRef.current) return
      const status = errorStatus(error)
      stopPollTimer()
      if (status === 401 || status === 403) setPhase("auth_lost")
      else if (status === 404) setPhase("missing")
      else if (status === 429) setPhase("throttled")
      else setPhase("outage")
      setSafeError(status === 404 ? "generation_not_found" : status === 401 || status === 403 ? "authentication_required" : status === 429 ? "generation_status_throttled" : "generation_status_unavailable")
    }
  }, [finishReceipt, schedulePoll, stopPollTimer])
  pollGenerationRef.current = pollGeneration

  const acceptReceipt = React.useCallback((receipt: PresentationGenerationReceipt, key: string, request: PresentationGenerationRequest) => {
    if (finishReceipt(receipt)) return
    const updated: ResumeRecord = {
      generationId: receipt.generation_id,
      idempotencyKey: key,
      requestDigest: digestRequest(request),
      timestamp: Date.now()
    }
    persistResume(updated)
    waitingRef.current = true
    pollAttemptRef.current = 0
    void pollGeneration(receipt.generation_id)
  }, [finishReceipt, persistResume, pollGeneration])

  const dispatch = React.useCallback(async (request: PresentationGenerationRequest, key: string) => {
    setPhase("submitting")
    setSafeError(null)
    try {
      const receipt = await tldwClient.submitPresentationGeneration(request, { idempotencyKey: key })
      if (!mountedRef.current) return
      acceptReceipt(receipt, key, request)
    } catch (error) {
      if (!mountedRef.current) return
      const status = errorStatus(error)
      if (status !== null && [400, 409, 413, 415, 422].includes(status)) {
        setPhase("rejected")
        setSafeError(safeErrorCode(error, "generation_request_rejected"))
        removeRecords()
        snapshotRef.current = null
        setSnapshot(null)
      } else {
        setPhase(status === 429 ? "throttled" : "ambiguous")
        setSafeError(status === 429 ? "generation_submission_throttled" : "generation_submission_unknown")
        setRecoveryAvailable(true)
      }
    }
  }, [acceptReceipt, removeRecords])

  const submit = React.useCallback(async () => {
    if (!scopeRef.current || snapshotRef.current || !validateForSubmit()) return
    const request = freezeRequest(buildRequest(cloneDraft(draftRef.current), capability.generation_config_revision))
    const key = createIdempotencyKey()
    const resume: ResumeRecord = { generationId: null, idempotencyKey: key, requestDigest: digestRequest(request), timestamp: Date.now() }
    snapshotRef.current = request
    draftRevisionRef.current = request.generation_config_revision
    setSnapshot(request)
    persistDraft(draftRef.current, request.generation_config_revision)
    persistResume(resume)
    await dispatch(request, key)
  }, [capability.generation_config_revision, dispatch, persistDraft, persistResume, validateForSubmit])

  const resume = React.useCallback(async () => {
    const request = snapshotRef.current
    const recovery = resumeRef.current
    if (!request || !recovery || digestRequest(request) !== recovery.requestDigest) {
      removeRecords()
      return
    }
    if (recovery.generationId) {
      setPhase("polling")
      waitingRef.current = true
      pollAttemptRef.current = 0
      await pollGeneration(recovery.generationId)
    } else {
      await dispatch(request, recovery.idempotencyKey)
    }
  }, [dispatch, pollGeneration, removeRecords])

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
    removeRecords()
    snapshotRef.current = null
    setSnapshot(null)
    setBackendStatus(null)
    setProgressText(null)
    setSafeError(null)
    setPhase("idle")
  }, [removeRecords])

  const tryAgain = React.useCallback(async () => {
    snapshotRef.current = null
    setSnapshot(null)
    setPhase("idle")
    await Promise.resolve()
    const request = freezeRequest(buildRequest(cloneDraft(draftRef.current), capability.generation_config_revision))
    const key = createIdempotencyKey()
    const recovery: ResumeRecord = { generationId: null, idempotencyKey: key, requestDigest: digestRequest(request), timestamp: Date.now() }
    snapshotRef.current = request
    setSnapshot(request)
    persistDraft(draftRef.current, request.generation_config_revision)
    persistResume(recovery)
    await dispatch(request, key)
  }, [capability.generation_config_revision, dispatch, persistDraft, persistResume])

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
