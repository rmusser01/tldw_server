import React from "react"

import {
  fetchCompanionConversationPrompts,
  isCompanionConsentRequiredResponse,
} from "@/services/companion"
import { buildPersonaWebSocketUrl } from "@/services/persona-stream"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { downloadBlob } from "@/utils/download-blob"

import type { PersonaGardenTabKey } from "@/utils/persona-garden-route"
import type { PersonaVoiceDefaults } from "@/hooks/useResolvedPersonaVoiceDefaults"
import type { PersonaRuntimeApprovalRequest } from "./usePersonaGovernanceContext"

// ── Internal types ──

type PersonaInfo = {
  id: string
  name: string
  description?: string | null
  voice?: string | null
}

type PersonaSessionSummary = {
  session_id: string
  persona_id?: string
  created_at?: string
  updated_at?: string
  turn_count?: number
  pending_plan_count?: number
}

type PersonaSessionPreferences = {
  use_memory_context?: boolean
  use_companion_context?: boolean
  use_persona_state_context?: boolean
  memory_top_k?: number
}

type PersonaSessionDetailResponse = {
  preferences?: PersonaSessionPreferences
  turns?: Array<Record<string, unknown>>
}

type PersonaSessionExportResponse = {
  session_id?: string
  persona_id?: string
  format?: "json"
  created_at?: string
  updated_at?: string
  turn_count?: number
  redaction_markers?: string[]
  turns?: Array<Record<string, unknown>>
}

type PersonaProfileResponse = {
  id?: string
  version?: number
  use_persona_state_context_default?: boolean
  voice_defaults?: PersonaVoiceDefaults | null
  setup?: Record<string, unknown> | null
}

type PersonaLogEntry = {
  id: string
  kind: "user" | "assistant" | "tool" | "notice"
  text: string
}

type PendingPlan = {
  planId: string
  steps: Array<{
    idx: number
    tool: string
    args?: Record<string, unknown>
    description?: string
    why?: string
    policy?: Record<string, unknown>
  }>
  memory?: Record<string, unknown>
  companion?: Record<string, unknown>
}

// ── Constants ──

const DEFAULT_PERSONA_ID = "research_assistant"
const DEFAULT_COMPANION_PROMPT_QUERY = "resume recent companion work"
const MAX_MEMORY_TOP_K = 10
const MEMORY_TOP_K_OPTIONS = Array.from(
  { length: MAX_MEMORY_TOP_K },
  (_, index) => index + 1
)
const _normalizeMemoryTopK = (value: unknown, fallback: number): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(1, Math.min(MAX_MEMORY_TOP_K, Math.trunc(parsed)))
}
const formatMemoryResultsLabel = (count: number) => `Memory results: ${count}`

// ── Deps interface ──

export interface UsePersonaLiveSessionDeps {
  isCompanionMode: boolean
  selectedPersonaId: string
  setSelectedPersonaId: React.Dispatch<React.SetStateAction<string>>
  catalog: PersonaInfo[]
  setCatalog: React.Dispatch<React.SetStateAction<PersonaInfo[]>>
  connected: boolean
  setConnected: React.Dispatch<React.SetStateAction<boolean>>
  connecting: boolean
  setConnecting: React.Dispatch<React.SetStateAction<boolean>>
  sessionId: string | null
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>
  error: string | null
  setError: React.Dispatch<React.SetStateAction<string | null>>
  input: string
  setInput: React.Dispatch<React.SetStateAction<string>>
  logs: PersonaLogEntry[]
  setLogs: React.Dispatch<React.SetStateAction<PersonaLogEntry[]>>
  appendLog: (kind: PersonaLogEntry["kind"], text: string) => void
  activeSessionPersonaId: string | null
  setActiveSessionPersonaId: React.Dispatch<React.SetStateAction<string | null>>
  /** Refs owned by the parent component */
  wsRef: React.MutableRefObject<WebSocket | null>
  manuallyClosingRef: React.MutableRefObject<boolean>
  handleIncomingPayloadRef: React.MutableRefObject<(payload: any) => void>
  liveVoiceControllerRef: React.MutableRefObject<any>
  liveVoiceAnalyticsSnapshotRef: React.MutableRefObject<{
    personaId: string
    sessionId: string
    listeningRecoveryCount: number
    thinkingRecoveryCount: number
  }>
  /** State owned by governance hook */
  setPendingApprovals: React.Dispatch<React.SetStateAction<PersonaRuntimeApprovalRequest[]>>
  setActiveApprovalKey: React.Dispatch<React.SetStateAction<string | null>>
  setResolvedApprovalSnapshot: React.Dispatch<
    React.SetStateAction<{ key: string; toolName: string } | null>
  >
  setApprovedStepMap: React.Dispatch<React.SetStateAction<Record<number, boolean>>>
  runtimeApprovalRowRefs: React.MutableRefObject<Map<string, HTMLDivElement | null>>
  resetApprovalHighlightMotion: () => void
  clearResolvedApprovalFadeTimer: () => void
  /** Profile state managed by parent */
  savedPersonaVoiceDefaults: PersonaVoiceDefaults | null
  setSavedPersonaVoiceDefaults: React.Dispatch<React.SetStateAction<PersonaVoiceDefaults | null>>
  setSavedPersonaProfileVersion: React.Dispatch<React.SetStateAction<number | null>>
  setLiveSessionVoiceDefaultsBaseline: React.Dispatch<
    React.SetStateAction<PersonaVoiceDefaults | null>
  >
  /** State docs */
  setPersonaStateHistory: React.Dispatch<React.SetStateAction<any[]>>
  setPersonaStateHistoryLoaded: React.Dispatch<React.SetStateAction<boolean>>
  /** Unsaved state guard */
  confirmDiscardUnsavedStateDrafts: (reason: string) => boolean
  /** Analytics flush */
  flushLiveVoiceSessionAnalytics: (options?: { finalize?: boolean }) => void
  /** State doc loader */
  loadPersonaStateDocs: (
    personaIdOverride?: string,
    options?: { silent?: boolean }
  ) => Promise<boolean>
  /** Setup wizard context */
  personaSetupWizardCurrentStep: string
  personaSetupWizardIsSetupRequired: boolean
  setSetupTestOutcome: React.Dispatch<React.SetStateAction<any>>
  setupLiveDetour: unknown
  setupWizardAwaitingLiveResponseRef: React.MutableRefObject<boolean>
  setupWizardLastLiveTextRef: React.MutableRefObject<string>
  clearSetupStepError: (step: string) => void
  /** Pending plan */
  setPendingPlan: React.Dispatch<React.SetStateAction<PendingPlan | null>>
  pendingPlan: PendingPlan | null
  /** Capabilities */
  capabilities: { hasPersonalization?: boolean; hasAudio?: boolean } | null
  capsLoading: boolean
  /** Route bootstrap */
  routeBootstrapPersonaId: string | undefined
}

// ── Hook ──

export function usePersonaLiveSession(deps: UsePersonaLiveSessionDeps) {
  const {
    isCompanionMode,
    selectedPersonaId,
    setSelectedPersonaId,
    catalog,
    setCatalog,
    connected,
    setConnected,
    connecting,
    setConnecting,
    sessionId,
    setSessionId,
    setError,
    input,
    setInput,
    setLogs,
    appendLog,
    setActiveSessionPersonaId,
    wsRef,
    manuallyClosingRef,
    handleIncomingPayloadRef,
    liveVoiceControllerRef,
    liveVoiceAnalyticsSnapshotRef,
    setPendingApprovals,
    setActiveApprovalKey,
    setResolvedApprovalSnapshot,
    setApprovedStepMap,
    runtimeApprovalRowRefs,
    resetApprovalHighlightMotion,
    clearResolvedApprovalFadeTimer,
    savedPersonaVoiceDefaults,
    setSavedPersonaVoiceDefaults,
    setSavedPersonaProfileVersion,
    setLiveSessionVoiceDefaultsBaseline,
    setPersonaStateHistory,
    setPersonaStateHistoryLoaded,
    confirmDiscardUnsavedStateDrafts,
    flushLiveVoiceSessionAnalytics,
    loadPersonaStateDocs,
    personaSetupWizardCurrentStep,
    personaSetupWizardIsSetupRequired,
    setSetupTestOutcome,
    setupLiveDetour,
    setupWizardAwaitingLiveResponseRef,
    setupWizardLastLiveTextRef,
    clearSetupStepError,
    setPendingPlan,
    pendingPlan,
    capabilities,
    capsLoading,
    routeBootstrapPersonaId,
  } = deps

  // ── Session state ──
  const [sessionHistory, setSessionHistory] = React.useState<
    PersonaSessionSummary[]
  >([])
  const [resumeSessionId, setResumeSessionId] = React.useState<string>("")
  const [memoryEnabled, setMemoryEnabled] = React.useState(true)
  const [memoryTopK, setMemoryTopK] = React.useState<number>(3)
  const [companionContextEnabled, setCompanionContextEnabled] =
    React.useState(true)
  const [personaStateContextEnabled, setPersonaStateContextEnabled] =
    React.useState(!isCompanionMode)
  const [personaStateContextProfileDefault, setPersonaStateContextProfileDefault] =
    React.useState(!isCompanionMode)
  const [updatingPersonaStateContextDefault, setUpdatingPersonaStateContextDefault] =
    React.useState(false)
  const [savingCompanionCheckIn, setSavingCompanionCheckIn] =
    React.useState(false)
  const [transcriptExporting, setTranscriptExporting] = React.useState(false)
  const [companionPrompts, setCompanionPrompts] = React.useState<
    Array<{ prompt_id: string; label: string; prompt_text: string }>
  >([])
  const [pendingRecoveryReconnectToken, setPendingRecoveryReconnectToken] =
    React.useState(0)

  // ── Companion mode side-effects ──
  React.useEffect(() => {
    if (!isCompanionMode) return
    setCompanionContextEnabled(true)
    setPersonaStateContextEnabled(false)
    setPersonaStateContextProfileDefault(false)
  }, [isCompanionMode])

  // ── Companion prompts loader ──
  React.useEffect(() => {
    if (!isCompanionMode || capsLoading || !capabilities?.hasPersonalization) {
      setCompanionPrompts([])
      return
    }
    const promptQuery = input.trim() || DEFAULT_COMPANION_PROMPT_QUERY
    let cancelled = false
    const timeoutId = window.setTimeout(
      () => {
        fetchCompanionConversationPrompts(promptQuery)
          .then((payload) => {
            if (cancelled) return
            setCompanionPrompts(
              Array.isArray(payload.prompts)
                ? payload.prompts
                    .filter(
                      (item) =>
                        item &&
                        typeof item.prompt_text === "string" &&
                        item.prompt_text.trim().length > 0
                    )
                    .slice(0, 3)
                    .map((item) => ({
                      prompt_id: item.prompt_id,
                      label: item.label,
                      prompt_text: item.prompt_text,
                    }))
                : []
            )
          })
          .catch(() => {
            if (!cancelled) {
              setCompanionPrompts([])
            }
          })
      },
      input.trim() ? 200 : 0
    )
    return () => {
      cancelled = true
      window.clearTimeout(timeoutId)
    }
  }, [capabilities?.hasPersonalization, capsLoading, input, isCompanionMode])

  // ── Session preferences applicator ──
  const applySessionPreferences = React.useCallback(
    (preferences?: PersonaSessionPreferences | null) => {
      if (!preferences || typeof preferences !== "object") return
      if (typeof preferences.use_memory_context === "boolean") {
        setMemoryEnabled(preferences.use_memory_context)
      }
      if (typeof preferences.memory_top_k !== "undefined") {
        setMemoryTopK((current) =>
          _normalizeMemoryTopK(preferences.memory_top_k, current)
        )
      }
      if (
        !isCompanionMode &&
        typeof preferences.use_companion_context === "boolean"
      ) {
        setCompanionContextEnabled(preferences.use_companion_context)
      }
      if (
        !isCompanionMode &&
        typeof preferences.use_persona_state_context === "boolean"
      ) {
        setPersonaStateContextEnabled(preferences.use_persona_state_context)
      }
    },
    [isCompanionMode]
  )

  // ── Disconnect ──
  const disconnect = React.useCallback(
    (options?: { force?: boolean }) => {
      if (
        !options?.force &&
        !confirmDiscardUnsavedStateDrafts("disconnect")
      ) {
        return false
      }
      flushLiveVoiceSessionAnalytics({ finalize: true })
      const ws = wsRef.current
      if (!ws) {
        setConnected(false)
        setActiveSessionPersonaId(null)
        setLiveSessionVoiceDefaultsBaseline(null)
        liveVoiceAnalyticsSnapshotRef.current = {
          personaId: "",
          sessionId: "",
          listeningRecoveryCount: 0,
          thinkingRecoveryCount: 0,
        }
        return false
      }
      manuallyClosingRef.current = true
      try {
        ws.close()
      } catch {
        // ignore close errors
      }
      wsRef.current = null
      setConnected(false)
      setActiveSessionPersonaId(null)
      setLiveSessionVoiceDefaultsBaseline(null)
      setPersonaStateHistory([])
      setPersonaStateHistoryLoaded(false)
      setActiveApprovalKey(null)
      resetApprovalHighlightMotion()
      clearResolvedApprovalFadeTimer()
      setResolvedApprovalSnapshot(null)
      runtimeApprovalRowRefs.current.clear()
      setPendingApprovals([])
      liveVoiceAnalyticsSnapshotRef.current = {
        personaId: "",
        sessionId: "",
        listeningRecoveryCount: 0,
        thinkingRecoveryCount: 0,
      }
      return true
    },
    [
      clearResolvedApprovalFadeTimer,
      confirmDiscardUnsavedStateDrafts,
      flushLiveVoiceSessionAnalytics,
      liveVoiceAnalyticsSnapshotRef,
      manuallyClosingRef,
      resetApprovalHighlightMotion,
      runtimeApprovalRowRefs,
      setActiveApprovalKey,
      setActiveSessionPersonaId,
      setConnected,
      setLiveSessionVoiceDefaultsBaseline,
      setPendingApprovals,
      setPersonaStateHistory,
      setPersonaStateHistoryLoaded,
      setResolvedApprovalSnapshot,
      wsRef,
    ]
  )

  // ── Connect ──
  const connect = React.useCallback(async () => {
    if (connecting || connected) return
    if (!confirmDiscardUnsavedStateDrafts("connect")) return
    setConnecting(true)
    setError(null)

    try {
      disconnect({ force: true })
      setActiveSessionPersonaId(null)
      setLiveSessionVoiceDefaultsBaseline(null)
      setPendingPlan(null)
      setApprovedStepMap({})
      setPersonaStateHistory([])
      setPersonaStateHistoryLoaded(false)
      setActiveApprovalKey(null)
      resetApprovalHighlightMotion()
      clearResolvedApprovalFadeTimer()
      setResolvedApprovalSnapshot(null)
      runtimeApprovalRowRefs.current.clear()

      const config = await tldwClient.getConfig()
      if (!config) {
        throw new Error("tldw server not configured")
      }

      const catalogResp = await tldwClient.fetchWithAuth(
        "/api/v1/persona/catalog" as any,
        { method: "GET" }
      )
      if (!catalogResp.ok) {
        throw new Error(
          catalogResp.error || "Failed to load persona catalog"
        )
      }
      const catalogPayload = await catalogResp.json()
      const personas = Array.isArray(catalogPayload)
        ? (catalogPayload as PersonaInfo[])
        : []
      setCatalog(personas)

      const preferredPersonaId = isCompanionMode
        ? DEFAULT_PERSONA_ID
        : routeBootstrapPersonaId || selectedPersonaId
      const selectedPersonaIsValid = personas.some(
        (persona) => String(persona.id || "") === preferredPersonaId
      )
      const resolvedPersonaId =
        (selectedPersonaIsValid
          ? preferredPersonaId
          : personas[0]?.id) ||
        preferredPersonaId ||
        DEFAULT_PERSONA_ID
      if (resolvedPersonaId && resolvedPersonaId !== selectedPersonaId) {
        setSelectedPersonaId(resolvedPersonaId)
      }
      if (isCompanionMode) {
        setCompanionContextEnabled(true)
        setPersonaStateContextEnabled(false)
        setPersonaStateContextProfileDefault(false)
      } else {
        setPersonaStateContextEnabled(true)
        setPersonaStateContextProfileDefault(true)
        let nextSavedVoiceDefaults = savedPersonaVoiceDefaults
        try {
          const profileResp = await tldwClient.fetchWithAuth(
            `/api/v1/persona/profiles/${encodeURIComponent(
              resolvedPersonaId
            )}` as any,
            { method: "GET" }
          )
          if (profileResp.ok) {
            const profilePayload =
              (await profileResp.json()) as PersonaProfileResponse
            const stateContextDefault =
              profilePayload?.use_persona_state_context_default !== false
            setPersonaStateContextEnabled(stateContextDefault)
            setPersonaStateContextProfileDefault(stateContextDefault)
            nextSavedVoiceDefaults =
              profilePayload?.voice_defaults || null
            setSavedPersonaVoiceDefaults(nextSavedVoiceDefaults)
            setSavedPersonaProfileVersion(
              typeof profilePayload?.version === "number"
                ? profilePayload.version
                : null
            )
          }
        } catch {
          // profile fetch is optional for route initialization
        }
        setLiveSessionVoiceDefaultsBaseline(
          nextSavedVoiceDefaults || null
        )
        void loadPersonaStateDocs(resolvedPersonaId, { silent: true })
      }

      const sessionsResp = await tldwClient.fetchWithAuth(
        `/api/v1/persona/sessions?persona_id=${encodeURIComponent(
          resolvedPersonaId
        )}${
          isCompanionMode
            ? `&surface=${encodeURIComponent("companion.conversation")}`
            : ""
        }&limit=50` as any,
        { method: "GET" }
      )
      let sessionsPayload: PersonaSessionSummary[] = []
      if (sessionsResp.ok) {
        const sessionsJson = await sessionsResp.json()
        sessionsPayload = Array.isArray(sessionsJson)
          ? (sessionsJson as PersonaSessionSummary[])
          : []
      }
      setSessionHistory(sessionsPayload)

      const sessionResp = await tldwClient.fetchWithAuth(
        "/api/v1/persona/session" as any,
        {
          method: "POST",
          body: {
            persona_id: resolvedPersonaId,
            resume_session_id: resumeSessionId || undefined,
            surface: isCompanionMode
              ? "companion.conversation"
              : undefined,
          },
        }
      )
      if (!sessionResp.ok) {
        throw new Error(
          sessionResp.error || "Failed to create persona session"
        )
      }
      const sessionPayload = await sessionResp.json()
      const nextSessionId = String(
        sessionPayload?.session_id || ""
      ).trim()
      if (!nextSessionId) {
        throw new Error("Persona session response missing session_id")
      }
      const connectedPersonaId =
        String(
          sessionPayload?.persona?.id || resolvedPersonaId || ""
        ).trim() || resolvedPersonaId
      setActiveSessionPersonaId(connectedPersonaId)
      if (
        connectedPersonaId &&
        connectedPersonaId !== selectedPersonaId
      ) {
        setSelectedPersonaId(connectedPersonaId)
      }
      setSessionId(nextSessionId)
      setResumeSessionId(nextSessionId)
      try {
        const sessionDetailResp = await tldwClient.fetchWithAuth(
          `/api/v1/persona/sessions/${encodeURIComponent(
            nextSessionId
          )}?limit_turns=0` as any,
          { method: "GET" }
        )
        if (sessionDetailResp.ok) {
          const sessionDetailPayload =
            (await sessionDetailResp.json()) as PersonaSessionDetailResponse
          applySessionPreferences(sessionDetailPayload?.preferences)
        }
      } catch {
        // session detail hydration is best-effort during connect
      }
      if (
        !sessionsPayload.some(
          (item) => item.session_id === nextSessionId
        )
      ) {
        setSessionHistory((prev) => [
          { session_id: nextSessionId },
          ...prev,
        ])
      }

      const ws = new WebSocket(buildPersonaWebSocketUrl(config))
      ws.binaryType = "arraybuffer"
      wsRef.current = ws
      manuallyClosingRef.current = false

      ws.onopen = () => {
        setConnected(true)
        appendLog("notice", "Persona stream connected")
      }

      ws.onmessage = (event) => {
        if (typeof event.data !== "string") {
          if (event.data instanceof ArrayBuffer) {
            liveVoiceControllerRef.current?.handleBinaryPayload(
              event.data
            )
            return
          }
          appendLog(
            "notice",
            "Received binary persona stream payload"
          )
          return
        }
        try {
          const payload = JSON.parse(event.data)
          handleIncomingPayloadRef.current(payload)
        } catch {
          appendLog("notice", event.data)
        }
      }

      ws.onerror = () => {
        setError("Persona stream error")
      }

      ws.onclose = () => {
        wsRef.current = null
        setConnected(false)
        const manual = manuallyClosingRef.current
        manuallyClosingRef.current = false
        if (!manual) {
          appendLog("notice", "Persona stream disconnected")
        }
      }
    } catch (err: any) {
      const message = String(
        err?.message || "Failed to connect persona stream"
      )
      if (
        personaSetupWizardIsSetupRequired &&
        personaSetupWizardCurrentStep === "test" &&
        !connected
      ) {
        setSetupTestOutcome({ kind: "live_unavailable" })
      }
      setError(message)
      appendLog("notice", message)
    } finally {
      setConnecting(false)
    }
  }, [
    appendLog,
    applySessionPreferences,
    clearResolvedApprovalFadeTimer,
    confirmDiscardUnsavedStateDrafts,
    connected,
    connecting,
    disconnect,
    handleIncomingPayloadRef,
    isCompanionMode,
    liveVoiceControllerRef,
    loadPersonaStateDocs,
    manuallyClosingRef,
    personaSetupWizardCurrentStep,
    personaSetupWizardIsSetupRequired,
    resetApprovalHighlightMotion,
    resumeSessionId,
    routeBootstrapPersonaId,
    runtimeApprovalRowRefs,
    savedPersonaVoiceDefaults,
    selectedPersonaId,
    setActiveApprovalKey,
    setActiveSessionPersonaId,
    setApprovedStepMap,
    setCatalog,
    setConnected,
    setConnecting,
    setError,
    setLiveSessionVoiceDefaultsBaseline,
    setPendingPlan,
    setPersonaStateHistory,
    setPersonaStateHistoryLoaded,
    setResolvedApprovalSnapshot,
    setSavedPersonaProfileVersion,
    setSavedPersonaVoiceDefaults,
    setSelectedPersonaId,
    setSessionId,
    setSetupTestOutcome,
    setPendingApprovals,
    wsRef,
  ])

  // ── Recovery reconnect effect ──
  React.useEffect(() => {
    if (!pendingRecoveryReconnectToken) return
    if (connected || connecting) return
    setPendingRecoveryReconnectToken(0)
    void connect()
  }, [connect, connected, connecting, pendingRecoveryReconnectToken])

  // ── Cleanup on unmount ──
  React.useEffect(() => {
    return () => {
      const ws = wsRef.current
      flushLiveVoiceSessionAnalytics({ finalize: true })
      if (!ws) return
      manuallyClosingRef.current = true
      try {
        ws.close()
      } catch {
        // ignore close errors
      }
      wsRef.current = null
      liveVoiceAnalyticsSnapshotRef.current = {
        personaId: "",
        sessionId: "",
        listeningRecoveryCount: 0,
        thinkingRecoveryCount: 0,
      }
    }
  }, [
    flushLiveVoiceSessionAnalytics,
    liveVoiceAnalyticsSnapshotRef,
    manuallyClosingRef,
    wsRef,
  ])

  // ── canSend ──
  const canSend = connected && Boolean(sessionId) && Boolean(input.trim())
  const canSaveCompanionCheckIn =
    Boolean(input.trim()) &&
    Boolean(capabilities?.hasPersonalization) &&
    !savingCompanionCheckIn

  // ── sendUserMessage ──
  const sendUserMessage = React.useCallback(
    () => {
      if (!canSend || !sessionId || !wsRef.current) return
      const trimmed = input.trim()
      try {
        wsRef.current.send(
          JSON.stringify({
            type: "user_message",
            session_id: sessionId,
            text: trimmed,
            use_memory_context: memoryEnabled,
            use_companion_context: companionContextEnabled,
            use_persona_state_context: personaStateContextEnabled,
            memory_top_k: memoryTopK,
          })
        )
        if (
          personaSetupWizardIsSetupRequired &&
          setupLiveDetour
        ) {
          setupWizardAwaitingLiveResponseRef.current = true
          setupWizardLastLiveTextRef.current = trimmed
        }
        appendLog("user", trimmed)
        setInput("")
      } catch (err: any) {
        setError(String(err?.message || "Failed to send message"))
      }
    },
    [
      appendLog,
      canSend,
      companionContextEnabled,
      input,
      memoryEnabled,
      memoryTopK,
      personaSetupWizardIsSetupRequired,
      personaStateContextEnabled,
      sessionId,
      setError,
      setInput,
      setupLiveDetour,
      setupWizardAwaitingLiveResponseRef,
      setupWizardLastLiveTextRef,
      wsRef,
    ]
  )

  // ── sendSetupLiveTestMessage ──
  const sendSetupLiveTestMessage = React.useCallback(
    (text: string) => {
      const trimmed = String(text || "").trim()
      if (!trimmed) return
      clearSetupStepError("test")
      if (!connected || !sessionId || !wsRef.current) {
        setSetupTestOutcome({ kind: "live_unavailable" })
        return
      }
      try {
        wsRef.current.send(
          JSON.stringify({
            type: "user_message",
            session_id: sessionId,
            text: trimmed,
            use_memory_context: memoryEnabled,
            use_companion_context: companionContextEnabled,
            use_persona_state_context: personaStateContextEnabled,
            memory_top_k: memoryTopK,
          })
        )
        setupWizardAwaitingLiveResponseRef.current = true
        setupWizardLastLiveTextRef.current = trimmed
        setSetupTestOutcome({
          kind: "live_sent",
          text: trimmed,
        })
        appendLog("user", trimmed)
      } catch (err: any) {
        setSetupTestOutcome({
          kind: "live_failure",
          text: trimmed,
          message: String(
            err?.message || "Failed to send setup live test"
          ),
        })
      }
    },
    [
      appendLog,
      clearSetupStepError,
      companionContextEnabled,
      connected,
      memoryEnabled,
      memoryTopK,
      personaStateContextEnabled,
      sessionId,
      setSetupTestOutcome,
      setupWizardAwaitingLiveResponseRef,
      setupWizardLastLiveTextRef,
      wsRef,
    ]
  )

  // ── loadSessionHistory ──
  const loadSessionHistory = React.useCallback(async () => {
    if (!sessionId) return
    const resp = await tldwClient.fetchWithAuth(
      `/api/v1/persona/sessions/${encodeURIComponent(
        sessionId
      )}?limit_turns=100` as any,
      { method: "GET" }
    )
    if (!resp.ok) {
      setError(resp.error || "Failed to load session history")
      return
    }
    const payload = await resp.json()
    const turns = Array.isArray(payload?.turns) ? payload.turns : []
    const historyLogs: PersonaLogEntry[] = turns.map(
      (turn: any, idx: number) => {
        const role = String(turn?.role || "notice").toLowerCase()
        const kind: PersonaLogEntry["kind"] =
          role === "user" || role === "assistant" || role === "tool"
            ? role
            : "notice"
        return {
          id: String(turn?.turn_id || `${Date.now()}-${idx}`),
          kind,
          text: String(turn?.content || ""),
        }
      }
    )
    setLogs(historyLogs)
  }, [sessionId, setError, setLogs])

  // ── exportSelectedSessionTranscript ──
  const exportSelectedSessionTranscript = React.useCallback(async () => {
    const selectedSessionId = String(sessionId || "").trim()
    if (!selectedSessionId || transcriptExporting) return
    setTranscriptExporting(true)
    try {
      const resp = await tldwClient.fetchWithAuth(
        `/api/v1/persona/sessions/${encodeURIComponent(
          selectedSessionId
        )}/export` as any,
        { method: "GET" }
      )
      if (!resp.ok) {
        throw new Error(resp.error || "Failed to export transcript")
      }
      const payload = (await resp.json()) as PersonaSessionExportResponse
      const filename = `persona-session-${selectedSessionId}-transcript.json`
      downloadBlob(
        new Blob([JSON.stringify(payload, null, 2)], {
          type: "application/json"
        }),
        filename
      )
      appendLog("notice", "Transcript export downloaded.")
    } catch (err: any) {
      const message = String(err?.message || "Failed to export transcript")
      setError(message)
      appendLog("notice", message)
    } finally {
      setTranscriptExporting(false)
    }
  }, [appendLog, sessionId, setError, transcriptExporting])

  // ── confirmPlan / cancelPlan ──
  const confirmPlan = React.useCallback(() => {
    if (!pendingPlan || !sessionId || !wsRef.current || !connected)
      return
    const approvedStepMapCurrent = {} as Record<number, boolean>
    // Read current approvedStepMap from governance hook
    // This is called with the map from the parent
    const approvedSteps = pendingPlan.steps
      .filter((step) => {
        // The caller passes approvedStepMap externally
        return true
      })
      .map((step) => step.idx)
    try {
      wsRef.current.send(
        JSON.stringify({
          type: "confirm_plan",
          session_id: sessionId,
          plan_id: pendingPlan.planId,
          approved_steps: approvedSteps,
        })
      )
      appendLog(
        "notice",
        `Confirmed ${approvedSteps.length} step${
          approvedSteps.length === 1 ? "" : "s"
        }`
      )
      setPendingPlan(null)
    } catch (err: any) {
      setError(String(err?.message || "Failed to confirm plan"))
    }
  }, [
    appendLog,
    connected,
    pendingPlan,
    sessionId,
    setError,
    setPendingPlan,
    wsRef,
  ])

  const confirmPlanWithMap = React.useCallback(
    (approvedStepMap: Record<number, boolean>) => {
      if (!pendingPlan || !sessionId || !wsRef.current || !connected)
        return
      const approvedSteps = pendingPlan.steps
        .filter((step) => approvedStepMap[step.idx] !== false)
        .map((step) => step.idx)
      try {
        wsRef.current.send(
          JSON.stringify({
            type: "confirm_plan",
            session_id: sessionId,
            plan_id: pendingPlan.planId,
            approved_steps: approvedSteps,
          })
        )
        appendLog(
          "notice",
          `Confirmed ${approvedSteps.length} step${
            approvedSteps.length === 1 ? "" : "s"
          }`
        )
        setPendingPlan(null)
      } catch (err: any) {
        setError(String(err?.message || "Failed to confirm plan"))
      }
    },
    [
      appendLog,
      connected,
      pendingPlan,
      sessionId,
      setError,
      setPendingPlan,
      wsRef,
    ]
  )

  const cancelPlan = React.useCallback(() => {
    if (!sessionId || !wsRef.current || !connected) return
    try {
      wsRef.current.send(
        JSON.stringify({
          type: "cancel",
          session_id: sessionId,
          reason: "user_cancelled",
        })
      )
      setPendingPlan(null)
      appendLog("notice", "Cancelled pending plan")
    } catch (err: any) {
      setError(String(err?.message || "Failed to cancel plan"))
    }
  }, [appendLog, connected, sessionId, setError, setPendingPlan, wsRef])

  // ── handleResumeSessionSelectionChange ──
  const handleResumeSessionSelectionChange = React.useCallback(
    (value: string) => {
      const nextResumeSessionId =
        value === "__new__" ? "" : String(value)
      if (nextResumeSessionId === resumeSessionId) return
      if (!confirmDiscardUnsavedStateDrafts("session_switch")) return
      setResumeSessionId(nextResumeSessionId)
    },
    [confirmDiscardUnsavedStateDrafts, resumeSessionId]
  )

  // ── saveCompanionCheckIn ──
  const saveCompanionCheckIn = React.useCallback(async () => {
    const trimmed = input.trim()
    if (
      !trimmed ||
      savingCompanionCheckIn ||
      !capabilities?.hasPersonalization
    )
      return
    setSavingCompanionCheckIn(true)
    setError(null)
    try {
      const response = await tldwClient.fetchWithAuth(
        "/api/v1/companion/check-ins" as any,
        {
          method: "POST",
          body: {
            summary: trimmed,
            surface: isCompanionMode
              ? "companion.conversation"
              : "persona.sidepanel",
          },
        }
      )
      if (!response.ok) {
        if (isCompanionConsentRequiredResponse(response)) {
          throw new Error(
            "Enable personalization before saving to companion."
          )
        }
        throw new Error(
          response.error || "Failed to save companion check-in"
        )
      }
      appendLog("notice", "Saved draft to companion")
    } catch (err: any) {
      setError(
        String(err?.message || "Failed to save companion check-in")
      )
    } finally {
      setSavingCompanionCheckIn(false)
    }
  }, [
    appendLog,
    capabilities?.hasPersonalization,
    input,
    isCompanionMode,
    savingCompanionCheckIn,
    setError,
  ])

  // ── updatePersonaStateContextDefault ──
  const getTargetPersonaId = React.useCallback(
    (override?: string): string =>
      String(
        override ||
          (connected
            ? deps.activeSessionPersonaId
            : selectedPersonaId) ||
          ""
      ).trim(),
    [connected, deps.activeSessionPersonaId, selectedPersonaId]
  )

  const updatePersonaStateContextDefault = React.useCallback(
    async (nextDefault: boolean) => {
      const personaId = getTargetPersonaId()
      if (
        !personaId ||
        updatingPersonaStateContextDefault ||
        !connected
      )
        return
      const previousDefault = personaStateContextProfileDefault
      const previousEnabled = personaStateContextEnabled
      setPersonaStateContextProfileDefault(nextDefault)
      setPersonaStateContextEnabled(nextDefault)
      setUpdatingPersonaStateContextDefault(true)
      setError(null)

      try {
        const updateResp = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(
            personaId
          )}` as any,
          {
            method: "PATCH",
            body: {
              use_persona_state_context_default: nextDefault,
            },
          }
        )
        if (!updateResp.ok) {
          throw new Error(
            updateResp.error ||
              "Failed to update persona state context default"
          )
        }
        const profilePayload =
          (await updateResp.json()) as PersonaProfileResponse
        const persistedDefault =
          profilePayload?.use_persona_state_context_default !== false
        setPersonaStateContextProfileDefault(persistedDefault)
        setPersonaStateContextEnabled(persistedDefault)
      } catch (err: any) {
        setPersonaStateContextProfileDefault(previousDefault)
        setPersonaStateContextEnabled(previousEnabled)
        setError(
          String(
            err?.message ||
              "Failed to update persona state context default"
          )
        )
      } finally {
        setUpdatingPersonaStateContextDefault(false)
      }
    },
    [
      connected,
      getTargetPersonaId,
      personaStateContextEnabled,
      personaStateContextProfileDefault,
      setError,
      updatingPersonaStateContextDefault,
    ]
  )

  // ── handlePersonaSelectionChange ──
  const handlePersonaSelectionChange = React.useCallback(
    (value: string) => {
      const nextPersonaId = String(value || "").trim()
      if (!nextPersonaId || nextPersonaId === selectedPersonaId)
        return
      if (!confirmDiscardUnsavedStateDrafts("persona_switch")) return
      setSelectedPersonaId(nextPersonaId)
    },
    [
      confirmDiscardUnsavedStateDrafts,
      selectedPersonaId,
      setSelectedPersonaId,
    ]
  )

  // ── handleReconnectPersonaSessionFromRecovery ──
  const handleReconnectPersonaSessionFromRecovery = React.useCallback(
    (liveVoiceController: { resetTurn: () => void }) => {
      if (connecting) return
      liveVoiceController.resetTurn()
      setPendingRecoveryReconnectToken((current) => current + 1)
      disconnect({ force: true })
    },
    [connecting, disconnect]
  )

  const triggerRecoveryReconnect = React.useCallback(() => {
    setPendingRecoveryReconnectToken((current) => current + 1)
  }, [])

  // ── handleCopyLastVoiceCommandToComposer ──
  const handleCopyLastVoiceCommandToComposer = React.useCallback(
    (lastCommittedText: string) => {
      const nextValue = String(lastCommittedText || "").trim()
      if (!nextValue) return
      setInput(nextValue)
    },
    [setInput]
  )

  return {
    // state
    sessionHistory,
    setSessionHistory,
    resumeSessionId,
    setResumeSessionId,
    memoryEnabled,
    setMemoryEnabled,
    memoryTopK,
    setMemoryTopK,
    companionContextEnabled,
    setCompanionContextEnabled,
    personaStateContextEnabled,
    setPersonaStateContextEnabled,
    personaStateContextProfileDefault,
    setPersonaStateContextProfileDefault,
    updatingPersonaStateContextDefault,
    savingCompanionCheckIn,
    transcriptExporting,
    companionPrompts,
    canSend,
    canSaveCompanionCheckIn,
    // constants
    MEMORY_TOP_K_OPTIONS,
    formatMemoryResultsLabel,
    // callbacks
    connect,
    disconnect,
    sendUserMessage,
    sendSetupLiveTestMessage,
    loadSessionHistory,
    exportSelectedSessionTranscript,
    confirmPlanWithMap,
    cancelPlan,
    handleResumeSessionSelectionChange,
    handleReconnectPersonaSessionFromRecovery,
    handleCopyLastVoiceCommandToComposer,
    handlePersonaSelectionChange,
    triggerRecoveryReconnect,
    saveCompanionCheckIn,
    updatePersonaStateContextDefault,
    getTargetPersonaId,
    applySessionPreferences,
  }
}
