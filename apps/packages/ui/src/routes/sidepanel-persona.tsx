import React from "react"
import { Button, Checkbox, Input, Select, Tag, Typography } from "antd"
import { CheckCircle2, Send, XCircle } from "lucide-react"
import {
  useLocation,
  useNavigate
} from "react-router-dom"
import { useTranslation } from "react-i18next"

import {
  resolvePersonaVisualState,
  useSetBuddyShellRenderContext
} from "@/components/Common/PersonaBuddy"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { PersonaPolicySummary } from "@/components/Option/MCPHub"
import type { PersonaTurnDetectionValues } from "@/components/PersonaGarden/PersonaTurnDetectionControls"
import { AssistantVoiceCard } from "@/components/PersonaGarden/AssistantVoiceCard"
import { AssistantDefaultsPanel } from "@/components/PersonaGarden/AssistantDefaultsPanel"
import {
  PersonaSetupHandoffCard,
} from "@/components/PersonaGarden/PersonaSetupHandoffCard"
import { AssistantSetupWizard } from "@/components/PersonaGarden/AssistantSetupWizard"
import { PersonaGardenTabs } from "@/components/PersonaGarden/PersonaGardenTabs"
import {
  SetupSafetyConnectionsStep,
} from "@/components/PersonaGarden/SetupSafetyConnectionsStep"
import { SetupStarterCommandsStep } from "@/components/PersonaGarden/SetupStarterCommandsStep"
import {
  SetupTestAndFinishStep,
} from "@/components/PersonaGarden/SetupTestAndFinishStep"
import {
  type CommandDraftSource
} from "@/components/PersonaGarden/CommandsPanel"
import {
  type TestLabDryRunCompletedResult
} from "@/components/PersonaGarden/TestLabPanel"
import { useConnectionUxState } from "@/hooks/useConnectionState"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import {
  type PersonaGardenTabKey
} from "@/utils/persona-garden-route"
import { usePersonaGardenRouteBootstrap } from "@/hooks/usePersonaGardenRouteBootstrap"
import {
  usePersonaLiveVoiceController
} from "@/hooks/usePersonaLiveVoiceController"
import {
  useResolvedPersonaVoiceDefaults,
  type PersonaVoiceDefaults
} from "@/hooks/useResolvedPersonaVoiceDefaults"
import { SidepanelHeaderSimple } from "~/components/Sidepanel/Chat/SidepanelHeaderSimple"
import { usePersonaAnalytics } from "./hooks/usePersonaAnalytics"
import {
  usePersonaGovernanceContext,
  type PersonaRuntimeApprovalDuration,
} from "./hooks/usePersonaGovernanceContext"
import { usePersonaLiveSession } from "./hooks/usePersonaLiveSession"
import { usePersonaIncomingPayload } from "./hooks/usePersonaIncomingPayload"
import { usePersonaStateDocs } from "./hooks/usePersonaStateDocs"
import { usePersonaSetupOrchestrator } from "./hooks/usePersonaSetupOrchestrator"
import {
  type PersonaInfo,
  type PersonaLogEntry,
  type PendingPlan,
  type PersonaProfileResponse,
  type SetupHandoffState,
  type SetupHandoffFocusRequest,
  type SetupLiveDetourState,
  type SidepanelPersonaProps,
  DEFAULT_PERSONA_ID,
  buildTurnDetectionValuesFromSavedDefaults,
  areTurnDetectionValuesEqual,
} from "./personaTypes"

const LazyCommandsPanel = React.lazy(() =>
  import("@/components/PersonaGarden/CommandsPanel").then((module) => ({
    default: module.CommandsPanel
  }))
)

const LazyConnectionsPanel = React.lazy(() =>
  import("@/components/PersonaGarden/ConnectionsPanel").then((module) => ({
    default: module.ConnectionsPanel
  }))
)

const LazyLiveSessionPanel = React.lazy(() =>
  import("@/components/PersonaGarden/LiveSessionPanel").then((module) => ({
    default: module.LiveSessionPanel
  }))
)

const LazyPoliciesPanel = React.lazy(() =>
  import("@/components/PersonaGarden/PoliciesPanel").then((module) => ({
    default: module.PoliciesPanel
  }))
)

const LazyProfilePanel = React.lazy(() =>
  import("@/components/PersonaGarden/ProfilePanel").then((module) => ({
    default: module.ProfilePanel
  }))
)

const LazyScopesPanel = React.lazy(() =>
  import("@/components/PersonaGarden/ScopesPanel").then((module) => ({
    default: module.ScopesPanel
  }))
)

const LazyStateDocsPanel = React.lazy(() =>
  import("@/components/PersonaGarden/StateDocsPanel").then((module) => ({
    default: module.StateDocsPanel
  }))
)

const LazyTestLabPanel = React.lazy(() =>
  import("@/components/PersonaGarden/TestLabPanel").then((module) => ({
    default: module.TestLabPanel
  }))
)

const LazyVoiceExamplesPanel = React.lazy(() =>
  import("@/components/PersonaGarden/VoiceExamplesPanel").then((module) => ({
    default: module.VoiceExamplesPanel
  }))
)

const normalizeWakeTriggerPhrases = (phrases?: string[] | null): string[] => {
  const seen = new Set<string>()
  const next: string[] = []
  for (const phrase of phrases || []) {
    const trimmed = String(phrase || "").trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    next.push(trimmed)
  }
  return next
}

const SidepanelPersona = ({
  mode = "persona",
  shell = "sidepanel"
}: SidepanelPersonaProps) => {
  const { t } = useTranslation(["sidepanel", "common", "option"])
  const navigate = useNavigate()
  const location = useLocation()
  const isOnline = useServerOnline()
  const { uxState, hasCompletedFirstRun } = useConnectionUxState()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const isCompanionMode = mode === "companion"
  const routeTitle = isCompanionMode
    ? t("option:header.companion", "Companion")
    : t("sidepanel:persona.title", "Persona Garden")
  const routeRootClassName =
    shell === "sidepanel"
      ? "flex bg-bg flex-col min-h-screen mx-auto max-w-7xl"
      : "flex bg-bg flex-col gap-3 mx-auto max-w-7xl"

  const wsRef = React.useRef<WebSocket | null>(null)
  const manuallyClosingRef = React.useRef(false)
  const setupLiveDetourRef = React.useRef<SetupLiveDetourState | null>(null)
  const setupHandoffRef = React.useRef<SetupHandoffState | null>(null)
  const setupHandoffFocusRequestRef = React.useRef<SetupHandoffFocusRequest | null>(null)
  const activeTabRef = React.useRef<PersonaGardenTabKey>("live")
  const handleIncomingPayloadRef = React.useRef<(payload: any) => void>(() => {})
  const liveVoiceControllerRef = React.useRef<any>(null)
  const setupWizardAwaitingLiveResponseRef = React.useRef(false)
  const setupWizardLastLiveTextRef = React.useRef("")

  const [catalog, setCatalog] = React.useState<PersonaInfo[]>([])
  const [selectedPersonaId, setSelectedPersonaId] =
    React.useState<string>(DEFAULT_PERSONA_ID)
  const [savedPersonaVoiceDefaults, setSavedPersonaVoiceDefaults] =
    React.useState<PersonaVoiceDefaults | null>(null)
  const [savedPersonaSetup, setSavedPersonaSetup] =
    React.useState<any>(null)
  const [savedPersonaBuddySummary, setSavedPersonaBuddySummary] = React.useState<
    PersonaInfo["buddy_summary"] | null
  >(null)
  const [savedPersonaBuddySummaryPersonaId, setSavedPersonaBuddySummaryPersonaId] =
    React.useState<string | null>(null)
  const [savedPersonaProfileVersion, setSavedPersonaProfileVersion] = React.useState<
    number | null
  >(null)
  const [personaProfileLoading, setPersonaProfileLoading] = React.useState(false)
  const [liveSessionVoiceDefaultsBaseline, setLiveSessionVoiceDefaultsBaseline] =
    React.useState<PersonaVoiceDefaults | null>(null)
  const [activeTab, setActiveTab] = React.useState<PersonaGardenTabKey>("live")
  const [openCommandId, setOpenCommandId] = React.useState<string | null>(null)
  const [draftCommandPhrase, setDraftCommandPhrase] = React.useState<string | null>(null)
  const [draftCommandSource, setDraftCommandSource] =
    React.useState<CommandDraftSource | null>(null)
  const [rerunAfterSaveCommandId, setRerunAfterSaveCommandId] =
    React.useState<string | null>(null)
  const [lastTestLabPhrase, setLastTestLabPhrase] = React.useState("")
  const [testLabRerunToken, setTestLabRerunToken] = React.useState(0)
  const [savingLiveVoiceDefaults, setSavingLiveVoiceDefaults] = React.useState(false)
  const [sessionId, setSessionId] = React.useState<string | null>(null)
  const [connected, setConnected] = React.useState(false)
  const [connecting, setConnecting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [input, setInput] = React.useState("")
  const [logs, setLogs] = React.useState<PersonaLogEntry[]>([])
  const [pendingPlan, setPendingPlan] = React.useState<PendingPlan | null>(null)
  const [activeSessionPersonaId, setActiveSessionPersonaId] = React.useState<string | null>(
    null
  )
  const routeBootstrap = usePersonaGardenRouteBootstrap({
    search: location.search,
    setActiveTab,
    setSelectedPersonaId
  })
  const setBuddyShellRenderContext = useSetBuddyShellRenderContext()
  const visualRuntimeOverride = usePersonaVisualRuntimeStore((state) => state.override)
  const [buddyShellLiveContext, setBuddyShellLiveContext] = React.useState({
    live_voice_state: "idle",
    active_tool_status: "",
    wake_armed: false,
    recovery_mode: "none"
  })
  const selectedCatalogPersona = React.useMemo(
    () =>
      catalog.find((persona) => String(persona.id || "") === String(selectedPersonaId || "")) ??
      null,
    [catalog, selectedPersonaId]
  )

  React.useEffect(() => {
    activeTabRef.current = activeTab
  }, [activeTab])

  React.useEffect(() => {
    const activePersonaId = String(selectedPersonaId || "").trim() || null
    const personaSurfaceActive =
      !isCompanionMode &&
      !capsLoading &&
      Boolean(capabilities?.hasPersona) &&
      isOnline &&
      uxState === "connected_ok"

    let context: Parameters<typeof setBuddyShellRenderContext>[0] = null

    if (personaSurfaceActive && activePersonaId) {
      context = {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: activePersonaId,
        position_bucket:
          shell === "sidepanel" ? "sidepanel-desktop" : "web-desktop",
        buddy_summary:
          (savedPersonaBuddySummaryPersonaId === activePersonaId
            ? savedPersonaBuddySummary
            : null) ??
          selectedCatalogPersona?.buddy_summary ??
          null,
        live_session_id: sessionId,
        live_voice_state: buddyShellLiveContext.live_voice_state,
        active_tool_status: buddyShellLiveContext.active_tool_status,
        wake_armed: buddyShellLiveContext.wake_armed,
        recovery_mode: buddyShellLiveContext.recovery_mode,
        persona_source: "route-local"
      }
    }

    setBuddyShellRenderContext(context)
    return () => {
      setBuddyShellRenderContext(null)
    }
  }, [
    capabilities?.hasPersona,
    capsLoading,
    isCompanionMode,
    isOnline,
    selectedPersonaId,
    selectedCatalogPersona,
    savedPersonaBuddySummary,
    savedPersonaBuddySummaryPersonaId,
    setBuddyShellRenderContext,
    shell,
    sessionId,
    buddyShellLiveContext,
    uxState
  ])

  // ── Profile fetch ──
  React.useEffect(() => {
    const normalizedPersonaId = String(selectedPersonaId || "").trim()
    if (!normalizedPersonaId || isCompanionMode) {
      setSavedPersonaBuddySummary(null)
      setSavedPersonaBuddySummaryPersonaId(null)
      setSavedPersonaVoiceDefaults(null)
      setSavedPersonaSetup(null)
      setSavedPersonaProfileVersion(null)
      setPersonaProfileLoading(false)
      if (!connected) {
        setLiveSessionVoiceDefaultsBaseline(null)
      }
      return
    }

    let cancelled = false
    setSavedPersonaBuddySummary(null)
    setSavedPersonaBuddySummaryPersonaId(null)
    setPersonaProfileLoading(true)
    ;(async () => {
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(normalizedPersonaId)}` as any,
          { method: "GET" }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to load persona profile")
        }
        const payload = (await response.json()) as PersonaProfileResponse
        if (!cancelled) {
          setSavedPersonaBuddySummary(payload?.buddy_summary ?? null)
          setSavedPersonaBuddySummaryPersonaId(normalizedPersonaId)
          setSavedPersonaVoiceDefaults(payload?.voice_defaults || null)
          setSavedPersonaSetup(payload?.setup || null)
          setSavedPersonaProfileVersion(
            typeof payload?.version === "number" ? payload.version : null
          )
        }
      } catch {
        if (!cancelled) {
          setSavedPersonaBuddySummary(null)
          setSavedPersonaBuddySummaryPersonaId(null)
          setSavedPersonaVoiceDefaults(null)
          setSavedPersonaSetup(null)
          setSavedPersonaProfileVersion(null)
        }
      } finally {
        if (!cancelled) {
          setPersonaProfileLoading(false)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [connected, isCompanionMode, selectedPersonaId])

  const appendLog = React.useCallback(
    (kind: PersonaLogEntry["kind"], text: string) => {
      const trimmed = String(text || "").trim()
      if (!trimmed) return
      setLogs((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          kind,
          text: trimmed
        }
      ])
    },
    []
  )

  // ── State docs hook ──
  const stateDocsGetTargetPersonaId = React.useCallback(
    (override?: string) => {
      const personaId = String(override || activeSessionPersonaId || selectedPersonaId || "").trim()
      return personaId || null
    },
    [activeSessionPersonaId, selectedPersonaId]
  )
  const stateDocs = usePersonaStateDocs({
    getTargetPersonaId: stateDocsGetTargetPersonaId,
    appendLog,
    setError,
  })

  // ── Refs for late-bound deps (break circular init order) ──
  const emitSetupAnalyticsEventRef = React.useRef<(event: Record<string, any>) => void>(() => {})
  const confirmDiscardUnsavedStateDraftsRef = React.useRef<(reason?: string) => boolean>(
    () => true
  )
  const triggerRecoveryReconnectRef = React.useRef<() => void>(() => {})

  // Keep refs in sync
  confirmDiscardUnsavedStateDraftsRef.current = stateDocs.confirmDiscardUnsavedStateDrafts

  // ── Setup orchestrator hook ──
  const setupOrch = usePersonaSetupOrchestrator({
    selectedPersonaId,
    setSelectedPersonaId,
    isCompanionMode,
    activeTab,
    setActiveTab,
    connected,
    connecting,
    catalog,
    setCatalog,
    personaProfileLoading,
    savedPersonaSetup,
    setSavedPersonaSetup,
    savedPersonaVoiceDefaults,
    setSavedPersonaVoiceDefaults,
    savedPersonaProfileVersion,
    setSavedPersonaProfileVersion,
    emitSetupAnalyticsEventRef,
    confirmDiscardUnsavedStateDraftsRef,
    triggerRecoveryReconnectRef,
    setupLiveDetourRef,
    setupHandoffRef,
    setupHandoffFocusRequestRef,
    activeTabRef,
    setupWizardAwaitingLiveResponseRef,
    setupWizardLastLiveTextRef,
  })

  // ── Analytics hook ──
  const analytics = usePersonaAnalytics({
    selectedPersonaId,
    activeTab,
    currentSetupRunId: setupOrch.currentSetupRunId,
  })
  const {
    voiceAnalytics,
    voiceAnalyticsLoading,
    setupAnalytics,
    setupAnalyticsLoading,
    liveVoiceAnalyticsSnapshotRef,
    emitSetupAnalyticsEvent,
    flushLiveVoiceSessionAnalytics,
  } = analytics

  // Wire emitSetupAnalyticsEvent into setupOrch via the ref
  emitSetupAnalyticsEventRef.current = emitSetupAnalyticsEvent

  // ── Governance hook ──
  const governance = usePersonaGovernanceContext({
    connected,
    sessionId,
    wsRef,
    appendLog,
    setError
  })
  const {
    pendingApprovals, setPendingApprovals,
    activeApprovalKey, setActiveApprovalKey,
    approvalHighlightPhase, approvalHighlightSequence,
    resolvedApprovalSnapshot, setResolvedApprovalSnapshot,
    approvedStepMap, setApprovedStepMap,
    submittingApprovalKey,
    runtimeApprovalCardRef, runtimeApprovalRowRefs,
    activePendingApproval, pendingApprovalSummary,
    clearResolvedApprovalFadeTimer, resetApprovalHighlightMotion,
    triggerApprovalHighlightPhase,
    updateApprovalDuration, submitApprovalDecision,
    registerRuntimeApprovalRow, handleJumpToRuntimeApproval,
  } = governance

  // ── Live session hook ──
  const liveSession = usePersonaLiveSession({
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
    error,
    setError,
    input,
    setInput,
    logs,
    setLogs,
    appendLog,
    activeSessionPersonaId,
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
    setPersonaStateHistory: stateDocs.setPersonaStateHistory,
    setPersonaStateHistoryLoaded: stateDocs.setPersonaStateHistoryLoaded,
    confirmDiscardUnsavedStateDrafts: stateDocs.confirmDiscardUnsavedStateDrafts,
    flushLiveVoiceSessionAnalytics,
    loadPersonaStateDocs: stateDocs.loadPersonaStateDocs,
    personaSetupWizardCurrentStep: setupOrch.personaSetupWizard.currentStep,
    personaSetupWizardIsSetupRequired: setupOrch.personaSetupWizard.isSetupRequired,
    setSetupTestOutcome: setupOrch.setSetupTestOutcome,
    setupLiveDetour: setupOrch.setupLiveDetour,
    setupWizardAwaitingLiveResponseRef,
    setupWizardLastLiveTextRef,
    clearSetupStepError: setupOrch.clearSetupStepError,
    setPendingPlan,
    pendingPlan,
    capabilities,
    capsLoading,
    routeBootstrapPersonaId: routeBootstrap.personaId,
  })
  const {
    sessionHistory,
    resumeSessionId,
    memoryEnabled, setMemoryEnabled,
    memoryTopK, setMemoryTopK,
    companionContextEnabled, setCompanionContextEnabled,
    personaStateContextEnabled, setPersonaStateContextEnabled,
    personaStateContextProfileDefault,
    updatingPersonaStateContextDefault,
    savingCompanionCheckIn,
    companionPrompts,
    canSend,
    canSaveCompanionCheckIn,
    MEMORY_TOP_K_OPTIONS,
    formatMemoryResultsLabel,
    connect,
    disconnect,
    sendUserMessage,
    sendSetupLiveTestMessage,
    loadSessionHistory,
    confirmPlanWithMap,
    cancelPlan,
    handleResumeSessionSelectionChange,
    handlePersonaSelectionChange,
    triggerRecoveryReconnect,
    saveCompanionCheckIn,
    updatePersonaStateContextDefault,
    getTargetPersonaId,
  } = liveSession

  // Wire triggerRecoveryReconnect into setupOrch via the ref
  triggerRecoveryReconnectRef.current = triggerRecoveryReconnect

  const confirmPlan = React.useCallback(() => {
    confirmPlanWithMap(approvedStepMap)
  }, [approvedStepMap, confirmPlanWithMap])

  // ── Voice controller ──
  const resolvedLivePersonaVoiceDefaults = useResolvedPersonaVoiceDefaults(
    connected ? liveSessionVoiceDefaultsBaseline : savedPersonaVoiceDefaults
  )
  const wakeTriggerPhrases = React.useMemo(
    () =>
      normalizeWakeTriggerPhrases(
        savedPersonaVoiceDefaults?.voice_chat_trigger_phrases
      ),
    [savedPersonaVoiceDefaults]
  )
  const livePersonaId = connected ? activeSessionPersonaId || selectedPersonaId : selectedPersonaId
  const liveVoiceController = usePersonaLiveVoiceController({
    ws: wsRef.current,
    connected,
    sessionId: sessionId || "",
    personaId: String(livePersonaId || "").trim(),
    resolvedDefaults: resolvedLivePersonaVoiceDefaults,
    canUseServerStt: Boolean(capabilities?.hasAudio),
    wakeTriggerPhrases
  })
  liveVoiceControllerRef.current = liveVoiceController

  React.useEffect(() => {
    setBuddyShellLiveContext({
      live_voice_state: liveVoiceController.state,
      active_tool_status: liveVoiceController.activeToolStatus,
      wake_armed: liveVoiceController.wakeArmed,
      recovery_mode: liveVoiceController.recoveryMode
    })
  }, [
    liveVoiceController.activeToolStatus,
    liveVoiceController.recoveryMode,
    liveVoiceController.state,
    liveVoiceController.wakeArmed
  ])

  React.useEffect(() => {
    liveVoiceAnalyticsSnapshotRef.current = {
      personaId: String(livePersonaId || "").trim(),
      sessionId: String(sessionId || "").trim(),
      listeningRecoveryCount: liveVoiceController.listeningRecoveryCount,
      thinkingRecoveryCount: liveVoiceController.thinkingRecoveryCount
    }
  }, [
    livePersonaId,
    liveVoiceAnalyticsSnapshotRef,
    liveVoiceController.listeningRecoveryCount,
    liveVoiceController.thinkingRecoveryCount,
    sessionId
  ])

  // ── Incoming payload handler ──
  const handleIncomingPayload = usePersonaIncomingPayload({
    appendLog,
    clearResolvedApprovalFadeTimer,
    consumeSetupHandoffAction: setupOrch.consumeSetupHandoffAction,
    emitSetupAnalyticsEvent,
    liveVoiceController,
    personaId: String(selectedPersonaId || "").trim(),
    personaSetupWizardCurrentStep: setupOrch.personaSetupWizard.currentStep,
    personaSetupWizardIsSetupRequired: setupOrch.personaSetupWizard.isSetupRequired,
    resolvedApprovalSnapshot,
    sessionId,
    setApprovedStepMap,
    setPendingApprovals,
    setPendingPlan,
    setResolvedApprovalSnapshot,
    setSetupTestOutcome: setupOrch.setSetupTestOutcome,
    setSetupLiveDetour: setupOrch.setSetupLiveDetour,
    setSetupTestResumeNote: setupOrch.setSetupTestResumeNote,
    setupLiveDetourRef,
    setupHandoffRef,
    activeTabRef,
    setupWizardAwaitingLiveResponseRef,
    setupWizardLastLiveTextRef,
  })
  handleIncomingPayloadRef.current = handleIncomingPayload

  // ── Command / test-lab navigation helpers ──

  const handleOpenCommandFromTestLab = React.useCallback((
    commandId: string,
    heardText: string
  ) => {
    const normalizedCommandId = String(commandId || "").trim()
    const normalizedHeardText = String(heardText || "").trim()
    if (!normalizedCommandId) return
    setDraftCommandPhrase(null)
    setDraftCommandSource(null)
    setOpenCommandId(normalizedCommandId)
    setRerunAfterSaveCommandId(normalizedCommandId)
    setLastTestLabPhrase(normalizedHeardText)
    setActiveTab("commands")
  }, [setActiveTab])

  const handleCreateCommandFromTestLab = React.useCallback((heardText: string) => {
    const normalizedHeardText = String(heardText || "").trim()
    if (!normalizedHeardText) return
    setOpenCommandId(null)
    setRerunAfterSaveCommandId(null)
    setDraftCommandPhrase(normalizedHeardText)
    setDraftCommandSource("test_lab")
    setLastTestLabPhrase(normalizedHeardText)
    setActiveTab("commands")
  }, [setActiveTab])

  const handleCreateCommandFromSetupNoMatch = React.useCallback((heardText: string) => {
    const normalizedHeardText = String(heardText || "").trim()
    if (!normalizedHeardText) return
    setOpenCommandId(null)
    setRerunAfterSaveCommandId(null)
    setDraftCommandPhrase(normalizedHeardText)
    setDraftCommandSource("setup_no_match")
    setupOrch.handleCreateCommandFromSetupNoMatch(normalizedHeardText)
    setActiveTab("commands")
  }, [setActiveTab, setupOrch])

  const handleOpenCommandHandled = React.useCallback((commandId: string) => {
    const normalizedCommandId = String(commandId || "").trim()
    if (!normalizedCommandId) return
    setOpenCommandId((current) =>
      current === normalizedCommandId ? null : current
    )
  }, [])

  const handleDraftCommandPhraseHandled = React.useCallback((heardText: string) => {
    const normalizedHeardText = String(heardText || "").trim()
    if (!normalizedHeardText) return
    setDraftCommandPhrase((current) =>
      current === normalizedHeardText ? null : current
    )
    setDraftCommandSource(null)
  }, [])

  const handleRerunAfterCommandSave = React.useCallback((commandId: string) => {
    const normalizedCommandId = String(commandId || "").trim()
    if (!normalizedCommandId) return
    if (rerunAfterSaveCommandId !== normalizedCommandId) return
    if (String(lastTestLabPhrase || "").trim()) {
      setActiveTab("test-lab")
      setTestLabRerunToken((previous) => previous + 1)
    }
    setRerunAfterSaveCommandId(null)
  }, [lastTestLabPhrase, rerunAfterSaveCommandId, setActiveTab])

  const handleCommandSaved = React.useCallback(
    (commandId: string, context: { fromDraft: boolean }) => {
      const consumed = setupOrch.handleSetupDetourCommandSaved(commandId, context)
      if (consumed) {
        setDraftCommandPhrase(null)
        setDraftCommandSource(null)
        setOpenCommandId(null)
        setRerunAfterSaveCommandId(null)
      }
      setupOrch.consumeSetupHandoffAction("command_saved")
    },
    [setupOrch]
  )

  const handleCopyLastVoiceCommandToComposer = React.useCallback(() => {
    const nextValue = String(liveVoiceController.lastCommittedText || "").trim()
    if (!nextValue) return
    setInput(nextValue)
  }, [liveVoiceController.lastCommittedText, setInput])

  const handleReconnectPersonaSessionFromRecovery = React.useCallback(() => {
    if (connecting) return
    void liveVoiceController.stopWakeListening("stop_live_voice")
    liveVoiceController.resetTurn()
    triggerRecoveryReconnect()
    disconnect({ force: true })
  }, [connecting, disconnect, liveVoiceController, triggerRecoveryReconnect])

  // ── Live voice defaults save ──
  const savedTurnDetectionValues = React.useMemo(
    () => buildTurnDetectionValuesFromSavedDefaults(savedPersonaVoiceDefaults),
    [savedPersonaVoiceDefaults]
  )
  const liveTurnDetectionValues = React.useMemo<PersonaTurnDetectionValues>(
    () => ({
      autoCommitEnabled: liveVoiceController.autoCommitEnabled,
      vadThreshold: liveVoiceController.vadThreshold,
      minSilenceMs: liveVoiceController.minSilenceMs,
      turnStopSecs: liveVoiceController.turnStopSecs,
      minUtteranceSecs: liveVoiceController.minUtteranceSecs
    }),
    [
      liveVoiceController.autoCommitEnabled,
      liveVoiceController.minSilenceMs,
      liveVoiceController.minUtteranceSecs,
      liveVoiceController.turnStopSecs,
      liveVoiceController.vadThreshold
    ]
  )
  const showSaveCurrentSettingsAsDefaults =
    connected &&
    !isCompanionMode &&
    (!savedTurnDetectionValues ||
      !areTurnDetectionValuesEqual(savedTurnDetectionValues, liveTurnDetectionValues))

  const handleSaveCurrentLiveTurnDetectionDefaults = React.useCallback(async () => {
    const personaId = getTargetPersonaId()
    if (!personaId || savingLiveVoiceDefaults) return
    setSavingLiveVoiceDefaults(true)
    setError(null)
    try {
      const mergedVoiceDefaults: PersonaVoiceDefaults = {
        ...(savedPersonaVoiceDefaults || {}),
        auto_commit_enabled: liveVoiceController.autoCommitEnabled,
        vad_threshold: liveVoiceController.vadThreshold,
        min_silence_ms: liveVoiceController.minSilenceMs,
        turn_stop_secs: liveVoiceController.turnStopSecs,
        min_utterance_secs: liveVoiceController.minUtteranceSecs
      }
      const response = await tldwClient.fetchWithAuth(
        `/api/v1/persona/profiles/${encodeURIComponent(personaId)}` as any,
        {
          method: "PATCH",
          body: {
            voice_defaults: mergedVoiceDefaults
          }
        }
      )
      if (!response.ok) {
        throw new Error(response.error || "Failed to save current live settings as defaults")
      }
      const payload = (await response.json()) as PersonaProfileResponse
      setSavedPersonaVoiceDefaults(payload?.voice_defaults || mergedVoiceDefaults)
      appendLog("notice", "Saved current live turn detection defaults")
    } catch (err: any) {
      setError(String(err?.message || "Failed to save current live settings as defaults"))
    } finally {
      setSavingLiveVoiceDefaults(false)
    }
  }, [
    appendLog,
    getTargetPersonaId,
    liveVoiceController.autoCommitEnabled,
    liveVoiceController.minSilenceMs,
    liveVoiceController.minUtteranceSecs,
    liveVoiceController.turnStopSecs,
    liveVoiceController.vadThreshold,
    savedPersonaVoiceDefaults,
    savingLiveVoiceDefaults
  ])

  // ── Handoff callbacks ──
  const handleProfileDefaultsSaved = React.useCallback(() => {
    setupOrch.consumeSetupHandoffAction("voice_defaults_saved")
  }, [setupOrch])

  const handleConnectionSaved = React.useCallback(() => {
    setupOrch.consumeSetupHandoffAction("connection_saved")
  }, [setupOrch])

  const handleConnectionTestSucceeded = React.useCallback(() => {
    setupOrch.consumeSetupHandoffAction("connection_test_succeeded")
  }, [setupOrch])

  const handleTestLabDryRunCompleted = React.useCallback((result: TestLabDryRunCompletedResult) => {
    if (!result.matched) return
    setupOrch.consumeSetupHandoffAction("dry_run_match")
  }, [setupOrch])

  const renderSetupHandoffCard = React.useCallback(
    (tab: PersonaGardenTabKey) => {
      if (!setupOrch.setupHandoff || setupOrch.setupHandoff.targetTab !== tab) return null
      return (
        <PersonaSetupHandoffCard
          targetTab={setupOrch.setupHandoff.targetTab}
          completionType={setupOrch.setupHandoff.completionType}
          reviewSummary={setupOrch.setupHandoff.reviewSummary}
          recommendedAction={setupOrch.setupHandoff.recommendedAction}
          compact={setupOrch.setupHandoff.compact}
          onDismiss={setupOrch.dismissSetupHandoff}
          onAddCommand={() =>
            setupOrch.openSetupHandoffTarget({ tab: "commands", section: "command_form" })
          }
          onOpenCommands={() =>
            setupOrch.openSetupHandoffTarget({ tab: "commands", section: "command_list" })
          }
          onOpenTestLab={() =>
            setupOrch.openSetupHandoffTarget({ tab: "test-lab", section: "dry_run_form" })
          }
          onOpenLive={() => setupOrch.openSetupHandoffTarget({ tab: "live" })}
          onOpenProfiles={() =>
            setupOrch.openSetupHandoffTarget({
              tab: "profiles",
              section: "confirmation_mode"
            })
          }
          onOpenConnections={() =>
            setupOrch.setupHandoff!.reviewSummary.connection.mode === "skipped"
              ? setupOrch.openSetupHandoffTarget({
                  tab: "connections",
                  section: "connection_form"
                })
              : setupOrch.openSetupHandoffTarget({
                  tab: "connections",
                  section: "saved_connections",
                  connectionName:
                    setupOrch.setupHandoff!.reviewSummary.connection.mode === "created" ||
                    setupOrch.setupHandoff!.reviewSummary.connection.mode === "available"
                      ? setupOrch.setupHandoff!.reviewSummary.connection.name
                      : null
                })
          }
        />
      )
    },
    [setupOrch]
  )

  const withSetupHandoff = React.useCallback(
    (tab: PersonaGardenTabKey, content: React.ReactNode) => (
      <div className="space-y-3">
        {renderSetupHandoffCard(tab)}
        {content}
      </div>
    ),
    [renderSetupHandoffCard]
  )

  const renderLazyPersonaTab = React.useCallback(
    (
      tab: PersonaGardenTabKey,
      content: React.ReactNode,
      options?: {
        includeSetupHandoff?: boolean
      }
    ) => {
      if (activeTab !== tab) {
        return null
      }

      const tabContent = (
        <React.Suspense fallback={null}>
          {content}
        </React.Suspense>
      )

      if (options?.includeSetupHandoff === false) {
        return tabContent
      }

      return withSetupHandoff(tab, tabContent)
    },
    [activeTab, withSetupHandoff]
  )

  // ── Persona unsupported check ──
  const personaUnsupported =
    !capsLoading &&
    capabilities &&
    (!capabilities.hasPersona ||
      (isCompanionMode && !capabilities.hasPersonalization))

  const selectedPersonaName =
    selectedCatalogPersona?.name || selectedPersonaId

  // ── Route header ──
  const routeHeader =
    shell === "sidepanel" ? (
      <div className="sticky bg-surface top-0 z-10">
        <SidepanelHeaderSimple activeTitle={routeTitle} />
      </div>
    ) : (
      <div className="rounded-lg border border-border bg-surface px-4 py-3">
        <Typography.Text strong>{routeTitle}</Typography.Text>
        <Typography.Text type="secondary" className="mt-1 block text-sm">
          {isCompanionMode
            ? "A dedicated conversation surface that keeps companion context in the loop."
            : "Live persona sessions run against your connected tldw server."}
        </Typography.Text>
      </div>
    )
  const settingsRoute = shell === "options" ? "/settings/tldw" : "/settings"
  const diagnosticsRoute = shell === "options" ? "/settings/health" : settingsRoute
  const settingsLabel = t("sidepanel:header.settingsShortLabel", "Settings")
  const setupActionLabel =
    shell === "options" && !hasCompletedFirstRun ? "Finish Setup" : settingsLabel
  const openSettings = () => navigate(settingsRoute)
  const openDiagnostics = () => navigate(diagnosticsRoute)
  const openSetup = () => {
    if (shell === "options" && !hasCompletedFirstRun) {
      navigate("/")
      return
    }
    openSettings()
  }
  const handlePersonaTabChange = React.useCallback(
    (key: string) => {
      const nextTab = key as PersonaGardenTabKey
      if (activeTabRef.current === "live" && nextTab !== "live") {
        void liveVoiceController.stopWakeListening("tab_switch")
      }
      setActiveTab(nextTab)
    },
    [liveVoiceController]
  )

  // ── JSX: live session controls ──
  const liveSessionControls = (
    <div className="flex flex-wrap items-center gap-2">
      {!isCompanionMode ? (
        <Select
          size="small"
          className="min-w-[180px]"
          value={selectedPersonaId}
          disabled={connected}
          aria-label={t("sidepanel:persona.select", "Select persona")}
          onChange={(value) => {
            void liveVoiceController.stopWakeListening("persona_switch")
            handlePersonaSelectionChange(String(value))
          }}
          options={catalog.map((persona) => ({
            label: persona.name || persona.id,
            value: persona.id
          }))}
          placeholder={t("sidepanel:persona.select", "Select persona")}
        />
      ) : null}
      <Select
        data-testid="persona-resume-session-select"
        size="small"
        className="min-w-[180px]"
        value={resumeSessionId || "__new__"}
        aria-label={t("sidepanel:persona.resume", "Resume session")}
        disabled={connected}
        onChange={(value) => {
          void liveVoiceController.stopWakeListening("persona_switch")
          handleResumeSessionSelectionChange(String(value))
        }}
        options={[
          { label: t("sidepanel:persona.newSession", "New session"), value: "__new__" },
          ...sessionHistory.map((session) => ({
            label: session.session_id,
            value: session.session_id
          }))
        ]}
        placeholder={t("sidepanel:persona.resume", "Resume session")}
      />
      <Checkbox
        data-testid="persona-memory-toggle"
        checked={memoryEnabled}
        onChange={(event) => setMemoryEnabled(event.target.checked)}
      >
        {t("sidepanel:persona.memoryToggle", "Memory")}
      </Checkbox>
      {!isCompanionMode ? (
        <Checkbox
          data-testid="persona-state-context-toggle"
          checked={personaStateContextEnabled}
          onChange={(event) => setPersonaStateContextEnabled(event.target.checked)}
        >
          {t("sidepanel:persona.stateContextToggle", "State context")}
        </Checkbox>
      ) : null}
      {!isCompanionMode ? (
        <Checkbox
          data-testid="persona-companion-context-toggle"
          checked={companionContextEnabled}
          onChange={(event) => setCompanionContextEnabled(event.target.checked)}
        >
          {t("sidepanel:persona.companionContextToggle", "Companion context")}
        </Checkbox>
      ) : null}
      {!isCompanionMode ? (
        <Checkbox
          data-testid="persona-state-context-default-toggle"
          checked={personaStateContextProfileDefault}
          disabled={!connected || updatingPersonaStateContextDefault}
          onChange={(event) => {
            void updatePersonaStateContextDefault(event.target.checked)
          }}
        >
          {t("sidepanel:persona.stateContextDefaultToggle", "Profile default")}
        </Checkbox>
      ) : null}
      <Select
        data-testid="persona-memory-topk-select"
        size="small"
        className="w-[150px]"
        value={memoryTopK}
        aria-label={t("sidepanel:persona.memoryTopK", "Memory results")}
        disabled={!memoryEnabled}
        onChange={(value) => setMemoryTopK(Number(value))}
        options={MEMORY_TOP_K_OPTIONS.map((k) => ({
          label: formatMemoryResultsLabel(k),
          value: k
        }))}
        placeholder={t("sidepanel:persona.memoryTopK", "Memory results")}
      />
      {!connected ? (
        <Button
          size="small"
          type="primary"
          loading={connecting}
          onClick={() => {
            void liveVoiceController.stopWakeListening("persona_switch")
            void connect()
          }}
        >
          {t("sidepanel:persona.connect", "Connect")}
        </Button>
      ) : (
        <Button size="small" onClick={() => {
          void liveVoiceController.stopWakeListening("stop_live_voice")
          disconnect()
        }}>
          {t("sidepanel:persona.disconnect", "Disconnect")}
        </Button>
      )}
      {sessionId ? <Tag color="blue">{`session: ${sessionId.slice(0, 8)}`}</Tag> : null}
      {sessionId ? (
        <Button size="small" onClick={() => void loadSessionHistory()}>
          {t("sidepanel:persona.loadHistory", "Load history")}
        </Button>
      ) : null}
    </div>
  )

  const errorBanner = error ? (
    <div className="rounded-md border border-danger/30 bg-danger/10 p-2 text-xs text-danger">
      {error}
    </div>
  ) : null

  const normalizedLivePersonaId = String(livePersonaId || "").trim()
  const applicableVisualOverride =
    visualRuntimeOverride &&
    visualRuntimeOverride.personaId === normalizedLivePersonaId &&
    (!sessionId ||
      !visualRuntimeOverride.sessionId ||
      visualRuntimeOverride.sessionId === sessionId) &&
    visualRuntimeOverride.expiresAt > Date.now()
      ? visualRuntimeOverride
      : null
  const resolvedLiveVisualState = resolvePersonaVisualState({
    liveVoiceState: liveVoiceController.state,
    activeToolStatus: liveVoiceController.activeToolStatus,
    wakeArmed: liveVoiceController.wakeArmed,
    recovering: liveVoiceController.recoveryMode !== "none",
    runtimeOverride: applicableVisualOverride,
    mcpRuntimeReason: applicableVisualOverride?.reason
  })
  const visualStateFeedback: React.ComponentProps<
    typeof AssistantVoiceCard
  >["visualStateFeedback"] = !isCompanionMode
    ? {
        state: resolvedLiveVisualState,
        source: applicableVisualOverride
          ? "override"
          : liveVoiceController.recoveryMode !== "none" ||
              liveVoiceController.state === "error"
            ? "error_recovery"
            : liveVoiceController.activeToolStatus
              ? "live"
              : "default",
        reason: applicableVisualOverride?.reason ?? null,
        fallbackReason: null
      }
    : undefined

  const assistantVoiceCard = (
    <AssistantVoiceCard
      resolvedDefaults={resolvedLivePersonaVoiceDefaults}
      connected={connected}
      state={liveVoiceController.state}
      speechAvailable={liveVoiceController.speechAvailable}
      isListening={liveVoiceController.isListening}
      heardText={liveVoiceController.heardText}
      lastCommittedText={liveVoiceController.lastCommittedText}
      activeToolStatus={liveVoiceController.activeToolStatus}
      pendingApprovalSummary={pendingApprovalSummary}
      warning={liveVoiceController.warning}
      recoveryMode={liveVoiceController.recoveryMode}
      manualModeRequired={liveVoiceController.manualModeRequired}
      canSendNow={liveVoiceController.canSendNow}
      textOnlyDueToTtsFailure={liveVoiceController.textOnlyDueToTtsFailure}
      showSaveCurrentSettingsAsDefaults={showSaveCurrentSettingsAsDefaults}
      savingCurrentSettingsAsDefaults={savingLiveVoiceDefaults}
      sessionAutoResume={liveVoiceController.sessionAutoResume}
      sessionBargeIn={liveVoiceController.sessionBargeIn}
      wakeArmed={liveVoiceController.wakeArmed}
      wakeDetectorState={liveVoiceController.wakeDetectorState}
      wakeWarning={liveVoiceController.wakeWarning}
      wakeTriggerPhrases={liveVoiceController.wakeTriggerPhrases}
      sessionWakeBehavior={liveVoiceController.sessionWakeBehavior}
      autoCommitEnabled={liveVoiceController.autoCommitEnabled}
      vadPreset={liveVoiceController.vadPreset}
      vadThreshold={liveVoiceController.vadThreshold}
      minSilenceMs={liveVoiceController.minSilenceMs}
      turnStopSecs={liveVoiceController.turnStopSecs}
      minUtteranceSecs={liveVoiceController.minUtteranceSecs}
      visualStateFeedback={visualStateFeedback}
      onToggleListening={liveVoiceController.toggleListening}
      onSendNow={liveVoiceController.sendCurrentTranscriptNow}
      onSessionAutoResumeChange={liveVoiceController.setSessionAutoResume}
      onSessionBargeInChange={liveVoiceController.setSessionBargeIn}
      onToggleWakeArmed={liveVoiceController.toggleWakeArmed}
      onSessionWakeBehaviorChange={liveVoiceController.setSessionWakeBehavior}
      onAutoCommitEnabledChange={liveVoiceController.setAutoCommitEnabled}
      onVadPresetChange={liveVoiceController.setVadPreset}
      onVadThresholdChange={liveVoiceController.setVadThreshold}
      onMinSilenceMsChange={liveVoiceController.setMinSilenceMs}
      onTurnStopSecsChange={liveVoiceController.setTurnStopSecs}
      onMinUtteranceSecsChange={liveVoiceController.setMinUtteranceSecs}
      onKeepListening={liveVoiceController.keepListening}
      onResetTurn={liveVoiceController.resetTurn}
      onWaitOnRecovery={liveVoiceController.waitOnRecovery}
      onCopyLastCommandToComposer={handleCopyLastVoiceCommandToComposer}
      onJumpToApproval={handleJumpToRuntimeApproval}
      onSaveCurrentSettingsAsDefaults={handleSaveCurrentLiveTurnDetectionDefaults}
      onReconnectPersonaSession={handleReconnectPersonaSessionFromRecovery}
    />
  )

  const runtimeApprovalCard = pendingApprovals.length || resolvedApprovalSnapshot ? (
    <div
      ref={runtimeApprovalCardRef}
      data-testid="persona-runtime-approval-card"
      className="rounded-lg border border-warning/40 bg-warning/5 p-3"
    >
      <Typography.Text strong>
        {t("sidepanel:persona.runtimeApproval", "Runtime approval required")}
      </Typography.Text>
      {resolvedApprovalSnapshot && !pendingApprovals.length ? (
        <div
          data-testid="persona-runtime-approval-answered"
          className="mt-2 rounded-md border border-success/30 bg-success/10 p-2 text-xs text-success"
        >
          {`Answered: ${resolvedApprovalSnapshot.toolName}`}
        </div>
      ) : null}
      <div className="mt-2 space-y-3">
        {pendingApprovals.map((approval) => {
          const isSubmitting = submittingApprovalKey === approval.key
          const isHighlighted = approval.key === activeApprovalKey
          const highlightPhase = isHighlighted ? approvalHighlightPhase : "none"
          return (
            <div
              key={approval.key}
              ref={(node) => {
                registerRuntimeApprovalRow(approval.key, node)
              }}
              data-testid={`persona-runtime-approval-row-${approval.key}`}
              data-approval-key={approval.key}
              data-highlighted={isHighlighted ? "true" : "false"}
              data-highlight-phase={highlightPhase}
              data-highlight-seq={isHighlighted ? String(approvalHighlightSequence) : "0"}
              className={`rounded-md border p-3 ${
                isHighlighted
                  ? "persona-runtime-approval-row border-warning/60 bg-warning/10"
                  : "persona-runtime-approval-row border-warning/30 bg-surface"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <Tag color="gold">{approval.tool_name}</Tag>
                {isHighlighted ? (
                  <Tag color="orange">Needs your approval</Tag>
                ) : null}
                {approval.mode ? <Tag color="blue">{approval.mode}</Tag> : null}
                {approval.reason ? <Tag color="red">{approval.reason}</Tag> : null}
              </div>
              {approval.scope_context?.server_name ||
              approval.scope_context?.server_id ||
              approval.scope_context?.workspace_id ||
              approval.scope_context?.workspace_bundle_ids?.length ? (
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                  {approval.scope_context?.server_name || approval.scope_context?.server_id ? (
                    <Tag color="cyan">
                      {approval.scope_context.server_name || approval.scope_context.server_id}
                    </Tag>
                  ) : null}
                  {approval.scope_context?.workspace_id ? (
                    <Tag color="cyan">{approval.scope_context.workspace_id}</Tag>
                  ) : null}
                  {(approval.scope_context?.workspace_bundle_ids || []).map((workspaceId) => (
                    <Tag
                      key={`${approval.key}-workspace-bundle-${workspaceId}`}
                      color="purple"
                    >
                      {workspaceId}
                    </Tag>
                  ))}
                  {approval.scope_context?.selected_workspace_trust_source ? (
                    <Tag color="blue">
                      {approval.scope_context.selected_workspace_trust_source}
                    </Tag>
                  ) : null}
                  {(approval.scope_context.requested_slots || []).map((slotName) => (
                    <Tag key={`${approval.key}-${slotName}`} color="geekblue">
                      {slotName}
                    </Tag>
                  ))}
                  {(approval.scope_context.normalized_paths || []).map((pathValue) => (
                    <Tag key={`${approval.key}-path-${pathValue}`} color="magenta">
                      {pathValue}
                    </Tag>
                  ))}
                </div>
              ) : null}
              {Object.keys(approval.arguments_summary).length ? (
                <pre className="mt-2 overflow-auto rounded bg-bg p-2 text-[11px] text-text">
                  {JSON.stringify(approval.arguments_summary, null, 2)}
                </pre>
              ) : null}
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                <label htmlFor={`persona-approval-duration-${approval.key}`}>
                  {t("sidepanel:persona.approvalDuration", "Approval duration")}
                </label>
                <select
                  id={`persona-approval-duration-${approval.key}`}
                  className="rounded border border-border bg-bg px-2 py-1 text-xs text-text"
                  value={approval.selected_duration}
                  disabled={isSubmitting}
                  onChange={(event) =>
                    updateApprovalDuration(
                      approval.key,
                      event.target.value as PersonaRuntimeApprovalDuration
                    )
                  }
                >
                  {approval.duration_options.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Button
                  size="small"
                  type="primary"
                  loading={isSubmitting}
                  onClick={() => {
                    void submitApprovalDecision(approval, "approved")
                  }}
                >
                  {t("sidepanel:persona.approveAndRetry", "Approve and retry")}
                </Button>
                <Button
                  size="small"
                  danger
                  disabled={isSubmitting}
                  onClick={() => {
                    void submitApprovalDecision(approval, "denied")
                  }}
                >
                  {t("sidepanel:persona.denyApproval", "Deny")}
                </Button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  ) : null

  const liveSessionStatusPanels = (
    <>
      {setupOrch.setupLiveDetour ? (
        <div className="rounded-lg border border-sky-500/30 bg-sky-500/10 p-3 text-sm text-sky-100">
          <div>Finish this live test, then return to setup.</div>
          <button
            type="button"
            className="mt-2 rounded-md border border-sky-500/40 px-3 py-2 text-sm font-medium text-sky-100"
            onClick={setupOrch.handleReturnToSetupFromLiveDetour}
          >
            Return to setup
          </button>
        </div>
      ) : null}
      {errorBanner}
      {!isCompanionMode ? (
        <PersonaPolicySummary personaId={selectedPersonaId || null} />
      ) : null}
      {runtimeApprovalCard}
    </>
  )

  const pendingPlanCard = pendingPlan ? (
    <div className="rounded-lg border border-border bg-surface p-3">
      <Typography.Text strong>
        {t("sidepanel:persona.pendingPlan", "Pending tool plan")}
      </Typography.Text>
      {pendingPlan.memory ? (
        <div className="mt-1 flex flex-wrap items-center gap-1">
          <Tag color={pendingPlan.memory.enabled ? "green" : "default"}>
            {pendingPlan.memory.enabled ? "memory on" : "memory off"}
          </Tag>
          {typeof pendingPlan.memory.requested_top_k === "number" ? (
            <Tag color="blue">
              {`requested ${formatMemoryResultsLabel(
                pendingPlan.memory.requested_top_k
              ).toLowerCase()}`}
            </Tag>
          ) : null}
          {typeof pendingPlan.memory.applied_count === "number" ? (
            <Tag color="purple">
              {`applied results: ${pendingPlan.memory.applied_count}`}
            </Tag>
          ) : null}
        </div>
      ) : null}
      {pendingPlan.companion ? (
        <div className="mt-1 flex flex-wrap items-center gap-1">
          <Tag color={pendingPlan.companion.enabled ? "green" : "default"}>
            {pendingPlan.companion.enabled ? "companion on" : "companion off"}
          </Tag>
          {typeof pendingPlan.companion.applied_card_count === "number" ? (
            <Tag color="cyan">
              {`applied cards: ${pendingPlan.companion.applied_card_count}`}
            </Tag>
          ) : null}
          {typeof pendingPlan.companion.applied_activity_count === "number" ? (
            <Tag color="geekblue">
              {`applied activity: ${pendingPlan.companion.applied_activity_count}`}
            </Tag>
          ) : null}
        </div>
      ) : null}
      <div className="mt-2 space-y-1">
        {pendingPlan.steps.map((step) => (
          <label key={step.idx} className="flex items-start gap-2 text-xs text-text">
            <Checkbox
              checked={approvedStepMap[step.idx] !== false}
              disabled={step.policy?.allow === false}
              onChange={(event) => {
                const nextChecked = event.target.checked
                setApprovedStepMap((prev) => ({
                  ...prev,
                  [step.idx]: nextChecked
                }))
              }}
            />
            <span>
              <span className="font-semibold">{`${step.idx}. ${step.tool}`}</span>
              {step.description ? ` - ${step.description}` : ""}
              <span className="ml-2 inline-flex flex-wrap gap-1 align-middle">
                {step.policy?.required_scope ? (
                  <Tag color="blue">{`scope: ${step.policy.required_scope}`}</Tag>
                ) : null}
                {step.policy?.requires_confirmation ? (
                  <Tag color="gold">confirm</Tag>
                ) : null}
                {step.policy?.allow === false ? (
                  <Tag color="red">{`blocked${step.policy.reason_code ? `: ${step.policy.reason_code}` : ""}`}</Tag>
                ) : null}
              </span>
              {step.policy?.allow === false && step.policy.reason ? (
                <div className="mt-1 text-[11px] text-danger">
                  {step.policy.reason}
                </div>
              ) : null}
            </span>
          </label>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-2">
        <Button
          size="small"
          type="primary"
          icon={<CheckCircle2 className="h-3.5 w-3.5" />}
          onClick={confirmPlan}
        >
          {t("sidepanel:persona.confirmPlan", "Confirm plan")}
        </Button>
        <Button
          size="small"
          icon={<XCircle className="h-3.5 w-3.5" />}
          onClick={cancelPlan}
        >
          {t("sidepanel:persona.cancelPlan", "Cancel")}
        </Button>
      </div>
    </div>
  ) : null

  const stateDocsCard = (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Typography.Text strong>
          {t("sidepanel:persona.stateDocs", "Persistent state docs")}
        </Typography.Text>
        <div className="flex flex-wrap items-center gap-1">
          <Tag
            data-testid="persona-state-dirty-tag"
            color={stateDocs.hasUnsavedPersonaStateChanges ? "gold" : "green"}
          >
            {stateDocs.stateDirtyLabel}
          </Tag>
          {stateDocs.stateLastModified ? (
            <Typography.Text type="secondary" className="text-xs">
              {`${t("sidepanel:persona.stateUpdatedPrefix", "updated")} ${stateDocs.stateLastModified}`}
            </Typography.Text>
          ) : null}
          <Button
            data-testid="persona-state-editor-toggle-button"
            size="small"
            onClick={() => {
              stateDocs.setPersonaStateEditorExpanded((prev) => !prev)
            }}
          >
            {stateDocs.stateEditorToggleLabel}
          </Button>
        </div>
      </div>
      {stateDocs.personaStateEditorExpanded ? (
        <>
          <div className="mt-2 grid gap-2">
            <Input.TextArea
              data-testid="persona-state-soul-input"
              value={stateDocs.soulMd}
              autoSize={{ minRows: 2, maxRows: 4 }}
              onChange={(event) => stateDocs.setSoulMd(event.target.value)}
              placeholder={t("sidepanel:persona.stateSoulPlaceholder", "soul.md")}
            />
            <Input.TextArea
              data-testid="persona-state-identity-input"
              value={stateDocs.identityMd}
              autoSize={{ minRows: 2, maxRows: 4 }}
              onChange={(event) => stateDocs.setIdentityMd(event.target.value)}
              placeholder={t("sidepanel:persona.stateIdentityPlaceholder", "identity.md")}
            />
            <Input.TextArea
              data-testid="persona-state-heartbeat-input"
              value={stateDocs.heartbeatMd}
              autoSize={{ minRows: 2, maxRows: 4 }}
              onChange={(event) => stateDocs.setHeartbeatMd(event.target.value)}
              placeholder={t("sidepanel:persona.stateHeartbeatPlaceholder", "heartbeat.md")}
            />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Button
              data-testid="persona-state-load-button"
              size="small"
              loading={stateDocs.personaStateLoading}
              disabled={!connected || stateDocs.personaStateSaving}
              onClick={() => {
                void stateDocs.loadPersonaStateDocs()
              }}
            >
              {t("sidepanel:persona.stateLoad", "Load state")}
            </Button>
            <Button
              data-testid="persona-state-save-button"
              size="small"
              type="primary"
              loading={stateDocs.personaStateSaving}
              disabled={!connected || !stateDocs.hasUnsavedPersonaStateChanges}
              onClick={() => {
                void stateDocs.savePersonaStateDocs()
              }}
            >
              {t("sidepanel:persona.stateSave", "Save state")}
            </Button>
            <Button
              data-testid="persona-state-revert-button"
              size="small"
              disabled={!stateDocs.hasUnsavedPersonaStateChanges || stateDocs.personaStateSaving}
              onClick={stateDocs.revertPersonaStateDraft}
            >
              {t("sidepanel:persona.stateRevert", "Revert")}
            </Button>
            <Button
              data-testid="persona-state-history-button"
              size="small"
              loading={stateDocs.personaStateHistoryLoading}
              disabled={!connected}
              onClick={() => {
                void stateDocs.loadPersonaStateHistory()
              }}
            >
              {t("sidepanel:persona.stateHistory", "Load history")}
            </Button>
          </div>
          {stateDocs.personaStateHistory.length > 0 ? (
            <div className="mt-3 space-y-2">
              {stateDocs.personaStateHistory.length > 1 ? (
                <div className="mb-1 flex items-center gap-2 text-xs">
                  <Typography.Text type="secondary" className="text-xs">
                    {t("sidepanel:persona.stateHistoryOrderLabel", "Order")}
                  </Typography.Text>
                  <Button
                    data-testid="persona-state-history-order-newest-button"
                    size="small"
                    type={stateDocs.personaStateHistoryOrder === "newest" ? "primary" : "default"}
                    onClick={() => {
                      stateDocs.setPersonaStateHistoryOrder("newest")
                    }}
                  >
                    {t("sidepanel:persona.stateHistoryOrderNewest", "Newest")}
                  </Button>
                  <Button
                    data-testid="persona-state-history-order-oldest-button"
                    size="small"
                    type={stateDocs.personaStateHistoryOrder === "oldest" ? "primary" : "default"}
                    onClick={() => {
                      stateDocs.setPersonaStateHistoryOrder("oldest")
                    }}
                  >
                    {t("sidepanel:persona.stateHistoryOrderOldest", "Oldest")}
                  </Button>
                </div>
              ) : null}
              {stateDocs.orderedPersonaStateHistory.map((entry) => (
                <div
                  data-testid={`persona-state-history-entry-${entry.entry_id}`}
                  key={entry.entry_id}
                  className="rounded border border-border bg-surface2 px-2 py-1.5 text-xs"
                >
                  <div className="flex flex-wrap items-center gap-1">
                    <Tag color={entry.is_active ? "green" : "default"}>
                      {entry.is_active
                        ? t("sidepanel:persona.stateHistoryActive", "active")
                        : t("sidepanel:persona.stateHistoryArchived", "archived")}
                    </Tag>
                    <Tag color="blue">{entry.field}</Tag>
                    {typeof entry.version === "number" ? (
                      <Tag color="purple">{`v${entry.version}`}</Tag>
                    ) : null}
                    <Button
                      data-testid={`persona-state-restore-${entry.entry_id}`}
                      size="small"
                      disabled={entry.is_active === true}
                      loading={stateDocs.restoringStateEntryId === entry.entry_id}
                      onClick={() => {
                        void stateDocs.restorePersonaStateHistoryEntry(entry.entry_id)
                      }}
                    >
                      {t("sidepanel:persona.stateRestore", "Restore")}
                    </Button>
                  </div>
                  <div className="mt-1 whitespace-pre-wrap text-text">
                    {String(entry.content || "")}
                  </div>
                  {entry.created_at || entry.last_modified ? (
                    <Typography.Text
                      data-testid={`persona-state-history-meta-${entry.entry_id}`}
                      type="secondary"
                      className="mt-1 block text-[11px]"
                    >
                      {[
                        entry.created_at
                          ? `${t("sidepanel:persona.stateHistoryCreated", "created")} ${entry.created_at}`
                          : null,
                        entry.last_modified
                          ? `${t("sidepanel:persona.stateHistoryUpdated", "updated")} ${entry.last_modified}`
                          : null
                      ]
                        .filter((item): item is string => Boolean(item))
                        .join(" · ")}
                    </Typography.Text>
                  ) : null}
                </div>
              ))}
            </div>
          ) : stateDocs.personaStateHistoryLoaded ? (
            <Typography.Text
              data-testid="persona-state-history-empty"
              type="secondary"
              className="mt-3 block text-xs"
            >
              {t(
                "sidepanel:persona.stateHistoryEmpty",
                "No state history entries yet."
              )}
            </Typography.Text>
          ) : null}
        </>
      ) : null}
    </div>
  )

  const transcriptPanel = (
    <div className="min-h-0 flex-1 overflow-auto rounded-lg border border-border bg-surface p-3">
      <div className="space-y-2">
        {logs.length === 0 ? (
          <Typography.Text type="secondary" className="text-xs">
            {isCompanionMode
              ? "Connect to companion and send a message to start."
              : t(
                  "sidepanel:persona.empty",
                  "Connect to persona and send a message to start."
                )}
          </Typography.Text>
        ) : (
          logs.map((entry) => (
            <div
              key={entry.id}
              className="rounded border border-border bg-surface2 px-2 py-1.5 text-xs"
            >
              <div className="mb-1 uppercase tracking-wide text-[10px] text-text-muted">
                {entry.kind}
              </div>
              <div className="whitespace-pre-wrap text-text">{entry.text}</div>
            </div>
          ))
        )}
      </div>
    </div>
  )

  const composerPanel = (
    <div className="flex flex-col gap-2">
      {isCompanionMode && companionPrompts.length ? (
        <div
          className="flex flex-wrap gap-2"
          data-testid="companion-conversation-prompts"
        >
          {companionPrompts.map((prompt) => (
            <Button
              key={prompt.prompt_id}
              size="small"
              onClick={() => {
                setInput(prompt.prompt_text)
              }}
              type="default"
            >
              {prompt.label}
            </Button>
          ))}
        </div>
      ) : null}
      <div className="flex items-end gap-2">
        <Input.TextArea
          value={input}
          autoSize={{ minRows: 2, maxRows: 4 }}
          onChange={(event) => setInput(event.target.value)}
          placeholder={
            isCompanionMode
              ? "Ask Companion..."
              : t("sidepanel:persona.inputPlaceholder", "Ask Persona...")
          }
          onPressEnter={(event) => {
            if (event.shiftKey) return
            event.preventDefault()
            sendUserMessage()
          }}
        />
        {capabilities?.hasPersonalization ? (
          <Button
            data-testid="persona-save-checkin-button"
            disabled={!canSaveCompanionCheckIn}
            loading={savingCompanionCheckIn}
            onClick={() => {
              void saveCompanionCheckIn()
            }}
          >
            {t("sidepanel:persona.saveCheckIn", "Save check-in")}
          </Button>
        ) : null}
        <Button
          type="primary"
          icon={<Send className="h-4 w-4" />}
          disabled={!canSend}
          onClick={sendUserMessage}
        >
          {t("common:send", "Send")}
        </Button>
      </div>
    </div>
  )

  const tabItems = [
    {
      key: "commands",
      label: t("sidepanel:persona.tabCommands", "Commands"),
      content: renderLazyPersonaTab(
        "commands",
        <LazyCommandsPanel
          selectedPersonaId={selectedPersonaId}
          selectedPersonaName={selectedPersonaName}
          isActive={activeTab === "commands"}
          analytics={voiceAnalytics}
          analyticsLoading={voiceAnalyticsLoading}
          handoffFocusRequest={
            setupOrch.setupHandoffFocusRequest?.tab === "commands"
              ? {
                  section: setupOrch.setupHandoffFocusRequest.section as "command_form" | "command_list",
                  token: setupOrch.setupHandoffFocusRequest.token
                }
              : null
          }
          onSetupHandoffFocusConsumed={setupOrch.handleSetupHandoffFocusConsumed}
          openCommandId={openCommandId}
          onOpenCommandHandled={handleOpenCommandHandled}
          draftCommandPhrase={draftCommandPhrase}
          draftCommandSource={draftCommandSource}
          onDraftCommandPhraseHandled={handleDraftCommandPhraseHandled}
          rerunAfterSaveCommandId={rerunAfterSaveCommandId}
          onRerunAfterSave={handleRerunAfterCommandSave}
          onCommandSaved={handleCommandSaved}
        />
      )
    },
    {
      key: "test-lab",
      label: t("sidepanel:persona.tabTestLab", "Test Lab"),
      content: renderLazyPersonaTab(
        "test-lab",
        <LazyTestLabPanel
          selectedPersonaId={selectedPersonaId}
          selectedPersonaName={selectedPersonaName}
          isActive={activeTab === "test-lab"}
          analytics={voiceAnalytics}
          handoffFocusRequest={
            setupOrch.setupHandoffFocusRequest?.tab === "test-lab"
              ? {
                  section: setupOrch.setupHandoffFocusRequest.section as "dry_run_form",
                  token: setupOrch.setupHandoffFocusRequest.token
                }
              : null
          }
          onSetupHandoffFocusConsumed={setupOrch.handleSetupHandoffFocusConsumed}
          initialHeardText={lastTestLabPhrase}
          rerunRequestToken={testLabRerunToken}
          onOpenCommand={handleOpenCommandFromTestLab}
          onCreateCommandDraft={handleCreateCommandFromTestLab}
          onDryRunCompleted={handleTestLabDryRunCompleted}
        />
      )
    },
    {
      key: "live",
      label: t("sidepanel:persona.tabLive", "Live Session"),
      content: renderLazyPersonaTab(
        "live",
        <LazyLiveSessionPanel
          controls={liveSessionControls}
          assistantVoice={assistantVoiceCard}
          error={liveSessionStatusPanels}
          pendingPlan={pendingPlanCard}
          transcript={transcriptPanel}
          composer={composerPanel}
        />
      )
    },
    {
      key: "profiles",
      label: t("sidepanel:persona.tabProfiles", "Profiles"),
      content: renderLazyPersonaTab(
        "profiles",
        <LazyProfilePanel
          selectedPersonaId={selectedPersonaId}
          selectedPersonaName={selectedPersonaName}
          personaCount={catalog.length}
          connected={connected}
          sessionId={sessionId}
          setup={savedPersonaSetup}
          onStartSetup={setupOrch.handleStartSetup}
          onResumeSetup={setupOrch.handleResumeSetup}
          onResetSetup={setupOrch.handleResetSetup}
          onRerunSetup={setupOrch.handleRerunSetup}
          onDefaultsSaved={handleProfileDefaultsSaved}
          isActive={activeTab === "profiles"}
          setupAnalytics={setupAnalytics}
          setupAnalyticsLoading={setupAnalyticsLoading}
          analytics={voiceAnalytics}
          analyticsLoading={voiceAnalyticsLoading}
          handoffFocusRequest={
            setupOrch.setupHandoffFocusRequest?.tab === "profiles"
              ? {
                  section: setupOrch.setupHandoffFocusRequest.section as
                    | "assistant_defaults"
                    | "confirmation_mode",
                  token: setupOrch.setupHandoffFocusRequest.token
                }
              : null
          }
          onSetupHandoffFocusConsumed={setupOrch.handleSetupHandoffFocusConsumed}
        />
      )
    },
    {
      key: "voice",
      label: t("sidepanel:persona.tabVoice", "Voice & Examples"),
      content: renderLazyPersonaTab(
        "voice",
        <LazyVoiceExamplesPanel
          selectedPersonaId={selectedPersonaId}
          selectedPersonaName={selectedPersonaName}
          isActive={activeTab === "voice"}
        />,
        {
          includeSetupHandoff: false
        }
      )
    },
    {
      key: "connections",
      label: t("sidepanel:persona.tabConnections", "Connections"),
      content: renderLazyPersonaTab(
        "connections",
        <LazyConnectionsPanel
          selectedPersonaId={selectedPersonaId}
          selectedPersonaName={selectedPersonaName}
          isActive={activeTab === "connections"}
          onConnectionSaved={handleConnectionSaved}
          onConnectionTestSucceeded={handleConnectionTestSucceeded}
          handoffFocusRequest={
            setupOrch.setupHandoffFocusRequest?.tab === "connections"
              ? {
                  section: setupOrch.setupHandoffFocusRequest.section as
                    | "connection_form"
                    | "saved_connections",
                  token: setupOrch.setupHandoffFocusRequest.token,
                  connectionId: setupOrch.setupHandoffFocusRequest.connectionId ?? null,
                  connectionName: setupOrch.setupHandoffFocusRequest.connectionName ?? null
                }
              : null
          }
          onSetupHandoffFocusConsumed={setupOrch.handleSetupHandoffFocusConsumed}
        />
      )
    },
    {
      key: "state",
      label: t("sidepanel:persona.tabStateDocs", "State Docs"),
      content: renderLazyPersonaTab(
        "state",
        <LazyStateDocsPanel>{stateDocsCard}</LazyStateDocsPanel>,
        {
          includeSetupHandoff: false
        }
      )
    },
    {
      key: "scopes",
      label: t("sidepanel:persona.tabScopes", "Scopes"),
      content: renderLazyPersonaTab(
        "scopes",
        <LazyScopesPanel selectedPersonaName={selectedPersonaName} />,
        {
          includeSetupHandoff: false
        }
      )
    },
    {
      key: "policies",
      label: t("sidepanel:persona.tabPolicies", "Policies"),
      content: renderLazyPersonaTab(
        "policies",
        <LazyPoliciesPanel hasPendingPlan={Boolean(pendingPlan)} />,
        {
          includeSetupHandoff: false
        }
      )
    }
  ]

  // ── Early-return gates ──
  if (uxState === "error_auth" || uxState === "configuring_auth") {
    return (
      <div data-testid="persona-route-root" className={routeRootClassName}>
        {routeHeader}
        <div className="p-4">
          <FeatureEmptyState
            title={
              isCompanionMode
                ? "Add your credentials to use Companion"
                : "Add your credentials to use Persona"
            }
            description={
              isCompanionMode
                ? "Companion conversation needs a reachable tldw server plus valid credentials before a session can start."
                : "Persona streaming needs a reachable tldw server plus valid credentials before a session can start."
            }
            primaryActionLabel={settingsLabel}
            onPrimaryAction={openSettings}
          />
        </div>
      </div>
    )
  }

  if (uxState === "unconfigured" || uxState === "configuring_url") {
    return (
      <div data-testid="persona-route-root" className={routeRootClassName}>
        {routeHeader}
        <div className="p-4">
          <FeatureEmptyState
            title={
              isCompanionMode
                ? "Finish setup to use Companion"
                : "Finish setup to use Persona"
            }
            description={
              isCompanionMode
                ? "Companion conversation depends on a configured tldw server before you can start a session."
                : "Persona streaming depends on a configured tldw server before you can start a session."
            }
            primaryActionLabel={setupActionLabel}
            onPrimaryAction={openSetup}
          />
        </div>
      </div>
    )
  }

  if (uxState === "error_unreachable") {
    return (
      <div data-testid="persona-route-root" className={routeRootClassName}>
        {routeHeader}
        <div className="p-4">
          <FeatureEmptyState
            title="Can't reach your tldw server right now"
            description={
              isCompanionMode
                ? "Companion conversation depends on a reachable tldw server. Review your server status and URL before trying again."
                : "Persona streaming depends on a reachable tldw server. Review your server status and URL before trying again."
            }
            primaryActionLabel={
              shell === "options"
                ? "Health & diagnostics"
                : settingsLabel
            }
            onPrimaryAction={
              shell === "options" ? openDiagnostics : openSettings
            }
            secondaryActionLabel={
              shell === "options" ? settingsLabel : undefined
            }
            onSecondaryAction={
              shell === "options" ? openSettings : undefined
            }
          />
        </div>
      </div>
    )
  }

  if (!isOnline) {
    return (
      <div data-testid="persona-route-root" className={routeRootClassName}>
        {routeHeader}
        <div className="p-4">
          <FeatureEmptyState
            title={
              isCompanionMode
                ? "Connect to use Companion"
                : t("sidepanel:persona.connectTitle", "Connect to use Persona")
            }
            description={
              isCompanionMode
                ? "Companion conversation runs on your tldw server. Connect to start a session."
                : t(
                    "sidepanel:persona.connectDescription",
                    "Persona streaming runs on your tldw server. Connect to a server to start a session."
                  )
            }
            primaryActionLabel={settingsLabel}
            onPrimaryAction={openSettings}
          />
        </div>
      </div>
    )
  }

  if (personaUnsupported) {
    return (
      <div data-testid="persona-route-root" className={routeRootClassName}>
        {routeHeader}
        <div className="p-4">
          <FeatureEmptyState
            title={
              isCompanionMode
                ? "Companion unavailable"
                : t("sidepanel:persona.unavailableTitle", "Persona unavailable")
            }
            description={
              isCompanionMode
                ? "This server does not currently advertise persona support for companion conversation."
                : t(
                    "sidepanel:persona.unavailableDescription",
                    "This server does not currently advertise persona support."
                  )
            }
            primaryActionLabel={t("sidepanel:header.settingsShortLabel", "Settings")}
            onPrimaryAction={() => navigate("/settings")}
          />
        </div>
      </div>
    )
  }

  return (
    <div
      data-testid="persona-route-root"
      className={routeRootClassName}
    >
      {routeHeader}
      {isCompanionMode ? (
        <div className="flex flex-1 flex-col gap-3 p-3">
          {liveSessionControls}
          {liveSessionStatusPanels}
          {pendingPlanCard}
          {transcriptPanel}
          {composerPanel}
        </div>
      ) : (
        <div className="flex flex-1 flex-col p-3">
          {setupOrch.personaSetupWizard.isSetupRequired && !setupOrch.setupCommandDetour && !setupOrch.setupLiveDetour ? (
            <AssistantSetupWizard
              catalog={catalog.map((persona) => ({
                id: String(persona.id || ""),
                name: String(persona.name || persona.id || "")
              }))}
              selectedPersonaId={selectedPersonaId}
              currentStep={setupOrch.personaSetupWizard.currentStep}
              postSetupTargetTab={setupOrch.setupIntentTargetTab || activeTab}
              progressItems={setupOrch.assistantSetupProgressItems}
              onResetSetup={setupOrch.handleResetSetup}
              voiceStepContent={
                setupOrch.personaSetupWizard.currentStep === "voice" ? (
                  <AssistantDefaultsPanel
                    selectedPersonaId={selectedPersonaId}
                    selectedPersonaName={selectedPersonaName}
                    isActive
                    analytics={null}
                    analyticsLoading={false}
                    onSaved={() => {
                      void setupOrch.handleSetupVoiceDefaultsSaved()
                    }}
                  />
                ) : undefined
              }
              commandsStepContent={
                setupOrch.personaSetupWizard.currentStep === "commands" ? (
                  <SetupStarterCommandsStep
                    saving={setupOrch.setupWizardSaving}
                    error={setupOrch.setupStepErrors.commands || null}
                    onCreateFromTemplate={(templateKey) => {
                      void setupOrch.handleCreateStarterCommandFromTemplate(templateKey)
                    }}
                    onCreateMcpStarter={(toolName, phrase) => {
                      void setupOrch.handleCreateMcpStarterCommand(toolName, phrase)
                    }}
                    onSkip={() => {
                      setupOrch.setSetupReviewSummaryDraft((current) => ({
                        ...current,
                        starterCommands: { mode: "skipped" }
                      }))
                      void setupOrch.advancePersonaSetupStep(
                        "safety",
                        "Failed to advance assistant setup",
                        undefined,
                        "commands"
                      )
                    }}
                  />
                ) : undefined
              }
              safetyStepContent={
                setupOrch.personaSetupWizard.currentStep === "safety" ? (
                  <SetupSafetyConnectionsStep
                    saving={setupOrch.setupWizardSaving}
                    error={setupOrch.setupStepErrors.safety || null}
                    currentConfirmationMode={
                      savedPersonaVoiceDefaults?.confirmation_mode || "destructive_only"
                    }
                    onContinue={(payload) => {
                      void setupOrch.handleSetupSafetyStepContinue(payload)
                    }}
                  />
                ) : undefined
              }
              testStepContent={
                setupOrch.personaSetupWizard.currentStep === "test" ? (
                  <SetupTestAndFinishStep
                    saving={setupOrch.setupWizardSaving}
                    dryRunLoading={setupOrch.setupWizardDryRunLoading}
                    liveConnected={connected}
                    error={setupOrch.setupStepErrors.test || null}
                    initialHeardText={setupOrch.setupNoMatchPhrase}
                    notice={setupOrch.setupTestResumeNote}
                    outcome={setupOrch.setupTestOutcome}
                    onRunDryRun={(heardText) => {
                      void setupOrch.handleRunSetupDryRun(heardText)
                    }}
                    onCreateCommandFromPhrase={handleCreateCommandFromSetupNoMatch}
                    onConnectLive={() => {
                      void connect()
                    }}
                    onRecoverInLiveSession={setupOrch.handleRecoverSetupInLiveSession}
                    onSendLive={(text) => {
                      sendSetupLiveTestMessage(text)
                    }}
                    onFinishWithDryRun={() => {
                      void setupOrch.completePersonaSetup("dry_run")
                    }}
                    onFinishWithLiveSession={() => {
                      void setupOrch.completePersonaSetup("live_session")
                    }}
                  />
                ) : undefined
              }
              saving={setupOrch.setupWizardSaving}
              error={setupOrch.currentSetupWizardError}
              onUsePersona={setupOrch.handleUsePersonaForSetup}
              onCreatePersona={setupOrch.handleCreatePersonaForSetup}
            />
          ) : (
            <PersonaGardenTabs
              activeKey={activeTab}
              onChange={handlePersonaTabChange}
              items={tabItems}
            />
          )}
        </div>
      )}
    </div>
  )
}

export default SidepanelPersona
