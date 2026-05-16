import React from "react"

import type { SetupReviewSummary } from "@/components/PersonaGarden/PersonaSetupHandoffCard"
import type { SetupSafetyConnectionDraft } from "@/components/PersonaGarden/SetupSafetyConnectionsStep"
import type { SetupTestOutcome } from "@/components/PersonaGarden/SetupTestAndFinishStep"
import { buildPersonaSetupProgress } from "@/components/PersonaGarden/personaSetupProgress"
import {
  getPersonaStarterCommandTemplate,
} from "@/components/PersonaGarden/personaStarterCommandTemplates"
import type { PersonaConfirmationMode, PersonaVoiceDefaults } from "@/hooks/useResolvedPersonaVoiceDefaults"
import type { PersonaSetupState, PersonaSetupStep } from "@/hooks/usePersonaSetupWizard"
import { usePersonaSetupWizard } from "@/hooks/usePersonaSetupWizard"
import type { PersonaGardenTabKey } from "@/utils/persona-garden-route"
import { tldwClient } from "@/services/tldw/TldwApiClient"

import type {
  PersonaInfo,
  PersonaProfileResponse,
  SetupStepErrors,
  SetupHandoffState,
  SetupHandoffSectionTarget,
  SetupHandoffFocusRequest,
  SetupHandoffConsumedAction,
  SetupCommandDetourState,
  SetupLiveDetourState,
  SetupVisualDetourState,
} from "../personaTypes"
import {
  DEFAULT_SETUP_REVIEW_SUMMARY,
  summarizeFallbackStarterCommands,
  pickAvailableConnectionName,
  deriveSetupHandoffRecommendedAction,
  toSetupHandoffActionTarget,
} from "../personaTypes"

// ── Types ──

export interface UsePersonaSetupOrchestratorDeps {
  selectedPersonaId: string
  setSelectedPersonaId: React.Dispatch<React.SetStateAction<string>>
  isCompanionMode: boolean
  activeTab: PersonaGardenTabKey
  setActiveTab: React.Dispatch<React.SetStateAction<PersonaGardenTabKey>>
  connected: boolean
  connecting: boolean
  catalog: PersonaInfo[]
  setCatalog: React.Dispatch<React.SetStateAction<PersonaInfo[]>>

  // Profile state managed by parent
  personaProfileLoading: boolean
  savedPersonaSetup: PersonaSetupState | null
  setSavedPersonaSetup: React.Dispatch<React.SetStateAction<PersonaSetupState | null>>
  savedPersonaVoiceDefaults: PersonaVoiceDefaults | null
  setSavedPersonaVoiceDefaults: React.Dispatch<React.SetStateAction<PersonaVoiceDefaults | null>>
  savedPersonaProfileVersion: number | null
  setSavedPersonaProfileVersion: React.Dispatch<React.SetStateAction<number | null>>

  // Analytics - use ref to break circular init order with analytics hook
  emitSetupAnalyticsEventRef: React.MutableRefObject<(event: Record<string, any>) => void>

  // State-docs guard - use ref to break circular init order with state docs hook
  confirmDiscardUnsavedStateDraftsRef: React.MutableRefObject<(reason?: string) => boolean>

  // Live-session helpers - use ref to break circular init order with live session hook
  triggerRecoveryReconnectRef: React.MutableRefObject<() => void>

  // Refs shared with live-session hook
  setupLiveDetourRef: React.MutableRefObject<SetupLiveDetourState | null>
  setupHandoffRef: React.MutableRefObject<SetupHandoffState | null>
  setupHandoffFocusRequestRef: React.MutableRefObject<SetupHandoffFocusRequest | null>
  activeTabRef: React.MutableRefObject<PersonaGardenTabKey>
  setupWizardAwaitingLiveResponseRef: React.MutableRefObject<boolean>
  setupWizardLastLiveTextRef: React.MutableRefObject<string>
}

export interface UsePersonaSetupOrchestratorReturn {
  // ── Wizard state ──
  personaSetupWizard: ReturnType<typeof usePersonaSetupWizard>
  assistantSetupProgressItems: ReturnType<typeof buildPersonaSetupProgress>
  currentSetupRunId: string | null

  // ── Setup step errors ──
  setupStepErrors: SetupStepErrors
  setSetupStepError: (step: PersonaSetupStep, message: string | null) => void
  clearSetupStepError: (step: PersonaSetupStep) => void
  clearAllSetupStepErrors: () => void
  currentSetupWizardError: string | null

  // ── Setup saving / dry-run ──
  setupWizardSaving: boolean
  setupWizardDryRunLoading: boolean
  setupTestOutcome: SetupTestOutcome | null
  setSetupTestOutcome: React.Dispatch<React.SetStateAction<SetupTestOutcome | null>>
  setupTestResumeNote: string | null
  setSetupTestResumeNote: React.Dispatch<React.SetStateAction<string | null>>

  // ── Detour state ──
  setupCommandDetour: SetupCommandDetourState | null
  setSetupCommandDetour: React.Dispatch<React.SetStateAction<SetupCommandDetourState | null>>
  setupLiveDetour: SetupLiveDetourState | null
  setSetupLiveDetour: React.Dispatch<React.SetStateAction<SetupLiveDetourState | null>>
  setupVisualDetour: SetupVisualDetourState | null
  setSetupVisualDetour: React.Dispatch<React.SetStateAction<SetupVisualDetourState | null>>
  setupNoMatchPhrase: string | null

  // ── Intent tracking ──
  setupIntentTargetTab: PersonaGardenTabKey | null

  // ── Handoff state ──
  setupHandoff: SetupHandoffState | null
  setSetupHandoff: React.Dispatch<React.SetStateAction<SetupHandoffState | null>>
  setupHandoffFocusRequest: SetupHandoffFocusRequest | null
  setSetupHandoffFocusRequest: React.Dispatch<React.SetStateAction<SetupHandoffFocusRequest | null>>
  setupReviewSummaryDraft: SetupReviewSummary
  setSetupReviewSummaryDraft: React.Dispatch<React.SetStateAction<SetupReviewSummary>>

  // ── Profile response helper ──
  applyPersonaProfileResponse: (
    payload: PersonaProfileResponse | null | undefined,
    fallback?: {
      voiceDefaults?: PersonaVoiceDefaults | null
      setup?: PersonaSetupState | null
    }
  ) => void

  // ── Setup step handlers ──
  handleSetupVoiceDefaultsSaved: () => Promise<void>
  advancePersonaSetupStep: (
    step: PersonaSetupState["current_step"],
    errorMessage: string,
    completedStep?: PersonaSetupStep,
    errorStep?: PersonaSetupStep
  ) => Promise<void>
  handleUsePersonaForSetup: (personaId: string) => Promise<void>
  handleCreatePersonaForSetup: (name: string) => Promise<void>
  handleCreateStarterCommandFromTemplate: (templateKey: string) => Promise<void>
  handleCreateMcpStarterCommand: (toolName: string, phrase: string) => Promise<void>
  handleSetupSafetyStepContinue: (payload: {
    confirmationMode: PersonaConfirmationMode
    connectionMode: "none" | "create"
    connection?: SetupSafetyConnectionDraft
  }) => Promise<void>
  completePersonaSetup: (testType: "dry_run" | "live_session") => Promise<void>
  handleStartSetup: () => void
  handleResumeSetup: () => void
  handleResetSetup: () => void
  handleRerunSetup: () => void
  handleRunSetupDryRun: (heardText: string) => Promise<void>

  // ── Command detour / live detour handlers ──
  handleCreateCommandFromSetupNoMatch: (heardText: string) => void
  handleRecoverSetupInLiveSession: (context: {
    source: "live_unavailable" | "live_failure"
    text: string
  }) => void
  handleReturnToSetupFromLiveDetour: () => void
  handleOpenVisualSetupDetour: () => void
  handleReturnToSetupFromVisualDetour: () => void
  /** Returns `true` when the detour was consumed and caller should clear command-editor state. */
  handleSetupDetourCommandSaved: (
    commandId: string,
    context: { fromDraft: boolean }
  ) => boolean
  consumeSetupHandoffAction: (action: SetupHandoffConsumedAction) => void

  // ── Handoff UI handlers ──
  dismissSetupHandoff: () => void
  openSetupHandoffTarget: (
    target: { tab: "live" } | SetupHandoffSectionTarget
  ) => void
  handleSetupHandoffFocusConsumed: (token: number) => void
  renderSetupHandoffCard: null // placeholder - rendered in parent
}

// ── Hook ──

export function usePersonaSetupOrchestrator(
  deps: UsePersonaSetupOrchestratorDeps
): UsePersonaSetupOrchestratorReturn {
  const {
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
  } = deps

  // Stable local wrappers that read from refs — avoids circular init dependency
  const emitSetupAnalyticsEvent = React.useCallback(
    (event: Record<string, any>) => emitSetupAnalyticsEventRef.current(event),
    [emitSetupAnalyticsEventRef]
  )
  const confirmDiscardUnsavedStateDrafts = React.useCallback(
    (reason?: string) => confirmDiscardUnsavedStateDraftsRef.current(reason),
    [confirmDiscardUnsavedStateDraftsRef]
  )
  const triggerRecoveryReconnect = React.useCallback(
    () => triggerRecoveryReconnectRef.current(),
    [triggerRecoveryReconnectRef]
  )

  // ── Setup wizard ──
  const personaSetupWizard = usePersonaSetupWizard({
    selectedPersonaId,
    isCompanionMode,
    loading: personaProfileLoading,
    setup: savedPersonaSetup,
  })

  // ── Catalog fetch for setup ──
  React.useEffect(() => {
    if (isCompanionMode || !personaSetupWizard.isSetupRequired || catalog.length > 0) {
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const response = await tldwClient.fetchWithAuth(
          "/api/v1/persona/catalog" as any,
          { method: "GET" }
        )
        if (!response.ok) return
        const payload = await response.json()
        if (!cancelled) {
          setCatalog(Array.isArray(payload) ? (payload as PersonaInfo[]) : [])
        }
      } catch {
        // Best-effort only for setup persona choice.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [catalog.length, isCompanionMode, personaSetupWizard.isSetupRequired, setCatalog])

  // ── Progress items ──
  const assistantSetupProgressItems = React.useMemo(
    () =>
      buildPersonaSetupProgress({
        ...(savedPersonaSetup || {}),
        status:
          savedPersonaSetup?.status ||
          (personaSetupWizard.isSetupRequired ? "in_progress" : "not_started"),
        current_step: personaSetupWizard.currentStep,
        completed_steps: Array.isArray(savedPersonaSetup?.completed_steps)
          ? savedPersonaSetup.completed_steps
          : [],
      }),
    [
      personaSetupWizard.currentStep,
      personaSetupWizard.isSetupRequired,
      savedPersonaSetup,
    ]
  )

  // ── Run ID ──
  const createSetupRunId = React.useCallback(() => {
    if (
      typeof globalThis !== "undefined" &&
      typeof globalThis.crypto?.randomUUID === "function"
    ) {
      return `setup-run-${globalThis.crypto.randomUUID()}`
    }
    return `setup-run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
  }, [])

  const currentSetupRunId = React.useMemo(() => {
    const normalized = String(savedPersonaSetup?.run_id || "").trim()
    return normalized || null
  }, [savedPersonaSetup?.run_id])

  // ── Step errors ──
  const [setupStepErrors, setSetupStepErrors] = React.useState<SetupStepErrors>({})

  const setSetupStepError = React.useCallback(
    (step: PersonaSetupStep, message: string | null) => {
      setSetupStepErrors((current) => ({
        ...current,
        [step]: message,
      }))
      if (message) {
        emitSetupAnalyticsEvent({
          eventType: "step_error",
          step,
          metadata: { message },
        })
      }
    },
    [emitSetupAnalyticsEvent]
  )

  const clearSetupStepError = React.useCallback(
    (step: PersonaSetupStep) => {
      setSetupStepErrors((current) => {
        if (!current[step]) return current
        return { ...current, [step]: null }
      })
    },
    []
  )

  const clearAllSetupStepErrors = React.useCallback(() => {
    setSetupStepErrors({})
  }, [])

  const currentSetupWizardError =
    personaSetupWizard.currentStep === "commands" ||
    personaSetupWizard.currentStep === "safety" ||
    personaSetupWizard.currentStep === "test"
      ? null
      : setupStepErrors[personaSetupWizard.currentStep] || null

  // ── Saving / test state ──
  const [setupWizardSaving, setSetupWizardSaving] = React.useState(false)
  const [setupWizardDryRunLoading, setSetupWizardDryRunLoading] = React.useState(false)
  const [setupTestOutcome, setSetupTestOutcome] =
    React.useState<SetupTestOutcome | null>(null)
  const [setupTestResumeNote, setSetupTestResumeNote] =
    React.useState<string | null>(null)

  // ── Detour state ──
  const [setupCommandDetour, setSetupCommandDetour] =
    React.useState<SetupCommandDetourState | null>(null)
  const [setupLiveDetour, setSetupLiveDetour] =
    React.useState<SetupLiveDetourState | null>(null)
  const [setupVisualDetour, setSetupVisualDetour] =
    React.useState<SetupVisualDetourState | null>(null)
  const [setupNoMatchPhrase, setSetupNoMatchPhrase] =
    React.useState<string | null>(null)

  // ── Intent tracking ──
  const [setupIntentTargetTab, setSetupIntentTargetTab] =
    React.useState<PersonaGardenTabKey | null>(null)
  const [setupIntentPersonaId, setSetupIntentPersonaId] = React.useState("")

  // ── Handoff state ──
  const [setupHandoff, setSetupHandoff] =
    React.useState<SetupHandoffState | null>(null)
  const [setupHandoffFocusRequest, setSetupHandoffFocusRequest] =
    React.useState<SetupHandoffFocusRequest | null>(null)
  const [setupReviewSummaryDraft, setSetupReviewSummaryDraft] =
    React.useState<SetupReviewSummary>(DEFAULT_SETUP_REVIEW_SUMMARY)

  const setupHandoffFocusTokenRef = React.useRef(0)
  const setupVisualDetourPersonaIdRef = React.useRef("")

  // ── Sync refs ──
  React.useEffect(() => {
    setupLiveDetourRef.current = setupLiveDetour
  }, [setupLiveDetour, setupLiveDetourRef])

  React.useEffect(() => {
    setupHandoffRef.current = setupHandoff
  }, [setupHandoff, setupHandoffRef])

  React.useEffect(() => {
    setupHandoffFocusRequestRef.current = setupHandoffFocusRequest
  }, [setupHandoffFocusRequest, setupHandoffFocusRequestRef])

  // ── Reset test state when step changes ──
  React.useEffect(() => {
    if (!personaSetupWizard.isSetupRequired || personaSetupWizard.currentStep === "test") {
      return
    }
    setSetupWizardDryRunLoading(false)
    setSetupTestOutcome(null)
    setSetupTestResumeNote(null)
    setSetupCommandDetour(null)
    setSetupLiveDetour(null)
    setSetupVisualDetour(null)
    setSetupNoMatchPhrase(null)
    setupWizardLastLiveTextRef.current = ""
    setupWizardAwaitingLiveResponseRef.current = false
  }, [personaSetupWizard.currentStep, personaSetupWizard.isSetupRequired, setupWizardAwaitingLiveResponseRef, setupWizardLastLiveTextRef])

  React.useEffect(() => {
    const normalizedPersonaId = String(selectedPersonaId || "").trim()
    if (
      setupVisualDetourPersonaIdRef.current &&
      setupVisualDetourPersonaIdRef.current !== normalizedPersonaId
    ) {
      setSetupVisualDetour(null)
    }
    setupVisualDetourPersonaIdRef.current = normalizedPersonaId
  }, [selectedPersonaId])

  // ── Emit step_viewed analytics ──
  React.useEffect(() => {
    if (!personaSetupWizard.isSetupRequired) return
    if (!currentSetupRunId) return
    void emitSetupAnalyticsEvent({
      eventType: "step_viewed",
      step: personaSetupWizard.currentStep,
      runId: currentSetupRunId,
    })
  }, [
    currentSetupRunId,
    emitSetupAnalyticsEvent,
    personaSetupWizard.currentStep,
    personaSetupWizard.isSetupRequired,
  ])

  // ── Intent tracking effect ──
  React.useEffect(() => {
    const normalizedPersonaId = String(selectedPersonaId || "").trim()
    if (!personaSetupWizard.isSetupRequired || !normalizedPersonaId) {
      return
    }
    if (setupIntentPersonaId !== normalizedPersonaId) {
      setSetupIntentPersonaId(normalizedPersonaId)
      setSetupIntentTargetTab(activeTab)
      setSetupReviewSummaryDraft(DEFAULT_SETUP_REVIEW_SUMMARY)
      return
    }
    if (!setupIntentTargetTab) {
      setSetupIntentTargetTab(activeTab)
    }
  }, [
    activeTab,
    personaSetupWizard.isSetupRequired,
    selectedPersonaId,
    setupIntentPersonaId,
    setupIntentTargetTab,
  ])

  // ── Profile helpers ──

  const buildSetupProfileUpdatePath = React.useCallback(
    (personaId: string) => {
      const normalizedPersonaId = String(personaId || "").trim()
      const basePath = `/api/v1/persona/profiles/${encodeURIComponent(normalizedPersonaId)}`
      if (!normalizedPersonaId || !savedPersonaProfileVersion) {
        return basePath
      }
      return `${basePath}?expected_version=${encodeURIComponent(
        String(savedPersonaProfileVersion)
      )}`
    },
    [savedPersonaProfileVersion]
  )

  const applyPersonaProfileResponse = React.useCallback(
    (
      payload: PersonaProfileResponse | null | undefined,
      fallback?: {
        voiceDefaults?: PersonaVoiceDefaults | null
        setup?: PersonaSetupState | null
      }
    ) => {
      if (payload && typeof payload.version === "number") {
        setSavedPersonaProfileVersion(payload.version)
      }
      if (payload?.voice_defaults !== undefined) {
        setSavedPersonaVoiceDefaults(payload.voice_defaults || null)
      } else if (fallback?.voiceDefaults !== undefined) {
        setSavedPersonaVoiceDefaults(fallback.voiceDefaults || null)
      }
      if (payload?.setup !== undefined) {
        setSavedPersonaSetup(payload.setup || null)
      } else if (fallback?.setup !== undefined) {
        setSavedPersonaSetup(fallback.setup || null)
      }
    },
    [setSavedPersonaProfileVersion, setSavedPersonaSetup, setSavedPersonaVoiceDefaults]
  )

  const buildPersonaSetupInProgress = React.useCallback(
    (
      step: PersonaSetupState["current_step"] = "voice",
      completedSteps: PersonaSetupStep[] = [],
      options?: { runId?: string | null }
    ): PersonaSetupState => ({
      status: "in_progress",
      version: 1,
      run_id:
        String(options?.runId || savedPersonaSetup?.run_id || "").trim() ||
        createSetupRunId(),
      current_step: step || "voice",
      completed_steps: completedSteps,
      completed_at: null,
      last_test_type: null,
    }),
    [createSetupRunId, savedPersonaSetup?.run_id]
  )

  const mergeCompletedSetupSteps = React.useCallback(
    (...steps: Array<PersonaSetupStep | null | undefined>) => {
      const next = new Set<PersonaSetupStep>(
        Array.isArray(savedPersonaSetup?.completed_steps)
          ? savedPersonaSetup.completed_steps
          : []
      )
      for (const step of steps) {
        if (!step) continue
        next.add(step)
      }
      return Array.from(next)
    },
    [savedPersonaSetup?.completed_steps]
  )

  // ── Review summary resolver ──

  const resolveSetupReviewSummary = React.useCallback(
    async (personaId: string): Promise<SetupReviewSummary> => {
      if (setupReviewSummaryDraft.confirmationMode !== null) {
        return setupReviewSummaryDraft
      }
      const fallbackSummary: SetupReviewSummary = {
        starterCommands: { mode: "skipped" },
        confirmationMode: savedPersonaVoiceDefaults?.confirmation_mode || null,
        connection: { mode: "skipped" },
      }
      try {
        const [commandsResult, connectionsResult] = await Promise.allSettled([
          tldwClient.fetchWithAuth(
            `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/voice-commands` as any,
            { method: "GET" }
          ),
          tldwClient.fetchWithAuth(
            `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/connections` as any,
            { method: "GET" }
          ),
        ])
        if (commandsResult.status === "fulfilled" && commandsResult.value.ok) {
          fallbackSummary.starterCommands = summarizeFallbackStarterCommands(
            await commandsResult.value.json()
          )
        }
        if (connectionsResult.status === "fulfilled" && connectionsResult.value.ok) {
          const connectionName = pickAvailableConnectionName(
            await connectionsResult.value.json()
          )
          fallbackSummary.connection = connectionName
            ? { mode: "available", name: connectionName }
            : { mode: "skipped" }
        }
      } catch {
        // Handoff summary enrichment is best-effort.
      }
      return fallbackSummary
    },
    [savedPersonaVoiceDefaults?.confirmation_mode, setupReviewSummaryDraft]
  )

  // ── Step advancement ──

  const handleSetupVoiceDefaultsSaved = React.useCallback(async () => {
    const personaId = String(selectedPersonaId || "").trim()
    if (!personaId) return
    setSetupWizardSaving(true)
    clearSetupStepError("voice")
    const nextSetup = buildPersonaSetupInProgress(
      "commands",
      mergeCompletedSetupSteps("voice")
    )
    try {
      const response = await tldwClient.fetchWithAuth(
        buildSetupProfileUpdatePath(personaId) as any,
        { method: "PATCH", body: { setup: nextSetup } }
      )
      if (!response.ok) {
        throw new Error(response.error || "Failed to advance assistant setup")
      }
      const payload = (await response.json()) as PersonaProfileResponse
      applyPersonaProfileResponse(payload, { setup: nextSetup })
      void emitSetupAnalyticsEvent({
        personaId,
        runId: nextSetup.run_id || undefined,
        eventType: "step_completed",
        step: "voice",
      })
    } catch (setupError: any) {
      setSetupStepError(
        "voice",
        String(setupError?.message || "Failed to advance assistant setup")
      )
    } finally {
      setSetupWizardSaving(false)
    }
  }, [
    applyPersonaProfileResponse,
    buildPersonaSetupInProgress,
    buildSetupProfileUpdatePath,
    clearSetupStepError,
    emitSetupAnalyticsEvent,
    mergeCompletedSetupSteps,
    setSetupStepError,
    selectedPersonaId,
  ])

  const advancePersonaSetupStep = React.useCallback(
    async (
      step: PersonaSetupState["current_step"],
      errorMessage: string,
      completedStep?: PersonaSetupStep,
      errorStep: PersonaSetupStep = step || "persona"
    ) => {
      const personaId = String(selectedPersonaId || "").trim()
      if (!personaId) return
      setSetupWizardSaving(true)
      clearSetupStepError(errorStep)
      const nextSetup = buildPersonaSetupInProgress(
        step,
        mergeCompletedSetupSteps(completedStep)
      )
      try {
        const response = await tldwClient.fetchWithAuth(
          buildSetupProfileUpdatePath(personaId) as any,
          { method: "PATCH", body: { setup: nextSetup } }
        )
        if (!response.ok) {
          throw new Error(response.error || errorMessage)
        }
        const payload = (await response.json()) as PersonaProfileResponse
        applyPersonaProfileResponse(payload, { setup: nextSetup })
        if (completedStep) {
          void emitSetupAnalyticsEvent({
            personaId,
            runId: nextSetup.run_id || undefined,
            eventType: "step_completed",
            step: completedStep,
          })
        }
      } catch (setupError: any) {
        setSetupStepError(errorStep, String(setupError?.message || errorMessage))
      } finally {
        setSetupWizardSaving(false)
      }
    },
    [
      applyPersonaProfileResponse,
      buildPersonaSetupInProgress,
      buildSetupProfileUpdatePath,
      clearSetupStepError,
      emitSetupAnalyticsEvent,
      mergeCompletedSetupSteps,
      setSetupStepError,
      selectedPersonaId,
    ]
  )

  // ── Use / create persona for setup ──

  const handleUsePersonaForSetup = React.useCallback(
    async (personaId: string) => {
      const nextPersonaId = String(personaId || "").trim()
      if (!nextPersonaId) return
      if (
        nextPersonaId !== selectedPersonaId &&
        !confirmDiscardUnsavedStateDrafts("persona_switch")
      ) {
        return
      }
      setSetupWizardSaving(true)
      clearSetupStepError("persona")
      const nextSetup = buildPersonaSetupInProgress("voice", ["persona"], {
        runId: createSetupRunId(),
      })
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(nextPersonaId)}` as any,
          { method: "PATCH", body: { setup: nextSetup } }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to update persona setup")
        }
        const payload = (await response.json()) as PersonaProfileResponse
        setSelectedPersonaId(nextPersonaId)
        applyPersonaProfileResponse(payload, {
          setup: nextSetup,
          voiceDefaults: payload?.voice_defaults || null,
        })
        void emitSetupAnalyticsEvent({
          personaId: nextPersonaId,
          runId: nextSetup.run_id || undefined,
          eventType: "setup_started",
        })
        void emitSetupAnalyticsEvent({
          personaId: nextPersonaId,
          runId: nextSetup.run_id || undefined,
          eventType: "step_completed",
          step: "persona",
        })
      } catch (setupError: any) {
        setSetupStepError(
          "persona",
          String(setupError?.message || "Failed to update persona setup")
        )
      } finally {
        setSetupWizardSaving(false)
      }
    },
    [
      applyPersonaProfileResponse,
      buildPersonaSetupInProgress,
      clearSetupStepError,
      confirmDiscardUnsavedStateDrafts,
      createSetupRunId,
      emitSetupAnalyticsEvent,
      setSelectedPersonaId,
      setSetupStepError,
      selectedPersonaId,
    ]
  )

  const handleCreatePersonaForSetup = React.useCallback(
    async (name: string) => {
      const normalizedName = String(name || "").trim()
      if (!normalizedName) return
      setSetupWizardSaving(true)
      clearSetupStepError("persona")
      const nextSetup = buildPersonaSetupInProgress("voice", ["persona"], {
        runId: createSetupRunId(),
      })
      try {
        const response = await tldwClient.fetchWithAuth(
          "/api/v1/persona/profiles" as any,
          {
            method: "POST",
            body: {
              name: normalizedName,
              mode: "persistent_scoped",
              setup: nextSetup,
            },
          }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to create persona")
        }
        const payload = (await response.json()) as PersonaProfileResponse
        const createdPersonaId = String(payload?.id || "").trim()
        if (createdPersonaId) {
          setCatalog((current) => {
            const exists = current.some(
              (persona) => String(persona.id || "") === createdPersonaId
            )
            if (exists) return current
            return [
              ...current,
              { id: createdPersonaId, name: normalizedName },
            ]
          })
          setSelectedPersonaId(createdPersonaId)
        }
        applyPersonaProfileResponse(payload, {
          setup: nextSetup,
          voiceDefaults: payload?.voice_defaults || null,
        })
        void emitSetupAnalyticsEvent({
          personaId: createdPersonaId || selectedPersonaId,
          runId: nextSetup.run_id || undefined,
          eventType: "setup_started",
        })
        void emitSetupAnalyticsEvent({
          personaId: createdPersonaId || selectedPersonaId,
          runId: nextSetup.run_id || undefined,
          eventType: "step_completed",
          step: "persona",
        })
      } catch (setupError: any) {
        setSetupStepError(
          "persona",
          String(setupError?.message || "Failed to create persona")
        )
      } finally {
        setSetupWizardSaving(false)
      }
    },
    [
      applyPersonaProfileResponse,
      buildPersonaSetupInProgress,
      clearSetupStepError,
      createSetupRunId,
      emitSetupAnalyticsEvent,
      selectedPersonaId,
      setCatalog,
      setSelectedPersonaId,
      setSetupStepError,
    ]
  )

  // ── Starter commands ──

  const handleCreateStarterCommandFromTemplate = React.useCallback(
    async (templateKey: string) => {
      const personaId = String(selectedPersonaId || "").trim()
      if (!personaId) return
      const template = getPersonaStarterCommandTemplate(templateKey)
      if (!template) return
      setSetupWizardSaving(true)
      clearSetupStepError("commands")
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/voice-commands` as any,
          {
            method: "POST",
            body: {
              name: template.name,
              description: template.commandDescription,
              phrases: template.phrases,
              action_type: "mcp_tool",
              action_config: { tool_name: template.toolName },
              priority: 50,
              enabled: true,
              requires_confirmation: template.requiresConfirmation,
            },
          }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to create starter command")
        }
        setSetupReviewSummaryDraft((current) => ({
          ...current,
          starterCommands: { mode: "added", count: 1 },
        }))
        await advancePersonaSetupStep(
          "safety",
          "Failed to advance assistant setup",
          "commands",
          "commands"
        )
      } catch (setupError: any) {
        setSetupStepError(
          "commands",
          String(setupError?.message || "Failed to create starter command")
        )
        setSetupWizardSaving(false)
      }
    },
    [advancePersonaSetupStep, clearSetupStepError, selectedPersonaId, setSetupStepError]
  )

  const handleCreateMcpStarterCommand = React.useCallback(
    async (toolName: string, phrase: string) => {
      const personaId = String(selectedPersonaId || "").trim()
      const normalizedToolName = String(toolName || "").trim()
      const normalizedPhrase = String(phrase || "").trim()
      if (!personaId || !normalizedToolName || !normalizedPhrase) return
      setSetupWizardSaving(true)
      clearSetupStepError("commands")
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/voice-commands` as any,
          {
            method: "POST",
            body: {
              name:
                normalizedPhrase.charAt(0).toUpperCase() +
                normalizedPhrase.slice(1),
              description: `Run ${normalizedToolName} from assistant setup`,
              phrases: [normalizedPhrase],
              action_type: "mcp_tool",
              action_config: { tool_name: normalizedToolName },
              priority: 50,
              enabled: true,
              requires_confirmation: false,
            },
          }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to create starter command")
        }
        setSetupReviewSummaryDraft((current) => ({
          ...current,
          starterCommands: { mode: "added", count: 1 },
        }))
        await advancePersonaSetupStep(
          "safety",
          "Failed to advance assistant setup",
          "commands",
          "commands"
        )
      } catch (setupError: any) {
        setSetupStepError(
          "commands",
          String(setupError?.message || "Failed to create starter command")
        )
        setSetupWizardSaving(false)
      }
    },
    [advancePersonaSetupStep, clearSetupStepError, selectedPersonaId, setSetupStepError]
  )

  // ── Safety step ──

  const handleSetupSafetyStepContinue = React.useCallback(
    async ({
      confirmationMode,
      connectionMode,
      connection,
    }: {
      confirmationMode: PersonaConfirmationMode
      connectionMode: "none" | "create"
      connection?: SetupSafetyConnectionDraft
    }) => {
      const personaId = String(selectedPersonaId || "").trim()
      if (!personaId) return
      setSetupWizardSaving(true)
      clearSetupStepError("safety")
      try {
        if (connectionMode === "create") {
          const normalizedName = String(connection?.name || "").trim()
          const normalizedBaseUrl = String(connection?.baseUrl || "").trim()
          if (!normalizedName || !normalizedBaseUrl) {
            throw new Error("Connection name and base URL are required.")
          }
          const connectionResponse = await tldwClient.fetchWithAuth(
            `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/connections` as any,
            {
              method: "POST",
              body: {
                name: normalizedName,
                base_url: normalizedBaseUrl,
                auth_type:
                  String(connection?.authType || "none").trim() || "none",
                secret:
                  String(connection?.secret || "").trim() || undefined,
              },
            }
          )
          if (!connectionResponse.ok) {
            throw new Error(
              connectionResponse.error || "Failed to create setup connection"
            )
          }
        }
        const mergedVoiceDefaults: PersonaVoiceDefaults = {
          ...(savedPersonaVoiceDefaults || {}),
          confirmation_mode: confirmationMode,
        }
        const response = await tldwClient.fetchWithAuth(
          buildSetupProfileUpdatePath(personaId) as any,
          {
            method: "PATCH",
            body: {
              voice_defaults: mergedVoiceDefaults,
              setup: buildPersonaSetupInProgress(
                "test",
                mergeCompletedSetupSteps("safety")
              ),
            },
          }
        )
        if (!response.ok) {
          throw new Error(
            response.error || "Failed to save assistant safety settings"
          )
        }
        const payload = (await response.json()) as PersonaProfileResponse
        setSetupReviewSummaryDraft((current) => ({
          ...current,
          confirmationMode,
          connection:
            connectionMode === "create" && String(connection?.name || "").trim()
              ? {
                  mode: "created",
                  name: String(connection?.name || "").trim(),
                }
              : { mode: "skipped" },
        }))
        applyPersonaProfileResponse(payload, {
          voiceDefaults: mergedVoiceDefaults,
          setup: buildPersonaSetupInProgress(
            "test",
            mergeCompletedSetupSteps("safety")
          ),
        })
        void emitSetupAnalyticsEvent({
          personaId,
          eventType: "step_completed",
          step: "safety",
        })
      } catch (setupError: any) {
        setSetupStepError(
          "safety",
          String(
            setupError?.message || "Failed to save assistant safety settings"
          )
        )
      } finally {
        setSetupWizardSaving(false)
      }
    },
    [
      applyPersonaProfileResponse,
      buildPersonaSetupInProgress,
      buildSetupProfileUpdatePath,
      clearSetupStepError,
      emitSetupAnalyticsEvent,
      mergeCompletedSetupSteps,
      savedPersonaVoiceDefaults,
      setSetupStepError,
      selectedPersonaId,
    ]
  )

  // ── Complete setup ──

  const completePersonaSetup = React.useCallback(
    async (testType: "dry_run" | "live_session") => {
      const personaId = String(selectedPersonaId || "").trim()
      if (!personaId) return
      setSetupWizardSaving(true)
      clearSetupStepError("test")
      try {
        const resolvedReviewSummary = await resolveSetupReviewSummary(personaId)
        const resolvedRunId =
          String(savedPersonaSetup?.run_id || "").trim() || createSetupRunId()
        const completedSetup: PersonaSetupState = {
          status: "completed",
          version: 1,
          run_id: resolvedRunId,
          current_step: "test",
          completed_steps: ["persona", "voice", "commands", "safety", "test"],
          completed_at: new Date().toISOString(),
          last_test_type: testType,
        }
        const response = await tldwClient.fetchWithAuth(
          buildSetupProfileUpdatePath(personaId) as any,
          { method: "PATCH", body: { setup: completedSetup } }
        )
        if (!response.ok) {
          throw new Error(
            response.error || "Failed to complete assistant setup"
          )
        }
        const payload = (await response.json()) as PersonaProfileResponse
        applyPersonaProfileResponse(payload, { setup: completedSetup })
        const handoffTargetTab = setupIntentTargetTab || activeTab
        const recommendedAction = deriveSetupHandoffRecommendedAction({
          completionType: testType,
          reviewSummary: resolvedReviewSummary,
        })
        setActiveTab(handoffTargetTab)
        setSetupHandoff({
          runId: resolvedRunId,
          targetTab: handoffTargetTab,
          completionType: testType,
          reviewSummary: resolvedReviewSummary,
          recommendedAction,
          consumedAction: null,
          compact: false,
        })
        void emitSetupAnalyticsEvent({
          personaId,
          runId: resolvedRunId,
          eventType: "step_completed",
          step: "test",
        })
        void emitSetupAnalyticsEvent({
          personaId,
          runId: resolvedRunId,
          eventType: "setup_completed",
          step: "test",
          completionType: testType,
        })
        setSetupIntentPersonaId("")
        setSetupIntentTargetTab(null)
        setSetupTestOutcome(null)
        setSetupTestResumeNote(null)
        setSetupCommandDetour(null)
        setSetupLiveDetour(null)
        setSetupVisualDetour(null)
        setSetupNoMatchPhrase(null)
        setupWizardLastLiveTextRef.current = ""
      } catch (setupError: any) {
        setSetupStepError(
          "test",
          String(setupError?.message || "Failed to complete assistant setup")
        )
      } finally {
        setSetupWizardSaving(false)
      }
    },
    [
      applyPersonaProfileResponse,
      activeTab,
      buildSetupProfileUpdatePath,
      clearSetupStepError,
      createSetupRunId,
      emitSetupAnalyticsEvent,
      resolveSetupReviewSummary,
      savedPersonaSetup?.run_id,
      selectedPersonaId,
      setActiveTab,
      setupIntentTargetTab,
      setSetupStepError,
      setupWizardLastLiveTextRef,
    ]
  )

  // ── Restart setup ──

  const restartPersonaSetupFromPersonaStep = React.useCallback(
    async (errorMessage: string) => {
      const personaId = String(selectedPersonaId || "").trim()
      if (!personaId) return
      const nextSetup = buildPersonaSetupInProgress("persona", [], {
        runId: createSetupRunId(),
      })
      setSetupWizardSaving(true)
      clearAllSetupStepErrors()
      setSetupIntentPersonaId(personaId)
      setSetupIntentTargetTab(activeTab)
      setSetupTestOutcome(null)
      setSetupTestResumeNote(null)
      setSetupCommandDetour(null)
      setSetupLiveDetour(null)
      setSetupVisualDetour(null)
      setSetupNoMatchPhrase(null)
      setSetupReviewSummaryDraft(DEFAULT_SETUP_REVIEW_SUMMARY)
      setupWizardLastLiveTextRef.current = ""
      setSetupHandoff(null)
      setupWizardAwaitingLiveResponseRef.current = false
      try {
        const response = await tldwClient.fetchWithAuth(
          buildSetupProfileUpdatePath(personaId) as any,
          { method: "PATCH", body: { setup: nextSetup } }
        )
        if (!response.ok) {
          throw new Error(response.error || errorMessage)
        }
        const payload = (await response.json()) as PersonaProfileResponse
        applyPersonaProfileResponse(payload, { setup: nextSetup })
        void emitSetupAnalyticsEvent({
          personaId,
          runId: nextSetup.run_id || undefined,
          eventType: "setup_started",
        })
      } catch (setupError: any) {
        setSetupStepError("persona", String(setupError?.message || errorMessage))
      } finally {
        setSetupWizardSaving(false)
      }
    },
    [
      activeTab,
      applyPersonaProfileResponse,
      buildPersonaSetupInProgress,
      buildSetupProfileUpdatePath,
      clearAllSetupStepErrors,
      createSetupRunId,
      emitSetupAnalyticsEvent,
      selectedPersonaId,
      setSetupStepError,
      setupWizardAwaitingLiveResponseRef,
      setupWizardLastLiveTextRef,
    ]
  )

  const handleStartSetup = React.useCallback(() => {
    void restartPersonaSetupFromPersonaStep("Failed to start assistant setup")
  }, [restartPersonaSetupFromPersonaStep])

  const handleResumeSetup = React.useCallback(() => {
    const personaId = String(selectedPersonaId || "").trim()
    if (!personaId) return
    setSetupIntentPersonaId(personaId)
    setSetupIntentTargetTab(activeTab)
    clearSetupStepError(savedPersonaSetup?.current_step || "persona")
  }, [activeTab, clearSetupStepError, savedPersonaSetup?.current_step, selectedPersonaId])

  const handleResetSetup = React.useCallback(() => {
    void restartPersonaSetupFromPersonaStep("Failed to reset assistant setup")
  }, [restartPersonaSetupFromPersonaStep])

  const handleRerunSetup = React.useCallback(() => {
    void restartPersonaSetupFromPersonaStep("Failed to rerun assistant setup")
  }, [restartPersonaSetupFromPersonaStep])

  // ── Dry run ──

  const handleRunSetupDryRun = React.useCallback(
    async (heardText: string) => {
      const personaId = String(selectedPersonaId || "").trim()
      const normalizedHeardText = String(heardText || "").trim()
      if (!personaId || !normalizedHeardText) return
      setSetupWizardDryRunLoading(true)
      clearSetupStepError("test")
      setSetupTestResumeNote(null)
      setSetupTestOutcome(null)
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/voice-commands/test` as any,
          { method: "POST", body: { heard_text: normalizedHeardText } }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to run setup dry-run")
        }
        const payload = (await response.json()) as {
          heard_text?: string
          matched?: boolean
          command_name?: string | null
          failure_phase?: string | null
        }
        const resolvedHeardText = String(
          payload?.heard_text || normalizedHeardText
        )
        if (payload?.matched === false) {
          setSetupTestOutcome({
            kind: "dry_run_no_match",
            heardText: resolvedHeardText,
            failurePhase: payload?.failure_phase || null,
          })
        } else {
          setSetupTestOutcome({
            kind: "dry_run_match",
            heardText: resolvedHeardText,
            commandName: payload?.command_name || null,
          })
        }
      } catch (setupError: any) {
        setSetupTestOutcome({
          kind: "dry_run_failure",
          message: String(
            setupError?.message || "Failed to run setup dry-run"
          ),
        })
      } finally {
        setSetupWizardDryRunLoading(false)
      }
    },
    [clearSetupStepError, selectedPersonaId]
  )

  // ── Command detour / live detour ──

  const handleCreateCommandFromSetupNoMatch = React.useCallback(
    (heardText: string) => {
      const normalizedHeardText = String(heardText || "").trim()
      if (!normalizedHeardText) return
      setSetupNoMatchPhrase(normalizedHeardText)
      setSetupTestResumeNote(null)
      setSetupCommandDetour({
        phrase: normalizedHeardText,
        returnStep: "test",
      })
      void emitSetupAnalyticsEvent({
        eventType: "detour_started",
        step: "test",
        detourSource: "dry_run_no_match",
      })
    },
    [emitSetupAnalyticsEvent]
  )

  const handleRecoverSetupInLiveSession = React.useCallback(
    (context: {
      source: "live_unavailable" | "live_failure"
      text: string
    }) => {
      setSetupLiveDetour({
        source: context.source,
        lastText: String(context.text || "").trim(),
      })
      setSetupTestResumeNote(null)
      setActiveTab("live")
      void emitSetupAnalyticsEvent({
        eventType: "detour_started",
        step: "test",
        detourSource: context.source,
      })
      if (
        context.source === "live_unavailable" &&
        !connected &&
        !connecting
      ) {
        triggerRecoveryReconnect()
      }
    },
    [connected, connecting, emitSetupAnalyticsEvent, setActiveTab, triggerRecoveryReconnect]
  )

  const handleReturnToSetupFromLiveDetour = React.useCallback(() => {
    const detourSource = setupLiveDetour?.source || null
    setSetupLiveDetour(null)
    setupWizardAwaitingLiveResponseRef.current = false
    setSetupTestResumeNote(
      "Live session is still available if you want to retry."
    )
    if (detourSource) {
      void emitSetupAnalyticsEvent({
        eventType: "detour_returned",
        step: "test",
        detourSource,
      })
    }
  }, [emitSetupAnalyticsEvent, setupLiveDetour?.source, setupWizardAwaitingLiveResponseRef])

  const handleOpenVisualSetupDetour = React.useCallback(() => {
    setSetupVisualDetour({
      source: "wizard_optional_card",
      returnStep: personaSetupWizard.currentStep,
    })
    void emitSetupAnalyticsEvent({
      eventType: "detour_started",
      step: personaSetupWizard.currentStep,
      detourSource: "wizard_optional_visuals",
    })
  }, [emitSetupAnalyticsEvent, personaSetupWizard.currentStep])

  const handleReturnToSetupFromVisualDetour = React.useCallback(() => {
    const returnStep = setupVisualDetour?.returnStep || personaSetupWizard.currentStep
    setSetupVisualDetour(null)
    void emitSetupAnalyticsEvent({
      eventType: "detour_returned",
      step: returnStep,
      detourSource: "wizard_optional_visuals",
    })
  }, [emitSetupAnalyticsEvent, personaSetupWizard.currentStep, setupVisualDetour?.returnStep])

  /** Returns `true` when the detour was consumed and caller should clear command-editor state. */
  const handleSetupDetourCommandSaved = React.useCallback(
    (_commandId: string, context: { fromDraft: boolean }): boolean => {
      if (!setupCommandDetour || !context.fromDraft) return false
      if (setupCommandDetour.returnStep !== "test") return false
      void emitSetupAnalyticsEvent({
        eventType: "detour_returned",
        step: "test",
        detourSource: "dry_run_no_match",
      })
      setSetupCommandDetour(null)
      setSetupTestOutcome(null)
      setSetupWizardDryRunLoading(false)
      setSetupTestResumeNote(
        "Command saved. Run the same phrase again to confirm setup."
      )
      setActiveTab(setupIntentTargetTab || "live")
      return true
    },
    [emitSetupAnalyticsEvent, setActiveTab, setupCommandDetour, setupIntentTargetTab]
  )

  // ── Consume handoff action ──

  const consumeSetupHandoffAction = React.useCallback(
    (action: SetupHandoffConsumedAction) => {
      const currentHandoff = setupHandoffRef.current
      if (
        !currentHandoff ||
        currentHandoff.compact ||
        currentHandoff.consumedAction
      ) {
        return
      }
      void emitSetupAnalyticsEvent({
        runId: currentHandoff.runId,
        eventType: "first_post_setup_action",
        actionTarget: toSetupHandoffActionTarget(action),
      })
      setSetupHandoff((existing) => {
        if (
          !existing ||
          existing.runId !== currentHandoff.runId ||
          existing.compact ||
          existing.consumedAction
        ) {
          return existing
        }
        return {
          ...existing,
          compact: true,
          consumedAction: action,
        }
      })
    },
    [emitSetupAnalyticsEvent, setupHandoffRef]
  )

  // ── Handoff UI handlers ──

  const dismissSetupHandoff = React.useCallback(() => {
    if (setupHandoff) {
      void emitSetupAnalyticsEvent({
        runId: setupHandoff.runId,
        eventType: "handoff_dismissed",
      })
    }
    setSetupHandoffFocusRequest(null)
    setSetupHandoff(null)
  }, [emitSetupAnalyticsEvent, setupHandoff])

  const openSetupHandoffTarget = React.useCallback(
    (target: { tab: "live" } | SetupHandoffSectionTarget) => {
      const tab = target.tab
      const currentHandoff = setupHandoffRef.current
      if (currentHandoff) {
        void emitSetupAnalyticsEvent({
          runId: currentHandoff.runId,
          eventType: "handoff_action_clicked",
          actionTarget: tab,
        })
      }
      setActiveTab(tab)
      if ("section" in target) {
        setupHandoffFocusTokenRef.current += 1
        setSetupHandoffFocusRequest({
          tab: target.tab,
          section: target.section,
          token: setupHandoffFocusTokenRef.current,
          connectionId:
            "connectionId" in target ? (target.connectionId ?? null) : null,
          connectionName:
            "connectionName" in target
              ? (target.connectionName ?? null)
              : null,
        })
      } else {
        setSetupHandoffFocusRequest(null)
      }
      setSetupHandoff((current) => {
        if (!current) return null
        if (current.targetTab === tab) return current
        return { ...current, targetTab: tab }
      })
    },
    [emitSetupAnalyticsEvent, setActiveTab, setupHandoffRef]
  )

  const handleSetupHandoffFocusConsumed = React.useCallback(
    (token: number) => {
      const currentRequest =
        setupHandoffFocusRequestRef.current || setupHandoffFocusRequest
      if (!currentRequest || currentRequest.token !== token) return
      const currentHandoff = setupHandoffRef.current || setupHandoff
      const actionTarget = `${currentRequest.tab}.${currentRequest.section}`
      void emitSetupAnalyticsEvent({
        runId: currentHandoff?.runId,
        eventType: "handoff_target_reached",
        actionTarget,
        metadata: {
          connection_id: currentRequest.connectionId || undefined,
          connection_name: currentRequest.connectionName || undefined,
          recommended_action: currentHandoff?.recommendedAction || undefined,
          completion_type: currentHandoff?.completionType || undefined,
        },
      })
      setSetupHandoffFocusRequest((current) => {
        if (!current) return current
        if (current.token !== token) return current
        return null
      })
    },
    [
      emitSetupAnalyticsEvent,
      setupHandoff,
      setupHandoffFocusRequest,
      setupHandoffFocusRequestRef,
      setupHandoffRef,
    ]
  )

  return {
    personaSetupWizard,
    assistantSetupProgressItems,
    currentSetupRunId,

    setupStepErrors,
    setSetupStepError,
    clearSetupStepError,
    clearAllSetupStepErrors,
    currentSetupWizardError,

    setupWizardSaving,
    setupWizardDryRunLoading,
    setupTestOutcome,
    setSetupTestOutcome,
    setupTestResumeNote,
    setSetupTestResumeNote,

    setupCommandDetour,
    setSetupCommandDetour,
    setupLiveDetour,
    setSetupLiveDetour,
    setupVisualDetour,
    setSetupVisualDetour,
    setupNoMatchPhrase,

    setupIntentTargetTab,

    setupHandoff,
    setSetupHandoff,
    setupHandoffFocusRequest,
    setSetupHandoffFocusRequest,
    setupReviewSummaryDraft,
    setSetupReviewSummaryDraft,

    applyPersonaProfileResponse,

    handleSetupVoiceDefaultsSaved,
    advancePersonaSetupStep,
    handleUsePersonaForSetup,
    handleCreatePersonaForSetup,
    handleCreateStarterCommandFromTemplate,
    handleCreateMcpStarterCommand,
    handleSetupSafetyStepContinue,
    completePersonaSetup,
    handleStartSetup,
    handleResumeSetup,
    handleResetSetup,
    handleRerunSetup,
    handleRunSetupDryRun,

    handleCreateCommandFromSetupNoMatch,
    handleRecoverSetupInLiveSession,
    handleReturnToSetupFromLiveDetour,
    handleOpenVisualSetupDetour,
    handleReturnToSetupFromVisualDetour,
    handleSetupDetourCommandSaved,
    consumeSetupHandoffAction,

    dismissSetupHandoff,
    openSetupHandoffTarget,
    handleSetupHandoffFocusConsumed,
    renderSetupHandoffCard: null,
  }
}
