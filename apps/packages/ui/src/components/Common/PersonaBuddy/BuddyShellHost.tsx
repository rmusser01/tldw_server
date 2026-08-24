import React from "react"
import { createPortal } from "react-dom"

import { useSetting } from "@/hooks/useSetting"
import { useDesktop, useMediaQuery } from "@/hooks/useMediaQuery"
import { usePersonaLiveControl } from "@/hooks/usePersonaLiveControl"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import {
  getBuddyPreferences,
  getPersonaBuddyPreferences,
  updateBuddyPreferences,
  updatePersonaBuddyPreferences
} from "@/services/persona-buddy"
import {
  getPersonaVisualPack,
  listPersonaVisualPacks
} from "@/services/persona-visuals"
import { PERSONA_BUDDY_SHELL_ENABLED_SETTING } from "@/services/settings/ui-settings"
import {
  clampPersonaBuddyShellPosition,
  DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS,
  usePersonaBuddyShellStore,
  type PersonaBuddyShellPosition
} from "@/store/persona-buddy-shell"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import type {
  PersonaBuddyPositionBucket,
  PersonaBuddyPreferences,
  PersonaBuddyOverridePreferences,
  PersonaBuddyLiveControlView,
  PersonaBuddyRenderContext,
  PersonaBuddySummary
} from "@/types/persona-buddy"
import {
  isPersonaVisualCustomStateIdText,
  PERSONA_VISUAL_PACK_ACTIVATED_EVENT
} from "@/types/persona-visuals"
import type {
  PersonaAmbientMode,
  PersonaVisualPack,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { useBuddyShellRenderContext } from "./BuddyShellRenderContext"
import { BuddyShellDock } from "./BuddyShellDock"
import {
  getPrimaryPersonaVisualDiagnostic,
  type PersonaVisualDiagnostic,
  type PersonaVisualDiagnosticCode
} from "./personaVisualDiagnostics"
import { resolvePersonaVisualState } from "./personaVisualState"
import { resolveEffectiveAmbientMode } from "./personaCompanionPolicy"
import { usePersonaCompanion } from "./usePersonaCompanion"

type BuddyShellHostProps = {
  root: "web" | "sidepanel"
}

type DragState = {
  pointerId: number
  startClientX: number
  startClientY: number
  startPosition: PersonaBuddyShellPosition
  currentPosition: PersonaBuddyShellPosition
  dragging: boolean
  target: HTMLButtonElement
}

type ResolvedPersonaShellState = {
  hasTargetPersona: boolean
  activePersonaId: string | null
  fallbackName: string | null
  buddySummary: PersonaBuddySummary | null
}

const BUDDY_DRAG_THRESHOLD_PX = 8
const BUDDY_CLICK_DELAY_MS = 300
const BUDDY_NUDGE_DURATION_MS = 160

const ensurePortalRoot = () => {
  if (typeof document === "undefined") return null
  return document.getElementById("tldw-portal-root")
}

const isPersonaSelection = (
  value: unknown
): value is {
  kind: "persona"
  id: string
  name: string
  buddy_summary?: PersonaBuddySummary | null
} => {
  if (!value || typeof value !== "object") {
    return false
  }
  const candidate = value as Record<string, unknown>
  return (
    candidate.kind === "persona" &&
    typeof candidate.id === "string" &&
    typeof candidate.name === "string"
  )
}

const hasExplicitBuddySummary = (
  renderContext:
    | {
        buddy_summary?: PersonaBuddySummary | null
      }
    | null
    | undefined
) =>
  Boolean(
    renderContext &&
      Object.prototype.hasOwnProperty.call(renderContext, "buddy_summary")
  )

const resolveActivePersonaSelection = ({
  renderContext,
  selectedAssistant
}: {
  renderContext:
    | PersonaBuddyRenderContext
    | null
    | undefined
  selectedAssistant: unknown
}): ResolvedPersonaShellState => {
  if (!renderContext?.surface_active) {
    return {
      hasTargetPersona: false,
      activePersonaId: null,
      fallbackName: null,
      buddySummary: null
    }
  }

  const selectedPersona = isPersonaSelection(selectedAssistant)
    ? selectedAssistant
    : null
  const canUseSelectedAssistantFallback =
    renderContext.persona_source === "selected-assistant-fallback" &&
    selectedPersona
  const selectionMatches =
    canUseSelectedAssistantFallback &&
    (!renderContext.active_persona_id ||
      selectedPersona.id === renderContext.active_persona_id)
  const hasExplicitTargetPersona =
    Boolean(renderContext.active_persona_id) || Boolean(selectionMatches)

  if (hasExplicitBuddySummary(renderContext)) {
    if (renderContext.buddy_summary) {
      return {
        hasTargetPersona: true,
        activePersonaId: renderContext.active_persona_id || selectedPersona?.id || null,
        fallbackName: renderContext.buddy_summary.persona_name,
        buddySummary: renderContext.buddy_summary
      }
    }

    return {
      hasTargetPersona: hasExplicitTargetPersona,
      activePersonaId: renderContext.active_persona_id || selectedPersona?.id || null,
      fallbackName: selectionMatches ? selectedPersona.name : null,
      buddySummary: null
    }
  }

  if (selectionMatches) {
    return {
      hasTargetPersona: true,
      activePersonaId: selectedPersona.id,
      fallbackName: selectedPersona.name,
      buddySummary: selectedPersona.buddy_summary ?? null
    }
  }

  return {
    hasTargetPersona: false,
    activePersonaId: null,
    fallbackName: null,
    buddySummary: null
  }
}

const hasVisualPackAssetMap = (pack: PersonaVisualPack | null): boolean =>
  Boolean(pack?.assets_by_id && Object.keys(pack.assets_by_id).length > 0)

type PersonaVisualPackLoadStatus = "idle" | "loading" | "loaded" | "error"

type PersonaVisualRenderErrorState = {
  key: string
  error: PersonaVisualDiagnosticCode
}

type BuddyShellHostInnerProps = {
  root: "web" | "sidepanel"
  renderContext: NonNullable<ReturnType<typeof useBuddyShellRenderContext>>
  selectedAssistant: unknown
}

const BuddyShellHostInner: React.FC<BuddyShellHostInnerProps> = ({
  root,
  renderContext,
  selectedAssistant
}) => {
  const dockRef = React.useRef<HTMLDivElement | null>(null)
  const dragStateRef = React.useRef<DragState | null>(null)
  const clickTimerRef = React.useRef<number | null>(null)
  const nudgeTimerRef = React.useRef<number | null>(null)
  const [dragPosition, setDragPosition] = React.useState<PersonaBuddyShellPosition | null>(null)
  const [isDragging, setIsDragging] = React.useState(false)
  const [focusWithin, setFocusWithin] = React.useState(false)
  const [nudgeActive, setNudgeActive] = React.useState(false)
  const [visibility, setVisibility] = React.useState<"visible" | "hidden">(
    () => typeof document !== "undefined" && document.visibilityState === "hidden"
      ? "hidden"
      : "visible"
  )
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)")

  const positionBucket: PersonaBuddyPositionBucket =
    renderContext?.position_bucket ??
    (root === "sidepanel" ? "sidepanel-desktop" : "web-desktop")

  const isOpen = usePersonaBuddyShellStore((state) => state.isOpen)
  const setOpen = usePersonaBuddyShellStore((state) => state.setOpen)
  const resetSessionState = usePersonaBuddyShellStore(
    (state) => state.resetSessionState
  )
  const setPosition = usePersonaBuddyShellStore((state) => state.setPosition)
  const firstUseHintDismissed = usePersonaBuddyShellStore(
    (state) => state.firstUseHintDismissed
  )
  const dismissFirstUseHint = usePersonaBuddyShellStore(
    (state) => state.dismissFirstUseHint
  )
  const position = usePersonaBuddyShellStore(
    (state) =>
      state.positions[positionBucket] ??
      DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS[positionBucket]
  )

  React.useEffect(() => {
    resetSessionState()
  }, [resetSessionState])

  React.useEffect(() => {
    const handleVisibility = () => setVisibility(
      document.visibilityState === "hidden" ? "hidden" : "visible"
    )
    document.addEventListener("visibilitychange", handleVisibility)
    return () => document.removeEventListener("visibilitychange", handleVisibility)
  }, [])

  React.useEffect(() => {
    const clampPersistedPosition = () => {
      if (!dockRef.current) {
        return
      }

      const rect = dockRef.current.getBoundingClientRect()
      const clampedPosition = clampPersonaBuddyShellPosition(
        position,
        positionBucket,
        {
          viewportWidth: window.innerWidth,
          viewportHeight: window.innerHeight,
          shellWidth: rect.width,
          shellHeight: rect.height,
          margin: 16
        }
      )

      if (
        clampedPosition.x !== position.x ||
        clampedPosition.y !== position.y
      ) {
        setPosition(positionBucket, clampedPosition)
      }
    }

    clampPersistedPosition()
    window.addEventListener("resize", clampPersistedPosition)
    return () => {
      window.removeEventListener("resize", clampPersistedPosition)
    }
  }, [position, positionBucket, setPosition])

  const resolvedPersona = React.useMemo(
    () =>
      resolveActivePersonaSelection({
        renderContext,
        selectedAssistant
      }),
    [renderContext, selectedAssistant]
  )
  const [globalPreferences, setGlobalPreferences] =
    React.useState<PersonaBuddyPreferences>({
      ambient_mode: "expressive",
      version: null,
      stored: false
    })
  const [personaPreferences, setPersonaPreferences] =
    React.useState<PersonaBuddyOverridePreferences | null>(null)
  const [preferenceReadFailed, setPreferenceReadFailed] = React.useState(true)
  const [ambientPreferenceMessage, setAmbientPreferenceMessage] =
    React.useState<string | null>(null)

  const refreshAmbientPreferences = React.useCallback(async () => {
    const personaId = String(resolvedPersona.activePersonaId ?? "").trim()
    if (!personaId) {
      setPersonaPreferences(null)
      setPreferenceReadFailed(true)
      return
    }
    try {
      const [globalResult, personaResult] = await Promise.all([
        getBuddyPreferences(),
        getPersonaBuddyPreferences(personaId)
      ])
      setGlobalPreferences(globalResult)
      setPersonaPreferences(personaResult)
      setPreferenceReadFailed(false)
    } catch {
      setPreferenceReadFailed(true)
    }
  }, [resolvedPersona.activePersonaId])

  React.useEffect(() => {
    setPreferenceReadFailed(true)
    setAmbientPreferenceMessage(null)
    void refreshAmbientPreferences()
  }, [refreshAmbientPreferences])

  const effectiveAmbientMode = resolveEffectiveAmbientMode({
    persona: personaPreferences?.ambient_mode,
    global: globalPreferences.ambient_mode,
    readFailed: preferenceReadFailed,
    surface: root
  })

  const handleGlobalAmbientModeChange = React.useCallback(
    async (mode: PersonaAmbientMode) => {
      const previous = globalPreferences
      setGlobalPreferences({
        ambient_mode: mode,
        version: previous.version,
        stored: true
      })
      setAmbientPreferenceMessage(null)
      try {
        setGlobalPreferences(await updateBuddyPreferences({
          ambient_mode: mode,
          expected_version: previous.version
        }))
      } catch (error) {
        if ((error as { status?: number })?.status === 409) {
          await refreshAmbientPreferences()
          setAmbientPreferenceMessage("Settings changed elsewhere. Latest values were loaded.")
        } else {
          setGlobalPreferences(previous)
          setAmbientPreferenceMessage("Buddy settings could not be saved.")
        }
      }
    },
    [globalPreferences, refreshAmbientPreferences]
  )

  const handlePersonaAmbientModeChange = React.useCallback(
    async (mode: PersonaAmbientMode | null) => {
      const personaId = String(resolvedPersona.activePersonaId ?? "").trim()
      const previous = personaPreferences
      if (!personaId || !previous) return
      setPersonaPreferences({
        ambient_mode: mode,
        version: previous.version,
        stored: mode !== null
      })
      setAmbientPreferenceMessage(null)
      try {
        setPersonaPreferences(await updatePersonaBuddyPreferences(personaId, {
          ambient_mode: mode,
          expected_version: previous.version
        }))
      } catch (error) {
        if ((error as { status?: number })?.status === 409) {
          await refreshAmbientPreferences()
          setAmbientPreferenceMessage("Settings changed elsewhere. Latest values were loaded.")
        } else {
          setPersonaPreferences(previous)
          setAmbientPreferenceMessage("Buddy settings could not be saved.")
        }
      }
    },
    [personaPreferences, refreshAmbientPreferences, resolvedPersona.activePersonaId]
  )
  const { capabilities } = useServerCapabilities()
  const liveControlEnabled = Boolean(capabilities?.hasPersonaLiveControl)
  const liveControl = usePersonaLiveControl({
    autoLoad: resolvedPersona.hasTargetPersona && liveControlEnabled,
    defaultPersonaId: resolvedPersona.activePersonaId,
    surface: renderContext.surface_id
  })
  const [visualPack, setVisualPack] = React.useState<PersonaVisualPack | null>(null)
  const [visualPackLoadStatus, setVisualPackLoadStatus] =
    React.useState<PersonaVisualPackLoadStatus>("idle")
  const [visualPackLoadError, setVisualPackLoadError] =
    React.useState<unknown>(null)
  const [visualPackRefreshNonce, setVisualPackRefreshNonce] = React.useState(0)
  const [visualRenderError, setVisualRenderError] =
    React.useState<PersonaVisualRenderErrorState | null>(null)
  const runtimeOverride = usePersonaVisualRuntimeStore((state) => state.override)
  const setVisualRuntimeDiagnostics = usePersonaVisualRuntimeStore(
    (state) => state.setRuntimeDiagnostics
  )
  const clearVisualRuntimeDiagnostics = usePersonaVisualRuntimeStore(
    (state) => state.clearRuntimeDiagnostics
  )
  const clearExpiredVisualOverride = usePersonaVisualRuntimeStore(
    (state) => state.clearExpired
  )
  const visualDiagnosticsSourceId = `${root}:${
    renderContext.surface_id || "unknown"
  }`

  React.useEffect(() => {
    clearExpiredVisualOverride()
    if (typeof window === "undefined") return undefined
    const timer = window.setInterval(() => clearExpiredVisualOverride(), 1000)
    return () => window.clearInterval(timer)
  }, [clearExpiredVisualOverride])

  React.useEffect(() => {
    if (typeof window === "undefined") return undefined
    const handlePackActivated = (event: Event) => {
      const detail = (event as CustomEvent<{ personaId?: unknown }>).detail
      const eventPersonaId = String(detail?.personaId ?? "").trim()
      const activePersonaId = String(resolvedPersona.activePersonaId ?? "").trim()
      if (eventPersonaId && eventPersonaId === activePersonaId) {
        setVisualPackRefreshNonce((current) => current + 1)
      }
    }
    window.addEventListener(PERSONA_VISUAL_PACK_ACTIVATED_EVENT, handlePackActivated)
    return () => {
      window.removeEventListener(
        PERSONA_VISUAL_PACK_ACTIVATED_EVENT,
        handlePackActivated
      )
    }
  }, [resolvedPersona.activePersonaId])

  React.useEffect(() => {
    const activePersonaId = String(resolvedPersona.activePersonaId || "").trim()
    if (!resolvedPersona.hasTargetPersona || !activePersonaId) {
      setVisualPack(null)
      setVisualPackLoadStatus("idle")
      setVisualPackLoadError(null)
      return undefined
    }

    let cancelled = false
    setVisualPack(null)
    setVisualPackLoadStatus("loading")
    setVisualPackLoadError(null)
    ;(async () => {
      try {
        const response = await listPersonaVisualPacks(activePersonaId)
        let activePack =
          response.active_pack ??
          response.packs.find((pack) => pack.status === "active") ??
          null
        if (activePack && !hasVisualPackAssetMap(activePack)) {
          activePack = await getPersonaVisualPack(activePersonaId, activePack.id)
        }
        if (!cancelled) {
          setVisualPack(activePack)
          setVisualPackLoadStatus("loaded")
        }
      } catch (error) {
        if (!cancelled) {
          setVisualPack(null)
          setVisualPackLoadStatus("error")
          setVisualPackLoadError(error)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [
    resolvedPersona.activePersonaId,
    resolvedPersona.hasTargetPersona,
    visualPackRefreshNonce
  ])

  const buddySummary = resolvedPersona.buddySummary
  const isDormant = resolvedPersona.hasTargetPersona && !buddySummary?.has_buddy
  const applicableRuntimeOverride =
    runtimeOverride &&
    runtimeOverride.personaId === resolvedPersona.activePersonaId &&
    (!renderContext.live_session_id ||
      !runtimeOverride.sessionId ||
      runtimeOverride.sessionId === renderContext.live_session_id)
      ? runtimeOverride
      : null
  const runtimeStateIds = React.useMemo(() => {
    const states = visualPack?.manifest?.states
    const stateCatalog = visualPack?.manifest?.state_catalog
    return Array.from(
      new Set([...Object.keys(states || {}), ...Object.keys(stateCatalog || {})])
    ).filter(isPersonaVisualCustomStateIdText)
  }, [visualPack])
  const visualState =
    renderContext.visual_state ??
    resolvePersonaVisualState({
      liveVoiceState: renderContext.live_voice_state,
      activeToolName: renderContext.active_tool_name,
      activeToolStatus: renderContext.active_tool_status,
      wakeArmed: renderContext.wake_armed,
      recovering:
        Boolean(renderContext.recovery_mode) &&
        renderContext.recovery_mode !== "none",
      runtimeOverride: applicableRuntimeOverride,
      runtimeStateIds,
      authoredTriggers: visualPack?.manifest?.authored_triggers,
      mcpRuntimeReason: applicableRuntimeOverride?.reason
    })
  const availableStates = React.useMemo(
    () => Object.keys(visualPack?.manifest?.states ?? {}) as PersonaVisualStateId[],
    [visualPack?.manifest?.states]
  )
  const companion = usePersonaCompanion({
    personaId: resolvedPersona.activePersonaId,
    packId: visualPack?.id ?? null,
    packRevision: visualPack?.revision_number ?? visualPack?.version ?? null,
    semanticState: visualState,
    mode: effectiveAmbientMode,
    surface: root,
    visibility,
    controlsOpen: isOpen,
    focusWithin,
    dragging: isDragging,
    reducedMotion,
    behavior: visualPack?.companion_behavior ?? null,
    availableStates,
    mirrorSafeStates: [],
    horizontalBounds: {
      min: 16 - position.x,
      max:
        typeof window === "undefined"
          ? 16 - position.x
          : Math.max(16 - position.x, window.innerWidth - position.x - 80)
    }
  })
  const {
    snapshot: companionSnapshot,
    react: companionReact,
    completeAction: completeCompanionAction
  } = companion

  const acknowledgeWithoutState = React.useCallback(() => {
    if (reducedMotion || visualState !== "idle") return
    if (nudgeTimerRef.current !== null) window.clearTimeout(nudgeTimerRef.current)
    setNudgeActive(true)
    nudgeTimerRef.current = window.setTimeout(() => {
      setNudgeActive(false)
      nudgeTimerRef.current = null
    }, BUDDY_NUDGE_DURATION_MS)
  }, [reducedMotion, visualState])

  const reactToBuddy = React.useCallback(
    (trigger: "click" | "space" | "drag") => {
      if (!companionReact(trigger)) acknowledgeWithoutState()
    },
    [acknowledgeWithoutState, companionReact]
  )

  const scheduleBuddyClick = React.useCallback(() => {
    if (clickTimerRef.current !== null) {
      window.clearTimeout(clickTimerRef.current)
      clickTimerRef.current = null
      setOpen(true)
      return
    }
    clickTimerRef.current = window.setTimeout(() => {
      clickTimerRef.current = null
      reactToBuddy("click")
    }, BUDDY_CLICK_DELAY_MS)
  }, [reactToBuddy, setOpen])

  const handleBuddyPointerDown = React.useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0) return
      const startPosition = {
        x: position.x + companionSnapshot.transientOffsetX,
        y: position.y
      }
      dragStateRef.current = {
        pointerId: event.pointerId,
        startClientX: event.clientX,
        startClientY: event.clientY,
        startPosition,
        currentPosition: startPosition,
        dragging: false,
        target: event.currentTarget
      }
      event.currentTarget.setPointerCapture?.(event.pointerId)
      event.preventDefault()
    },
    [companionSnapshot.transientOffsetX, position]
  )

  React.useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const drag = dragStateRef.current
      if (!drag || drag.pointerId !== event.pointerId || !dockRef.current) return
      const deltaX = event.clientX - drag.startClientX
      const deltaY = event.clientY - drag.startClientY
      if (!drag.dragging && Math.hypot(deltaX, deltaY) < BUDDY_DRAG_THRESHOLD_PX) return
      if (!drag.dragging) {
        drag.dragging = true
        setIsDragging(true)
        if (clickTimerRef.current !== null) {
          window.clearTimeout(clickTimerRef.current)
          clickTimerRef.current = null
        }
      }
      const rect = dockRef.current.getBoundingClientRect()
      drag.currentPosition = clampPersonaBuddyShellPosition(
        {
          x: drag.startPosition.x + deltaX,
          y: drag.startPosition.y + deltaY
        },
        positionBucket,
        {
          viewportWidth: window.innerWidth,
          viewportHeight: window.innerHeight,
          shellWidth: rect.width,
          shellHeight: rect.height,
          margin: 16
        }
      )
      setDragPosition(drag.currentPosition)
    }
    const finishPointer = (event: PointerEvent, cancelled: boolean) => {
      const drag = dragStateRef.current
      if (!drag || drag.pointerId !== event.pointerId) return
      drag.target.releasePointerCapture?.(drag.pointerId)
      if (drag.dragging) {
        setPosition(positionBucket, drag.currentPosition)
        setDragPosition(null)
        setIsDragging(false)
        if (!cancelled) reactToBuddy("drag")
      } else if (!cancelled) {
        scheduleBuddyClick()
      }
      dragStateRef.current = null
    }
    const handlePointerUp = (event: PointerEvent) => finishPointer(event, false)
    const handlePointerCancel = (event: PointerEvent) => finishPointer(event, true)
    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    window.addEventListener("pointercancel", handlePointerCancel)
    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
      window.removeEventListener("pointercancel", handlePointerCancel)
    }
  }, [positionBucket, reactToBuddy, scheduleBuddyClick, setPosition])

  React.useEffect(() => () => {
    if (clickTimerRef.current !== null) window.clearTimeout(clickTimerRef.current)
    if (nudgeTimerRef.current !== null) window.clearTimeout(nudgeTimerRef.current)
  }, [])

  const handleBuddyKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === "Enter") {
        event.preventDefault()
        setOpen(true)
      } else if (event.key === " ") {
        event.preventDefault()
        reactToBuddy("space")
      }
    },
    [reactToBuddy, setOpen]
  )
  const visualRenderKey = `${resolvedPersona.activePersonaId || ""}:${
    visualPack?.id || ""
  }:${visualState}`
  const activeVisualRenderError =
    visualRenderError?.key === visualRenderKey ? visualRenderError.error : null
  const handleVisualRenderError = React.useCallback(
    (error: PersonaVisualDiagnosticCode | null) => {
      if (!error) {
        setVisualRenderError((current) =>
          current?.key === visualRenderKey ? null : current
        )
        return
      }
      setVisualRenderError({ key: visualRenderKey, error })
    },
    [visualRenderKey]
  )
  const rendererActionToken = companionSnapshot.actionToken
  const handleVisualFailure = React.useCallback(
    (error: PersonaVisualDiagnosticCode) => {
      handleVisualRenderError(error)
      if (rendererActionToken !== null) {
        completeCompanionAction(rendererActionToken, false)
      }
    },
    [completeCompanionAction, handleVisualRenderError, rendererActionToken]
  )
  const handleVisualComplete = React.useCallback(() => {
    if (rendererActionToken !== null) {
      completeCompanionAction(rendererActionToken, true)
    }
  }, [completeCompanionAction, rendererActionToken])
  const visualDiagnostic: PersonaVisualDiagnostic | null = React.useMemo(
    () =>
      getPrimaryPersonaVisualDiagnostic({
        pack: visualPack,
        visualState,
        loadStatus: visualPackLoadStatus,
        loadError: visualPackLoadError,
        renderError: activeVisualRenderError,
        includeNoActivePack: visualPackLoadStatus === "loaded"
      }),
    [
      activeVisualRenderError,
      visualPack,
      visualPackLoadError,
      visualPackLoadStatus,
      visualState
    ]
  )
  const liveControlView: PersonaBuddyLiveControlView | null =
    liveControlEnabled
      ? {
          ...liveControl,
          voiceIsListening: renderContext.live_voice_is_listening ?? undefined,
          voiceState: renderContext.live_voice_state ?? null
        }
      : null

  React.useEffect(() => {
    const activePersonaId = String(resolvedPersona.activePersonaId || "").trim()
    if (!resolvedPersona.hasTargetPersona || !activePersonaId) {
      clearVisualRuntimeDiagnostics(visualDiagnosticsSourceId)
      return undefined
    }

    setVisualRuntimeDiagnostics({
      sourceId: visualDiagnosticsSourceId,
      personaId: activePersonaId,
      sessionId: renderContext.live_session_id ?? null,
      packId: visualPack?.id ?? null,
      packTitle: visualPack?.title ?? null,
      packLoadStatus: visualPackLoadStatus,
      visualState,
      diagnostic: visualDiagnostic,
      updatedAt: Date.now()
    })
    return () => {
      clearVisualRuntimeDiagnostics(visualDiagnosticsSourceId)
    }
  }, [
    clearVisualRuntimeDiagnostics,
    renderContext.live_session_id,
    resolvedPersona.activePersonaId,
    resolvedPersona.hasTargetPersona,
    setVisualRuntimeDiagnostics,
    visualDiagnostic,
    visualDiagnosticsSourceId,
    visualPack?.id,
    visualPack?.title,
    visualPackLoadStatus,
    visualState
  ])
  const portalRoot = ensurePortalRoot()

  if (!resolvedPersona.hasTargetPersona) {
    return null
  }

  if (!portalRoot) {
    return null
  }

  const dockSummary: PersonaBuddySummary =
    buddySummary
      ? buddySummary
      : {
          has_buddy: false,
          persona_name: resolvedPersona.fallbackName || "Persona Buddy",
          role_summary: null,
          visual: null
        }
  const dockPosition = dragPosition ?? {
    x: position.x + companionSnapshot.transientOffsetX,
    y: position.y
  }

  return createPortal(
    <BuddyShellDock
      buddySummary={dockSummary}
      personaId={resolvedPersona.activePersonaId}
      isOpen={isOpen}
      isDormant={isDormant}
      visualPack={visualPack}
      visualState={companionSnapshot.requestedState}
      visualGeneration={companionSnapshot.generation}
      reducedMotion={reducedMotion}
      visualFacing={companionSnapshot.facing}
      nudgeActive={nudgeActive}
      visualDiagnostic={visualDiagnostic}
      liveControl={liveControlView}
      onVisualRenderError={handleVisualRenderError}
      onVisualFailure={handleVisualFailure}
      onVisualComplete={handleVisualComplete}
      position={dockPosition}
      onOpenControls={() => setOpen(true)}
      onBuddyPointerDown={handleBuddyPointerDown}
      onBuddyKeyDown={handleBuddyKeyDown}
      onFocusWithinChange={setFocusWithin}
      showFirstUseHint={!firstUseHintDismissed}
      onDismissFirstUseHint={dismissFirstUseHint}
      globalAmbientMode={globalPreferences.ambient_mode}
      personaAmbientMode={personaPreferences?.ambient_mode ?? null}
      effectiveAmbientMode={effectiveAmbientMode}
      ambientSurface={root}
      ambientPreferenceMessage={ambientPreferenceMessage}
      onGlobalAmbientModeChange={(mode) => void handleGlobalAmbientModeChange(mode)}
      onPersonaAmbientModeChange={(mode) => void handlePersonaAmbientModeChange(mode)}
      dockRef={dockRef}
    />,
    portalRoot
  )
}

export const BuddyShellHost: React.FC<BuddyShellHostProps> = ({ root }) => {
  const renderContext = useBuddyShellRenderContext()
  const [selectedAssistant] = useSelectedAssistant()
  const [buddyShellEnabled] = useSetting(PERSONA_BUDDY_SHELL_ENABLED_SETTING)
  const isDesktop = useDesktop()

  if (!buddyShellEnabled) {
    return null
  }

  if (!renderContext?.surface_active) {
    return null
  }

  if (root !== "sidepanel" && !isDesktop) {
    return null
  }

  return (
    <BuddyShellHostInner
      root={root}
      renderContext={renderContext}
      selectedAssistant={selectedAssistant}
    />
  )
}

export default BuddyShellHost
