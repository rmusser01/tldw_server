import React from "react"
import { createPortal } from "react-dom"

import { useSetting } from "@/hooks/useSetting"
import { useDesktop } from "@/hooks/useMediaQuery"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import {
  getPersonaVisualPack,
  listPersonaVisualPacks
} from "@/services/persona-visuals"
import { PERSONA_BUDDY_SHELL_ENABLED_SETTING } from "@/services/settings/ui-settings"
import {
  clampPersonaBuddyShellPosition,
  DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS,
  usePersonaBuddyShellStore
} from "@/store/persona-buddy-shell"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import type {
  PersonaBuddyPositionBucket,
  PersonaBuddyRenderContext,
  PersonaBuddySummary
} from "@/types/persona-buddy"
import type { PersonaVisualPack } from "@/types/persona-visuals"

import { useBuddyShellRenderContext } from "./BuddyShellRenderContext"
import { BuddyShellDock } from "./BuddyShellDock"
import {
  getPrimaryPersonaVisualDiagnostic,
  type PersonaVisualDiagnostic,
  type PersonaVisualDiagnosticCode
} from "./personaVisualDiagnostics"
import { resolvePersonaVisualState } from "./personaVisualState"

type BuddyShellHostProps = {
  root: "web" | "sidepanel"
}

type DragState = {
  offsetX: number
  offsetY: number
}

type ResolvedPersonaShellState = {
  hasTargetPersona: boolean
  activePersonaId: string | null
  fallbackName: string | null
  buddySummary: PersonaBuddySummary | null
}

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

  const positionBucket: PersonaBuddyPositionBucket =
    renderContext?.position_bucket ??
    (root === "sidepanel" ? "sidepanel-desktop" : "web-desktop")

  const isOpen = usePersonaBuddyShellStore((state) => state.isOpen)
  const setOpen = usePersonaBuddyShellStore((state) => state.setOpen)
  const resetSessionState = usePersonaBuddyShellStore(
    (state) => state.resetSessionState
  )
  const setPosition = usePersonaBuddyShellStore((state) => state.setPosition)
  const position = usePersonaBuddyShellStore(
    (state) =>
      state.positions[positionBucket] ??
      DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS[positionBucket]
  )

  React.useEffect(() => {
    resetSessionState()
  }, [resetSessionState])

  React.useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!dragStateRef.current || !dockRef.current) {
        return
      }

      const rect = dockRef.current.getBoundingClientRect()
      const nextPosition = clampPersonaBuddyShellPosition(
        {
          x: event.clientX - dragStateRef.current.offsetX,
          y: event.clientY - dragStateRef.current.offsetY
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

      setPosition(positionBucket, nextPosition)
    }

    const handlePointerUp = () => {
      dragStateRef.current = null
    }

    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
    }
  }, [positionBucket, setPosition])

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

  const handleDragHandlePointerDown = React.useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 || !dockRef.current) {
        return
      }

      const rect = dockRef.current.getBoundingClientRect()
      dragStateRef.current = {
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top
      }
      event.preventDefault()
    },
    []
  )

  const resolvedPersona = React.useMemo(
    () =>
      resolveActivePersonaSelection({
        renderContext,
        selectedAssistant
      }),
    [renderContext, selectedAssistant]
  )
  const [visualPack, setVisualPack] = React.useState<PersonaVisualPack | null>(null)
  const [visualPackLoadStatus, setVisualPackLoadStatus] =
    React.useState<PersonaVisualPackLoadStatus>("idle")
  const [visualPackLoadError, setVisualPackLoadError] =
    React.useState<unknown>(null)
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
  }, [resolvedPersona.activePersonaId, resolvedPersona.hasTargetPersona])

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
  const visualState =
    renderContext.visual_state ??
    resolvePersonaVisualState({
      liveVoiceState: renderContext.live_voice_state,
      activeToolStatus: renderContext.active_tool_status,
      wakeArmed: renderContext.wake_armed,
      recovering:
        Boolean(renderContext.recovery_mode) &&
        renderContext.recovery_mode !== "none",
      runtimeOverride: applicableRuntimeOverride,
      authoredTriggers: visualPack?.manifest?.authored_triggers,
      mcpRuntimeReason: applicableRuntimeOverride?.reason
    })
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

  return createPortal(
    <BuddyShellDock
      buddySummary={dockSummary}
      personaId={resolvedPersona.activePersonaId}
      isOpen={isOpen}
      isDormant={isDormant}
      visualPack={visualPack}
      visualState={visualState}
      visualDiagnostic={visualDiagnostic}
      onVisualRenderError={handleVisualRenderError}
      position={position}
      onToggle={() => setOpen(!isOpen)}
      onDragHandlePointerDown={handleDragHandlePointerDown}
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
