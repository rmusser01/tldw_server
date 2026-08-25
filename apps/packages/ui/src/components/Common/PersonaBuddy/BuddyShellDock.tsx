import React from "react"

import type { PersonaBuddyShellPosition } from "@/store/persona-buddy-shell"
import type {
  PersonaBuddyLiveControlView,
  PersonaBuddySummary
} from "@/types/persona-buddy"
import type {
  PersonaVisualPack,
  PersonaAmbientMode,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { BuddyShellPopover } from "./BuddyShellPopover"
import type { PersonaCompanionSnapshot } from "./personaCompanionEngine"
import {
  getPersonaVisualDiagnosticToneClassName,
  type PersonaVisualDiagnostic
} from "./personaVisualDiagnostics"
import {
  getPersonaVisualRenderer,
  PersonaVisualRendererHost
} from "./personaVisualRenderers"
import type { PersonaVisualRenderError } from "./personaVisualTypes"

type BuddyShellDockProps = {
  buddySummary: PersonaBuddySummary
  personaId?: string | null
  isOpen: boolean
  isDormant?: boolean
  visualPack?: PersonaVisualPack | null
  visualState?: PersonaVisualStateId
  visualGeneration?: number
  companionPhase?: PersonaCompanionSnapshot["phase"]
  companionSuspension?: PersonaCompanionSnapshot["suspension"]
  companionTransientOffsetX?: number
  reducedMotion?: boolean
  visualFacing?: "left" | "right"
  nudgeActive?: boolean
  visualDiagnostic?: PersonaVisualDiagnostic | null
  liveControl?: PersonaBuddyLiveControlView | null
  onVisualRenderError?: (error: PersonaVisualRenderError | null) => void
  onVisualReady?: () => void
  onVisualFailure?: (error: PersonaVisualRenderError) => void
  onVisualComplete?: () => void
  position: PersonaBuddyShellPosition
  onOpenControls: () => void
  onCloseControls: () => void
  onBuddyPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => void
  onBuddyKeyDown: (event: React.KeyboardEvent<HTMLButtonElement>) => void
  onFocusWithinChange?: (focused: boolean) => void
  showFirstUseHint?: boolean
  onDismissFirstUseHint?: () => void
  globalAmbientMode?: PersonaAmbientMode
  personaAmbientMode?: PersonaAmbientMode | null
  effectiveAmbientMode?: PersonaAmbientMode
  ambientSurface?: "web" | "sidepanel"
  ambientPreferenceMessage?: string | null
  onGlobalAmbientModeChange?: (mode: PersonaAmbientMode) => void
  onPersonaAmbientModeChange?: (mode: PersonaAmbientMode | null) => void
  dockRef: React.RefObject<HTMLDivElement | null>
}

export const BuddyShellDock: React.FC<BuddyShellDockProps> = ({
  buddySummary,
  personaId = null,
  isOpen,
  isDormant = false,
  visualPack = null,
  visualState = "idle",
  visualGeneration = 0,
  companionPhase = "idle",
  companionSuspension = "none",
  companionTransientOffsetX = 0,
  reducedMotion = false,
  visualFacing = "right",
  nudgeActive = false,
  visualDiagnostic = null,
  liveControl = null,
  onVisualRenderError,
  onVisualReady,
  onVisualFailure,
  onVisualComplete,
  position,
  onOpenControls,
  onCloseControls,
  onBuddyPointerDown,
  onBuddyKeyDown,
  onFocusWithinChange,
  showFirstUseHint = false,
  onDismissFirstUseHint,
  globalAmbientMode,
  personaAmbientMode,
  effectiveAmbientMode,
  ambientSurface,
  ambientPreferenceMessage,
  onGlobalAmbientModeChange,
  onPersonaAmbientModeChange,
  dockRef
}) => {
  const visualRenderer = visualPack
    ? getPersonaVisualRenderer(visualPack.renderer_type)
    : null
  const canMountVisualRenderer =
    !isDormant && Boolean(visualPack?.manifest && visualRenderer)
  const showVisualDiagnostic =
    Boolean(visualDiagnostic) &&
    (isOpen || visualDiagnostic?.severity !== "info")
  const focusedLiveSession = liveControl?.focusedSession ?? null
  const urgentCount = focusedLiveSession?.pendingApprovalCount ?? 0
  const liveStatusLabel = urgentCount > 0
    ? "Needs approval"
    : focusedLiveSession?.lifecycle === "recovering"
        ? "Recovering"
        : focusedLiveSession?.lifecycle === "offline"
          ? "Offline"
          : liveControl?.streamState === "error"
            ? "Error"
          : null

  return (
    <div
      ref={dockRef}
      data-testid="persona-buddy-dock"
      data-dormant={isDormant ? "true" : "false"}
      data-companion-phase={companionPhase}
      data-companion-suspension={companionSuspension}
      data-companion-generation={visualGeneration}
      data-companion-requested-state={visualState}
      data-companion-effective-mode={effectiveAmbientMode}
      data-companion-transient-offset-x={companionTransientOffsetX}
      className="group fixed z-[1100] flex flex-col gap-2"
      onFocusCapture={() => onFocusWithinChange?.(true)}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          onFocusWithinChange?.(false)
        }
      }}
      style={{
        left: position.x,
        top: position.y
      }}
    >
      <button
        type="button"
        onPointerDown={onBuddyPointerDown}
        onKeyDown={onBuddyKeyDown}
        disabled={isDormant}
        aria-expanded={isOpen}
        aria-label={`Buddy for ${buddySummary.persona_name}`}
        className="relative flex h-16 w-16 cursor-grab items-center justify-center rounded-2xl border border-border bg-bg/90 p-2 shadow-lg backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary active:cursor-grabbing"
      >
        {canMountVisualRenderer && visualPack ? (
          <div
            data-testid="persona-buddy-visual-wrapper"
            className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-surface2 transition-transform duration-150"
            style={{
              transform: `${visualFacing === "left" ? "scaleX(-1)" : "scaleX(1)"} ${nudgeActive ? "translateX(4px)" : "translateX(0)"}`
            }}
          >
            <PersonaVisualRendererHost
              pack={visualPack}
              requestedState={visualState}
              generation={visualGeneration}
              reducedMotion={reducedMotion}
              fallbackLabel={buddySummary.persona_name}
              className="max-h-10 max-w-10 object-contain"
              onRenderError={onVisualRenderError}
              onReady={onVisualReady}
              onFailure={onVisualFailure}
              onComplete={onVisualComplete}
            />
          </div>
        ) : (
          <span aria-hidden="true" className="h-3 w-3 rounded-full bg-text-muted/50" />
        )}
        {liveStatusLabel ? (
          <div
            data-testid="persona-buddy-live-status"
            className="absolute -bottom-6 left-0 whitespace-nowrap rounded-full border border-border bg-bg/95 px-2 py-0.5 text-[11px] font-medium text-text-muted shadow-sm"
          >
            {liveStatusLabel}
          </div>
        ) : null}
        {urgentCount > 0 ? (
          <span
            data-testid="persona-buddy-urgent-badge"
            className="inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full bg-danger px-1.5 text-[11px] font-semibold text-white"
          >
            {urgentCount}
          </span>
        ) : null}
      </button>

      <button
        type="button"
        aria-label="Open Buddy controls"
        onClick={onOpenControls}
        className="absolute -right-2 -top-2 inline-flex h-6 w-6 items-center justify-center rounded-full border border-border bg-bg text-sm text-text opacity-0 shadow-sm transition-opacity focus:opacity-100 group-focus-within:opacity-100 [@media(pointer:coarse)]:opacity-100"
      >
        ⋯
      </button>

      {showFirstUseHint ? (
        <div data-testid="persona-buddy-first-use-hint" className="max-w-[220px] rounded-lg border border-border bg-bg/95 px-3 py-2 text-xs text-text-muted shadow-sm">
          Click to react · double-click for controls · drag to move
          <button type="button" className="ml-2 font-medium text-text" onClick={onDismissFirstUseHint}>
            Got it
          </button>
        </div>
      ) : null}

      {showVisualDiagnostic && visualDiagnostic ? (
        <div
          data-testid="persona-buddy-visual-diagnostic"
          data-severity={visualDiagnostic.severity}
          className={`max-w-[220px] rounded-lg border px-3 py-2 text-xs leading-5 shadow-sm backdrop-blur ${getPersonaVisualDiagnosticToneClassName(visualDiagnostic.severity)}`}
        >
          <div className="font-medium text-inherit">{visualDiagnostic.title}</div>
          <div>{visualDiagnostic.message}</div>
        </div>
      ) : null}

      {isOpen && !isDormant ? (
        <BuddyShellPopover
          buddySummary={buddySummary}
          personaId={personaId}
          visualDiagnostic={visualDiagnostic}
          liveControl={liveControl}
          globalAmbientMode={globalAmbientMode}
          personaAmbientMode={personaAmbientMode}
          effectiveAmbientMode={effectiveAmbientMode}
          ambientSurface={ambientSurface}
          ambientPreferenceMessage={ambientPreferenceMessage}
          onGlobalAmbientModeChange={onGlobalAmbientModeChange}
          onPersonaAmbientModeChange={onPersonaAmbientModeChange}
          onClose={onCloseControls}
        />
      ) : null}
    </div>
  )
}

export default BuddyShellDock
