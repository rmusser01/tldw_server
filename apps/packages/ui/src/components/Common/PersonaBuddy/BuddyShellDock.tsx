import React from "react"

import type {
  PersonaBuddyShellPosition
} from "@/store/persona-buddy-shell"
import type { PersonaBuddySummary } from "@/types/persona-buddy"
import type {
  PersonaVisualPack,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { BuddyShellPopover } from "./BuddyShellPopover"
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
  visualDiagnostic?: PersonaVisualDiagnostic | null
  onVisualRenderError?: (error: PersonaVisualRenderError | null) => void
  position: PersonaBuddyShellPosition
  onToggle: () => void
  onDragHandlePointerDown: (event: React.PointerEvent<HTMLDivElement>) => void
  dockRef: React.RefObject<HTMLDivElement | null>
}

export const BuddyShellDock: React.FC<BuddyShellDockProps> = ({
  buddySummary,
  personaId = null,
  isOpen,
  isDormant = false,
  visualPack = null,
  visualState = "idle",
  visualDiagnostic = null,
  onVisualRenderError,
  position,
  onToggle,
  onDragHandlePointerDown,
  dockRef
}) => {
  const visualRenderer = visualPack
    ? getPersonaVisualRenderer(visualPack.renderer_type)
    : null
  const canMountVisualRenderer =
    !isDormant && Boolean(visualPack?.manifest && visualRenderer)

  return (
    <div
      ref={dockRef}
      data-testid="persona-buddy-dock"
      data-dormant={isDormant ? "true" : "false"}
      className="fixed z-[1100] flex flex-col gap-2"
      style={{
        left: position.x,
        top: position.y
      }}
    >
      <div
        data-testid="persona-buddy-drag-handle"
        onPointerDown={onDragHandlePointerDown}
        className="cursor-grab rounded-full border border-border bg-bg/95 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted shadow-sm backdrop-blur active:cursor-grabbing"
      >
        Drag Buddy
      </div>

      <button
        type="button"
        onClick={onToggle}
        disabled={isDormant}
        aria-expanded={isOpen}
        aria-label={`Toggle buddy for ${buddySummary.persona_name}`}
        className="flex min-w-[160px] items-center justify-between gap-3 rounded-2xl border border-border bg-bg/95 px-4 py-3 text-left shadow-xl backdrop-blur"
      >
        {canMountVisualRenderer && visualPack ? (
          <div className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded border border-border bg-surface2">
            <PersonaVisualRendererHost
              pack={visualPack}
              state={visualState}
              fallbackLabel={buddySummary.persona_name}
              className="max-h-10 max-w-10 object-contain"
              onRenderError={onVisualRenderError}
            />
          </div>
        ) : null}
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-text">
            {buddySummary.persona_name}
          </div>
          <div className="truncate text-xs text-text-muted">
            {buddySummary.visual?.species_id ?? "buddy unavailable"}
          </div>
        </div>
        <div className="text-lg leading-none text-text">
          {isDormant ? "·" : isOpen ? "−" : "+"}
        </div>
      </button>

      {visualDiagnostic ? (
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
        />
      ) : null}
    </div>
  )
}

export default BuddyShellDock
