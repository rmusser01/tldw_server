import React from "react"

import type {
  PersonaBuddyShellPosition
} from "@/store/persona-buddy-shell"
import type { PersonaBuddySummary } from "@/types/persona-buddy"
import type {
  PersonaVisualAsset,
  PersonaVisualPack,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { BuddyShellPopover } from "./BuddyShellPopover"
import { SpriteFrameRenderer } from "./SpriteFrameRenderer"

type BuddyShellDockProps = {
  buddySummary: PersonaBuddySummary
  personaId?: string | null
  isOpen: boolean
  isDormant?: boolean
  visualPack?: PersonaVisualPack | null
  visualState?: PersonaVisualStateId
  position: PersonaBuddyShellPosition
  onToggle: () => void
  onDragHandlePointerDown: (event: React.PointerEvent<HTMLDivElement>) => void
  dockRef: React.RefObject<HTMLDivElement | null>
}

const getPersonaVisualAssetsById = (
  visualPack: PersonaVisualPack | null | undefined
): Record<string, PersonaVisualAsset> => {
  if (!visualPack) return {}
  if (visualPack.assets_by_id && Object.keys(visualPack.assets_by_id).length > 0) {
    return visualPack.assets_by_id
  }
  const assets: Record<string, PersonaVisualAsset> = {}
  for (const asset of visualPack.assets || []) {
    if (asset?.id) assets[asset.id] = asset
  }
  return assets
}

export const BuddyShellDock: React.FC<BuddyShellDockProps> = ({
  buddySummary,
  personaId = null,
  isOpen,
  isDormant = false,
  visualPack = null,
  visualState = "idle",
  position,
  onToggle,
  onDragHandlePointerDown,
  dockRef
}) => {
  const assetsById = getPersonaVisualAssetsById(visualPack)
  const canRenderVisualPack =
    !isDormant &&
    visualPack?.renderer_type === "sprite_frames" &&
    Boolean(visualPack.manifest) &&
    Object.keys(assetsById).length > 0

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
        {canRenderVisualPack ? (
          <div className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded border border-border bg-surface2">
            <SpriteFrameRenderer
              manifest={visualPack.manifest}
              assets={assetsById}
              state={visualState}
              fallbackLabel={buddySummary.persona_name}
              className="max-h-10 max-w-10 object-contain"
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

      {isOpen && !isDormant ? (
        <BuddyShellPopover buddySummary={buddySummary} personaId={personaId} />
      ) : null}
    </div>
  )
}

export default BuddyShellDock
