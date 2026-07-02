import React from "react"
import type { VisualIdentityResolveResponse } from "@/types/visual-identities"
import { getVisualIdentityExpressionDisplayLabel } from "@/utils/visual-identity-expressions"
import { VisualIdentityImage } from "./VisualIdentityImage"

export type VisualIdentityStageProps = {
  actorName: string
  resolution: VisualIdentityResolveResponse | null
  className?: string
}

export const VisualIdentityStage = ({
  actorName,
  resolution,
  className = ""
}: VisualIdentityStageProps) => {
  if (!resolution?.asset_url) return null

  const expressionLabel =
    getVisualIdentityExpressionDisplayLabel(resolution.expression_key) ||
    "Expression"
  const alt = `${actorName} ${expressionLabel.toLowerCase()}`

  return (
    <section
      className={`w-full border-b border-border/60 bg-surface/60 px-4 py-3 ${className}`}
      aria-label="Visual identity stage"
      data-testid="visual-identity-stage"
    >
      <div className="mx-auto flex max-w-5xl items-end justify-center">
        <div className="relative h-48 w-36 overflow-hidden rounded-md border border-border/70 bg-surface2 shadow-sm sm:h-64 sm:w-48">
          <VisualIdentityImage
            assetUrl={resolution.asset_url}
            isAnimated={resolution.is_animated}
            alt={alt}
            className="h-full w-full object-contain"
          />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/55 to-transparent px-3 py-2">
            <div className="truncate text-sm font-medium text-white">
              {actorName}
            </div>
            <div className="text-xs text-white/80">{expressionLabel}</div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default VisualIdentityStage
