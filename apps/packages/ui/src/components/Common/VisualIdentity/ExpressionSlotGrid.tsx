import React from "react"
import { Button, Tag } from "antd"
import { Trash2 } from "lucide-react"

import { getVisualIdentityExpressionDisplayLabel } from "@/utils/visual-identity-expressions"
import { ExpressionAssetUploader } from "./ExpressionAssetUploader"

const CANONICAL_SLOT_ORDER = [
  "neutral",
  "happy",
  "excited",
  "sad",
  "angry",
  "thinking",
  "confused",
  "surprised"
]

const usePrefersReducedMotion = (): boolean => {
  const [reducedMotion, setReducedMotion] = React.useState(false)

  React.useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return
    }
    const media = window.matchMedia("(prefers-reduced-motion: reduce)")
    const update = () => setReducedMotion(media.matches)
    update()
    media.addEventListener?.("change", update)
    return () => media.removeEventListener?.("change", update)
  }, [])

  return reducedMotion
}

export type ExpressionSlotGridAsset = {
  id?: number | null
  asset_url?: string | null
  content_type?: string | null
  is_animated?: boolean
  display_label?: string
  source_filename?: string
}

export type ExpressionSlotGridSlot = {
  key: string
  label?: string
  canonical?: boolean
  aliases?: string[]
  asset?: ExpressionSlotGridAsset | null
  assetUrl?: string | null
}

export type ExpressionSlotGridProps = {
  slots: ExpressionSlotGridSlot[]
  uploadingSlotKey?: string | null
  readOnly?: boolean
  onUploadAsset?: (slotKey: string, file: File) => void
  onClearSlot?: (slotKey: string) => void
}

const slotSortRank = (slot: ExpressionSlotGridSlot): [number, number, string] => {
  const canonicalIndex = CANONICAL_SLOT_ORDER.indexOf(slot.key)
  if (canonicalIndex >= 0) return [0, canonicalIndex, slot.key]
  return [1, Number.MAX_SAFE_INTEGER, slot.label || slot.key]
}

export const sortExpressionSlots = (
  slots: ExpressionSlotGridSlot[]
): ExpressionSlotGridSlot[] =>
  [...slots].sort((left, right) => {
    const leftRank = slotSortRank(left)
    const rightRank = slotSortRank(right)
    if (leftRank[0] !== rightRank[0]) return leftRank[0] - rightRank[0]
    if (leftRank[1] !== rightRank[1]) return leftRank[1] - rightRank[1]
    return leftRank[2].localeCompare(rightRank[2])
  })

export const ExpressionSlotGrid: React.FC<ExpressionSlotGridProps> = ({
  slots,
  uploadingSlotKey = null,
  readOnly = false,
  onUploadAsset,
  onClearSlot
}) => {
  const orderedSlots = React.useMemo(() => sortExpressionSlots(slots), [slots])
  const reducedMotion = usePrefersReducedMotion()

  if (orderedSlots.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border bg-bg px-3 py-3 text-xs text-text-muted">
        No expression slots available.
      </div>
    )
  }

  return (
    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
      {orderedSlots.map((slot) => {
        const label = slot.label || getVisualIdentityExpressionDisplayLabel(slot.key) || slot.key
        const asset = slot.asset || null
        const assetUrl = slot.assetUrl || asset?.asset_url || ""
        const hasAsset = Boolean(assetUrl || asset?.id)
        const shouldPausePreview =
          Boolean(assetUrl) && reducedMotion && Boolean(asset?.is_animated)
        const isUploading = uploadingSlotKey === slot.key
        return (
          <div
            key={slot.key}
            data-testid="expression-slot"
            className="rounded-md border border-border bg-bg p-2"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div
                  data-testid="expression-slot-label"
                  className="truncate text-sm font-medium text-text"
                >
                  {label}
                </div>
                <div className="mt-0.5 text-[11px] text-text-subtle">
                  {hasAsset ? asset?.content_type || "asset attached" : "No asset"}
                </div>
              </div>
              {slot.canonical === false ? <Tag>custom</Tag> : null}
            </div>
            <div className="mt-2 aspect-square overflow-hidden rounded-md border border-border bg-surface2">
              {assetUrl && !shouldPausePreview ? (
                <img
                  src={assetUrl}
                  alt={`${label} expression`}
                  className="h-full w-full object-contain"
                />
              ) : shouldPausePreview ? (
                <div className="flex h-full w-full items-center justify-center px-2 text-center text-[11px] text-text-subtle">
                  Animated preview paused
                </div>
              ) : hasAsset ? (
                <div className="flex h-full w-full items-center justify-center px-2 text-center text-[11px] text-text-subtle">
                  Preview unavailable
                </div>
              ) : (
                <div className="flex h-full w-full items-center justify-center px-2 text-center text-[11px] text-text-subtle">
                  No asset
                </div>
              )}
            </div>
            {!readOnly ? (
              <div className="mt-2 flex items-center justify-between gap-2">
                {onUploadAsset ? (
                  <ExpressionAssetUploader
                    label={hasAsset ? "Replace" : "Upload"}
                    loading={isUploading}
                    onSelectFile={(file) => onUploadAsset(slot.key, file)}
                  />
                ) : null}
                {hasAsset && onClearSlot ? (
                  <Button
                    size="small"
                    type="text"
                    danger
                    icon={<Trash2 className="h-3.5 w-3.5" />}
                    aria-label={`Clear ${label} expression asset`}
                    onClick={() => onClearSlot(slot.key)}
                  />
                ) : null}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
