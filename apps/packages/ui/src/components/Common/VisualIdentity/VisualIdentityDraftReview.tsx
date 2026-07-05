import React from "react"
import { Button, Tag } from "antd"
import { CheckCircle2, RefreshCw } from "lucide-react"

import type {
  VisualIdentityActorKind,
  VisualIdentityAssetResponse,
  VisualIdentityDraftActivateRequest,
  VisualIdentityDraftResponse,
  VisualIdentityExpressionSlotResponse
} from "@/types/visual-identities"
import { getVisualIdentityExpressionDisplayLabel } from "@/utils/visual-identity-expressions"
import {
  ExpressionSlotGrid,
  type ExpressionSlotGridSlot
} from "./ExpressionSlotGrid"

export type VisualIdentityDraftReviewProps = {
  actorKind: VisualIdentityActorKind
  actorId: number | string
  draft: VisualIdentityDraftResponse
  expressionSlots: VisualIdentityExpressionSlotResponse[]
  activating?: boolean
  uploadingSlotKey?: string | null
  embedded?: boolean
  buildAssetUrl?: (asset: VisualIdentityAssetResponse) => string
  onActivate: (draftId: number, request: VisualIdentityDraftActivateRequest) => void
  onRefreshDraft?: (draftId: number) => void
  onUploadAsset?: (slotKey: string, file: File) => void
  onClearSlot?: (slotKey: string) => void
}

const assetToSlot = (
  asset: VisualIdentityAssetResponse,
  buildAssetUrl?: (asset: VisualIdentityAssetResponse) => string
): ExpressionSlotGridSlot => ({
  key: asset.expression_key,
  label:
    asset.display_label ||
    getVisualIdentityExpressionDisplayLabel(asset.expression_key) ||
    asset.expression_key,
  canonical: false,
  asset: {
    id: asset.id,
    asset_url: buildAssetUrl?.(asset) || null,
    content_type: asset.content_type,
    is_animated: asset.is_animated,
    display_label: asset.display_label,
    source_filename: asset.source_filename
  }
})

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

const stringFromEntry = (
  entry: Record<string, unknown> | null,
  field: string
): string | null => {
  const value = entry?.[field]
  return typeof value === "string" && value.trim() ? value : null
}

const assetIdFromEntry = (entry: Record<string, unknown> | null): number | null => {
  const value = entry?.asset_id
  if (value === null || value === undefined) return null
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0 ? numeric : null
}

export const buildDraftExpressionSlots = (
  draft: VisualIdentityDraftResponse,
  expressionSlots: VisualIdentityExpressionSlotResponse[],
  buildAssetUrl?: (asset: VisualIdentityAssetResponse) => string
): ExpressionSlotGridSlot[] => {
  const slotMap = draft.slot_map || {}
  const hasExplicitSlotMap = Object.keys(slotMap).length > 0
  if (hasExplicitSlotMap) {
    const assetsById = new Map(draft.assets.map((asset) => [asset.id, asset]))
    const slotFromEntry = (
      slotKey: string,
      fallbackLabel: string,
      canonical: boolean,
      aliases: string[] = []
    ): ExpressionSlotGridSlot => {
      const entry = isRecord(slotMap[slotKey]) ? slotMap[slotKey] : null
      const expressionKey = stringFromEntry(entry, "expression_key") || slotKey
      const label =
        stringFromEntry(entry, "display_label") ||
        fallbackLabel ||
        getVisualIdentityExpressionDisplayLabel(expressionKey) ||
        expressionKey
      const assetId = assetIdFromEntry(entry)
      const asset = assetId != null ? assetsById.get(assetId) : null
      return {
        key: slotKey,
        label,
        canonical,
        aliases,
        asset: asset ? assetToSlot(asset, buildAssetUrl).asset : null
      }
    }

    const slots = expressionSlots.map((slot) =>
      slotFromEntry(slot.key, slot.label, slot.canonical, slot.aliases)
    )
    const knownKeys = new Set(slots.map((slot) => slot.key))
    for (const [slotKey, rawEntry] of Object.entries(slotMap)) {
      if (knownKeys.has(slotKey)) continue
      const entry = isRecord(rawEntry) ? rawEntry : null
      const expressionKey = stringFromEntry(entry, "expression_key") || slotKey
      slots.push(
        slotFromEntry(
          slotKey,
          getVisualIdentityExpressionDisplayLabel(expressionKey) || expressionKey,
          false
        )
      )
    }
    return slots
  }

  const assetsByExpression = new Map(
    draft.assets.map((asset) => [asset.expression_key, asset])
  )
  const slots: ExpressionSlotGridSlot[] = expressionSlots.map((slot) => {
    const asset = assetsByExpression.get(slot.key)
    return {
      key: slot.key,
      label: slot.label,
      canonical: slot.canonical,
      aliases: slot.aliases,
      asset: asset ? assetToSlot(asset, buildAssetUrl).asset : null
    }
  })
  const knownKeys = new Set(slots.map((slot) => slot.key))
  for (const asset of draft.assets) {
    if (!knownKeys.has(asset.expression_key)) {
      slots.push(assetToSlot(asset, buildAssetUrl))
    }
  }
  return slots
}

export const VisualIdentityDraftReview: React.FC<VisualIdentityDraftReviewProps> = ({
  actorKind,
  actorId,
  draft,
  expressionSlots,
  activating = false,
  uploadingSlotKey = null,
  embedded = false,
  buildAssetUrl,
  onActivate,
  onRefreshDraft,
  onUploadAsset,
  onClearSlot
}) => {
  const isReady = draft.status === "ready_for_review"
  const slotViews = React.useMemo(
    () => buildDraftExpressionSlots(draft, expressionSlots, buildAssetUrl),
    [buildAssetUrl, draft, expressionSlots]
  )
  const assetCount = draft.assets.length
  const canUploadAssets = draft.pack_id != null

  return (
    <section
      data-testid="visual-identity-draft-review"
      className={
        embedded
          ? "border-t border-border pt-3"
          : "rounded-lg border border-border bg-surface p-3"
      }
      aria-label="Expression pack draft review"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            Draft review
          </div>
          <div className="mt-1 text-sm font-medium text-text">{draft.title}</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-text-muted">
            <Tag color={isReady ? "green" : draft.status === "failed" ? "red" : "blue"}>
              {draft.status}
            </Tag>
            <span>{assetCount} assets</span>
            {draft.import_job_id ? <span>Job {draft.import_job_id}</span> : null}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {onRefreshDraft ? (
            <Button
              size="small"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              onClick={() => onRefreshDraft(draft.id)}
            >
              Refresh
            </Button>
          ) : null}
          <Button
            type="primary"
            size="small"
            icon={<CheckCircle2 className="h-3.5 w-3.5" />}
            disabled={!isReady || activating}
            loading={activating}
            onClick={() =>
              onActivate(draft.id, {
                actor_kind: actorKind,
                actor_id: actorId
              })
            }
          >
            Activate
          </Button>
        </div>
      </div>

      {draft.status === "failed" ? (
        <div className="mt-3 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
          {String(draft.error?.message || "Import failed. Review the archive and try again.")}
        </div>
      ) : null}

      <div className="mt-3">
        <ExpressionSlotGrid
          slots={slotViews}
          uploadingSlotKey={uploadingSlotKey}
          onUploadAsset={canUploadAssets ? onUploadAsset : undefined}
          onClearSlot={onClearSlot}
        />
      </div>
    </section>
  )
}
