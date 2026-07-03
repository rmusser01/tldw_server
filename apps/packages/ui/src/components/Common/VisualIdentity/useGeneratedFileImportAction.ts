import { useCallback } from "react"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  VisualIdentityDraftSlotUpdate,
  VisualIdentityGeneratedFileAssetRequest
} from "@/types/visual-identities"

export type GeneratedFileImportActionResult =
  | { status: "assigned"; assetId: number; slotKey: string }
  | {
      status: "imported_unassigned"
      assetId: number
      slotKey: string
      error: unknown
    }
  | { status: "failed"; error: unknown }

export type GeneratedFileImportActionClient = {
  createVisualIdentityAssetFromGeneratedFile: (
    packId: number,
    request: VisualIdentityGeneratedFileAssetRequest
  ) => Promise<{ id: number }>
  updateVisualIdentityDraftSlot: (
    draftId: number,
    slotKey: string,
    request: VisualIdentityDraftSlotUpdate
  ) => Promise<unknown>
}

export type ImportGeneratedFileAndAssignSlotArgs = {
  client: GeneratedFileImportActionClient
  packId: number
  draftId: number
  slotKey: string
  generatedFileId: number
  sourceFeature?: string
  sourceContext?: Record<string, unknown>
  idempotencyKey?: string
}

export type UseGeneratedFileImportActionArgs = Omit<
  ImportGeneratedFileAndAssignSlotArgs,
  "client"
>

const buildGeneratedFileImportIdempotencyKey = ({
  generatedFileId,
  packId,
  draftId,
  slotKey
}: UseGeneratedFileImportActionArgs): string =>
  `vn-assets:${generatedFileId}:pack:${packId}:draft:${draftId}:${slotKey}`

export async function importGeneratedFileAndAssignSlot(
  args: ImportGeneratedFileAndAssignSlotArgs
): Promise<GeneratedFileImportActionResult> {
  try {
    const asset = await args.client.createVisualIdentityAssetFromGeneratedFile(
      args.packId,
      {
        generated_file_id: args.generatedFileId,
        expression_key: args.slotKey,
        draft_id: args.draftId,
        source_feature: args.sourceFeature ?? "vn_assets",
        source_context: args.sourceContext ?? {},
        idempotency_key:
          args.idempotencyKey ?? buildGeneratedFileImportIdempotencyKey(args)
      }
    )

    try {
      await args.client.updateVisualIdentityDraftSlot(args.draftId, args.slotKey, {
        asset_id: asset.id,
        expression_key: args.slotKey
      })
      return { status: "assigned", assetId: asset.id, slotKey: args.slotKey }
    } catch (error) {
      return {
        status: "imported_unassigned",
        assetId: asset.id,
        slotKey: args.slotKey,
        error
      }
    }
  } catch (error) {
    return { status: "failed", error }
  }
}

export function useGeneratedFileImportAction(
  client: GeneratedFileImportActionClient = tldwClient
): (args: UseGeneratedFileImportActionArgs) => Promise<GeneratedFileImportActionResult> {
  return useCallback(
    (args: UseGeneratedFileImportActionArgs) =>
      importGeneratedFileAndAssignSlot({ ...args, client }),
    [client]
  )
}
