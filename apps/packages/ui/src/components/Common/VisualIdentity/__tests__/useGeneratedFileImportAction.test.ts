import { describe, expect, it, vi } from "vitest"

import { importGeneratedFileAndAssignSlot } from "../useGeneratedFileImportAction"

const makeClient = () => ({
  createVisualIdentityAssetFromGeneratedFile: vi.fn(async () => ({ id: 44 })),
  updateVisualIdentityDraftSlot: vi.fn(async () => ({ id: 7 }))
})

describe("importGeneratedFileAndAssignSlot", () => {
  it("returns assigned after import and slot update succeed", async () => {
    const client = makeClient()

    const result = await importGeneratedFileAndAssignSlot({
      client,
      packId: 5,
      draftId: 7,
      slotKey: "happy",
      generatedFileId: 42
    })

    expect(result).toEqual({ status: "assigned", assetId: 44, slotKey: "happy" })
    expect(client.createVisualIdentityAssetFromGeneratedFile).toHaveBeenCalledWith(5, {
      generated_file_id: 42,
      expression_key: "happy",
      draft_id: 7,
      source_feature: "vn_assets",
      source_context: {},
      idempotency_key: "vn-assets:42:pack:5:draft:7:happy"
    })
    expect(client.updateVisualIdentityDraftSlot).toHaveBeenCalledWith(7, "happy", {
      asset_id: 44,
      expression_key: "happy"
    })
  })

  it("returns imported_unassigned when slot update throws after import", async () => {
    const error = new Error("slot update failed")
    const client = makeClient()
    client.updateVisualIdentityDraftSlot.mockRejectedValue(error)

    const result = await importGeneratedFileAndAssignSlot({
      client,
      packId: 5,
      draftId: 7,
      slotKey: "happy",
      generatedFileId: 42
    })

    expect(result).toEqual({
      status: "imported_unassigned",
      assetId: 44,
      slotKey: "happy",
      error
    })
  })

  it("returns failed when import throws", async () => {
    const error = new Error("import failed")
    const client = makeClient()
    client.createVisualIdentityAssetFromGeneratedFile.mockRejectedValue(error)

    const result = await importGeneratedFileAndAssignSlot({
      client,
      packId: 5,
      draftId: 7,
      slotKey: "happy",
      generatedFileId: 42
    })

    expect(result).toEqual({ status: "failed", error })
    expect(client.updateVisualIdentityDraftSlot).not.toHaveBeenCalled()
  })
})
