import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { clearVisualIdentityResolverCaches } from "@/hooks/useVisualIdentityResolver"
import { VisualIdentityPackPanel } from "../VisualIdentityPackPanel"
import type { VisualIdentityDraftResponse } from "@/types/visual-identities"

vi.mock("@/hooks/useVisualIdentityResolver", () => ({
  clearVisualIdentityResolverCaches: vi.fn()
}))

const readyDraft: VisualIdentityDraftResponse = {
  id: 42,
  owner_user_id: 1,
  pack_id: 10,
  title: "Imported expressions",
  status: "ready_for_review",
  source_kind: "zip",
  source_filename: "expressions.zip",
  import_job_id: "job-1",
  validation_summary: {},
  slot_map: {},
  default_expression_key: "neutral",
  error: {},
  created_at: "2026-07-02T00:00:00Z",
  updated_at: "2026-07-02T00:00:00Z",
  version: 1,
  assets: [],
  pack_version_id: null,
  asset_ids: [],
  binding_id: null
}

const makeClient = () => ({
  getVisualIdentityCapabilities: vi.fn(async () => ({
    upload_max_bytes: 1024 * 1024,
    archive_max_bytes: 4 * 1024 * 1024,
    max_dimension: 2048,
    max_frame_count: 120,
    supported_mime_types: ["image/png", "image/webp"],
    avif_enabled: false
  })),
  listVisualIdentityExpressionSlots: vi.fn(async () => [
    { key: "neutral", label: "Neutral", canonical: true, aliases: [] }
  ]),
  resolveVisualIdentityBinding: vi.fn(async () => ({
    actor_kind: "character" as const,
    actor_id: 7,
    pack_id: null,
    pack_version_id: null,
    expression_key: null,
    requested_expression_key: "neutral",
    asset_id: null,
    storage_relpath: null,
    fallback_reason: "no_binding",
    is_animated: false,
    content_type: null,
    asset_url: null
  })),
  listVisualIdentityPacks: vi.fn(async () => []),
  startVisualIdentityZipImport: vi.fn(async () => ({
    draft_id: 42,
    job_id: "job-1",
    status: "queued",
    source_filename: "expressions.zip",
    import_job_id: "job-1"
  })),
  getVisualIdentityDraft: vi.fn(async () => readyDraft),
  activateVisualIdentityDraft: vi.fn(async () => ({
    ...readyDraft,
    status: "activated",
    binding_id: 5
  })),
  uploadVisualIdentityPackAsset: vi.fn(async () => ({
    id: 77,
    owner_user_id: 1,
    pack_id: 10,
    draft_id: 42,
    pack_version_id: null,
    expression_key: "neutral",
    original_expression_key: "neutral",
    display_label: "Neutral",
    source_filename: "neutral-new.webp",
    source_context: {},
    content_type: "image/webp",
    bytes: 512,
    sha256: "sha-new",
    width: 256,
    height: 256,
    is_animated: false,
    frame_count: null,
    duration_ms: null,
    preview_relpath: null,
    created_at: "2026-07-02T00:00:00Z",
    updated_at: "2026-07-02T00:00:00Z"
  })),
  updateVisualIdentityDraftSlot: vi.fn(async () => ({
    ...readyDraft,
    slot_map: {
      neutral: { asset_id: 77, expression_key: "neutral", display_label: "Neutral" }
    }
  })),
  getVisualIdentityAssetContentPath: vi.fn((packId: number, assetId: number) =>
    `/api/v1/visual-identities/packs/${packId}/assets/${assetId}/content`
  )
})

describe("VisualIdentityPackPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("activates an imported ready draft with the current character binding", async () => {
    const client = makeClient()

    render(
      <VisualIdentityPackPanel
        actorKind="character"
        actorId={7}
        actorName="Ari"
        client={client}
      />
    )

    const archiveInput = await screen.findByLabelText("Import expression pack ZIP")
    const file = new File(["zip"], "expressions.zip", { type: "application/zip" })
    fireEvent.change(archiveInput, { target: { files: [file] } })

    await screen.findByTestId("visual-identity-draft-review")
    fireEvent.click(screen.getByRole("button", { name: /activate/i }))

    await waitFor(() => {
      expect(client.activateVisualIdentityDraft).toHaveBeenCalledWith(
        42,
        expect.objectContaining({ actor_kind: "character", actor_id: 7 })
      )
    })
    expect(clearVisualIdentityResolverCaches).toHaveBeenCalledTimes(1)
  })

  it("persists uploaded slot replacements into the draft slot map", async () => {
    const client = makeClient()

    render(
      <VisualIdentityPackPanel
        actorKind="character"
        actorId={7}
        actorName="Ari"
        client={client}
      />
    )

    const archiveInput = await screen.findByLabelText("Import expression pack ZIP")
    fireEvent.change(archiveInput, {
      target: { files: [new File(["zip"], "expressions.zip", { type: "application/zip" })] }
    })

    await screen.findByTestId("visual-identity-draft-review")
    const uploadInput = screen.getByLabelText("Upload")
    fireEvent.change(uploadInput, {
      target: { files: [new File(["image"], "neutral-new.webp", { type: "image/webp" })] }
    })

    await waitFor(() => {
      expect(client.updateVisualIdentityDraftSlot).toHaveBeenCalledWith(
        42,
        "neutral",
        expect.objectContaining({ asset_id: 77, expression_key: "neutral" })
      )
    })
  })
})
