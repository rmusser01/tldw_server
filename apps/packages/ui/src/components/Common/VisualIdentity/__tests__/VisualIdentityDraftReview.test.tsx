import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { VisualIdentityDraftReview } from "../VisualIdentityDraftReview"
import type { VisualIdentityDraftResponse } from "@/types/visual-identities"

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
  assets: [
    {
      id: 9,
      owner_user_id: 1,
      pack_id: 10,
      draft_id: 42,
      pack_version_id: null,
      expression_key: "neutral",
      original_expression_key: "neutral",
      display_label: "Neutral",
      source_filename: "neutral.webp",
      source_context: {},
      content_type: "image/webp",
      bytes: 512,
      sha256: "sha",
      width: 256,
      height: 256,
      is_animated: true,
      frame_count: 8,
      duration_ms: 1200,
      preview_relpath: null,
      created_at: "2026-07-02T00:00:00Z",
      updated_at: "2026-07-02T00:00:00Z"
    }
  ],
  pack_version_id: null,
  asset_ids: [],
  binding_id: null
}

describe("VisualIdentityDraftReview", () => {
  it("activates a ready draft with the current character binding", () => {
    const activate = vi.fn()

    render(
      <VisualIdentityDraftReview
        actorKind="character"
        actorId={7}
        draft={readyDraft}
        expressionSlots={[{ key: "neutral", label: "Neutral", canonical: true, aliases: [] }]}
        onActivate={activate}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /activate/i }))

    expect(activate).toHaveBeenCalledWith(
      42,
      expect.objectContaining({ actor_kind: "character", actor_id: 7 })
    )
  })

  it("renders packless draft assets without replacement upload actions", () => {
    const clearSlot = vi.fn()

    render(
      <VisualIdentityDraftReview
        actorKind="character"
        actorId={7}
        draft={{ ...readyDraft, pack_id: null }}
        expressionSlots={[{ key: "neutral", label: "Neutral", canonical: true, aliases: [] }]}
        onActivate={vi.fn()}
        onUploadAsset={vi.fn()}
        onClearSlot={clearSlot}
      />
    )

    expect(screen.getByText("Preview unavailable")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /replace|upload/i })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /clear neutral expression asset/i }))

    expect(clearSlot).toHaveBeenCalledWith("neutral")
  })

  it("uses slot map asset ids as the selected draft assets", () => {
    render(
      <VisualIdentityDraftReview
        actorKind="character"
        actorId={7}
        draft={{
          ...readyDraft,
          slot_map: {
            neutral: { asset_id: 10, expression_key: "neutral", display_label: "Neutral" }
          },
          assets: [
            {
              ...readyDraft.assets[0],
              id: 9,
              source_filename: "neutral-old.webp",
              is_animated: false
            },
            {
              ...readyDraft.assets[0],
              id: 10,
              source_filename: "neutral-new.webp",
              is_animated: false
            }
          ]
        }}
        expressionSlots={[{ key: "neutral", label: "Neutral", canonical: true, aliases: [] }]}
        buildAssetUrl={(asset) => `/assets/${asset.id}`}
        onActivate={vi.fn()}
      />
    )

    expect(screen.getByAltText("Neutral expression")).toHaveAttribute("src", "/assets/10")
  })

  it("renders cleared slot map entries as empty slots", () => {
    render(
      <VisualIdentityDraftReview
        actorKind="character"
        actorId={7}
        draft={{
          ...readyDraft,
          slot_map: {
            neutral: { asset_id: null, expression_key: "neutral", display_label: "Neutral" }
          }
        }}
        expressionSlots={[{ key: "neutral", label: "Neutral", canonical: true, aliases: [] }]}
        buildAssetUrl={(asset) => `/assets/${asset.id}`}
        onActivate={vi.fn()}
        onClearSlot={vi.fn()}
      />
    )

    expect(screen.getAllByText("No asset")).toHaveLength(2)
    expect(
      screen.queryByRole("button", { name: /clear neutral expression asset/i })
    ).not.toBeInTheDocument()
  })
})
