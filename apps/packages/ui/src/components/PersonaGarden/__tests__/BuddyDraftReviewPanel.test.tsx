import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  asPersonaVisualCustomStateId,
  type PersonaVisualAsset,
  type PersonaVisualImportPreviewResponse,
  type PersonaVisualManifest
} from "@/types/persona-visuals"

import { BuddyDraftReviewPanel } from "../BuddyDraftReviewPanel"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? _key
  })
}))

vi.mock("@/components/Common/PersonaBuddy/SpriteFrameRenderer", () => ({
  SpriteFrameRenderer: ({ state }: { state: string }) => (
    <div data-testid="sprite-frame-renderer">Sprite preview: {state}</div>
  )
}))

const manifest: PersonaVisualManifest = {
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {
    idle: { animation_id: "idle" },
    listening: { animation_id: "listening" },
    thinking: { animation_id: "thinking" },
    speaking: { animation_id: "speaking" },
    error: { animation_id: "error" },
    [asPersonaVisualCustomStateId("moving_right")]: { animation_id: "move-r" }
  },
  animations: {
    idle: {
      frames: [{ asset_id: "idle-frame", duration_ms: 1000 }]
    },
    listening: { frames: [] },
    thinking: { frames: [] },
    speaking: { frames: [] },
    error: { frames: [] },
    "move-r": { frames: [] }
  },
  state_catalog: {
    [asPersonaVisualCustomStateId("moving_right")]: {
      label: "Moving right",
      kind: "live_variant"
    },
    [asPersonaVisualCustomStateId("sparkle_review")]: {
      label: "Sparkle review",
      kind: "tool_variant"
    }
  }
}

const asset: PersonaVisualAsset = {
  id: "idle-frame",
  asset_role: "frame",
  url: "https://example.test/idle.png",
  mime_type: "image/png"
}

const preview = (
  overrides: Partial<PersonaVisualImportPreviewResponse> = {}
): PersonaVisualImportPreviewResponse => ({
  preview_id: "preview-1",
  job_id: "job-1",
  portability_job_id: "portability-1",
  operation: "import_preview",
  target_persona_id: "persona-1",
  status: "completed",
  visual_status: "completed",
  stage: "completed",
  schema_version: "codex.pet.v1",
  bundle_summary: {
    pack_title: "Imported Buddy",
    assets: [
      {
        source_asset_id: "atlas",
        asset_role: "sprite_sheet",
        asset_group: "animation_atlas",
        width: 1536,
        height: 1872
      }
    ]
  },
  validation_warnings: [],
  conflicts: [],
  proposed_plan: {},
  quota_estimate: {},
  required_choices: [],
  target_warnings: [],
  ...overrides
})

describe("BuddyDraftReviewPanel", () => {
  it("shows Codex import semantics as a Persona Visual draft with atlas metadata", () => {
    render(
      <BuddyDraftReviewPanel
        manifest={manifest}
        assetsById={{ "idle-frame": asset }}
        importPreview={preview()}
      />
    )

    const panel = screen.getByTestId("buddy-draft-review-panel")
    expect(panel).toHaveTextContent("Codex/Petdex pet")
    expect(panel).toHaveTextContent("imported as a Persona Visual draft")
    expect(panel).toHaveTextContent("1536x1872")
    expect(panel).toHaveTextContent("atlas")
  })

  it("distinguishes native archives from Codex previews using preview data", () => {
    render(
      <BuddyDraftReviewPanel
        manifest={manifest}
        assetsById={{ "idle-frame": asset }}
        importPreview={preview({ schema_version: "persona_visual_pack.v1" })}
      />
    )

    expect(screen.getByTestId("buddy-draft-review-panel")).toHaveTextContent(
      "Persona Visual pack"
    )
  })

  it("disables the activation path when blockers are present", () => {
    render(
      <BuddyDraftReviewPanel
        manifest={{ ...manifest, states: { idle: { animation_id: "idle" } } }}
        assetsById={{}}
        importPreview={preview()}
      />
    )

    expect(screen.getByTestId("buddy-draft-review-blockers")).toHaveTextContent(
      "Missing required state: listening"
    )
    expect(screen.getByTestId("buddy-draft-review-activation-path")).toBeDisabled()
  })

  it("renders movement and custom state coverage separately", () => {
    render(
      <BuddyDraftReviewPanel
        manifest={manifest}
        assetsById={{ "idle-frame": asset }}
        importPreview={preview()}
      />
    )

    expect(screen.getByTestId("buddy-draft-review-movement-states")).toHaveTextContent(
      "moving_right"
    )
    expect(screen.getByTestId("buddy-draft-review-custom-states")).toHaveTextContent(
      "Sparkle review"
    )
  })

  it("uses SpriteFrameRenderer only when draft preview assets are available", () => {
    const { rerender } = render(
      <BuddyDraftReviewPanel
        manifest={manifest}
        assetsById={{ "idle-frame": asset }}
        importPreview={preview()}
      />
    )

    expect(screen.getByTestId("sprite-frame-renderer")).toHaveTextContent(
      "Sprite preview: idle"
    )

    rerender(
      <BuddyDraftReviewPanel
        manifest={manifest}
        assetsById={{}}
        importPreview={preview()}
      />
    )

    expect(screen.queryByTestId("sprite-frame-renderer")).not.toBeInTheDocument()
    expect(
      within(screen.getByTestId("buddy-draft-review-preview")).getByText(
        "Preview unavailable"
      )
    ).toBeVisible()
  })
})
