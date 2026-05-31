import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"

import { asPersonaVisualCustomStateId } from "@/types/persona-visuals"
import { BASIC_BUDDY_STARTER_IDS } from "../buddyBuilderState"
import { BuddyGuidedBuilder } from "../BuddyGuidedBuilder"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: Record<string, unknown>) => {
      let value = String(options?.defaultValue ?? _key)
      for (const [name, replacement] of Object.entries(options ?? {})) {
        if (name !== "defaultValue") {
          value = value.replaceAll(`{{${name}}}`, String(replacement))
        }
      }
      return value
    }
  })
}))

const starter = (
  id: string,
  title: string,
  tier: PersonaVisualStarterPackSummary["complexity_tier"] = "basic",
  status: PersonaVisualStarterPackSummary["production_status"] = "art_ready"
): PersonaVisualStarterPackSummary => ({
  id,
  title,
  description: `${title} description`,
  renderer_type: "sprite_frames",
  manifest_version: 1,
  states_offered: ["idle", "running", "review"],
  asset_count: 1,
  total_bytes: 1024,
  tags: ["starter"],
  license_label: "bundled",
  complexity_tier: tier,
  production_status: status,
  neutral_anchor_required: true,
  expected_asset_groups: ["neutral_anchor", "required_state_loops"],
  animation_coverage_notes: ["Reviewed row loops."]
})

const basicStarters = BASIC_BUDDY_STARTER_IDS.map((id) =>
  starter(
    id,
    id
      .split("-")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  )
)

const renderBuilder = (
  overrides: Partial<React.ComponentProps<typeof BuddyGuidedBuilder>> = {}
) => {
  const onCopyStarterPack = vi.fn()
  const onStartBlank = vi.fn()
  const onOpenLibrary = vi.fn()
  const onOpenDuplicate = vi.fn()
  const onSaveManifest = vi.fn()
  const result = render(
    <BuddyGuidedBuilder
      selectedPersonaId="persona-1"
      selectedPersonaName="Garden Helper"
      hasActiveVisual={false}
      packCount={0}
      starterPacks={basicStarters}
      copyingStarterId={null}
      importPreviewPanel={<div data-testid="import-panel">Import panel</div>}
      onCopyStarterPack={onCopyStarterPack}
      onStartBlank={onStartBlank}
      onOpenLibrary={onOpenLibrary}
      onOpenDuplicate={onOpenDuplicate}
      onSaveManifest={onSaveManifest}
      {...overrides}
    />
  )
  return {
    ...result,
    onCopyStarterPack,
    onStartBlank,
    onOpenLibrary,
    onOpenDuplicate,
    onSaveManifest
  }
}

describe("BuddyGuidedBuilder", () => {
  it("renders accessible source and step navigation labels", () => {
    renderBuilder()

    expect(screen.getByRole("heading", { name: "Buddy builder" })).toBeVisible()
    const stepper = screen.getByLabelText("Buddy builder steps")
    expect(stepper).toHaveTextContent("Choose a source")
    expect(stepper).toHaveTextContent("Create a draft")
    expect(stepper).toHaveTextContent("Review readiness")
    expect(stepper).toHaveTextContent("Configure states")
    expect(stepper).toHaveTextContent("Activate")

    expect(screen.getByRole("button", { name: "Bundled Buddy" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Bundled Buddy" })).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(
      screen.getByRole("button", { name: "Import Codex/Petdex pet" })
    ).toBeVisible()
    expect(
      screen.getByRole("button", { name: "Import Persona Visual pack" })
    ).toBeVisible()
  })

  it("shows the six Basic defaults first in the expected order", () => {
    renderBuilder()

    const basicList = screen.getByTestId("buddy-builder-tier-basic")
    const titles = within(basicList)
      .getAllByTestId("buddy-builder-starter-title")
      .map((node) => node.textContent)

    expect(titles).toEqual([
      "Search Lens Basic",
      "Index Card Basic",
      "Archive Cube Basic",
      "Paperclip Basic",
      "Terminal Tile Basic",
      "Migu Marker Basic"
    ])
    expect(within(basicList).getAllByText("Recommended")).toHaveLength(6)
  })

  it("keeps scaffold production packets visible but distinct from reviewed Basic defaults", () => {
    renderBuilder({
      starterPacks: [
        ...basicStarters.slice(0, 1),
        starter("lofi-study-intermediate", "Lofi Study", "intermediate", "scaffold")
      ]
    })

    const scaffold = screen.getByTestId("buddy-builder-starter-lofi-study-intermediate")
    expect(scaffold).toHaveTextContent("Production packet")
    expect(
      within(scaffold).getByRole("button", { name: "Copy production packet" })
    ).toBeVisible()
    expect(within(scaffold).queryByText("Recommended")).not.toBeInTheDocument()
  })

  it("separates Codex/Petdex and native Persona Visual import paths", () => {
    renderBuilder()

    fireEvent.click(screen.getByRole("button", { name: "Import Codex/Petdex pet" }))
    expect(
      screen.getByRole("button", { name: "Import Codex/Petdex pet" })
    ).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByTestId("buddy-builder-import-panel")).toHaveTextContent(
      "Codex/Petdex pet"
    )
    expect(screen.getByTestId("buddy-builder-import-panel")).toHaveTextContent(
      "pet.json"
    )

    fireEvent.click(screen.getByRole("button", { name: "Import Persona Visual pack" }))
    expect(screen.getByTestId("buddy-builder-import-panel")).toHaveTextContent(
      "Persona Visual pack"
    )
    expect(screen.getByTestId("buddy-builder-import-panel")).toHaveTextContent(
      ".tldw-persona-vpack"
    )
  })

  it("still renders as the primary surface when a buddy is active", () => {
    renderBuilder({
      hasActiveVisual: true,
      packCount: 2,
      activePackTitle: "Active Lens"
    })

    expect(screen.getByTestId("buddy-guided-builder")).toHaveTextContent(
      "Active Lens"
    )
    expect(screen.getByTestId("buddy-guided-builder-active-pack")).toHaveTextContent(
      "active"
    )
  })

  it("renders draft review diagnostics inside the builder when a draft manifest exists", () => {
    const { onSaveManifest } = renderBuilder({
      draftManifest: {
        manifest_version: 1,
        renderer_type: "sprite_frames",
        states: {
          idle: { animation_id: "idle" },
          [asPersonaVisualCustomStateId("moving_left")]: {
            animation_id: "move-left"
          }
        },
        animations: {
          idle: { frames: [] },
          "move-left": { frames: [] }
        },
        state_catalog: {
          [asPersonaVisualCustomStateId("moving_left")]: {
            label: "Moving left",
            kind: "live_variant"
          }
        }
      }
    })

    expect(screen.getByTestId("buddy-draft-review-panel")).toHaveTextContent(
      "Review draft readiness"
    )
    expect(screen.getByTestId("buddy-draft-review-movement-states")).toHaveTextContent(
      "moving_left"
    )
    expect(screen.getByTestId("buddy-state-configuration-panel")).toHaveTextContent(
      "Configure visual states"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Save visual state configuration" })
    )

    expect(onSaveManifest).toHaveBeenCalledTimes(1)
  })
})
