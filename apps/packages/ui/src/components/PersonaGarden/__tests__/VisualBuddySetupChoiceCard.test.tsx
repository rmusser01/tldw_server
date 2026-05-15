import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { VisualBuddySetupChoiceCard } from "../VisualBuddySetupChoiceCard"

const starter = {
  id: "research-buddy-starter",
  title: "Research Buddy Starter",
  description: "Starter sprite pack",
  renderer_type: "sprite_frames" as const,
  manifest_version: 1,
  states_offered: ["idle", "thinking"],
  asset_count: 1,
  total_bytes: 512,
  tags: ["starter"],
  license_label: "bundled"
}

describe("VisualBuddySetupChoiceCard", () => {
  it("renders first-run setup actions", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={starter}
        starterCount={1}
        onUseDefault={vi.fn()}
        onImportPack={vi.fn()}
        onStartBlank={vi.fn()}
      />
    )

    expect(screen.getByTestId("visual-buddy-setup-choice-card")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /use default/i })).toBeEnabled()
    expect(screen.getByRole("button", { name: /import pack/i })).toBeEnabled()
    expect(screen.getByRole("button", { name: /start blank/i })).toBeEnabled()
    expect(screen.getByText(/no visual buddy is active/i)).toBeInTheDocument()
  })

  it("frames existing drafts as reviewable but inactive", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={2}
        recommendedStarter={starter}
        starterCount={1}
      />
    )

    expect(screen.getByText(/draft/i)).toBeInTheDocument()
    expect(screen.getByText(/activate/i)).toBeInTheDocument()
  })

  it("disables default copy without blocking import or blank", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={null}
        starterCatalogError="Starter catalog unavailable"
        onImportPack={vi.fn()}
        onStartBlank={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: /use default/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /import pack/i })).toBeEnabled()
    expect(screen.getByRole("button", { name: /start blank/i })).toBeEnabled()
  })

  it("invokes compact open visuals action without rendering mutation buttons", () => {
    const onOpenVisuals = vi.fn()

    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        compact
        onOpenVisuals={onOpenVisuals}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /set up visual buddy/i }))

    expect(onOpenVisuals).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("button", { name: /use default/i })).not.toBeInTheDocument()
  })
})
