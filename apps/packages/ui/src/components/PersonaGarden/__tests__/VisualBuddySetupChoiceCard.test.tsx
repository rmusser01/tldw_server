import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"

const mocks = vi.hoisted(() => ({
  translate: vi.fn(
    (
      _key: string,
      options?:
        | string
        | {
            defaultValue?: string
            count?: number
          }
    ) => {
      if (typeof options === "string") return options
      if (typeof options?.count === "number" && options.defaultValue) {
        return options.defaultValue.replace("{{count}}", String(options.count))
      }
      return options?.defaultValue ?? _key
    }
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (...args: Parameters<typeof mocks.translate>) => mocks.translate(...args)
  })
}))

import { VisualBuddySetupChoiceCard } from "../VisualBuddySetupChoiceCard"

const starter: PersonaVisualStarterPackSummary = {
  id: "search-lens-basic",
  title: "Search Lens Buddy",
  description: "Reviewed Basic sprite pack",
  renderer_type: "sprite_frames" as const,
  manifest_version: 1,
  states_offered: ["idle", "thinking"],
  asset_count: 1,
  total_bytes: 512,
  tags: ["starter"],
  license_label: "bundled",
  complexity_tier: "basic",
  production_status: "art_ready",
  neutral_anchor_required: true,
  expected_asset_groups: [
    "identity_brief",
    "neutral_anchor",
    "preview_image",
    "required_state_loops"
  ],
  animation_coverage_notes: [
    "Reviewed bundled basic default with neutral-anchor-derived required-state loops."
  ]
}

describe("VisualBuddySetupChoiceCard", () => {
  beforeEach(() => {
    mocks.translate.mockClear()
  })

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
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.metadata.productionStatus.art_ready",
      expect.objectContaining({ defaultValue: "Art Ready" })
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.metadata.complexityTier.basic",
      expect.objectContaining({ defaultValue: "Basic" })
    )
  })

  it("surfaces starter production readiness metadata", () => {
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

    const setupCard = screen.getByTestId("visual-buddy-setup-choice-card")
    expect(setupCard).toHaveTextContent(/art ready/i)
    expect(setupCard).toHaveTextContent(/basic/i)
    expect(setupCard).toHaveTextContent(/neutral anchor/i)
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

  it("disables action buttons when handlers are omitted", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={starter}
        starterCount={2}
      />
    )

    expect(screen.getByRole("button", { name: /use default/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /import pack/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /start blank/i })).toBeDisabled()
    expect(
      screen.queryByRole("button", { name: /choose another default/i })
    ).not.toBeInTheDocument()
  })

  it("distinguishes unknown starter counts from actual zero", () => {
    const { rerender } = render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={starter}
        onUseDefault={vi.fn()}
      />
    )

    expect(screen.getByText(/bundled default count unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/no bundled defaults available/i)).not.toBeInTheDocument()

    rerender(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={starter}
        starterCount={0}
        onUseDefault={vi.fn()}
      />
    )

    expect(screen.getByText(/no bundled defaults available/i)).toBeInTheDocument()

    rerender(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={null}
        starterCatalogLoading
      />
    )

    expect(screen.getByText(/checking bundled defaults/i)).toBeInTheDocument()
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

    fireEvent.click(screen.getByRole("button", { name: /open buddy builder/i }))

    expect(onOpenVisuals).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("button", { name: /use default/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /import pack/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /start blank/i })).not.toBeInTheDocument()
    expect(screen.getByTestId("visual-buddy-setup-choice-card")).not.toHaveTextContent(
      /no visual buddy is active|starter catalog unavailable|no bundled defaults available/i
    )
  })

  it("disables compact setup action when the open handler is omitted", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        compact
      />
    )

    expect(screen.getByRole("button", { name: /open buddy builder/i })).toBeDisabled()
  })
})
