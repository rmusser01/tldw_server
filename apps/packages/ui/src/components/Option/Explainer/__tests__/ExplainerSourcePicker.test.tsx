import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { ExplainerSourceCandidate } from "../types"
import { ExplainerSourcePicker } from "../ExplainerSourcePicker"

const candidates: ExplainerSourceCandidate[] = [
  { sourceId: "m1", sourceType: "media", title: "Attention PDF", description: "pdf" },
  { sourceId: "n1", sourceType: "note", title: "Reading notes", description: null }
]

const renderPicker = (overrides: Partial<React.ComponentProps<typeof ExplainerSourcePicker>> = {}) =>
  render(
    <ExplainerSourcePicker
      query="attention"
      results={candidates}
      selectedSources={[]}
      grounding="source_led"
      outputIntent="explain"
      depthPreset="standard"
      onQueryChange={vi.fn()}
      onSearch={vi.fn()}
      onAddSource={vi.fn()}
      onRemoveSource={vi.fn()}
      onGroundingChange={vi.fn()}
      onOutputIntentChange={vi.fn()}
      onDepthPresetChange={vi.fn()}
      onCreate={vi.fn()}
      {...overrides}
    />
  )

describe("ExplainerSourcePicker", () => {
  it("shows a result count above the list", () => {
    renderPicker()

    expect(screen.getByText("2 results")).toBeInTheDocument()
  })

  it("marks already-selected sources as Added instead of a disabled Add", () => {
    renderPicker({
      selectedSources: [{ sourceId: "m1", sourceType: "media", title: "Attention PDF" }]
    })

    expect(screen.getByRole("button", { name: "Added Attention PDF" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Add Reading notes" })).toBeEnabled()
  })

  it("explains the empty selection in terms of the active grounding mode", () => {
    renderPicker({ grounding: "source_only" })
    expect(
      screen.getByText("Add at least one source — source-only grounding requires it.")
    ).toBeInTheDocument()
  })

  it("uses neutral guidance for non-source-only grounding", () => {
    renderPicker({ grounding: "source_led" })
    expect(
      screen.getByText("Search and add sources to ground the explanation in your library.")
    ).toBeInTheDocument()
  })
})
