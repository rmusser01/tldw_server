// @vitest-environment jsdom
/**
 * XS-02 (#3111): a fresh side panel picks a model on its own, as the full
 * chat page does, so the first send does not fail with "Please select a
 * model" while a working model exists.
 */
import React from "react"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const state = vi.hoisted(() => ({
  selectedModel: null as string | null,
  selectedModelIsLoading: false,
  favorites: [] as string[],
  favoritesLoading: false,
  setSelectedModel: vi.fn(async (_model: string) => true)
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: state.selectedModel,
    setSelectedModel: state.setSelectedModel,
    selectedModelIsLoading: state.selectedModelIsLoading
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [state.favorites, vi.fn(), { isLoading: state.favoritesLoading }]
}))

import { useSidepanelStartupModel } from "../useSidepanelStartupModel"

const MODELS = [
  { model: "tldw:gemma3:1b", provider: "ollama" },
  { model: "tldw:mock-gpt-4o", provider: "custom-openai-api", is_server_default: true }
]

describe("useSidepanelStartupModel (XS-02)", () => {
  beforeEach(() => {
    state.selectedModel = null
    state.selectedModelIsLoading = false
    state.favorites = []
    state.favoritesLoading = false
    state.setSelectedModel.mockClear()
  })

  it("selects the server default when no model is chosen", async () => {
    renderHook(() => useSidepanelStartupModel({ enabled: true, models: MODELS }))

    await waitFor(() =>
      expect(state.setSelectedModel).toHaveBeenCalledWith("custom-openai-api:mock-gpt-4o")
    )
  })

  it("prefers a favourite model", async () => {
    state.favorites = ["ollama:gemma3:1b"]
    renderHook(() => useSidepanelStartupModel({ enabled: true, models: MODELS }))

    await waitFor(() => expect(state.setSelectedModel).toHaveBeenCalledWith("ollama:gemma3:1b"))
  })

  it("keeps a model the person already chose", async () => {
    state.selectedModel = "ollama:gemma3:1b"
    renderHook(() => useSidepanelStartupModel({ enabled: true, models: MODELS }))

    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(state.setSelectedModel).not.toHaveBeenCalled()
  })

  it.each([
    ["the saved choice is still loading", () => { state.selectedModelIsLoading = true }],
    ["favourites are still loading", () => { state.favoritesLoading = true }]
  ])("waits while %s", async (_label, arrange) => {
    arrange()
    renderHook(() => useSidepanelStartupModel({ enabled: true, models: MODELS }))

    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(state.setSelectedModel).not.toHaveBeenCalled()
  })

  it("does nothing until connected and the catalog is verified", async () => {
    const { rerender } = renderHook(
      ({ enabled, models }: { enabled: boolean; models: typeof MODELS | undefined }) =>
        useSidepanelStartupModel({ enabled, models }),
      { initialProps: { enabled: false, models: MODELS as typeof MODELS | undefined } }
    )
    rerender({ enabled: true, models: undefined })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(state.setSelectedModel).not.toHaveBeenCalled()
  })
})
