import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { RagSearchBar } from "../RagSearchBar"

const { ragSearch, optionState } = vi.hoisted(() => ({
  ragSearch: vi.fn(),
  optionState: {
    ragPinnedResults: [],
    setRagPinnedResults: vi.fn(),
    setRagSearchMode: vi.fn(),
    setRagTopK: vi.fn(),
    setRagEnableGeneration: vi.fn(),
    setRagEnableCitations: vi.fn(),
    setRagSources: vi.fn(),
    setRagAdvancedOptions: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: function useStorage<T>(_key: string, initialValue: T) {
    return React.useState(initialValue)
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: typeof optionState) => unknown) =>
    selector(optionState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { initialize: vi.fn().mockResolvedValue(undefined), ragSearch }
}))

vi.mock("@/components/Knowledge/hooks", () => ({
  withFullMediaTextIfAvailable: vi.fn(async (result: unknown) => result)
}))

describe("RagSearchBar input ownership", () => {
  const props = { onInsert: vi.fn(), onAsk: vi.fn() }
  let frames: Map<number, FrameRequestCallback>
  let nextFrame: number

  const flushFrames = () => {
    const pending = [...frames.values()]
    frames.clear()
    act(() => pending.forEach((callback) => callback(0)))
  }

  beforeEach(() => {
    ragSearch.mockReset().mockResolvedValue({ results: [] })
    frames = new Map()
    nextFrame = 0
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      frames.set(++nextFrame, callback)
      return nextFrame
    })
    vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id))
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("focuses the query input after opening and reopening the panel", () => {
    const { rerender } = render(<RagSearchBar {...props} open={false} />)
    rerender(<RagSearchBar {...props} open />)
    flushFrames()
    expect(screen.getByRole("textbox", { name: "Search query" })).toHaveFocus()

    rerender(<RagSearchBar {...props} open={false} />)
    rerender(<RagSearchBar {...props} open />)
    flushFrames()
    expect(screen.getByRole("textbox", { name: "Search query" })).toHaveFocus()
  })

  it("leaves focus alone when autofocus is disabled", () => {
    render(<RagSearchBar {...props} open autoFocus={false} />)
    flushFrames()
    expect(screen.getByRole("textbox", { name: "Search query" })).not.toHaveFocus()
  })

  it("cancels pending input focus when autofocus is disabled", () => {
    const { rerender } = render(<RagSearchBar {...props} open />)
    const input = screen.getByRole("textbox", { name: "Search query" })
    const focus = vi.spyOn(input, "focus")
    rerender(<RagSearchBar {...props} open autoFocus={false} />)
    flushFrames()
    expect(focus).not.toHaveBeenCalled()
  })

  it("submits the edited query and filters through the search state hook", async () => {
    render(<RagSearchBar {...props} open autoFocus={false} />)
    const input = screen.getByRole("textbox", { name: "Search query" })
    fireEvent.change(input, { target: { value: "  orbital mechanics  " } })
    fireEvent.change(screen.getByRole("textbox", { name: "Keyword filter" }), {
      target: { value: "space" }
    })
    fireEvent.keyDown(input, { key: "Enter", code: "Enter", charCode: 13 })

    await waitFor(() => expect(ragSearch).toHaveBeenCalledWith(
      "orbital mechanics",
      expect.objectContaining({ keyword_filter: "space" })
    ))
  })
})
