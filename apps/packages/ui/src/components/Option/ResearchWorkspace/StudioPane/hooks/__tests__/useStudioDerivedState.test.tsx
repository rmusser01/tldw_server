import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useStudioDerivedState } from "../useStudioDerivedState"

const options = {
  ragTopK: undefined, ragAdvancedOptions: {}, temperature: undefined, topP: undefined,
  numPredict: undefined, setRagTopK: vi.fn(), setRagAdvancedOptions: vi.fn(),
  setTemperature: vi.fn(), setTopP: vi.fn(), generatedArtifacts: [],
  outputListScrollTop: 0, outputListViewportHeight: 600,
}

describe("Studio output budget defaults", () => {
  it("uses the DeepSeek reasoning budget when unset", () => {
    const { result } = renderHook(() => useStudioDerivedState({ ...options, provider: "deepseek" }))
    expect(result.current.resolvedNumPredict).toBe(8192)
  })
  it("preserves an explicit DeepSeek output budget", () => {
    const { result } = renderHook(() => useStudioDerivedState({ ...options, provider: "deepseek", numPredict: 4096 }))
    expect(result.current.resolvedNumPredict).toBe(4096)
  })
  it("retains the ordinary fallback for other providers", () => {
    const { result } = renderHook(() => useStudioDerivedState({ ...options, provider: "other" }))
    expect(result.current.resolvedNumPredict).toBe(4096)
  })
})
