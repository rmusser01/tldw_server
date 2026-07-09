// @vitest-environment jsdom

import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const storageState = vi.hoisted(() => ({
  calls: [] as Array<{ key: string; defaultValue: unknown }>
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    storageState.calls.push({ key, defaultValue })
    return [defaultValue, vi.fn()]
  }
}))

import { useSttSettings } from "../useSttSettings"

describe("useSttSettings", () => {
  beforeEach(() => {
    storageState.calls = []
  })

  it("defaults to no concrete STT model so the server chooses its configured default", () => {
    const { result } = renderHook(() => useSttSettings())

    expect(result.current.model).toBe("")
    expect(
      storageState.calls.find((call) => call.key === "sttModel")?.defaultValue
    ).toBe("")
  })
})
