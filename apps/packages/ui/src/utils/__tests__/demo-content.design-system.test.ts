import type { TFunction } from "i18next"
import { describe, expect, it, vi } from "vitest"
import { getDemoMediaItems } from "../demo-content"

const registryLabels = vi.hoisted(() => ({
  ready: "Ready via registry"
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "ready" ? registryLabels.ready : state.label
        }
      }
    )
  }
})

const t = ((_: string, options?: { defaultValue?: string }) =>
  options?.defaultValue ?? "") as TFunction

describe("getDemoMediaItems", () => {
  it("uses the design-system ready label while preserving demo media order", () => {
    const items = getDemoMediaItems(t)

    expect(items.map((item) => item.id)).toEqual([
      "demo-media-1",
      "demo-media-2",
      "demo-media-3"
    ])
    expect(items.map((item) => item.statusKey)).toEqual([
      "ready",
      "processing",
      "ready"
    ])
    expect(items.map((item) => item.statusLabel)).toEqual([
      "Ready via registry",
      "Processing",
      "Ready via registry"
    ])
    expect(items.map((item) => item.title)).toEqual([
      "Demo media: Team call recording",
      "Demo media: Product walkthrough",
      "Demo media: Research article PDF"
    ])
  })
})
