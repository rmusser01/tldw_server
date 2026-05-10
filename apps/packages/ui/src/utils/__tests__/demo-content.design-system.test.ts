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

    expect(
      items.map(({ id, statusKey, statusLabel, title }) => ({
        id,
        statusKey,
        statusLabel,
        title
      }))
    ).toEqual([
      {
        id: "demo-media-1",
        statusKey: "ready",
        statusLabel: "Ready via registry",
        title: "Demo media: Team call recording"
      },
      {
        id: "demo-media-2",
        statusKey: "processing",
        statusLabel: "Processing",
        title: "Demo media: Product walkthrough"
      },
      {
        id: "demo-media-3",
        statusKey: "ready",
        statusLabel: "Ready via registry",
        title: "Demo media: Research article PDF"
      }
    ])
  })
})
