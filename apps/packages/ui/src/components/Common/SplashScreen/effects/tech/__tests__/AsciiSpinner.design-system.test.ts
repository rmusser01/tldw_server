import { describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
import AsciiSpinner from "../AsciiSpinner"

const registryLabels = vi.hoisted(() => ({
  loading: "Loading via registry"
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
          label: key === "loading" ? registryLabels.loading : state.label
        }
      }
    )
  }
})

describe("AsciiSpinner design-system labels", () => {
  it("renders the loading row from the design-system loading state label", () => {
    const effect = new AsciiSpinner()
    effect.init({} as CanvasRenderingContext2D, 800, 480)
    effect.update(0, 16)

    const grid = (effect as unknown as {
      grid: {
        renderToCanvas: (ctx: CanvasRenderingContext2D, cellW: number, cellH: number) => void
        writeCentered: (row: number, text: string, color: string) => void
      }
    }).grid
    const writeCenteredSpy = vi.spyOn(grid, "writeCentered")
    vi.spyOn(grid, "renderToCanvas").mockImplementation(() => {})
    vi.mocked(getDesignSystemState).mockClear()

    effect.render({} as CanvasRenderingContext2D)

    expect(writeCenteredSpy).toHaveBeenCalledWith(17, "Loading via registry   ", "#aaaaaa")
    expect(getDesignSystemState).not.toHaveBeenCalled()
  })
})
