import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ImageOcclusionPanel } from "../ImageOcclusionPanel"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

describe("ImageOcclusionPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.stubGlobal(
      "URL",
      Object.assign(globalThis.URL ?? {}, {
        createObjectURL: vi.fn(() => "blob:occlusion-source"),
        revokeObjectURL: vi.fn()
      })
    )
  })

  it("loads a source image, creates normalized regions, tracks labels, and keeps selection stable after removal", async () => {
    render(<ImageOcclusionPanel />)

    fireEvent.change(screen.getByLabelText("Upload image occlusion source"), {
      target: {
        files: [new File(["binary"], "diagram.png", { type: "image/png" })]
      }
    })

    expect(await screen.findByAltText("Image occlusion source preview")).toBeInTheDocument()

    const overlay = screen.getByTestId("image-occlusion-overlay")
    Object.defineProperty(overlay, "getBoundingClientRect", {
      value: () => ({
        x: 10,
        y: 20,
        left: 10,
        top: 20,
        right: 210,
        bottom: 120,
        width: 200,
        height: 100,
        toJSON: () => ({})
      })
    })

    fireEvent.pointerDown(overlay, { clientX: 30, clientY: 30 })
    fireEvent.pointerMove(overlay, { clientX: 31, clientY: 32 })
    fireEvent.pointerUp(overlay, { clientX: 31, clientY: 32 })
    expect(screen.queryByTestId("image-occlusion-region-row-region-1")).not.toBeInTheDocument()

    fireEvent.pointerDown(overlay, { clientX: 30, clientY: 30 })
    fireEvent.pointerMove(overlay, { clientX: 90, clientY: 70 })
    fireEvent.pointerUp(overlay, { clientX: 90, clientY: 70 })

    fireEvent.pointerDown(overlay, { clientX: 110, clientY: 40 })
    fireEvent.pointerMove(overlay, { clientX: 170, clientY: 90 })
    fireEvent.pointerUp(overlay, { clientX: 170, clientY: 90 })

    expect(await screen.findByTestId("image-occlusion-region-row-region-1")).toBeInTheDocument()
    expect(screen.getByTestId("image-occlusion-region-row-region-2")).toBeInTheDocument()
    expect(screen.getByTestId("image-occlusion-selected-region")).toHaveTextContent("Region 2")
    expect(screen.getByTestId("image-occlusion-region-geometry-region-1")).toHaveTextContent(
      "x: 10.0%, y: 10.0%, w: 30.0%, h: 40.0%"
    )

    fireEvent.change(screen.getByTestId("image-occlusion-region-label-region-2"), {
      target: { value: "Mitochondria" }
    })
    expect(screen.getByDisplayValue("Mitochondria")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("image-occlusion-remove-region-region-2"))

    await waitFor(() => {
      expect(screen.queryByTestId("image-occlusion-region-row-region-2")).not.toBeInTheDocument()
    })
    expect(screen.getByTestId("image-occlusion-selected-region")).toHaveTextContent("Region 1")
  })

  it("renders empty authoring notices with the design-system Alert", async () => {
    render(<ImageOcclusionPanel />)

    const awaitingImageNotice = screen.getByText(
      "Choose an image to begin authoring occlusions."
    )
    expect(awaitingImageNotice.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Upload image occlusion source"), {
      target: {
        files: [new File(["binary"], "diagram.png", { type: "image/png" })]
      }
    })

    expect(await screen.findByAltText("Image occlusion source preview")).toBeInTheDocument()
    const noRegionsNotice = screen.getByText(
      "No regions yet. Draw on the image to add one."
    )
    expect(noRegionsNotice.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })
})
