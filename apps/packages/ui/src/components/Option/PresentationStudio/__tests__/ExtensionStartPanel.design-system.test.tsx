// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ExtensionStartPanel } from "../ExtensionStartPanel"

const designSystemLabels = vi.hoisted(() => ({
  empty: "Registry Empty",
  ready: "Registry Ready"
}))

const translationMock = vi.hoisted(() => ({
  t: vi.fn((key: string, fallback: string) => fallback)
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: {
      hasPresentationStudio: true
    }
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    serverUrl: "http://127.0.0.1:8000"
  })
}))

vi.mock("@/libs/get-screenshot", () => ({
  getScreenshotFromCurrentTab: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createPresentation: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translationMock.t
  })
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    EMPTY_STATE_LABEL: designSystemLabels.empty,
    READY_STATE_LABEL: designSystemLabels.ready,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        if (key === "empty") {
          return { ...state, label: designSystemLabels.empty }
        }

        if (key === "ready") {
          return { ...state, label: designSystemLabels.ready }
        }

        return state
      }
    )
  }
})

describe("ExtensionStartPanel design-system labels", () => {
  it("renders seed statuses through translated design-system state labels", async () => {
    render(<ExtensionStartPanel />)

    expect(screen.getAllByText(designSystemLabels.empty)).toHaveLength(2)

    fireEvent.change(screen.getByLabelText("Narration seed"), {
      target: { value: "Open with the product story." }
    })

    const imageFile = new File(["seed"], "seed.png", { type: "image/png" })
    fireEvent.change(screen.getByLabelText("Upload image"), {
      target: { files: [imageFile] }
    })

    await waitFor(() => {
      expect(screen.getAllByText(designSystemLabels.ready)).toHaveLength(2)
    })
    expect(screen.queryAllByText("Ready")).toHaveLength(0)
    expect(screen.queryAllByText("Empty")).toHaveLength(0)
    expect(translationMock.t).toHaveBeenCalledWith(
      "presentationStudio.start.status.ready",
      designSystemLabels.ready
    )
    expect(translationMock.t).toHaveBeenCalledWith(
      "presentationStudio.start.status.empty",
      designSystemLabels.empty
    )
    expect(screen.getByText("Server")).toBeInTheDocument()
    expect(screen.getAllByText("Narration seed").length).toBeGreaterThan(0)
    expect(screen.getByText("Image seed")).toBeInTheDocument()
  })
})
