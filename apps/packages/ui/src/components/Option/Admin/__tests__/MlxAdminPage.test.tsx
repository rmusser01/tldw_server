import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, within } from "@testing-library/react"
import MlxAdminPage from "../MlxAdminPage"

const apiMock = vi.hoisted(() => ({
  getMlxStatus: vi.fn(),
  getLlmProviders: vi.fn(),
  loadMlxModel: vi.fn(),
  unloadMlxModel: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return maybeOptions?.defaultValue || key
    }
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

const expectDesignSystemAlertForText = async (text: string) => {
  const title = await screen.findByText(text)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  return alert as HTMLElement
}

describe("MlxAdminPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    if (!(window as any).ResizeObserver) {
      ;(window as any).ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    }

    apiMock.getMlxStatus.mockResolvedValue({
      active: false,
      model: null,
      max_concurrent: 2
    })
    apiMock.getLlmProviders.mockResolvedValue({
      providers: []
    })
    apiMock.loadMlxModel.mockResolvedValue({})
    apiMock.unloadMlxModel.mockResolvedValue({})
  })

  it("clarifies inactive concurrency semantics", async () => {
    render(<MlxAdminPage />)

    expect(await screen.findByText(/Configured concurrency \(inactive\)/)).toBeTruthy()
    expect(
      await screen.findByText(
        "Concurrency is a configured limit and applies once a model is active."
      )
    ).toBeTruthy()
  })

  it("gates controls when admin APIs are unavailable", async () => {
    apiMock.getMlxStatus.mockRejectedValueOnce(
      new Error(
        "Request failed: 503 (GET /api/v1/admin/mlx/status) config=/Users/dev/.config/tldw/config.txt"
      )
    )

    render(<MlxAdminPage />)

    const alert = await expectDesignSystemAlertForText("Admin APIs not available")
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.queryByText("Load Model")).toBeNull()
  })

  it("disables model actions when MLX status is temporarily unavailable", async () => {
    apiMock.getMlxStatus.mockRejectedValueOnce(new Error("network down"))

    render(<MlxAdminPage />)

    const alert = await expectDesignSystemAlertForText(
      "MLX controls are temporarily unavailable until status checks succeed."
    )
    expect(alert).toHaveAttribute("role", "alert")

    expect(screen.getByRole("button", { name: "Load Model" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Unload Model" })).toBeDisabled()
  })

  it("renders the active-model notice through the design-system Alert primitive", async () => {
    apiMock.getMlxStatus.mockResolvedValueOnce({
      active: true,
      model: "mlx-community/test-model",
      max_concurrent: 2
    })

    render(<MlxAdminPage />)

    const alert = await expectDesignSystemAlertForText(
      "A model is currently loaded. Unload it first or load a different model to replace it."
    )
    expect(alert).toHaveAttribute("role", "status")
  })

  it("renders the trust-remote-code warning through the design-system Badge primitive", async () => {
    render(<MlxAdminPage />)

    fireEvent.click(await screen.findByText("Advanced Settings"))
    await screen.findByText("Trust remote code:")

    const trustRemoteCodeLabel = screen.getByText("Trust remote code:")
    const trustRemoteCodeRow = trustRemoteCodeLabel.closest("div")
    expect(trustRemoteCodeRow).not.toBeNull()

    fireEvent.click(within(trustRemoteCodeRow as HTMLElement).getByRole("switch"))

    const warning = await screen.findByText("Security risk")
    const badge = warning.closest('[data-ds-component="Badge"]')
    expect(badge).not.toBeNull()
  })
})
