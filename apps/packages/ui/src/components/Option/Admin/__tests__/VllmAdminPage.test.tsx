import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import VllmAdminPage from "../VllmAdminPage"

const apiMock = vi.hoisted(() => ({
  listVllmInstances: vi.fn(),
  createVllmInstance: vi.fn(),
  setDefaultVllmInstance: vi.fn(),
  startVllmInstance: vi.fn(),
  stopVllmInstance: vi.fn(),
  restartVllmInstance: vi.fn(),
  probeVllmInstance: vi.fn(),
  deleteVllmInstance: vi.fn()
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

describe("VllmAdminPage", () => {
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

    apiMock.listVllmInstances.mockResolvedValue({
      backend: "vllm",
      default_instance_id: null,
      instances: [
        {
          instance_id: "vision-id",
          name: "vision-a100",
          execution_mode: "local",
          transport_config: {},
          launch_spec: { model: "Qwen/Qwen2.5-VL-7B-Instruct", port: 8000 },
          routing_policy: {},
          declared_capabilities: { chat: true, vision: true },
          desired_state: "stopped",
          observed_state: "stopped",
          created_at: "2026-03-10T00:00:00Z",
          updated_at: "2026-03-10T00:00:00Z",
          probed_capabilities: {},
          effective_capabilities: { chat: true, vision: true },
          last_known_base_url: null,
          last_error: null,
          executor_handle: {}
        }
      ]
    })
    apiMock.createVllmInstance.mockResolvedValue({})
    apiMock.setDefaultVllmInstance.mockResolvedValue({ backend: "vllm", default_instance_id: "vision-id" })
    apiMock.startVllmInstance.mockResolvedValue({
      backend: "vllm",
      instance_id: "vision-id",
      requested_action: "start",
      job_id: 101,
      status: "queued"
    })
    apiMock.stopVllmInstance.mockResolvedValue({})
    apiMock.restartVllmInstance.mockResolvedValue({})
    apiMock.probeVllmInstance.mockResolvedValue({})
    apiMock.deleteVllmInstance.mockResolvedValue({})
  })

  it("loads managed instances and renders start controls", async () => {
    render(<VllmAdminPage />)

    expect(await screen.findByText("vision-a100")).toBeTruthy()
    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy()
  })
})
