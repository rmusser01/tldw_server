import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import LlamacppAdminPage from "../LlamacppAdminPage"

const apiMock = vi.hoisted(() => ({
  getLlamacppConfig: vi.fn(),
  getLlamacppStatus: vi.fn(),
  getLlamacppInventory: vi.fn(),
  getLlamacppHardware: vi.fn(),
  listLlamacppModels: vi.fn(),
  startLlamacppModel: vi.fn(),
  startLlamacppServer: vi.fn(),
  stopLlamacppServer: vi.fn(),
  useLlamacppInChat: vi.fn(),
  registerLlamacppModelPath: vi.fn()
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

vi.mock("../LlamacppLaunchPanel", async () => {
  const React = await import("react")

  interface ChatActionState {
    visible: boolean
    loading?: boolean
    notice?: string | null
    warnings?: string[]
    onUse: () => void
  }

  interface MockLaunchPanelProps {
    selectedModelId?: string
    isRunning: boolean
    actionLoading: boolean
    inventoryUnavailable: boolean
    adminUnavailable: boolean
    onStart: () => void
    onStartWithDefaults: () => void
    chatAction: ChatActionState | null
  }

  const LlamacppLaunchPanel = ({
    selectedModelId,
    isRunning,
    actionLoading,
    inventoryUnavailable,
    adminUnavailable,
    onStart,
    onStartWithDefaults,
    chatAction
  }: MockLaunchPanelProps) => {
    const startDisabled =
      !selectedModelId ||
      isRunning ||
      inventoryUnavailable ||
      adminUnavailable ||
      actionLoading

    return React.createElement(
      "section",
      { "aria-label": "Launch" },
      React.createElement(
        "button",
        { disabled: startDisabled, onClick: onStart },
        "Start Server"
      ),
      React.createElement(
        "button",
        { disabled: startDisabled, onClick: onStartWithDefaults },
        "Start with Defaults"
      ),
      chatAction?.visible
        ? React.createElement(
            "button",
            { disabled: chatAction.loading, onClick: chatAction.onUse },
            "Use this in Chat"
          )
        : null,
      chatAction?.notice
        ? React.createElement("div", null, chatAction.notice)
        : null,
      chatAction?.warnings?.map((warning) =>
        React.createElement("div", { key: warning }, warning)
      )
    )
  }

  return {
    LlamacppLaunchPanel,
    default: LlamacppLaunchPanel
  }
})

const mockConfig = {
  saved_config: {
    enabled: true,
    executable_path: "/opt/llama-server",
    models_dir: "/srv/models/gguf",
    default_host: "127.0.0.1",
    default_port: 8080,
    default_threads: 8,
    default_n_gpu_layers: 0,
    default_ctx_size: 4096,
    allowed_paths: ["/srv/models"],
    registered_model_paths: [],
    log_output_file: null
  },
  active_config: {
    handler_configured: false,
    enabled: null,
    executable_path: null,
    models_dir: null,
    default_host: null,
    default_port: null,
    active_model: null,
    active_host: null,
    active_port: null,
    active_pid: null
  },
  restart_required: true,
  restart_reasons: ["handler_not_configured"],
  env_overrides: {
    models_dir: true,
    default_port: false
  },
  warnings: ["Saved config is loaded on API server restart."]
}

const mockInventory = {
  models: [
    {
      model_id: "gguf:toy-model-id",
      display_name: "Toy 7B Q4_K_M",
      basename: "toy-7b-q4_k_m.gguf",
      source: "models_dir",
      path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
      size_bytes: 4_200_000_000,
      modified_at: "2026-05-15T10:00:00Z",
      metadata: {
        quantization: "Q4_K_M",
        parameter_hint: "7B",
        context_hint: 4096
      },
      warnings: ["Metadata is filename-derived."]
    }
  ],
  warnings: ["One registered path was skipped."],
  scan_limited: false
}

describe("LlamacppAdminPage", () => {
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

    apiMock.getLlamacppConfig.mockResolvedValue(mockConfig)
    apiMock.getLlamacppStatus.mockResolvedValue({
      state: "stopped",
      model: null,
      port: 8080
    })
    apiMock.getLlamacppInventory.mockResolvedValue(mockInventory)
    apiMock.getLlamacppHardware.mockResolvedValue({
      ram_total_bytes: 16_000_000_000,
      ram_available_bytes: 8_000_000_000,
      cpu_count: 8,
      gpus: [],
      warnings: ["GPU probe unavailable."]
    })
    apiMock.startLlamacppModel.mockResolvedValue({
      status: "started",
      model_id: "gguf:toy-model-id"
    })
    apiMock.stopLlamacppServer.mockResolvedValue({
      status: "stopped"
    })
    apiMock.useLlamacppInChat.mockResolvedValue({
      provider: "llamacpp",
      endpoint: "http://127.0.0.1:8080",
      updated: true,
      effective: true,
      warnings: []
    })
    apiMock.registerLlamacppModelPath.mockResolvedValue(mockInventory.models[0])
  })

  it("renders readiness from saved config and restart-required state", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Readiness")).toBeTruthy()
    expect(await screen.findByText("/srv/models/gguf")).toBeTruthy()
    expect(await screen.findByText("API server restart required")).toBeTruthy()
    expect(await screen.findByText("handler_not_configured")).toBeTruthy()
    expect(await screen.findByText("models_dir override")).toBeTruthy()
    expect(screen.queryByText("Active handler configured")).toBeNull()
  })

  it("renders inventory display names warnings and stable model selection", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Toy 7B Q4_K_M")).toBeTruthy()
    expect(screen.getByText("toy-7b-q4_k_m.gguf")).toBeTruthy()
    expect(screen.getByText("/srv/models/gguf/toy-7b-q4_k_m.gguf")).toBeTruthy()
    expect(screen.getByText("Metadata is filename-derived.")).toBeTruthy()
    expect(screen.getByText("Selected")).toBeTruthy()
  })

  it("starts selected inventory model by stable model_id", async () => {
    render(<LlamacppAdminPage />)

    fireEvent.click(await screen.findByRole("button", { name: "Start Server" }))

    await waitFor(() => {
      expect(apiMock.startLlamacppModel).toHaveBeenCalled()
    })

    expect(apiMock.startLlamacppModel).toHaveBeenCalledWith(
      "gguf:toy-model-id",
      expect.objectContaining({
        ctx_size: 4096,
        n_gpu_layers: 0,
        cache_type_k: "f16",
        cache_type_v: "f16"
      })
    )
    expect(apiMock.startLlamacppServer).not.toHaveBeenCalled()
  })

  it("shows explicit chat wiring after start and never calls it automatically", async () => {
    apiMock.getLlamacppStatus
      .mockResolvedValueOnce({
        state: "stopped",
        model: null,
        port: 8080
      })
      .mockResolvedValueOnce({
        state: "running",
        model: "toy-7b-q4_k_m.gguf",
        port: 8080
      })

    render(<LlamacppAdminPage />)

    expect(await screen.findByRole("button", { name: "Start Server" })).toBeTruthy()
    expect(apiMock.useLlamacppInChat).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Start Server" }))

    expect(await screen.findByRole("button", { name: "Use this in Chat" })).toBeTruthy()
    expect(apiMock.useLlamacppInChat).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Use this in Chat" }))

    await waitFor(() => {
      expect(apiMock.useLlamacppInChat).toHaveBeenCalledTimes(1)
    })
    expect(await screen.findByText("Chat provider updated.")).toBeTruthy()
  })

  it("shows chat wiring when status already reports a running managed server", async () => {
    apiMock.getLlamacppStatus.mockResolvedValueOnce({
      state: "running",
      model: "toy-7b-q4_k_m.gguf",
      port: 8080
    })

    render(<LlamacppAdminPage />)

    expect(await screen.findByRole("button", { name: "Use this in Chat" })).toBeTruthy()
    expect(apiMock.useLlamacppInChat).not.toHaveBeenCalled()
  })

  it("hides chat wiring when a later status refresh reports stopped", async () => {
    apiMock.getLlamacppStatus
      .mockResolvedValueOnce({
        state: "running",
        model: "toy-7b-q4_k_m.gguf",
        port: 8080
      })
      .mockResolvedValueOnce({
        state: "stopped",
        model: null,
        port: 8080
      })

    render(<LlamacppAdminPage />)

    expect(await screen.findByRole("button", { name: "Use this in Chat" })).toBeTruthy()

    fireEvent.click(await screen.findByTitle("Refresh status"))

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Use this in Chat" })).toBeNull()
    })
  })

  it("keeps launch available when only hardware probing fails", async () => {
    apiMock.getLlamacppHardware.mockRejectedValueOnce(
      new Error("Request failed: 503 (GET /api/v1/admin/llamacpp/hardware)")
    )

    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Inventory")).toBeTruthy()
    expect(await screen.findByRole("button", { name: "Start Server" })).toBeTruthy()
    expect(screen.queryByText("Admin APIs not available")).toBeNull()
  })

  it("registers a local model path and reloads inventory", async () => {
    render(<LlamacppAdminPage />)

    fireEvent.change(await screen.findByLabelText("Register local GGUF path"), {
      target: { value: "/external/model.gguf" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Register path" }))

    await waitFor(() => {
      expect(apiMock.registerLlamacppModelPath).toHaveBeenCalledWith("/external/model.gguf")
      expect(apiMock.getLlamacppInventory).toHaveBeenCalledTimes(2)
    })
  })

  it("loads config status inventory and hardware only once during strict-mode mount", async () => {
    render(
      <React.StrictMode>
        <LlamacppAdminPage />
      </React.StrictMode>
    )

    await waitFor(() => {
      expect(apiMock.getLlamacppConfig).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppStatus).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppInventory).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppHardware).toHaveBeenCalledTimes(1)
    })
  })

  it("gates controls when admin APIs are unavailable", async () => {
    apiMock.getLlamacppStatus.mockRejectedValueOnce(
      new Error(
        "Request failed: 503 (GET /api/v1/admin/llamacpp/status) config=/Users/dev/.config/tldw/config.txt"
      )
    )

    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Admin APIs not available")).toBeTruthy()
    expect(screen.queryByText("Inventory")).toBeNull()
  })
})
