import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import LlamacppAdminPage from "../LlamacppAdminPage"

// Render counters observed on two sibling panels while the real launch panel
// (unmocked here) handles keystrokes. Counting pass-through wrappers count
// every React invocation, i.e. every page-driven re-render of the sibling.
const panelRenderCounts = vi.hoisted(() => ({ assets: 0, inventory: 0 }))

vi.mock("../LlamacppAssetsPanel", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../LlamacppAssetsPanel")>()

  const CountingAssetsPanel = (
    props: React.ComponentProps<typeof actual.LlamacppAssetsPanel>
  ) => {
    panelRenderCounts.assets += 1
    return <actual.LlamacppAssetsPanel {...props} />
  }

  return {
    LlamacppAssetsPanel: CountingAssetsPanel,
    default: CountingAssetsPanel
  }
})

vi.mock("../LlamacppInventoryPanel", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../LlamacppInventoryPanel")>()

  const CountingInventoryPanel = (
    props: React.ComponentProps<typeof actual.LlamacppInventoryPanel>
  ) => {
    panelRenderCounts.inventory += 1
    return <actual.LlamacppInventoryPanel {...props} />
  }

  return {
    LlamacppInventoryPanel: CountingInventoryPanel,
    default: CountingInventoryPanel
  }
})

const apiMock = vi.hoisted(() => ({
  getLlamacppConfig: vi.fn(),
  getLlamacppStatus: vi.fn(),
  getLlamacppInventory: vi.fn(),
  getLlamacppAssets: vi.fn(),
  getLlamacppHardware: vi.fn(),
  listLlamacppProfiles: vi.fn(),
  listLlamacppInstances: vi.fn(),
  startLlamacppModel: vi.fn(),
  listLlamacppAssetDownloads: vi.fn()
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
    imported_asset_folders: [],
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
  env_overrides: {},
  warnings: []
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
      metadata: {},
      warnings: []
    }
  ],
  warnings: [],
  scan_limited: false
}

const mockAssets = {
  assets: [],
  warnings: [],
  scan_limited: false
}

const mockProfiles = {
  profiles: []
}

const mockRuntimes = {
  runtimes: []
}

describe("LlamacppAdminPage render isolation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    panelRenderCounts.assets = 0
    panelRenderCounts.inventory = 0

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
    apiMock.getLlamacppAssets.mockResolvedValue(mockAssets)
    apiMock.getLlamacppHardware.mockResolvedValue({
      ram_total_bytes: 16_000_000_000,
      ram_available_bytes: 8_000_000_000,
      cpu_count: 8,
      gpus: [],
      warnings: ["GPU probe unavailable."]
    })
    apiMock.listLlamacppProfiles.mockResolvedValue(mockProfiles)
    apiMock.listLlamacppInstances.mockResolvedValue(mockRuntimes)
    apiMock.listLlamacppAssetDownloads.mockResolvedValue({ jobs: [] })
    apiMock.startLlamacppModel.mockResolvedValue({
      status: "started",
      model_id: "gguf:toy-model-id"
    })
  })

  const waitForInitialLoad = async () => {
    // "Selected" marks that the inventory loaded and a model is picked.
    expect(await screen.findByText("Selected")).toBeTruthy()
    await waitFor(() => {
      expect(apiMock.getLlamacppConfig).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppStatus).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppInventory).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlamacppHardware).toHaveBeenCalledTimes(1)
      expect(apiMock.listLlamacppProfiles).toHaveBeenCalledTimes(1)
      expect(apiMock.listLlamacppInstances).toHaveBeenCalledTimes(1)
      expect(apiMock.listLlamacppAssetDownloads).toHaveBeenCalledTimes(1)
    })
    // Flush follow-up renders triggered by the initial load settling.
    await act(async () => {
      await Promise.resolve()
    })
    await act(async () => {
      await Promise.resolve()
    })
  }

  it("test_keystroke_rerenders_only_launch_panel", async () => {
    render(<LlamacppAdminPage />)
    await waitForInitialLoad()

    fireEvent.click(screen.getByText("Other Options"))
    const tensorSplitInput = (await screen.findByPlaceholderText(
      "e.g. 38,62"
    )) as HTMLInputElement

    const assetsBefore = panelRenderCounts.assets
    const inventoryBefore = panelRenderCounts.inventory

    fireEvent.change(tensorSplitInput, { target: { value: "38,62" } })

    // The keystroke took effect inside the launch panel itself.
    expect(tensorSplitInput.value).toBe("38,62")
    await act(async () => {
      await Promise.resolve()
    })

    // Sibling panels must not have re-rendered for a launch-form keystroke.
    expect(panelRenderCounts.assets).toBe(assetsBefore)
    expect(panelRenderCounts.inventory).toBe(inventoryBefore)
  })

  it("test_launch_submits_current_args", async () => {
    render(<LlamacppAdminPage />)
    await waitForInitialLoad()

    fireEvent.click(screen.getByText("Other Options"))
    fireEvent.change(await screen.findByPlaceholderText("e.g. 38,62"), {
      target: { value: "38,62" }
    })

    fireEvent.click(screen.getByText("Network & Runtime"))
    fireEvent.change(await screen.findByPlaceholderText("127.0.0.1"), {
      target: { value: "0.0.0.0" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Start Server" }))

    await waitFor(() => {
      expect(apiMock.startLlamacppModel).toHaveBeenCalledTimes(1)
    })
    expect(apiMock.startLlamacppModel).toHaveBeenCalledWith(
      "gguf:toy-model-id",
      expect.objectContaining({
        ctx_size: 4096,
        n_gpu_layers: 0,
        tensor_split: [38, 62],
        host: "0.0.0.0"
      })
    )
  })
})
