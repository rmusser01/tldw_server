import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import LlamacppAdminPage from "../LlamacppAdminPage"

const apiMock = vi.hoisted(() => ({
  getLlamacppConfig: vi.fn(),
  getLlamacppStatus: vi.fn(),
  getLlamacppInventory: vi.fn(),
  getLlamacppAssets: vi.fn(),
  getLlamacppHardware: vi.fn(),
  listLlamacppProfiles: vi.fn(),
  listLlamacppInstances: vi.fn(),
  listLlamacppModels: vi.fn(),
  startLlamacppProfile: vi.fn(),
  stopLlamacppProfile: vi.fn(),
  pauseLlamacppProfile: vi.fn(),
  resumeLlamacppProfile: vi.fn(),
  useLlamacppProfileInChat: vi.fn(),
  createLlamacppProfile: vi.fn(),
  updateLlamacppProfile: vi.fn(),
  deleteLlamacppProfile: vi.fn(),
  startLlamacppModel: vi.fn(),
  startLlamacppServer: vi.fn(),
  stopLlamacppServer: vi.fn(),
  useLlamacppInChat: vi.fn(),
  registerLlamacppModelPath: vi.fn(),
  registerLlamacppAssetPath: vi.fn(),
  importLlamacppAssetFolder: vi.fn(),
  previewLlamacppAssetFolder: vi.fn(),
  startLlamacppAssetDownload: vi.fn(),
  listLlamacppAssetDownloads: vi.fn(),
  getLlamacppAssetDownload: vi.fn(),
  cancelLlamacppAssetDownload: vi.fn()
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

vi.mock("../LlamacppRuntimePanel", async () => {
  const React = await import("react")

  interface RuntimePanelProps {
    profiles: Array<{ profile_id: string; name: string }>
    runtimes: Array<{ profile_id: string; state: string; port?: number | null }>
    onStart: (profileId: string) => void
    onStop: (profileId: string) => void
    onUseInChat: (profileId: string) => void
  }

  const LlamacppRuntimePanel = ({
    profiles,
    runtimes,
    onStart,
    onStop,
    onUseInChat
  }: RuntimePanelProps) => {
    const nameByProfile = new Map(profiles.map((profile) => [profile.profile_id, profile.name]))
    return React.createElement(
      "section",
      { "aria-label": "Runtime instances" },
      runtimes.map((runtime) => {
        const label = nameByProfile.get(runtime.profile_id) || runtime.profile_id
        return React.createElement(
          "div",
          { key: runtime.profile_id },
          React.createElement("span", null, label),
          React.createElement("span", null, runtime.state),
          React.createElement("span", null, runtime.port),
          React.createElement(
            "button",
            { onClick: () => onStart(runtime.profile_id) },
            `Start runtime ${runtime.profile_id}`
          ),
          React.createElement(
            "button",
            { onClick: () => onStop(runtime.profile_id) },
            `Stop runtime ${runtime.profile_id}`
          ),
          React.createElement(
            "button",
            { onClick: () => onUseInChat(runtime.profile_id) },
            `Use runtime ${runtime.profile_id}`
          )
        )
      })
    )
  }

  return {
    LlamacppRuntimePanel,
    default: LlamacppRuntimePanel
  }
})

vi.mock("../LlamacppProfilesPanel", async () => {
  const React = await import("react")

  interface ProfilesPanelProps {
    onCreate: (payload: any) => void
    onUpdate: (profileId: string, payload: any) => void
    onDelete: (profileId: string) => void
  }

  const LlamacppProfilesPanel = ({
    onCreate,
    onUpdate,
    onDelete
  }: ProfilesPanelProps) =>
    React.createElement(
      "section",
      { "aria-label": "Profiles" },
      React.createElement(
        "button",
        {
          onClick: () =>
            onCreate({
              name: "Created profile",
              model_id: "gguf:toy-model-id",
              host: "127.0.0.1",
              port: 8190,
              port_policy: "explicit",
              server_args: {}
            })
        },
        "Create saved profile"
      ),
      React.createElement(
        "button",
        {
          onClick: () =>
            onUpdate("analysis", {
              name: "Edited profile",
              port: 8191,
              server_args: { ctx_size: 8192 }
            })
        },
        "Update saved profile"
      ),
      React.createElement(
        "button",
        { onClick: () => onDelete("analysis") },
        "Delete saved profile"
      )
    )

  return {
    LlamacppProfilesPanel,
    default: LlamacppProfilesPanel
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

const mockAssets = {
  assets: [
    {
      asset_id: "gguf:toy-model-id",
      kind: "gguf",
      identity_basis: "resolved_path",
      path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
      resolved_path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
      display_name: "Toy 7B Q4_K_M",
      source: "models_dir",
      size_bytes: 4_200_000_000,
      modified_at: "2026-05-15T10:00:00Z",
      metadata: {
        quantization: "Q4_K_M",
        parameter_hint: "7B",
        context_hint: 4096,
        family_hint: "toy"
      },
      capabilities: ["unknown"],
      mmproj_asset_ids: ["mmproj:toy-vision"],
      base_model_asset_ids: [],
      warnings: ["Projector pairing is inferred."]
    },
    {
      asset_id: "mmproj:toy-vision",
      kind: "mmproj",
      identity_basis: "resolved_path",
      path: "/srv/models/gguf/mmproj-toy.gguf",
      resolved_path: "/srv/models/gguf/mmproj-toy.gguf",
      display_name: "mmproj-toy",
      source: "models_dir",
      size_bytes: 50_000_000,
      modified_at: null,
      metadata: {},
      capabilities: ["vision_projector"],
      mmproj_asset_ids: [],
      base_model_asset_ids: ["gguf:toy-model-id"],
      warnings: []
    },
    {
      asset_id: "folder:external",
      kind: "folder",
      identity_basis: "resolved_path",
      path: "/srv/models/imported",
      resolved_path: "/srv/models/imported",
      display_name: "imported",
      source: "imported_folder",
      size_bytes: null,
      modified_at: null,
      metadata: {},
      capabilities: ["asset_folder"],
      mmproj_asset_ids: [],
      base_model_asset_ids: [],
      warnings: []
    }
  ],
  warnings: ["One imported folder could not be read."],
  scan_limited: false
}

const mockProfiles = {
  profiles: [
    {
      profile_id: "default",
      name: "Default runtime",
      enabled: true,
      mode: "chat",
      model_id: "gguf:toy-model-id",
      model_path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
      host: "127.0.0.1",
      port: 8080,
      port_policy: "explicit",
      server_args: {},
      autostart: false,
      restart_policy: {},
      tags: []
    },
    {
      profile_id: "analysis",
      name: "Analysis runtime",
      enabled: true,
      mode: "chat",
      model_id: "gguf:analysis",
      model_path: "/srv/models/gguf/analysis.gguf",
      host: "127.0.0.1",
      port: 8081,
      port_policy: "explicit",
      server_args: {},
      autostart: false,
      restart_policy: {},
      tags: []
    }
  ]
}

const mockRuntimes = {
  runtimes: [
    {
      profile_id: "default",
      state: "running",
      host: "127.0.0.1",
      port: 8080,
      endpoint: "http://127.0.0.1:8080",
      model_id: "gguf:toy-model-id",
      model_path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
      resolved_args: [],
      restart_count: 0,
      warnings: [],
      health: {},
      log_tail_available: true
    },
    {
      profile_id: "analysis",
      state: "stopped",
      host: "127.0.0.1",
      port: 8081,
      endpoint: null,
      model_id: "gguf:analysis",
      model_path: "/srv/models/gguf/analysis.gguf",
      resolved_args: [],
      restart_count: 0,
      warnings: [],
      health: {},
      log_tail_available: false
    }
  ]
}

const completedDownloadJob = {
  job_id: "99",
  status: "completed",
  operation: "download",
  queue: "acquisition",
  source_label: "Done model",
  destination_path: "/srv/models/done.gguf",
  asset_id: "gguf:done",
  progress: {
    progress_percent: 100
  },
  warnings: [],
  error_message: null
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
    apiMock.startLlamacppProfile.mockResolvedValue({
      profile_id: "analysis",
      action: "start",
      state: "running",
      accepted: true
    })
    apiMock.stopLlamacppProfile.mockResolvedValue({
      profile_id: "default",
      action: "stop",
      state: "stopped",
      accepted: true
    })
    apiMock.pauseLlamacppProfile.mockResolvedValue({
      profile_id: "default",
      action: "pause",
      state: "paused",
      accepted: true
    })
    apiMock.resumeLlamacppProfile.mockResolvedValue({
      profile_id: "default",
      action: "resume",
      state: "stopped",
      accepted: true
    })
    apiMock.useLlamacppProfileInChat.mockResolvedValue({
      provider: "llamacpp",
      endpoint: "http://127.0.0.1:8080",
      updated: true,
      effective: true,
      warnings: []
    })
    apiMock.createLlamacppProfile.mockResolvedValue(mockProfiles.profiles[0])
    apiMock.updateLlamacppProfile.mockResolvedValue(mockProfiles.profiles[1])
    apiMock.deleteLlamacppProfile.mockResolvedValue({
      profile_id: "analysis",
      deleted: true
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
    apiMock.registerLlamacppAssetPath.mockResolvedValue(mockAssets.assets[0])
    apiMock.importLlamacppAssetFolder.mockResolvedValue(mockAssets.assets[2])
    apiMock.previewLlamacppAssetFolder.mockResolvedValue({
      folder: mockAssets.assets[2],
      assets: [mockAssets.assets[0], mockAssets.assets[1]],
      asset_counts: {
        gguf: 1,
        mmproj: 1
      },
      warnings: ["Preview skipped unreadable sidecar file."],
      scan_limited: false,
      will_persist: false
    })
    apiMock.startLlamacppAssetDownload.mockResolvedValue({
      job_id: "42",
      status: "queued",
      operation: "download",
      queue: "acquisition",
      source_label: "Toy model",
      destination_path: "/srv/models/toy.gguf",
      asset_id: null,
      progress: {},
      warnings: [],
      error_message: null
    })
    apiMock.listLlamacppAssetDownloads.mockResolvedValue({ jobs: [] })
    apiMock.getLlamacppAssetDownload.mockResolvedValue({
      job_id: "42",
      status: "running",
      operation: "download",
      queue: "acquisition",
      source_label: "Toy model",
      destination_path: "/srv/models/toy.gguf",
      asset_id: null,
      progress: {
        progress_percent: 25
      },
      warnings: [],
      error_message: null
    })
    apiMock.cancelLlamacppAssetDownload.mockResolvedValue({
      job_id: "42",
      status: "canceled",
      operation: "download",
      queue: "acquisition",
      source_label: "Toy model",
      destination_path: "/srv/models/toy.gguf",
      asset_id: null,
      progress: {},
      warnings: [],
      error_message: null
    })
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

    expect((await screen.findAllByText("Toy 7B Q4_K_M")).length).toBeGreaterThan(0)
    expect(screen.getByText("toy-7b-q4_k_m.gguf")).toBeTruthy()
    expect(screen.getAllByText("/srv/models/gguf/toy-7b-q4_k_m.gguf").length).toBeGreaterThan(0)
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

  it("renders runtime instances and routes profile lifecycle actions", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByLabelText("Runtime instances")).toBeTruthy()
    expect(screen.getByText("Default runtime")).toBeTruthy()
    expect(screen.getByText("Analysis runtime")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Stop runtime default" }))
    fireEvent.click(screen.getByRole("button", { name: "Start runtime analysis" }))
    fireEvent.click(screen.getByRole("button", { name: "Use runtime default" }))

    await waitFor(() => {
      expect(apiMock.stopLlamacppProfile).toHaveBeenCalledWith("default")
      expect(apiMock.startLlamacppProfile).toHaveBeenCalledWith("analysis")
      expect(apiMock.useLlamacppProfileInChat).toHaveBeenCalledWith("default")
    })
  })

  it("routes saved profile mutations without lifecycle side effects", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByLabelText("Profiles")).toBeTruthy()

    apiMock.listLlamacppProfiles.mockClear()
    apiMock.listLlamacppInstances.mockClear()
    apiMock.startLlamacppProfile.mockClear()
    apiMock.useLlamacppProfileInChat.mockClear()
    apiMock.startLlamacppModel.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Create saved profile" }))

    await waitFor(() => {
      expect(apiMock.createLlamacppProfile).toHaveBeenCalledWith({
        name: "Created profile",
        model_id: "gguf:toy-model-id",
        host: "127.0.0.1",
        port: 8190,
        port_policy: "explicit",
        server_args: {}
      })
      expect(apiMock.listLlamacppProfiles).toHaveBeenCalledTimes(1)
      expect(apiMock.listLlamacppInstances).toHaveBeenCalledTimes(1)
    })

    apiMock.listLlamacppProfiles.mockClear()
    apiMock.listLlamacppInstances.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Update saved profile" }))

    await waitFor(() => {
      expect(apiMock.updateLlamacppProfile).toHaveBeenCalledWith("analysis", {
        name: "Edited profile",
        port: 8191,
        server_args: { ctx_size: 8192 }
      })
      expect(apiMock.listLlamacppProfiles).toHaveBeenCalledTimes(1)
      expect(apiMock.listLlamacppInstances).toHaveBeenCalledTimes(1)
    })

    apiMock.listLlamacppProfiles.mockClear()
    apiMock.listLlamacppInstances.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Delete saved profile" }))

    await waitFor(() => {
      expect(apiMock.deleteLlamacppProfile).toHaveBeenCalledWith("analysis")
      expect(apiMock.listLlamacppProfiles).toHaveBeenCalledTimes(1)
      expect(apiMock.listLlamacppInstances).toHaveBeenCalledTimes(1)
    })

    expect(apiMock.startLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.useLlamacppProfileInChat).not.toHaveBeenCalled()
    expect(apiMock.startLlamacppModel).not.toHaveBeenCalled()
  })

  it("keeps single-server controls when runtime instance APIs are unavailable", async () => {
    apiMock.listLlamacppInstances.mockRejectedValueOnce(
      new Error("Request failed: 404 (GET /api/v1/llamacpp/instances)")
    )

    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Inventory")).toBeTruthy()
    expect(await screen.findByRole("button", { name: "Start Server" })).toBeTruthy()
    expect(screen.queryByLabelText("Runtime instances")).toBeNull()
    expect(screen.queryByText("Admin APIs not available")).toBeNull()
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

  it("renders asset inventory groups and warnings next to legacy inventory", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Assets")).toBeTruthy()
    expect(screen.getByText("GGUF models")).toBeTruthy()
    expect(screen.getByText("mmproj projectors")).toBeTruthy()
    expect(screen.getByText("Imported folders")).toBeTruthy()
    expect(screen.getByText("One imported folder could not be read.")).toBeTruthy()
    expect(screen.getByText("Projector candidates: mmproj:toy-vision")).toBeTruthy()
    expect(screen.getByText("Base model candidates: gguf:toy-model-id")).toBeTruthy()
    expect(screen.getByText("Inventory")).toBeTruthy()
  })

  it("registers an asset path and reloads assets plus legacy GGUF inventory", async () => {
    render(<LlamacppAdminPage />)

    fireEvent.change(await screen.findByLabelText("Register local asset path"), {
      target: { value: "/external/model.gguf" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Register asset" }))

    await waitFor(() => {
      expect(apiMock.registerLlamacppAssetPath).toHaveBeenCalledWith("/external/model.gguf")
      expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(2)
      expect(apiMock.getLlamacppInventory).toHaveBeenCalledTimes(2)
    })
  })

  it("imports an asset folder and reloads assets", async () => {
    render(<LlamacppAdminPage />)

    fireEvent.change(await screen.findByLabelText("Import local asset folder"), {
      target: { value: "/srv/models/imported" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Preview folder" }))

    await waitFor(() => {
      expect(apiMock.previewLlamacppAssetFolder).toHaveBeenCalledWith("/srv/models/imported")
    })

    expect(screen.getByText("Import preview")).toBeTruthy()
    expect(screen.getByText("GGUF: 1")).toBeTruthy()
    expect(screen.getByText("mmproj: 1")).toBeTruthy()
    expect(screen.getByText("Preview skipped unreadable sidecar file.")).toBeTruthy()
    expect(apiMock.importLlamacppAssetFolder).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(apiMock.importLlamacppAssetFolder).toHaveBeenCalledWith("/srv/models/imported")
      expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(2)
    })
  })

  it("queues asset downloads without creating profiles or wiring chat", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Assets")).toBeTruthy()

    fireEvent.change(screen.getByLabelText("Download source URL"), {
      target: { value: "https://example.com/toy.gguf" }
    })
    fireEvent.change(screen.getByLabelText("Download destination directory"), {
      target: { value: "/srv/models" }
    })
    fireEvent.change(screen.getByLabelText("Download filename"), {
      target: { value: "toy.gguf" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Queue download" }))

    await waitFor(() => {
      expect(apiMock.startLlamacppAssetDownload).toHaveBeenCalledWith({
        url: "https://example.com/toy.gguf",
        destination_dir: "/srv/models",
        filename: "toy.gguf"
      })
      expect(apiMock.listLlamacppAssetDownloads).toHaveBeenCalledTimes(2)
    })
    expect(apiMock.createLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.startLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.useLlamacppInChat).not.toHaveBeenCalled()
  })

  it("does not duplicate asset scans for completed downloads during initial load", async () => {
    apiMock.listLlamacppAssetDownloads.mockResolvedValueOnce({
      jobs: [completedDownloadJob]
    })

    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Done model")).toBeTruthy()

    await waitFor(() => {
      expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(1)
    })
    expect(apiMock.createLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.startLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.useLlamacppInChat).not.toHaveBeenCalled()
  })

  it("refreshes assets when a download completes after initial load without profile side effects", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Assets")).toBeTruthy()

    await waitFor(() => {
      expect(apiMock.listLlamacppAssetDownloads).toHaveBeenCalledTimes(1)
    })

    apiMock.getLlamacppAssets.mockClear()
    apiMock.listLlamacppAssetDownloads.mockResolvedValueOnce({
      jobs: [completedDownloadJob]
    })

    fireEvent.click(screen.getByRole("button", { name: /Refresh downloads/ }))

    await waitFor(() => {
      expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(1)
    })
    expect(apiMock.createLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.startLlamacppProfile).not.toHaveBeenCalled()
    expect(apiMock.useLlamacppInChat).not.toHaveBeenCalled()
  })

  it("retries asset refresh for completed downloads after a failed refresh", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Assets")).toBeTruthy()

    await waitFor(() => {
      expect(apiMock.listLlamacppAssetDownloads).toHaveBeenCalledTimes(1)
    })

    apiMock.getLlamacppAssets.mockClear()
    apiMock.getLlamacppAssets
      .mockRejectedValueOnce(new Error("asset refresh failed"))
      .mockResolvedValue(mockAssets)
    apiMock.listLlamacppAssetDownloads.mockResolvedValueOnce({
      jobs: [completedDownloadJob]
    })

    fireEvent.click(screen.getByRole("button", { name: /Refresh downloads/ }))

    expect(await screen.findByText("asset refresh failed")).toBeTruthy()
    expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(1)

    apiMock.listLlamacppAssetDownloads.mockResolvedValueOnce({
      jobs: [completedDownloadJob]
    })

    fireEvent.click(screen.getByRole("button", { name: /Refresh downloads/ }))

    await waitFor(() => {
      expect(apiMock.getLlamacppAssets).toHaveBeenCalledTimes(2)
    })
    await waitFor(() => {
      expect(screen.queryByText("asset refresh failed")).toBeNull()
    })
  })

  it("clears a stale download-list error after a successful downloads refresh", async () => {
    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Assets")).toBeTruthy()

    await waitFor(() => {
      expect(apiMock.listLlamacppAssetDownloads).toHaveBeenCalledTimes(1)
    })

    apiMock.listLlamacppAssetDownloads.mockRejectedValueOnce(
      new Error("download list failed")
    )

    fireEvent.click(screen.getByRole("button", { name: /Refresh downloads/ }))

    expect(await screen.findByText("download list failed")).toBeTruthy()

    apiMock.listLlamacppAssetDownloads.mockResolvedValueOnce({ jobs: [] })

    fireEvent.click(screen.getByRole("button", { name: /Refresh downloads/ }))

    await waitFor(() => {
      expect(apiMock.listLlamacppAssetDownloads).toHaveBeenCalledTimes(3)
    })
    await waitFor(() => {
      expect(screen.queryByText("download list failed")).toBeNull()
    })
  })

  it("shows asset load failure without hiding legacy inventory", async () => {
    apiMock.getLlamacppAssets.mockRejectedValueOnce(new Error("asset scan failed"))

    render(<LlamacppAdminPage />)

    expect(await screen.findByText("Inventory")).toBeTruthy()
    expect(await screen.findByText("asset scan failed")).toBeTruthy()
    expect(screen.getAllByText("Toy 7B Q4_K_M").length).toBeGreaterThan(0)
  })

  it("loads config status inventory assets and hardware only once during strict-mode mount", async () => {
    render(
      <React.StrictMode>
        <LlamacppAdminPage />
      </React.StrictMode>
    )

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
