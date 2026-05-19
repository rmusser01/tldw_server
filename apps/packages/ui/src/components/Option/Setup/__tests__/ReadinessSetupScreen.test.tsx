// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import optionMessages from "@/assets/locale/en/option.json"
import { ReadinessSetupScreen } from "../ReadinessSetupScreen"

const mocks = vi.hoisted(() => ({
  useSetupReadiness: vi.fn(),
  previewSelection: vi.fn(),
  provision: vi.fn(),
  verify: vi.fn(),
  refresh: vi.fn()
}))

vi.mock("../hooks/useSetupReadiness", () => ({
  useSetupReadiness: (...args: unknown[]) => mocks.useSetupReadiness(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string, options?: Record<string, unknown>) => {
      const translations: Record<string, string> = {
        "setupReadiness.errors.load": "Localized setup readiness load failure."
      }
      if (translations[key]) return translations[key]
      const template = typeof defaultValue === "string" ? defaultValue : key
      return template.replace(/{{(\w+)}}/g, (_match, token) => String(options?.[token] ?? ""))
    }
  })
}))

const readinessProfiles = {
  setup_access: {
    mode: "first_run",
    needs_setup: true,
    setup_completed: false,
    remote_access_active: false
  },
  machine_profile: {
    platform: "darwin",
    arch: "arm64",
    apple_silicon: true,
    free_disk_gb: 128
  },
  lane_ids: ["chat", "embeddings_rag", "speech"],
  supported_statuses: ["not_configured", "ready_with_warnings"],
  supported_overlays: ["restart_required"],
  active_overlays: [],
  lanes: [
    {
      lane_id: "chat",
      label: "Chat",
      status: "not_configured",
      consequences: ["Chat defaults can be configured later."]
    },
    {
      lane_id: "embeddings_rag",
      label: "Embeddings/RAG",
      status: "ready_with_warnings",
      selection: { provider: "sentence_transformers", model: "all-MiniLM-L6-v2" }
    },
    {
      lane_id: "speech",
      label: "Speech",
      status: "ready_with_warnings",
      selection: {
        bundle_id: "apple_silicon_local",
        resource_profile: "balanced",
        tts_choice: "kokoro"
      },
      warnings: ["Speech bundle still needs provisioning."]
    }
  ],
  profiles: [
    {
      profile_id: "local_light",
      label: "Local Light",
      description: "Lower disk and memory footprint.",
      lanes: {}
    },
    {
      profile_id: "local_balanced",
      label: "Local Balanced",
      description: "Recommended local-first default.",
      lanes: {}
    },
    {
      profile_id: "local_performance",
      label: "Local Performance",
      description: "Larger local footprint for throughput.",
      lanes: {}
    },
    {
      profile_id: "advanced_custom",
      label: "Advanced Custom",
      description: "Expose exact provider, endpoint, and model controls.",
      lanes: {},
      advanced: true
    }
  ],
  recommended_profile_id: "local_balanced"
}

const baseHookState = {
  error: null,
  errorKey: null,
  fallbackUrl: "/setup",
  guard: null,
  loading: false,
  mode: "first-run",
  preview: null,
  previewSelection: mocks.previewSelection,
  previewing: false,
  profiles: readinessProfiles,
  provision: mocks.provision,
  provisionResult: null,
  provisioning: false,
  refresh: mocks.refresh,
  refreshStatus: vi.fn(),
  status: {
    ...readinessProfiles,
    readiness_status: "not_started",
    operation_status: null
  },
  verification: null,
  verify: mocks.verify,
  verifying: false
}

describe("ReadinessSetupScreen", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.previewSelection.mockResolvedValue({
      preview_id: "preview-1",
      profile_id: "local_balanced",
      lane_ids: ["chat", "embeddings_rag", "speech"],
      lanes: {},
      overlays: [],
      config_updates: {},
      secret_fields: [],
      install_plan: {},
      operation_required: true
    })
    mocks.provision.mockResolvedValue({
      operation_id: "operation-1",
      operation_status: "queued",
      status_url: "/api/v1/setup/readiness/status",
      status: "provisioning"
    })
    mocks.verify.mockResolvedValue({ status: "ready_with_warnings" })
    mocks.useSetupReadiness.mockReturnValue(baseHookState)
  })

  it("renders profile picker, canonical lanes, and secondary TTS copy", async () => {
    render(<ReadinessSetupScreen />)

    expect(
      await screen.findByRole("heading", { level: 1, name: "Setup readiness" })
    ).toBeInTheDocument()
    expect(await screen.findAllByText("Local Balanced")).not.toHaveLength(0)
    expect(screen.getByText("Chat")).toBeInTheDocument()
    expect(screen.getByText("Embeddings/RAG")).toBeInTheDocument()
    expect(screen.getByText("Speech")).toBeInTheDocument()
    expect(screen.getByText(/TTS: kokoro/i)).toHaveClass("secondary")
    expect(
      screen
        .getByText("Review before provisioning")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(
      screen
        .getByText("Skipped-lane consequences")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /Open backend setup/i })).toHaveAttribute(
      "href",
      "/setup"
    )
  })

  it("has English translation keys for setup readiness copy", () => {
    expect(optionMessages.setupReadiness.title).toBe("Setup readiness")
    expect(optionMessages.setupReadiness.actions.provision).toBe("Provision now")
    expect(optionMessages.setupReadiness.provision.description).toContain("status cards")
  })

  it("does not provision until Provision now is clicked", async () => {
    render(<ReadinessSetupScreen />)

    fireEvent.click(await screen.findByText("Local Performance"))

    await waitFor(() =>
      expect(mocks.previewSelection).toHaveBeenCalledWith({
        profile_id: "local_performance"
      })
    )
    expect(mocks.provision).not.toHaveBeenCalled()

    await userEvent.click(screen.getByRole("button", { name: /Provision now/i }))
    expect(mocks.provision).toHaveBeenCalledTimes(1)
  })

  it("shows previewed lanes immediately and treats empty install plans as no work", async () => {
    mocks.useSetupReadiness.mockReturnValue({
      ...baseHookState,
      preview: {
        preview_id: "preview-1",
        profile_id: "local_balanced",
        lane_ids: ["chat", "embeddings_rag", "speech"],
        lanes: {
          chat: {
            lane_id: "chat",
            label: "Chat",
            status: "skipped",
            consequences: ["Chat can be configured later."]
          },
          embeddings_rag: {
            lane_id: "embeddings_rag",
            label: "Embeddings/RAG",
            status: "previewed",
            selection: { provider: "huggingface", model: "Qwen/Qwen3-Embedding-0.6B" }
          },
          speech: {
            lane_id: "speech",
            label: "Speech",
            status: "not_configured"
          }
        },
        overlays: [],
        config_updates: {},
        secret_fields: [],
        install_plan: { stt: [], tts: [], embeddings: { huggingface: [], custom: [], onnx: [] } },
        operation_required: false
      }
    })

    render(<ReadinessSetupScreen />)

    expect(await screen.findByText("previewed")).toBeInTheDocument()
    expect(screen.getByText("Qwen/Qwen3-Embedding-0.6B")).toBeInTheDocument()
    expect(screen.getByText("No downloads needed")).toBeInTheDocument()
  })

  it("shows remote setup guard failures with the backend setup fallback", async () => {
    mocks.useSetupReadiness.mockReturnValue({
      ...baseHookState,
      error: "Setup access is restricted to local requests.",
      guard: "remote_setup_blocked",
      profiles: null,
      status: null
    })

    render(<ReadinessSetupScreen />)

    expect(await screen.findByText("Local setup required")).toBeInTheDocument()
    expect(
      screen.getByText("Local setup required").closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByText("Setup access is restricted to local requests.")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /Open backend setup/i })).toHaveAttribute(
      "href",
      "/setup"
    )
  })

  it("does not display raw provisioning status endpoints to users", async () => {
    mocks.useSetupReadiness.mockReturnValue({
      ...baseHookState,
      provisionResult: {
        operation_id: "operation-1",
        operation_status: "queued",
        status_url: "/api/v1/setup/readiness/status",
        status: "provisioning"
      }
    })

    render(<ReadinessSetupScreen />)

    expect(screen.queryByText("/api/v1/setup/readiness/status")).not.toBeInTheDocument()
    expect(screen.getByText(/watch the status cards/i)).toBeInTheDocument()
    expect(
      screen
        .getByText(/watch the status cards/i)
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })

  it("prefers localized error copy when the readiness hook exposes an error key", async () => {
    mocks.useSetupReadiness.mockReturnValue({
      ...baseHookState,
      error: "Request failed: 500 backend detail",
      errorKey: "setupReadiness.errors.load"
    })

    render(<ReadinessSetupScreen />)

    expect(screen.getByText("Localized setup readiness load failure.")).toBeInTheDocument()
    expect(
      screen
        .getByText("Readiness request failed")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.queryByText("Request failed: 500 backend detail")).not.toBeInTheDocument()
  })

  it("renders the profile-empty state with the design-system EmptyState", async () => {
    mocks.useSetupReadiness.mockReturnValue({
      ...baseHookState,
      profiles: {
        ...readinessProfiles,
        profiles: []
      }
    })

    render(<ReadinessSetupScreen />)

    const emptyTitle = await screen.findByText("No setup readiness profiles are available.")
    expect(emptyTitle.closest('[data-ds-component="EmptyState"]')).toBeInTheDocument()
  })

  it("passes admin mode through to the setup readiness hook", () => {
    render(<ReadinessSetupScreen mode="admin" />)

    expect(mocks.useSetupReadiness).toHaveBeenCalledWith({ mode: "admin" })
  })
})
