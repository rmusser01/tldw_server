// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

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

    expect(await screen.findAllByText("Local Balanced")).not.toHaveLength(0)
    expect(screen.getByText("Chat")).toBeInTheDocument()
    expect(screen.getByText("Embeddings/RAG")).toBeInTheDocument()
    expect(screen.getByText("Speech")).toBeInTheDocument()
    expect(screen.getByText(/TTS: kokoro/i)).toHaveClass("secondary")
    expect(screen.getByRole("link", { name: /Open backend setup/i })).toHaveAttribute(
      "href",
      "/setup"
    )
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
    expect(screen.getByText("Setup access is restricted to local requests.")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /Open backend setup/i })).toHaveAttribute(
      "href",
      "/setup"
    )
  })

  it("passes admin mode through to the setup readiness hook", () => {
    render(<ReadinessSetupScreen mode="admin" />)

    expect(mocks.useSetupReadiness).toHaveBeenCalledWith({ mode: "admin" })
  })
})
