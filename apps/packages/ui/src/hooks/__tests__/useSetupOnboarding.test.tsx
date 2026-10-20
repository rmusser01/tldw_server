// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { FirstRunState } from "@/types/setup-onboarding"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn().mockResolvedValue({ serverUrl: "https://declared-setup.test", authMode: "single-user", apiKey: "declared-setup-key" }),
    initialize: vi.fn().mockResolvedValue(undefined),
    getFirstRunState: vi.fn().mockResolvedValue({
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    }),
    getFirstRunMetadata: vi.fn().mockResolvedValue({
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    }),
    saveSetupProvider: vi.fn(),
    updateFirstRunState: vi.fn(),
    getMcpToolsCatalog: vi.fn(),
    applyMcpTools: vi.fn(),
    validateMcpTools: vi.fn()
  }
}))

describe("useSetupOnboarding", () => {
  beforeEach(async () => {
    vi.resetModules()
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    vi.mocked(tldwClient.getFirstRunState)
      .mockReset()
      .mockResolvedValue({
        status: "not_started",
        completed_steps: [],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: [],
        first_chat: { completed: false }
      })
    vi.mocked(tldwClient.getFirstRunMetadata)
      .mockReset()
      .mockResolvedValue({
        auth_mode: "single_user",
        bundled_single_user_auth_available: true,
        manual_auth_required: false,
        setup_required: true,
        setup_completed: false,
        remote_setup_enabled: false,
        connection: { browser_access: "local" },
        setup_paths: [],
        multi_user_exit: { guide_path: "/docs/multi-user" }
      })
    vi.mocked(tldwClient.saveSetupProvider).mockReset()
    vi.mocked(tldwClient.updateFirstRunState).mockReset()
    vi.mocked(tldwClient.getMcpToolsCatalog).mockReset()
    vi.mocked(tldwClient.applyMcpTools).mockReset()
    vi.mocked(tldwClient.validateMcpTools).mockReset()
  })

  afterEach(() => {
    cleanup()
  })


  it("persists a saved default provider draft without acknowledging or copying secrets", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const state: FirstRunState = {
      status: "in_progress",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    }
    const data = {
      acknowledged: false,
      default_provider: "llamacpp",
      default_model: "real-model",
      default_base_url: "http://127.0.0.1:8010/v1",
      default_provider_credential_configured: true
    }
    vi.mocked(tldwClient.saveSetupProvider).mockResolvedValue({
      provider_key: "llamacpp",
      status: "saved",
      model: "real-model",
      base_url: data.default_base_url,
      make_default: true,
      credential_configured: true
    })
    vi.mocked(tldwClient.updateFirstRunState).mockResolvedValue({
      ...state,
      step_data: { providers: data }
    })
    const { result } = renderHook(() =>
      useSetupOnboarding({ initialState: state, autoLoad: false })
    )
    await act(async () => {
      await result.current.saveProvider({
        provider_key: "llamacpp",
        model: "real-model",
        base_url: data.default_base_url,
        make_default: true,
        api_key: "synthetic-private-key"
      })
    })
    expect(tldwClient.updateFirstRunState).toHaveBeenCalledWith({
      step: "providers",
      data
    })
    expect(result.current.state?.step_data.providers).toEqual(data)
    expect(result.current.state?.completed_steps).not.toContain("providers")
  })

  it("surfaces a provider draft write failure instead of claiming saved progress", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const state: FirstRunState = {
      status: "in_progress",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    }
    vi.mocked(tldwClient.saveSetupProvider).mockResolvedValue({
      provider_key: "openai",
      status: "saved",
      model: "working",
      make_default: true,
      credential_configured: true
    })
    vi.mocked(tldwClient.updateFirstRunState).mockRejectedValue(
      new Error("progress unavailable")
    )
    const { result } = renderHook(() =>
      useSetupOnboarding({ initialState: state, autoLoad: false })
    )
    await act(async () => {
      await expect(
        result.current.saveProvider({
          provider_key: "openai",
          model: "working",
          make_default: true
        })
      ).rejects.toThrow("progress unavailable")
    })
  })

  it.each([
    {
      status: "failed" as const,
      makeDefault: true,
      stateStatus: "in_progress" as const
    },
    {
      status: "saved" as const,
      makeDefault: false,
      stateStatus: "in_progress" as const
    },
    {
      status: "saved" as const,
      makeDefault: true,
      stateStatus: "blocked" as const
    },
    {
      status: "saved" as const,
      makeDefault: true,
      stateStatus: "skipped" as const
    },
    {
      status: "saved" as const,
      makeDefault: true,
      stateStatus: "first_chat_complete" as const
    },
    {
      status: "saved" as const,
      makeDefault: true,
      stateStatus: "completed" as const
    }
  ])(
    "does not rewrite provider progress for %j",
    async ({ status, makeDefault, stateStatus }) => {
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      const { useSetupOnboarding } = await import("../useSetupOnboarding")
      const state: FirstRunState = {
        status: stateStatus,
        completed_steps: [],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: [],
        first_chat: { completed: false }
      }
      vi.mocked(tldwClient.saveSetupProvider).mockResolvedValue({
        provider_key: "openai",
        status
      })
      const { result } = renderHook(() =>
        useSetupOnboarding({ initialState: state, autoLoad: false })
      )
      await act(async () => {
        await result.current.saveProvider({
          provider_key: "openai",
          model: "working",
          make_default: makeDefault
        })
      })
      expect(tldwClient.updateFirstRunState).not.toHaveBeenCalled()
    }
  )

  it.each([
    "http://user:synthetic-secret@127.0.0.1:8010/v1",
    "http://127.0.0.1:8010/v1?api_key=synthetic-secret",
    "http://127.0.0.1:8010/v1#synthetic-secret",
    "not-a-url"
  ])(
    "keeps unsafe endpoint URLs out of public provider progress (%s)",
    async (baseUrl) => {
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      const { useSetupOnboarding } = await import("../useSetupOnboarding")
      const state: FirstRunState = {
        status: "in_progress",
        completed_steps: [],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: [],
        first_chat: { completed: false }
      }
      vi.mocked(tldwClient.saveSetupProvider).mockResolvedValue({
        provider_key: "llamacpp",
        status: "saved",
        model: "working",
        base_url: baseUrl,
        make_default: true
      })
      const { result } = renderHook(() =>
        useSetupOnboarding({ initialState: state, autoLoad: false })
      )
      await act(async () => {
        await expect(
          result.current.saveProvider({
            provider_key: "llamacpp",
            model: "working",
            base_url: baseUrl
          })
        ).rejects.toThrow(/saved.*plain.*URL/i)
      })
      expect(tldwClient.updateFirstRunState).not.toHaveBeenCalled()
    }
  )

  it("loads backend first-run state", async () => {
    const { useSetupOnboarding } = await import("../useSetupOnboarding")

    const { result } = renderHook(() => useSetupOnboarding())

    await waitFor(() =>
      expect(result.current.state?.status).toBe("not_started")
    )
  })

  it("does not restart initial loading while metadata is still in flight", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const metadata = {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    }
    let resolveMetadata: (value: typeof metadata) => void = () => undefined
    vi.mocked(tldwClient.getFirstRunMetadata).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveMetadata = resolve
        })
    )
    const { useSetupOnboarding } = await import("../useSetupOnboarding")

    const { result } = renderHook(() => useSetupOnboarding())

    await waitFor(() =>
      expect(result.current.state?.status).toBe("not_started")
    )
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(tldwClient.getFirstRunState).toHaveBeenCalledTimes(1)
    expect(tldwClient.getFirstRunMetadata).toHaveBeenCalledTimes(1)

    resolveMetadata(metadata)

    await waitFor(() => expect(result.current.metadata).toEqual(metadata))
  })

  it("continues the initial setup load across a remount", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const state: FirstRunState = {
      status: "completed",
      completed_steps: ["first_chat"],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: ["first_chat"],
      first_chat: { completed: true }
    }
    const metadata = {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: false,
      setup_completed: true,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    }
    let resolveMetadata: (value: typeof metadata) => void = () => undefined
    vi.mocked(tldwClient.getFirstRunState).mockResolvedValue(state)
    vi.mocked(tldwClient.getFirstRunMetadata).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveMetadata = resolve
        })
    )
    const { useSetupOnboarding } = await import("../useSetupOnboarding")

    const first = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(first.result.current.state).toEqual(state))
    first.unmount()

    const second = renderHook(() => useSetupOnboarding())
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(tldwClient.getFirstRunState).toHaveBeenCalledTimes(1)
    expect(tldwClient.getFirstRunMetadata).toHaveBeenCalledTimes(1)

    resolveMetadata(metadata)

    await waitFor(() =>
      expect(second.result.current.metadata).toEqual(metadata)
    )
    expect(second.result.current.state).toEqual(state)
  })

  it("keeps completed backend state when metadata loading fails", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    vi.mocked(tldwClient.getFirstRunState).mockResolvedValue({
      status: "completed",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: true }
    })
    vi.mocked(tldwClient.getFirstRunMetadata).mockRejectedValue(
      new Error("metadata failed")
    )
    const { useSetupOnboarding } = await import("../useSetupOnboarding")

    const { result } = renderHook(() => useSetupOnboarding())

    await waitFor(() => expect(result.current.state?.status).toBe("completed"))
    expect(result.current.error?.message).toBe("metadata failed")
  })

  it("loads first-run MCP tools catalog into state", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const catalog: Awaited<ReturnType<typeof tldwClient.getMcpToolsCatalog>> = {
      catalog_version: "2026-07-04",
      confirmation_version: "v1",
      packs: [
        {
          pack_id: "research",
          label: "Research",
          purpose: "Search saved knowledge",
          default_selected: true,
          available: true,
          legacy: false,
          module_targets: ["mcp_discovery"],
          tool_patterns: ["mcp.tools.list"],
          available_tools: [{ tool_name: "mcp.tools.list", available: true }],
          unavailable_tools: [],
          add_on_ids: [],
          sample_validation_candidates: ["mcp.tools.list"],
          catalog_version: "2026-07-04"
        }
      ],
      add_ons: [],
      validation_states: ["not_run"]
    }
    vi.mocked(tldwClient.getMcpToolsCatalog).mockResolvedValue(catalog)
    const { useSetupOnboarding } = await import("../useSetupOnboarding")

    const { result } = renderHook(() =>
      useSetupOnboarding({ autoLoad: false })
    )

    await act(async () => {
      await result.current.loadMcpToolsCatalog()
    })

    expect(result.current.mcpToolsCatalog).toEqual(catalog)
  })

  it("refreshes first-run state after applying MCP tools", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const response: Awaited<ReturnType<typeof tldwClient.applyMcpTools>> = {
      status: "applied",
      profile_id: 7,
      assignment_id: 9,
      catalog_version: "2026-07-04",
      selected_pack_ids: ["research"],
      selected_addon_ids: [],
      effective_tool_count: 1,
      effective_tools: ["mcp.tools.list"],
      disabled_addons: [],
      validation_state: "not_run",
      conflict: null
    }
    vi.mocked(tldwClient.applyMcpTools).mockResolvedValue(response)
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const { result } = renderHook(() =>
      useSetupOnboarding({ autoLoad: false })
    )

    let returned: typeof response | undefined
    await act(async () => {
      returned = await result.current.applyMcpTools({
        selected_pack_ids: ["research"]
      })
    })

    expect(returned).toBe(response)
    expect(tldwClient.getFirstRunState).toHaveBeenCalledTimes(1)
    expect(tldwClient.getFirstRunMetadata).toHaveBeenCalledTimes(1)
  })

  it("refreshes first-run state after validating MCP tools", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const response: Awaited<ReturnType<typeof tldwClient.validateMcpTools>> = {
      status: "validated",
      validation_state: "built_in_passed",
      profile_id: 7,
      assignment_id: 9,
      catalog_version: "2026-07-04",
      selected_pack_ids: ["research"],
      selected_addon_ids: [],
      effective_tool_count: 1
    }
    vi.mocked(tldwClient.validateMcpTools).mockResolvedValue(response)
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const { result } = renderHook(() =>
      useSetupOnboarding({ autoLoad: false })
    )

    let returned: typeof response | undefined
    await act(async () => {
      returned = await result.current.validateMcpTools()
    })

    expect(returned).toBe(response)
    expect(tldwClient.getFirstRunState).toHaveBeenCalledTimes(1)
    expect(tldwClient.getFirstRunMetadata).toHaveBeenCalledTimes(1)
  })

  it("bubbles first-run MCP tools method errors", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    vi.mocked(tldwClient.getMcpToolsCatalog).mockRejectedValue(
      new Error("catalog failed")
    )
    vi.mocked(tldwClient.applyMcpTools).mockRejectedValue(
      new Error("apply failed")
    )
    vi.mocked(tldwClient.validateMcpTools).mockRejectedValue(
      new Error("validate failed")
    )
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const { result } = renderHook(() =>
      useSetupOnboarding({ autoLoad: false })
    )

    await expect(result.current.loadMcpToolsCatalog()).rejects.toThrow(
      "catalog failed"
    )
    await expect(
      result.current.applyMcpTools({ selected_pack_ids: ["research"] })
    ).rejects.toThrow("apply failed")
    await expect(result.current.validateMcpTools()).rejects.toThrow(
      "validate failed"
    )
  })
})
