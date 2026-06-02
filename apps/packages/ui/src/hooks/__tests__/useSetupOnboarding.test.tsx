// @vitest-environment jsdom
import { cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { FirstRunState } from "@/types/setup-onboarding"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
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
    })
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
  })

  afterEach(() => {
    cleanup()
  })

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
})
