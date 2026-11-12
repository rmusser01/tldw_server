// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import type { FirstRunMetadata, FirstRunState } from "@/types/setup-onboarding"

const io = vi.hoisted(() => ({
  config: null as TldwConfig | null,
  getConfig: vi.fn(),
  initialize: vi.fn(),
  state: vi.fn(),
  metadata: vi.fn(),
  saveProvider: vi.fn(),
  saveIngest: vi.fn(),
  saveAudio: vi.fn(),
  applyMcp: vi.fn(),
  validateMcp: vi.fn(),
  saveAdvanced: vi.fn(),
  verifyChat: vi.fn(),
  complete: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: io.getConfig,
    initialize: io.initialize,
    getFirstRunState: io.state,
    getFirstRunMetadata: io.metadata,
    saveSetupProvider: io.saveProvider,
    saveIngestDefaults: io.saveIngest,
    saveAudioDefaults: io.saveAudio,
    applyMcpTools: io.applyMcp,
    validateMcpTools: io.validateMcp,
    saveOptionalAdvanced: io.saveAdvanced,
    verifyFirstRunChat: io.verifyChat,
    completeFirstRun: io.complete
  }
}))

const backendState: FirstRunState = {
  status: "not_started",
  completed_steps: [],
  skipped_steps: [],
  step_data: {},
  acknowledged_steps: [],
  first_chat: { completed: false }
}
const backendMetadata = {
  auth_mode: "single_user",
  bundled_single_user_auth_available: false,
  manual_auth_required: true,
  setup_required: true,
  setup_completed: false,
  remote_setup_enabled: false,
  connection: { browser_access: "remote" },
  setup_paths: [],
  multi_user_exit: { guide_path: "/docs/multi-user" }
} as FirstRunMetadata
const connection = (host: string): TldwConfig =>
  ({
    serverUrl: `https://${host}.test`,
    authMode: "single-user",
    apiKey: `declared-${host}`
  }) as TldwConfig
const surface = (protocol: string) =>
  vi.spyOn(window, "location", "get").mockReturnValue({
    protocol,
    origin:
      protocol === "https:"
        ? "https://webui.test"
        : "chrome-extension://declared",
    pathname: "/options.html"
  } as Location)
const notifyConnection = () =>
  window.dispatchEvent(
    new CustomEvent("tldw:config-updated", {
      detail: { authorityChanged: true }
    })
  )

// Only configuration and backend IO are declared. Transport selection and account
// notifications remain production code; no fake readiness or saved setup is supplied.
describe("setup initial loading follows configured connection authority", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "self_host")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    io.config = null
    io.getConfig.mockImplementation(async () => io.config)
    io.initialize.mockResolvedValue(undefined)
    io.state.mockResolvedValue(backendState)
    io.metadata.mockResolvedValue(backendMetadata)
    surface("chrome-extension:")
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
  })

  it("does not request or report backend errors before an extension has a server", async () => {
    io.state.mockRejectedValue(new Error("No configured server"))
    io.metadata.mockRejectedValue(new Error("No configured server"))
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(mounted.result.current.loading).toBe(false))
    expect(io.state).not.toHaveBeenCalled()
    expect(io.metadata).not.toHaveBeenCalled()
    expect(mounted.result.current.state).toBeNull()
    expect(mounted.result.current.metadata).toBeNull()
    expect(mounted.result.current.error).toBeNull()
    await act(async () => {
      expect(await mounted.result.current.refresh()).toBeNull()
    })
    expect(io.state).not.toHaveBeenCalled()
    expect(io.metadata).not.toHaveBeenCalled()
  })

  it("loads real setup state after connection is saved while mounted", async () => {
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(mounted.result.current.loading).toBe(false))
    expect(io.state).not.toHaveBeenCalled()
    await act(async () => {
      io.config = connection("first")
      notifyConnection()
    })
    await waitFor(() =>
      expect(mounted.result.current.metadata).toEqual(backendMetadata)
    )
    expect(io.state).toHaveBeenCalledOnce()
    expect(io.metadata).toHaveBeenCalledOnce()
    expect(mounted.result.current.state).toEqual(backendState)
    expect(mounted.result.current.error).toBeNull()
  })

  it("keeps a supplied but unreachable server error retryable", async () => {
    io.config = connection("unreachable")
    io.state.mockRejectedValue(new Error("Declared backend unavailable"))
    io.metadata.mockRejectedValue(new Error("Declared backend unavailable"))
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(mounted.result.current.loading).toBe(false))
    expect(mounted.result.current.error?.message).toBe(
      "Declared backend unavailable"
    )
    io.state.mockResolvedValue(backendState)
    io.metadata.mockResolvedValue(backendMetadata)
    await act(async () => {
      await mounted.result.current.refresh()
    })
    expect(mounted.result.current.state).toEqual(backendState)
    expect(mounted.result.current.error).toBeNull()
  })

  it("discards a held old-server state and its cache after connection replacement", async () => {
    io.config = connection("first")
    const oldState = {
      ...backendState,
      status: "completed",
      first_chat: { completed: true }
    } as FirstRunState
    let release!: (state: FirstRunState) => void
    const held = new Promise<FirstRunState>((resolve) => {
      release = resolve
    })
    io.state.mockImplementationOnce(() => held)
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(io.state).toHaveBeenCalledOnce())
    await act(async () => {
      io.config = connection("second")
      notifyConnection()
    })
    await waitFor(() => expect(io.state).toHaveBeenCalledTimes(2))
    await waitFor(() =>
      expect(mounted.result.current.metadata).toEqual(backendMetadata)
    )
    await act(async () => {
      release(oldState)
      await held
    })
    expect(mounted.result.current.state).toEqual(backendState)
    mounted.unmount()
    const reopened = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(reopened.result.current.loading).toBe(false))
    expect(reopened.result.current.state).toEqual(backendState)
    expect(reopened.result.current.error).toBeNull()
  })

  it("does not reuse cached setup for a different server on remount", async () => {
    io.config = connection("first")
    const oldState = {
      ...backendState,
      status: "completed",
      first_chat: { completed: true }
    } as FirstRunState
    io.state.mockResolvedValueOnce(oldState)
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const first = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(first.result.current.loading).toBe(false))
    expect(first.result.current.state).toEqual(oldState)
    first.unmount()
    io.config = connection("second")
    const second = renderHook(() => useSetupOnboarding())
    await waitFor(() => expect(second.result.current.loading).toBe(false))
    expect(second.result.current.state).toEqual(backendState)
    expect(io.state).toHaveBeenCalledTimes(2)
  })

  it("keeps a held mutation current during a same-authority provider config event", async () => {
    io.config = connection("first")
    let release!: (value: unknown) => void
    io.saveProvider.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve
        })
    )
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(() => useSetupOnboarding())
    await waitFor(() =>
      expect(mounted.result.current.metadata).toEqual(backendMetadata)
    )
    const response = { saved: true }
    const pending = mounted.result.current.saveProvider({} as never)
    await waitFor(() => expect(io.saveProvider).toHaveBeenCalledOnce())
    await act(async () => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: false }
        })
      )
    })
    await waitFor(() =>
      expect(mounted.result.current.connectionChecking).toBe(false)
    )
    await act(async () => {
      release(response)
      expect(await pending).toBe(response)
    })
    expect(mounted.result.current.state).toEqual(backendState)
  })

  it.each(["credential replacement", "account ABA"] as const)(
    "retires a held mutation after %s",
    async (change) => {
      io.config = connection("first")
      let release!: (value: unknown) => void
      io.saveProvider.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            release = resolve
          })
      )
      const { useSetupOnboarding } = await import("../useSetupOnboarding")
      const mounted = renderHook(() => useSetupOnboarding())
      await waitFor(() =>
        expect(mounted.result.current.metadata).toEqual(backendMetadata)
      )
      const pending = mounted.result.current
        .saveProvider({} as never)
        .catch((error) => error)
      await waitFor(() => expect(io.saveProvider).toHaveBeenCalledOnce())
      await act(async () => {
        io.config =
          change === "credential replacement"
            ? { ...connection("first"), apiKey: "declared-replacement" }
            : connection("second")
        notifyConnection()
        if (change === "account ABA") {
          io.config = connection("first")
          notifyConnection()
        }
      })
      await waitFor(() =>
        expect(mounted.result.current.metadata).toEqual(backendMetadata)
      )
      await act(async () => {
        release({ saved: true })
        expect(await pending).toBeInstanceOf(Error)
      })
      expect(mounted.result.current.state).toEqual(backendState)
    }
  )

  it("keeps newly supplied metadata when the parent finishes loading and disables autoLoad", async () => {
    io.config = connection("first")
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(
      (props: {
        initialState: FirstRunState
        initialMetadata: FirstRunMetadata | null
        autoLoad: boolean
      }) => useSetupOnboarding(props),
      {
        initialProps: {
          initialState: backendState,
          initialMetadata: null,
          autoLoad: true
        }
      }
    )
    await waitFor(() =>
      expect(mounted.result.current.metadata).toEqual(backendMetadata)
    )
    mounted.rerender({
      initialState: backendState,
      initialMetadata: backendMetadata,
      autoLoad: false
    })
    await waitFor(() =>
      expect(mounted.result.current.connectionChecking).toBe(false)
    )
    expect(mounted.result.current.metadata).toEqual(backendMetadata)
    expect(mounted.result.current.state).toEqual(backendState)
  })

  it("retires supplied initial setup data when its configured connection changes", async () => {
    io.config = connection("first")
    const oldState = {
      ...backendState,
      status: "completed",
      first_chat: { completed: true }
    } as FirstRunState
    const oldMetadata = { ...backendMetadata, setup_completed: true }
    const { useSetupOnboarding } = await import("../useSetupOnboarding")
    const mounted = renderHook(() =>
      useSetupOnboarding({
        autoLoad: false,
        initialState: oldState,
        initialMetadata: oldMetadata
      })
    )
    await waitFor(() =>
      expect(mounted.result.current.connectionChecking).toBe(false)
    )
    expect(mounted.result.current.state).toEqual(oldState)
    expect(io.state).not.toHaveBeenCalled()
    await act(async () => {
      io.config = connection("second")
      notifyConnection()
    })
    await waitFor(() =>
      expect(mounted.result.current.metadata).toEqual(backendMetadata)
    )
    expect(mounted.result.current.state).toEqual(backendState)
  })

  it.each([
    ["saveProvider", "saveProvider"],
    ["saveIngestDefaults", "saveIngest"],
    ["saveAudioDefaults", "saveAudio"],
    ["applyMcpTools", "applyMcp"],
    ["validateMcpTools", "validateMcp"],
    ["saveOptionalAdvanced", "saveAdvanced"],
    ["verifyFirstChat", "verifyChat"],
    ["complete", "complete"]
  ] as const)(
    "refuses the old %s result after replacement during its post-save refresh",
    async (method, ioMethod) => {
      io.config = connection("first")
      io[ioMethod].mockResolvedValueOnce({ saved: "old-server" })
      const { useSetupOnboarding } = await import("../useSetupOnboarding")
      const mounted = renderHook(() => useSetupOnboarding())
      await waitFor(() =>
        expect(mounted.result.current.metadata).toEqual(backendMetadata)
      )
      let release!: (value: FirstRunState) => void
      const held = new Promise<FirstRunState>((resolve) => {
        release = resolve
      })
      io.state.mockImplementationOnce(() => held)
      const pending = mounted.result.current[method]({} as never).catch(
        (error) => error
      )
      await waitFor(() => expect(io.state).toHaveBeenCalledTimes(2))
      await act(async () => {
        io.config = connection("second")
        notifyConnection()
      })
      await waitFor(() => expect(io.state).toHaveBeenCalledTimes(3))
      await waitFor(() =>
        expect(mounted.result.current.metadata).toEqual(backendMetadata)
      )
      await act(async () => {
        release({ ...backendState, status: "completed" } as FirstRunState)
        expect(await pending).toBeInstanceOf(Error)
      })
      expect(mounted.result.current.state).toEqual(backendState)
    }
  )

  it.each([
    "quickstart",
    "runtime cookie",
    "hosted",
    "advanced environment"
  ] as const)(
    "retains valid %s transport without a stored manual API key",
    async (mode) => {
      vi.restoreAllMocks()
      surface("https:")
      if (mode === "quickstart" || mode === "runtime cookie")
        vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      if (mode === "runtime cookie")
        io.config = {
          serverUrl: "https://webui.test",
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: ""
        } as TldwConfig
      if (mode === "hosted")
        vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
      if (mode === "advanced environment")
        vi.stubEnv("NEXT_PUBLIC_API_URL", "https://env-api.test")
      const { useSetupOnboarding } = await import("../useSetupOnboarding")
      const mounted = renderHook(() => useSetupOnboarding())
      await waitFor(() =>
        expect(mounted.result.current.metadata).toEqual(backendMetadata)
      )
      expect(io.state).toHaveBeenCalledOnce()
      expect(mounted.result.current.error).toBeNull()
    }
  )
})
