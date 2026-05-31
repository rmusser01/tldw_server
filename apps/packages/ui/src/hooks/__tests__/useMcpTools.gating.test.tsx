import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useMcpTools } from "@/hooks/useMcpTools"
import { useMcpToolsStore } from "@/store/mcp-tools"

const state = vi.hoisted(() => ({
  capabilities: {
    hasMcp: true
  } as any,
  loading: false,
  connectionConfig: {
    serverUrl: "http://127.0.0.1:8000",
    authMode: "single-user",
    apiKey: "test-key"
  } as any,
  settingValues: new Map<string, unknown>(),
  settingSetters: new Map<string, ReturnType<typeof vi.fn>>(),
  apiSend: vi.fn(),
  fetchMcpTools: vi.fn(),
  fetchMcpToolCatalogs: vi.fn(),
  fetchMcpToolCatalogsViaDiscovery: vi.fn(),
  fetchMcpModulesViaDiscovery: vi.fn(),
  fetchMcpToolsViaDiscovery: vi.fn()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: state.capabilities,
    loading: state.loading
  })
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: state.connectionConfig,
    loading: false
  })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { key: string; defaultValue: unknown }) => {
    const setter =
      state.settingSetters.get(setting.key) ??
      vi.fn(async (next: unknown) => {
        const current = state.settingValues.has(setting.key)
          ? state.settingValues.get(setting.key)
          : setting.defaultValue
        const resolved =
          typeof next === "function"
            ? (next as (value: unknown) => unknown)(current)
            : next
        state.settingValues.set(setting.key, resolved)
      })
    state.settingSetters.set(setting.key, setter)
    return [
      state.settingValues.has(setting.key)
        ? state.settingValues.get(setting.key)
        : setting.defaultValue,
      setter
    ]
  }
}))

vi.mock("@/services/api-send", () => ({
  apiSend: (...args: unknown[]) =>
    (state.apiSend as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tldw/mcp", () => ({
  fetchMcpTools: (...args: unknown[]) =>
    (state.fetchMcpTools as (...args: unknown[]) => unknown)(...args),
  fetchMcpToolCatalogs: (...args: unknown[]) =>
    (state.fetchMcpToolCatalogs as (...args: unknown[]) => unknown)(...args),
  fetchMcpToolCatalogsViaDiscovery: (...args: unknown[]) =>
    (state.fetchMcpToolCatalogsViaDiscovery as (...args: unknown[]) => unknown)(
      ...args
    ),
  fetchMcpModulesViaDiscovery: (...args: unknown[]) =>
    (state.fetchMcpModulesViaDiscovery as (...args: unknown[]) => unknown)(...args),
  fetchMcpToolsViaDiscovery: (...args: unknown[]) =>
    (state.fetchMcpToolsViaDiscovery as (...args: unknown[]) => unknown)(...args)
}))

const buildWrapper = () => {
  const { wrapper } = buildWrapperWithClient()
  return wrapper
}

const buildWrapperWithClient = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return { queryClient, wrapper }
}

describe("useMcpTools gating", () => {
  beforeEach(() => {
    state.capabilities = { hasMcp: true }
    state.loading = false
    state.connectionConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    state.settingValues.clear()
    state.settingSetters.clear()
    state.apiSend.mockReset()
    state.fetchMcpTools.mockReset()
    state.fetchMcpToolCatalogs.mockReset()
    state.fetchMcpToolCatalogsViaDiscovery.mockReset()
    state.fetchMcpModulesViaDiscovery.mockReset()
    state.fetchMcpToolsViaDiscovery.mockReset()
    useMcpToolsStore.setState({
      tools: [],
      discoveredTools: [],
      availableTools: [],
      chatTools: [],
      healthState: "unknown",
      toolsLoading: false,
      disabledToolPreferences: { version: 1, scopes: {} },
      activeToolPreferenceScope: "default",
      disabledToolNames: [],
      collisionToolNames: [],
      toolCounts: {
        discovered: 0,
        executable: 0,
        disabled: 0,
        colliding: 0,
        chatEnabled: 0
      },
      toolCatalog: "",
      toolCatalogId: null,
      toolModules: [],
      toolCatalogStrict: false
    })
  })

  it("does not query MCP endpoints until the tools surface is enabled", async () => {
    const { result } = renderHook(() => useMcpTools({ enabled: false }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.healthLoading).toBe(false)
      expect(result.current.toolsLoading).toBe(false)
      expect(result.current.catalogsLoading).toBe(false)
    })

    expect(result.current.healthState).toBe("unknown")
    expect(result.current.tools).toEqual([])
    expect(state.apiSend).not.toHaveBeenCalled()
    expect(state.fetchMcpTools).not.toHaveBeenCalled()
    expect(state.fetchMcpToolCatalogs).not.toHaveBeenCalled()
    expect(state.fetchMcpToolCatalogsViaDiscovery).not.toHaveBeenCalled()
    expect(state.fetchMcpModulesViaDiscovery).not.toHaveBeenCalled()
    expect(state.fetchMcpToolsViaDiscovery).not.toHaveBeenCalled()
  })

  it("keeps discovered tools visible while filtering executable chat tools", async () => {
    state.apiSend.mockResolvedValue({ ok: true })
    state.fetchMcpToolCatalogsViaDiscovery.mockResolvedValue([])
    state.fetchMcpModulesViaDiscovery.mockResolvedValue([])
    state.fetchMcpToolsViaDiscovery.mockResolvedValue([
      { name: "notes.search", canExecute: true },
      { name: "media.search", canExecute: false },
      { name: "slides.list", canExecute: true }
    ])

    const { result } = renderHook(() => useMcpTools(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.discoveredTools).toHaveLength(3)
    })

    expect(result.current.discoveredTools.map((tool) => tool.rawName)).toEqual([
      "notes.search",
      "media.search",
      "slides.list"
    ])
    expect(result.current.availableTools.map((tool) => tool.rawName)).toEqual([
      "notes.search",
      "slides.list"
    ])
    expect(result.current.chatTools.map((tool) => tool.rawName)).toEqual([
      "notes.search",
      "slides.list"
    ])
    expect(result.current.toolCounts).toMatchObject({
      discovered: 3,
      executable: 2,
      disabled: 0,
      colliding: 0,
      chatEnabled: 2
    })
    expect(useMcpToolsStore.getState().tools.map((tool) => tool.name)).toEqual([
      "notes.search",
      "slides.list"
    ])
    expect(useMcpToolsStore.getState().chatTools.map((tool) => tool.rawName)).toEqual([
      "notes.search",
      "slides.list"
    ])
  })

  it("persists disabled tool names in the active connection scope", async () => {
    state.apiSend.mockResolvedValue({ ok: true })
    state.fetchMcpToolCatalogsViaDiscovery.mockResolvedValue([])
    state.fetchMcpModulesViaDiscovery.mockResolvedValue([])
    state.fetchMcpToolsViaDiscovery.mockResolvedValue([
      { name: "notes.search", canExecute: true },
      { name: "slides.list", canExecute: true }
    ])

    const { result } = renderHook(() => useMcpTools(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.chatTools).toHaveLength(2)
    })

    result.current.setToolEnabled("slides_list", false)

    await waitFor(() => {
      expect(result.current.chatTools.map((tool) => tool.rawName)).toEqual([
        "notes.search"
      ])
    })

    const persisted = state.settingValues.get("tldw:mcp:disabledTools:v1") as any
    expect(result.current.activeToolPreferenceScope).toBe(
      "server:http://127.0.0.1:8000|auth:single-user|org:none|principal:anonymous"
    )
    expect(
      persisted.scopes[result.current.activeToolPreferenceScope].disabledToolNames
    ).toEqual(["slides_list"])
  })

  it("isolates disabled preferences between server scopes", async () => {
    state.settingValues.set("tldw:mcp:disabledTools:v1", {
      version: 1,
      scopes: {
        "server:http://127.0.0.1:8000|auth:single-user|org:none|principal:anonymous": {
          disabledToolNames: ["notes_search"]
        }
      }
    })
    state.connectionConfig = {
      serverUrl: "http://127.0.0.1:9000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    state.apiSend.mockResolvedValue({ ok: true })
    state.fetchMcpToolCatalogsViaDiscovery.mockResolvedValue([])
    state.fetchMcpModulesViaDiscovery.mockResolvedValue([])
    state.fetchMcpToolsViaDiscovery.mockResolvedValue([
      { name: "notes.search", canExecute: true }
    ])

    const { result } = renderHook(() => useMcpTools(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.chatTools).toHaveLength(1)
    })

    expect(result.current.activeToolPreferenceScope).toBe(
      "server:http://127.0.0.1:9000|auth:single-user|org:none|principal:anonymous"
    )
    expect(result.current.disabledToolNames).toEqual([])
    expect(result.current.chatTools[0]?.rawName).toBe("notes.search")
  })

  it("scopes MCP query caches by the active connection identity", async () => {
    const firstScope =
      "server:http://127.0.0.1:8000|auth:single-user|org:none|principal:anonymous"
    const secondScope =
      "server:http://127.0.0.1:9000|auth:single-user|org:none|principal:anonymous"
    state.apiSend.mockResolvedValue({ ok: true })
    state.fetchMcpToolCatalogsViaDiscovery.mockResolvedValue([])
    state.fetchMcpModulesViaDiscovery.mockResolvedValue([])
    state.fetchMcpToolsViaDiscovery.mockImplementation(async () =>
      state.connectionConfig.serverUrl === "http://127.0.0.1:9000"
        ? [{ name: "slides.list", canExecute: true }]
        : [{ name: "notes.search", canExecute: true }]
    )
    const { queryClient, wrapper } = buildWrapperWithClient()

    const { result, rerender } = renderHook(() => useMcpTools(), {
      wrapper
    })

    await waitFor(() => {
      expect(result.current.chatTools.map((tool) => tool.rawName)).toEqual([
        "notes.search"
      ])
    })

    state.connectionConfig = {
      serverUrl: "http://127.0.0.1:9000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    rerender()

    await waitFor(() => {
      expect(result.current.chatTools.map((tool) => tool.rawName)).toEqual([
        "slides.list"
      ])
    })

    expect(state.fetchMcpToolsViaDiscovery).toHaveBeenCalledTimes(2)
    const queryKeys = queryClient
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey)
    expect(queryKeys).toEqual(
      expect.arrayContaining([
        ["mcp-health", firstScope],
        ["mcp-health", secondScope],
        ["mcp-tools", firstScope, "", null, [], false],
        ["mcp-tools", secondScope, "", null, [], false],
        ["mcp-tool-catalogs", firstScope],
        ["mcp-tool-catalogs", secondScope],
        ["mcp-tool-modules", firstScope],
        ["mcp-tool-modules", secondScope]
      ])
    )
  })

  it("defaults newly discovered executable tools to enabled in an existing scope", async () => {
    state.settingValues.set("tldw:mcp:disabledTools:v1", {
      version: 1,
      scopes: {
        "server:http://127.0.0.1:8000|auth:single-user|org:none|principal:anonymous": {
          disabledToolNames: ["notes_search"]
        }
      }
    })
    state.apiSend.mockResolvedValue({ ok: true })
    state.fetchMcpToolCatalogsViaDiscovery.mockResolvedValue([])
    state.fetchMcpModulesViaDiscovery.mockResolvedValue([])
    state.fetchMcpToolsViaDiscovery.mockResolvedValue([
      { name: "notes.search", canExecute: true },
      { name: "slides.list", canExecute: true }
    ])

    const { result } = renderHook(() => useMcpTools(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.chatTools.map((tool) => tool.rawName)).toEqual([
        "slides.list"
      ])
    })

    expect(result.current.disabledToolNames).toEqual(["notes_search"])
  })
})
