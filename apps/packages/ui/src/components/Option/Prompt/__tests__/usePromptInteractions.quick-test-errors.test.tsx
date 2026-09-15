import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { notification } from "antd"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

const storage = vi.hoisted(() => new Map<string, unknown>())
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: { serializer: JSON.stringify, deserializer: (value: unknown) => value },
  createSafeStorage: () => ({
    get: async (key: string) => storage.get(key) ?? null,
    set: async (key: string, value: unknown) => { storage.set(key, value) },
    remove: async (key: string) => { storage.delete(key) }
  })
}))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({ setSelectedQuickPrompt: vi.fn(), setSelectedSystemPrompt: vi.fn() })
}))
vi.mock("@/services/application", () => ({
  getAllCopilotPrompts: async () => [],
  upsertCopilotPrompts: vi.fn()
}))
vi.mock("@/services/prompt-studio", () => ({
  hasPromptStudio: async () => false,
  getPrompt: vi.fn(),
  getLlmProviders: async () => ({ providers: [] })
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { usePromptInteractions } from "../hooks/usePromptInteractions"

beforeEach(() => {
  storage.clear()
  storage.set("tldwConfig", {
    serverUrl: "http://127.0.0.1:19999",
    authMode: "single-user",
    credentialSource: "manual",
    apiKeyPersistence: "device",
    apiKeyServerOrigin: "http://127.0.0.1:19999",
    apiKey: "synthetic-test-key"
  })
  // Startup discovery is unrelated to this local Quick Test. Completion,
  // proxy, request parsing and notification ownership remain real.
  vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
  vi.spyOn(console, "warn").mockImplementation(() => undefined)
})
afterEach(() => vi.unstubAllGlobals())

const mountQuickTest = async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  )
  const hook = renderHook(() => usePromptInteractions({
    queryClient,
    isOnline: true,
    t: (key, opts) => String(opts?.defaultValue ?? key),
    getPromptTexts: () => ({ systemText: "Be concise.", userText: "Explain this example." }),
    getPromptKeywords: () => [],
    getPromptRecordById: () => null,
    getPromptModifiedAt: () => 0,
    getPromptUsageCount: () => 0,
    getPromptLastUsedAt: () => null,
    editorMarkPromptAsUsed: async () => undefined
  }), { wrapper })
  await act(async () => { await hook.result.current.handleQuickTest({ id: "local-prompt", name: "Example" }) })
  return { ...hook, queryClient }
}

it("shows an actionable Quick Test notification without raw failed-transport paths", async () => {
  const notify = vi.spyOn(notification, "error").mockImplementation(() => undefined)
  const fetchMock = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({
    detail: "Provider unavailable at /Users/private/provider.conf. Choose another model."
  }), { status: 500, headers: { "content-type": "application/json" } }))
  vi.stubGlobal("fetch", fetchMock)
  const { result, unmount, queryClient } = await mountQuickTest()
  try {
    await act(async () => { await result.current.runLocalQuickTest() })
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({
      message: "Quick test failed",
      description: expect.stringContaining("Choose another model")
    }))
    expect(notify.mock.calls[0]?.[0].description).not.toContain("/Users/private")
    expect(result.current.localQuickTestOutput).toBeNull()
    expect(result.current.isRunningLocalQuickTest).toBe(false)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  } finally {
    unmount()
    queryClient.clear()
  }
})

it("retains successful Quick Test output that explains errors and filesystem paths", async () => {
  const notify = vi.spyOn(notification, "error").mockImplementation(() => undefined)
  const content = "Error example:\ncat /Users/private/example.txt\nAn exception is expected here."
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
    choices: [{ message: { content } }]
  }), { status: 200, headers: { "content-type": "application/json" } })))
  const { result, unmount, queryClient } = await mountQuickTest()
  try {
    await act(async () => { await result.current.runLocalQuickTest() })
    expect(result.current.localQuickTestOutput).toBe(content)
    expect(notify).not.toHaveBeenCalled()
  } finally {
    unmount()
    queryClient.clear()
  }
})
