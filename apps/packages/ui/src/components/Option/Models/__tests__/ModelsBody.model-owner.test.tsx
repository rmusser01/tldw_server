import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useStoreMessageOption } from "@/store/option"
import { ModelsBody } from "../index"

const models = [
  { id: "old.gguf", model: "tldw:old.gguf", provider: "llama", is_configured: true },
  { id: "ready.gguf", model: "tldw:ready.gguf", provider: "llama", is_configured: true }
]
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string | { defaultValue?: string }) => typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key }) }))
vi.mock("@/hooks/useAntdNotification", () => ({ useAntdNotification: () => ({ error: vi.fn(), success: vi.fn(), info: vi.fn() }) }))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: async () => models }))
vi.mock("@/services/tldw", () => ({ tldwClient: {
  getOpenAIOAuthStatus: async () => ({}), listUserProviderKeys: async () => ({ items: [] })
}, tldwModels: { warmCache: vi.fn() } }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { initialize: async () => undefined, getModelsMetadata: async () => ({ models }) } }))
vi.mock("antd", async original => {
  const actual = await original<typeof import("antd")>()
  return { ...actual, Select: ({ value, options, onChange }: { value?: string; options: Array<{ value: string; label: string; options?: Array<{ value: string; label: string }> }>; onChange: (value: string) => void }) =>
    <select aria-label={options.some(option => option.label === "Auto (from model)") ? "Default provider" : "Default model"} value={value ?? ""} onChange={event => onChange(event.target.value)}>
      <option value="">None</option>{options.flatMap(option => option.options ?? [option]).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
    </select> }
})

const ChatModelOwner = () => { const { selectedModel } = useSelectedModel(); return <output aria-label="Chat selected model">{selectedModel}</output> }
let storage: Storage
const syncStorageValues = new Map<string, unknown>()
const syncStorageListeners = new Set<(
  changes: Record<string, { oldValue?: unknown; newValue?: unknown }>,
  areaName: string
) => void>()

const installSyncStorage = () => {
  vi.stubGlobal("chrome", {
    storage: {
      sync: {
        get: async (keys: string[] | string | null) => {
          const requested = Array.isArray(keys)
            ? keys
            : typeof keys === "string"
              ? [keys]
              : Array.from(syncStorageValues.keys())
          return Object.fromEntries(
            requested.flatMap((key) =>
              syncStorageValues.has(key)
                ? [[key, syncStorageValues.get(key)]]
                : []
            )
          )
        },
        set: async (values: Record<string, unknown>) => {
          const changes: Record<string, { oldValue?: unknown; newValue?: unknown }> = {}
          for (const [key, value] of Object.entries(values)) {
            const oldValue = syncStorageValues.get(key)
            syncStorageValues.set(key, value)
            changes[key] = { oldValue, newValue: value }
          }
          for (const listener of syncStorageListeners) listener(changes, "sync")
        }
      },
      onChanged: {
        addListener: (listener: (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, areaName: string) => void) => {
          syncStorageListeners.add(listener)
        },
        removeListener: (listener: (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, areaName: string) => void) => {
          syncStorageListeners.delete(listener)
        }
      }
    }
  })
}

describe("Settings model selection with a live Chat owner", () => {
  beforeEach(async () => {
    localStorage.clear()
    syncStorageValues.clear()
    syncStorageListeners.clear()
    installSyncStorage()
    useStoreMessageOption.setState({ selectedModel: "tldw:old.gguf" })
    storage = new Storage()
    await storage.set("selectedModel", "tldw:old.gguf")
    await storage.set("defaultApiProvider", "llama")
  })

  afterEach(() => {
    syncStorageValues.clear()
    syncStorageListeners.clear()
    vi.unstubAllGlobals()
  })

  it("commits a deliberate Settings selection to the live owner and survives remount", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={queryClient}><MemoryRouter><ModelsBody /><ChatModelOwner /></MemoryRouter></QueryClientProvider>)
    await screen.findByRole("option", { name: /ready.gguf/ })
    fireEvent.change(screen.getByRole("combobox", { name: "Default model" }), { target: { value: "tldw:ready.gguf" } })
    await waitFor(() => expect(screen.getByLabelText("Chat selected model")).toHaveTextContent("tldw:ready.gguf"))
    expect(screen.getByRole("combobox", { name: "Default model" })).toHaveValue("tldw:ready.gguf")
    expect(useStoreMessageOption.getState().selectedModel).toBe("tldw:ready.gguf")
    await waitFor(async () => expect(await storage.get("selectedModel")).toBe("tldw:ready.gguf"))
    view.unmount()
    await act(async () => { render(<ChatModelOwner />) })
    expect(screen.getByLabelText("Chat selected model")).toHaveTextContent("tldw:ready.gguf")
  })
})
