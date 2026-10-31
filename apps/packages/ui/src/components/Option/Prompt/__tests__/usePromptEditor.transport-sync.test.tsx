import React from "react"
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import i18next from "i18next"
import { I18nextProvider, initReactI18next } from "react-i18next"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { Prompt } from "@/db/dexie/types"
import { getAllPrompts } from "@/db/dexie/helpers"
import { createSafeStorage } from "@/utils/safe-storage"
import { clearRuntimeAuthOverride } from "@/services/tldw/runtime-auth-override"
import { setPromptStudioDefaults } from "@/services/prompt-studio-settings"
import { usePromptEditor } from "../hooks/usePromptEditor"
import { usePromptSync } from "../hooks/usePromptSync"
import { SyncStatusBadge } from "../SyncStatusBadge"

// Keep the production local-save methods and query readers; substitute only
// the IndexedDB table unavailable in jsdom, with durable rows across remounts.
const rows = vi.hoisted(() => new Map<string, Prompt>())
vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      get: async (id: string) => structuredClone(rows.get(id)),
      put: async (row: Prompt) => { rows.set(row.id, structuredClone(row)); return row.id },
      update: async (id: string, fields: Partial<Prompt>) => {
        const row = rows.get(id)
        if (!row) return 0
        rows.set(id, structuredClone({ ...row, ...fields }))
        return 1
      },
      filter: (predicate: (row: Prompt) => boolean) => ({
        reverse: () => ({
          sortBy: async () => structuredClone([...rows.values()].filter(predicate).reverse())
        })
      })
    }
  }
}))

const original: Prompt = {
  id: "linked-prompt", title: "Recovery prompt", name: "Recovery prompt",
  content: "Original instructions", system_prompt: "Original instructions",
  user_prompt: "", is_system: true, createdAt: 1, updatedAt: 1,
  promptFormat: "legacy", promptSchemaVersion: null,
  serverId: 1, studioPromptId: 1, studioProjectId: 1,
  versionNumber: 1, lastSyncedAt: 10, syncStatus: "synced"
}
const changedText = "Keep this local revision after the server disconnects."
const i18n = i18next.createInstance()
void i18n.use(initReactI18next).init({ lng: "en", resources: {}, initImmediate: false })
const t = (key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue || key).replace("{{error}}", String(options?.error || ""))

function SaveHarness({ queryClient }: { queryClient: QueryClient }) {
  const query = useQuery({ queryKey: ["fetchAllPrompts"], queryFn: getAllPrompts })
  const sync = usePromptSync({ queryClient, isOnline: true, t })
  const record = query.data?.[0]
  const editor = usePromptEditor({
    queryClient, isOnline: true, t, guardPrivateMode: () => false,
    getPromptTexts: (prompt: Prompt) => ({
      systemText: prompt.system_prompt || undefined,
      userText: prompt.user_prompt || undefined
    }),
    getPromptKeywords: (prompt: Prompt) => prompt.keywords || [],
    getPromptRecordById: (id: string) => query.data?.find(prompt => prompt.id === id),
    confirmDanger: async () => true,
    syncPromptAfterLocalSave: sync.syncPromptAfterLocalSave,
    recipePersistenceAvailable: false
  })
  return <>
    {record && <section aria-label="Saved prompt">
      <span>{record.system_prompt}</span>
      <SyncStatusBadge syncStatus={record.syncStatus} serverId={record.serverId} />
      <button onClick={() => editor.openFullEditor(record)}>Edit prompt</button>
    </section>}
    {editor.fullEditorOpen && <button onClick={() => void editor.handleFullEditorSubmit({
      name: "Recovery prompt", system_prompt: changedText,
      user_prompt: "", versionNumber: record?.versionNumber
    })}>Save prompt</button>}
  </>
}

function mountEditor() {
  const queryClient = new QueryClient({ defaultOptions: {
    queries: { retry: false }, mutations: { retry: false }
  } })
  const rendered = render(<ConfigProvider theme={{ token: { motion: false } }}>
    <App><I18nextProvider i18n={i18n}><MemoryRouter><QueryClientProvider client={queryClient}>
      <SaveHarness queryClient={queryClient} />
    </QueryClientProvider></MemoryRouter></I18nextProvider></App>
  </ConfigProvider>)
  return { ...rendered, queryClient }
}

function createStorageArea(initial: Record<string, unknown> = {}) {
  const values = structuredClone(initial)
  return {
    get: async (
      keys: string | string[] | null = null,
      callback?: (result: Record<string, unknown>) => void
    ) => {
      const requested = keys === null ? Object.keys(values) : typeof keys === "string" ? [keys] : keys
      const result = structuredClone(Object.fromEntries(
        requested.filter(key => key in values).map(key => [key, values[key]])
      ))
      callback?.(result)
      return result
    },
    set: async (items: Record<string, unknown>) => { Object.assign(values, structuredClone(items)) }
  }
}

beforeEach(async () => {
  rows.clear()
  rows.set(original.id, structuredClone(original))
  localStorage.clear()
  sessionStorage.clear()
  clearRuntimeAuthOverride()
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "")
  vi.stubEnv("NEXT_PUBLIC_TLDW_API_KEY", "")
  vi.stubEnv("VITE_TLDW_API_KEY", "")
  // Native Plasmo reads credentials from local and settings from sync; the
  // Firefox prompt mirror uses the callback API on the same local area.
  vi.stubGlobal("chrome", { storage: {
    local: createStorageArea({ prompts: [original] }),
    sync: createStorageArea(),
    session: createStorageArea()
  } })
  await createSafeStorage({ area: "local" }).set("tldwConfig", {
    serverUrl: "https://prompt.test", authMode: "single-user", apiKey: "test-prompt-key",
    credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: "https://prompt.test"
  })
  await setPromptStudioDefaults({ defaultProjectId: 1, autoSyncWorkspacePrompts: true })
})

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); clearRuntimeAuthOverride() })

it("persists and reloads Pending after an actual normalized fetch failure, then adopts only the acknowledged version", async () => {
  const fetcher = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))
  vi.stubGlobal("fetch", fetcher)
  const first = mountEditor()
  fireEvent.click(await screen.findByRole("button", { name: "Edit prompt" }))
  fireEvent.click(screen.getByRole("button", { name: "Save prompt" }))
  await screen.findByText("Failed to fetch Your changes are saved locally.")
  await waitFor(() => expect(screen.getByRole("region", { name: "Saved prompt" })).toHaveTextContent(changedText))
  expect(fetcher).toHaveBeenCalledTimes(1)
  expect(fetcher).toHaveBeenLastCalledWith("https://prompt.test/api/v1/prompt-studio/prompts/update/1", expect.objectContaining({ method: "PUT" }))
  expect(rows.get(original.id)).toMatchObject({
    system_prompt: changedText, syncStatus: "pending", serverId: 1,
    studioPromptId: 1, versionNumber: 1, lastSyncedAt: 10
  })
  expect(screen.getByRole("region", { name: "Saved prompt" })).toHaveTextContent("Pending")
  expect(screen.getByRole("region", { name: "Saved prompt" })).not.toHaveTextContent("Synced")
  first.unmount()
  first.queryClient.clear()

  const reloaded = mountEditor()
  await screen.findByText("Pending")
  expect(fetcher).toHaveBeenCalledTimes(1)
  let acknowledge!: (response: Response) => void
  fetcher.mockImplementationOnce(() => new Promise<Response>(resolve => { acknowledge = resolve }))
  fireEvent.click(screen.getByRole("button", { name: "Edit prompt" }))
  fireEvent.click(screen.getByRole("button", { name: "Save prompt" }))
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2))
  expect(rows.get(original.id)).toMatchObject({ syncStatus: "pending", serverId: 1 })
  acknowledge(new Response(JSON.stringify({ success: true, data: {
    id: 2, project_id: 1, version_number: 2, parent_version_id: 1,
    name: "Recovery prompt", system_prompt: changedText, user_prompt: "",
    prompt_format: "legacy", prompt_schema_version: null, prompt_definition: null,
    updated_at: "2026-09-16T05:29:02"
  }, error: null }), { status: 200, headers: { "Content-Type": "application/json" } }))
  await waitFor(() => expect(screen.getByRole("region", { name: "Saved prompt" })).toHaveTextContent("Synced#2"))
  expect(rows.get(original.id)).toMatchObject({
    system_prompt: changedText, syncStatus: "synced", serverId: 2,
    studioPromptId: 2, versionNumber: 2, serverParentVersionId: 1
  })
  expect(fetcher).toHaveBeenLastCalledWith("https://prompt.test/api/v1/prompt-studio/prompts/update/1", expect.objectContaining({ method: "PUT" }))
  expect(fetcher).toHaveBeenCalledTimes(2)
  reloaded.queryClient.clear()
})
