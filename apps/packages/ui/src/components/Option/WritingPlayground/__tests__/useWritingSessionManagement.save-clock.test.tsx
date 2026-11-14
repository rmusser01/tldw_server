import React from "react"
import type { JSONContent } from "@tiptap/react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, expect, it, vi } from "vitest"

import {
  useWritingSessionManagement,
  type UseWritingSessionManagementDeps
} from "../hooks/useWritingSessionManagement"
import {
  updateWritingSession,
  type WritingSessionResponse
} from "@/services/writing-playground"

vi.mock("@/services/writing-playground", () => ({
  cloneWritingSession: vi.fn(),
  createWritingSession: vi.fn(),
  deleteWritingSession: vi.fn(),
  getWritingSession: vi.fn(),
  listWritingSessions: vi.fn(),
  updateWritingSession: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it("records the save completion time only after the mutation succeeds", async () => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date("2026-09-10T12:00:00Z"))
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  const deps: UseWritingSessionManagementDeps = {
    isOnline: false,
    hasWriting: false,
    activeSessionId: "session-1",
    activeSessionName: "Draft",
    setActiveSessionId: vi.fn(),
    setActiveSessionName: vi.fn(),
    sessionUsageMap: {},
    setSessionUsageMap: vi.fn(),
    selectedModel: undefined,
    setSelectedModel: vi.fn(),
    apiProviderOverride: undefined,
    setApiProvider: vi.fn(),
    isGenerating: false,
    t: ((key: string) => key) as UseWritingSessionManagementDeps["t"]
  }
  let finishSave!: (value: WritingSessionResponse) => void
  vi.mocked(updateWritingSession).mockImplementation(
    () =>
      new Promise((resolve) => {
        finishSave = resolve
      })
  )
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const { result, unmount } = renderHook(
    () => useWritingSessionManagement(deps),
    { wrapper }
  )
  expect(result.current.lastSavedAt).toBeNull()

  let mutation!: Promise<WritingSessionResponse>
  await act(async () => {
    mutation = result.current.saveSessionMutation.mutateAsync({
      sessionId: "session-1",
      payload: { prompt: "Hello" },
      expectedVersion: 1
    })
    await Promise.resolve()
  })
  expect(result.current.lastSavedAt).toBeNull()
  vi.setSystemTime(new Date("2026-09-10T12:00:05Z"))
  await act(async () => {
    finishSave({
      id: "session-1",
      name: "Draft",
      payload: { prompt: "Hello" },
      schema_version: 1,
      version: 2,
      created_at: "2026-09-10T12:00:00Z",
      last_modified: "2026-09-10T12:00:05Z",
      deleted: false,
      client_id: "test"
    })
    await mutation
  })
  expect(result.current.lastSavedAt).toBe(Date.parse("2026-09-10T12:00:05Z"))
  unmount()
  client.clear()
})

it.each<[string, string, JSONContent | null]>([
  ["plain text", "Revised paragraph.", null],
  ["cleared text", "", null],
  ["rich text", "Revised paragraph.", { type: "doc", content: [{ type: "paragraph", content: [{ type: "text", text: "Revised paragraph." }] }] }]
])("keeps pending %s when revision metadata is saved in the same event", async (_label, prompt, promptRich) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const session: WritingSessionResponse = {
    id: "session-1", name: "Draft", payload: { prompt: "Original paragraph." },
    schema_version: 1, version: 1, version_parent_id: null,
    created_at: "2026-10-18T12:00:00Z", last_modified: "2026-10-18T12:00:00Z",
    deleted: false, client_id: "test"
  }
  client.setQueryData(["writing-session", "session-1"], session)
  client.setQueryData(["writing-sessions"], { sessions: [session], total: 1 })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  vi.mocked(updateWritingSession).mockImplementation(async (_id, patch) => ({ ...session, payload: patch.payload ?? {}, schema_version: patch.schema_version ?? 1, version: 2 }))
  const { result, unmount } = renderHook(() => useWritingSessionManagement({
    isOnline: false, hasWriting: false, activeSessionId: "session-1", activeSessionName: "Draft",
    setActiveSessionId: vi.fn(), setActiveSessionName: vi.fn(), sessionUsageMap: {}, setSessionUsageMap: vi.fn(),
    selectedModel: undefined, setSelectedModel: vi.fn(), apiProviderOverride: undefined, setApiProvider: vi.fn(),
    isGenerating: false, t: ((key: string) => key) as UseWritingSessionManagementDeps["t"]
  }), { wrapper })
  try {
    vi.useFakeTimers()
    act(() => {
      result.current.applyPromptValue(prompt, { promptRich })
      result.current.applySessionPayloadPatch(payload => ({ ...payload, revision_preset_id: "make_concise" }))
    })
    await act(async () => { await vi.advanceTimersByTimeAsync(800) })
    const patch = vi.mocked(updateWritingSession).mock.calls.at(-1)?.[1]
    expect(patch?.payload).toMatchObject({ prompt, revision_preset_id: "make_concise" })
    expect(patch?.payload?.prompt_rich ?? null).toEqual(promptRich)
  } finally {
    unmount()
    client.clear()
    vi.useRealTimers()
  }
})
