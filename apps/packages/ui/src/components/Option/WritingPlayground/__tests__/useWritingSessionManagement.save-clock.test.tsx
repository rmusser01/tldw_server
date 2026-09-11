import React from "react"
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
