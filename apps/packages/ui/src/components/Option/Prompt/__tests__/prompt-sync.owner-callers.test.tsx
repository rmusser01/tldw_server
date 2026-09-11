import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, expect, it, vi } from "vitest"

import { usePromptBulkActions } from "../hooks/usePromptBulkActions"
import { usePromptSync } from "../hooks/usePromptSync"

const mocks = vi.hoisted(() => ({
  owner: vi.fn(),
  auto: vi.fn(),
  push: vi.fn(),
  conflict: vi.fn(),
  all: vi.fn()
}))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: mocks.owner
}))
vi.mock("@/services/prompt-sync", () => ({
  autoSyncPrompt: mocks.auto,
  pushToStudio: mocks.push,
  resolveConflict: mocks.conflict,
  shouldAutoSyncWorkspacePrompts: async () => true,
  getAllPromptsWithSyncStatus: mocks.all,
  pullFromStudio: vi.fn(),
  unlinkPrompt: vi.fn(),
  getConflictInfo: vi.fn()
}))
vi.mock("@/db/dexie/helpers", () => ({
  deletePromptById: vi.fn(),
  updatePrompt: vi.fn(),
  restorePrompt: vi.fn(),
  exportPrompts: vi.fn()
}))
vi.mock("antd", () => ({
  notification: {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }
}))
const ownerA = "recipe-owner:sha256:" + "a".repeat(64)
const ownerB = "recipe-owner:sha256:" + "b".repeat(64)
beforeEach(() => {
  vi.resetAllMocks()
  mocks.owner.mockResolvedValue({
    ownerId: ownerA,
    authorizationRevision: "unused-revision"
  })
  mocks.auto.mockImplementation(async (id) => ({
    success: true,
    localId: id,
    syncStatus: "synced"
  }))
  mocks.push.mockImplementation(async (id) => ({
    success: true,
    localId: id,
    syncStatus: "synced"
  }))
})
function setup() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  const deps = { queryClient, isOnline: true, t: (key: string) => key }
  return { queryClient, wrapper, deps }
}
it("captures current owner for user save and manual push without exposing authorization revision", async () => {
  const { wrapper, deps } = setup()
  const { result } = renderHook(() => usePromptSync(deps), { wrapper })
  await act(() => result.current.syncPromptAfterLocalSave("recipe-1"))
  expect(mocks.auto).toHaveBeenCalledWith("recipe-1", undefined, {
    expectedOwnerId: ownerA
  })
  mocks.owner.mockResolvedValue({
    ownerId: ownerB,
    authorizationRevision: "unused"
  })
  act(() =>
    result.current.pushToStudioMutation({ localId: "recipe-2", projectId: 42 })
  )
  await waitFor(() =>
    expect(mocks.push).toHaveBeenCalledWith("recipe-2", 42, {
      expectedOwnerId: ownerB
    })
  )
})
it("does not require owner resolution for v1 user saves", async () => {
  mocks.owner.mockResolvedValue(null)
  const { wrapper, deps } = setup()
  const { result } = renderHook(() => usePromptSync(deps), { wrapper })
  await act(() => result.current.syncPromptAfterLocalSave("v1"))
  expect(mocks.auto).toHaveBeenCalledWith("v1", undefined, undefined)
})
it("freezes one owner for an entire outbox batch despite a connection change between items", async () => {
  const { wrapper, deps } = setup()
  const { result } = renderHook(() => usePromptSync(deps), { wrapper })
  mocks.auto.mockImplementation(async (id) => {
    mocks.owner.mockResolvedValue({
      ownerId: ownerB,
      authorizationRevision: "new"
    })
    return {
      success: false,
      localId: id,
      syncStatus: "error",
      failureKind: "invalid_server_payload"
    }
  })
  await act(() =>
    result.current.runBatchSync([
      { promptId: "one", direction: "push", preferredProjectId: 42 },
      { promptId: "two", direction: "push", preferredProjectId: 42 }
    ])
  )
  expect(mocks.owner).toHaveBeenCalledTimes(1)
  expect(mocks.auto.mock.calls).toEqual([
    ["one", 42, { expectedOwnerId: ownerA }],
    ["two", 42, { expectedOwnerId: ownerA }]
  ])
  expect(result.current.batchSyncState.failed.map((item) => item.task)).toEqual(
    [
      { promptId: "one", direction: "push", preferredProjectId: 42 },
      { promptId: "two", direction: "push", preferredProjectId: 42 }
    ]
  )
})
it("freezes one opaque owner for parallel bulk pushes", async () => {
  const { wrapper, deps } = setup()
  const { result } = renderHook(
    () =>
      usePromptBulkActions({
        ...deps,
        data: [],
        isFireFoxPrivateMode: false,
        guardPrivateMode: () => false,
        getPromptKeywords: () => [],
        buildPromptUpdatePayload: (p) => p,
        confirmDanger: async () => true
      }),
    { wrapper }
  )
  act(() => result.current.bulkPushToServer(["one", "two"]))
  await waitFor(() => expect(mocks.auto).toHaveBeenCalledTimes(2))
  expect(mocks.owner).toHaveBeenCalledTimes(1)
  expect(mocks.auto.mock.calls).toEqual([
    ["one", undefined, { expectedOwnerId: ownerA }],
    ["two", undefined, { expectedOwnerId: ownerA }]
  ])
})
