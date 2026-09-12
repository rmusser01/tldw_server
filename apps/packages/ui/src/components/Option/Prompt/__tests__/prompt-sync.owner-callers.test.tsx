import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { notification } from "antd"
import React from "react"
import { beforeEach, expect, it, vi } from "vitest"

import { usePromptBulkActions } from "../hooks/usePromptBulkActions"
import { usePromptSync } from "../hooks/usePromptSync"

const mocks = vi.hoisted(() => ({
  owner: vi.fn(),
  auto: vi.fn(),
  push: vi.fn(),
  conflict: vi.fn(),
  pull: vi.fn(),
  unlink: vi.fn(),
  info: vi.fn(),
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
  pullFromStudio: mocks.pull,
  unlinkPrompt: mocks.unlink,
  getConflictInfo: mocks.info
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
  mocks.pull.mockImplementation(async (_serverId, localId) => ({
    success: true,
    localId: localId || "pulled",
    syncStatus: "synced"
  }))
  mocks.unlink.mockImplementation(async (id) => ({
    success: true,
    localId: id,
    syncStatus: "local"
  }))
  mocks.info.mockResolvedValue({
    localPrompt: { id: "recipe" },
    serverPrompt: { id: 101 },
    localUpdatedAt: 1,
    serverUpdatedAt: "2026-09-11T00:00:00Z"
  })
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
it("keeps the project selector open and reports rejection instead of push success", async () => {
  const { wrapper, deps } = setup()
  const { result } = renderHook(() => usePromptSync(deps), { wrapper })
  mocks.push.mockResolvedValue({
    success: false,
    localId: "recipe",
    syncStatus: "error",
    error: "Unresolved recipe operation"
  })
  act(() => {
    result.current.setProjectSelectorOpen(true)
    result.current.setPromptToSync("recipe")
  })
  act(() =>
    result.current.pushToStudioMutation({ localId: "recipe", projectId: 42 })
  )
  await waitFor(() =>
    expect(notification.error).toHaveBeenCalledWith(
      expect.objectContaining({ description: "Unresolved recipe operation" })
    )
  )
  expect(result.current.projectSelectorOpen).toBe(true)
  expect(result.current.promptToSync).toBe("recipe")
  expect(notification.success).not.toHaveBeenCalled()
})
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

it.each([
  ["pull", "pullFromStudioMutation"],
  ["unlink", "unlinkPromptMutation"]
] as const)(
  "reports a false %s result as an error instead of success",
  async (operation, mutationName) => {
    const { wrapper, deps, queryClient } = setup()
    const invalidate = vi.spyOn(queryClient, "invalidateQueries")
    const failure = {
      success: false,
      localId: "recipe",
      serverId: 101,
      syncStatus: "error",
      error: "Recipe has an unresolved operation"
    }
    mocks[operation].mockResolvedValue(failure)
    const { result } = renderHook(() => usePromptSync(deps), { wrapper })

    act(() => {
      if (mutationName === "pullFromStudioMutation")
        result.current.pullFromStudioMutation({
          serverId: 101,
          localId: "recipe"
        })
      else result.current.unlinkPromptMutation("recipe")
    })

    await waitFor(() =>
      expect(notification.error).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "Recipe has an unresolved operation"
        })
      )
    )
    expect(notification.success).not.toHaveBeenCalled()
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["fetchAllPrompts"]
    })
  }
)

it("keeps failed keep-server recovery visible and refreshes its copied error row", async () => {
  const { wrapper, deps, queryClient } = setup()
  const invalidate = vi.spyOn(queryClient, "invalidateQueries")
  mocks.conflict.mockResolvedValue({
    success: false,
    localId: "recipe",
    serverId: 101,
    syncStatus: "error",
    error: "Recipe has an unresolved operation"
  })
  const { result } = renderHook(() => usePromptSync(deps), { wrapper })
  act(() => result.current.openConflictResolution("recipe"))
  await waitFor(() => expect(result.current.conflictPromptId).toBe("recipe"))
  act(() => result.current.handleResolveConflict("keep_server"))

  await waitFor(() =>
    expect(notification.error).toHaveBeenCalledWith(
      expect.objectContaining({
        description: "Recipe has an unresolved operation"
      })
    )
  )
  expect(result.current.conflictModalOpen).toBe(true)
  expect(notification.success).not.toHaveBeenCalled()
  expect(invalidate).toHaveBeenCalledWith({
    queryKey: ["fetchAllPrompts"]
  })
})

it("reports a false Prompt Studio import as an error and refreshes on settlement", async () => {
  const { wrapper, deps, queryClient } = setup()
  const invalidate = vi.spyOn(queryClient, "invalidateQueries")
  mocks.pull.mockResolvedValue({
    success: false,
    localId: "recipe",
    serverId: 101,
    syncStatus: "error",
    error: "Recipe has an unresolved operation"
  })
  const { result } = renderHook(() => usePromptSync(deps), { wrapper })

  act(() => result.current.importFromStudioMutation({ serverId: 101 }))

  await waitFor(() =>
    expect(notification.error).toHaveBeenCalledWith(
      expect.objectContaining({
        description: "Recipe has an unresolved operation"
      })
    )
  )
  expect(notification.success).not.toHaveBeenCalled()
  expect(invalidate).toHaveBeenCalledWith({
    queryKey: ["fetchAllPrompts"]
  })
})
