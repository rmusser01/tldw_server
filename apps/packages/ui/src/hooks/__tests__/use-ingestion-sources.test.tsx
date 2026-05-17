import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  ingestionSourceKeys,
  useCreateIngestionSourceMutation,
  useIngestionSourceDirectoryBrowseQuery,
  useIngestionSourceDetailQuery,
  useIngestionSourceItemsQuery,
  useIngestionSourcesQuery,
  useReattachIngestionSourceItemMutation,
  useSyncIngestionSourceMutation,
  useUpdateIngestionSourceMutation,
  useUploadIngestionSourceArchiveMutation
} from "@/hooks/use-ingestion-sources"

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

const createMockClient = () => ({
  listIngestionSources: vi.fn(),
  getIngestionSource: vi.fn(),
  listIngestionSourceItems: vi.fn(),
  browseIngestionSourceDirectories: vi.fn(),
  createIngestionSource: vi.fn(),
  updateIngestionSource: vi.fn(),
  syncIngestionSource: vi.fn(),
  uploadIngestionSourceArchive: vi.fn(),
  reattachIngestionSourceItem: vi.fn()
})

describe("use-ingestion-sources", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("loads ingestion source list, detail, and items queries", async () => {
    const client = createMockClient()
    client.listIngestionSources.mockResolvedValueOnce({
      sources: [{ id: "12", source_type: "archive_snapshot" }],
      total: 1
    })
    client.getIngestionSource.mockResolvedValueOnce({
      id: "12",
      source_type: "archive_snapshot"
    })
    client.listIngestionSourceItems.mockResolvedValueOnce({
      items: [{ id: "41", source_id: "12", normalized_relative_path: "docs/a.md" }],
      total: 1
    })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const wrapper = buildWrapper(queryClient)

    const listQuery = renderHook(() => useIngestionSourcesQuery(client as any), { wrapper })
    const detailQuery = renderHook(() => useIngestionSourceDetailQuery("12", client as any), {
      wrapper
    })
    const itemsQuery = renderHook(
      () =>
        useIngestionSourceItemsQuery(
          "12",
          { sync_status: "conflict_detached" },
          client as any
        ),
      { wrapper }
    )

    await waitFor(() => {
      expect(listQuery.result.current.isSuccess).toBe(true)
      expect(detailQuery.result.current.isSuccess).toBe(true)
      expect(itemsQuery.result.current.isSuccess).toBe(true)
    })

    expect(client.listIngestionSources).toHaveBeenCalledTimes(1)
    expect(client.getIngestionSource).toHaveBeenCalledWith("12")
    expect(client.listIngestionSourceItems).toHaveBeenCalledWith("12", {
      sync_status: "conflict_detached"
    })
  })

  it("loads directory browse results by server path", async () => {
    const client = createMockClient()
    client.browseIngestionSourceDirectories.mockResolvedValueOnce({
      roots: [{ name: "imports", path: "/srv/tldw/imports", is_root: true }],
      current_path: "/srv/tldw/imports",
      parent_path: null,
      entries: [{ name: "notes", path: "/srv/tldw/imports/notes", is_root: false }],
      error: null
    })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const wrapper = buildWrapper(queryClient)

    const browseQuery = renderHook(
      () => useIngestionSourceDirectoryBrowseQuery("/srv/tldw/imports", client as any),
      { wrapper }
    )

    await waitFor(() => {
      expect(browseQuery.result.current.isSuccess).toBe(true)
    })

    expect(client.browseIngestionSourceDirectories).toHaveBeenCalledWith("/srv/tldw/imports")
    expect(browseQuery.result.current.data?.entries[0]?.path).toBe("/srv/tldw/imports/notes")
  })

  it("invalidates the list query after creating a source", async () => {
    const client = createMockClient()
    client.createIngestionSource.mockResolvedValueOnce({ id: "12", source_type: "local_directory" })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    const { result } = renderHook(() => useCreateIngestionSourceMutation(client as any), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        source_type: "local_directory",
        sink_type: "notes"
      })
    })

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ingestionSourceKeys.list()
      })
    })
  })

  it("invalidates list and detail queries after updating a source", async () => {
    const client = createMockClient()
    client.updateIngestionSource.mockResolvedValueOnce({ id: "12", enabled: false })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    const { result } = renderHook(() => useUpdateIngestionSourceMutation("12", client as any), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        enabled: false
      })
    })

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ingestionSourceKeys.list()
      })
    })
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ingestionSourceKeys.detail("12")
    })
  })

  it("invalidates list and detail queries after syncing a source", async () => {
    const client = createMockClient()
    client.syncIngestionSource.mockResolvedValueOnce({ status: "queued", source_id: "12" })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    const { result } = renderHook(() => useSyncIngestionSourceMutation(client as any), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync("12")
    })

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ingestionSourceKeys.list()
      })
    })
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ingestionSourceKeys.detail("12")
    })
  })

  it("invalidates list, detail, and item queries after uploading an archive", async () => {
    const client = createMockClient()
    client.uploadIngestionSourceArchive.mockResolvedValueOnce({
      status: "queued",
      source_id: "12",
      snapshot_status: "staged"
    })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    const file = new File(["{}"], "notes.zip", { type: "application/zip" })

    const { result } = renderHook(
      () => useUploadIngestionSourceArchiveMutation("12", client as any),
      { wrapper: buildWrapper(queryClient) }
    )

    await act(async () => {
      await result.current.mutateAsync(file)
    })

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ingestionSourceKeys.list()
      })
    })
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ingestionSourceKeys.detail("12")
    })
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ingestionSourceKeys.itemsRoot("12")
    })
  })

  it("invalidates item queries after reattaching a detached item", async () => {
    const client = createMockClient()
    client.reattachIngestionSourceItem.mockResolvedValueOnce({
      id: "41",
      source_id: "12",
      sync_status: "sync_managed"
    })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    const { result } = renderHook(
      () => useReattachIngestionSourceItemMutation("12", client as any),
      { wrapper: buildWrapper(queryClient) }
    )

    await act(async () => {
      await result.current.mutateAsync("41")
    })

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ingestionSourceKeys.itemsRoot("12")
      })
    })
  })
})
