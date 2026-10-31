import { useQuickIngestSessionStore as store } from "@/store/quick-ingest-session"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MediaIngestJobsPanel } from '../MediaIngestJobsPanel'

const mocks = vi.hoisted(() => ({
  listMediaIngestJobs: vi.fn(),
  setSetting: vi.fn(),
  navigate: vi.fn(),
  owner: "verified-A",
  captureBlocked: false
}))

const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return (fallbackOrOptions?.defaultValue || key).replace(
        /{{(\w+)}}/g,
        (_match, name) => String(fallbackOrOptions?.[name] ?? `{{${name}}}`)
      )
    }
  })
}))

vi.mock('@plasmohq/storage/hook', async () => {
  const React = await import('react')
  return {
    useStorage: (key: string, initialValue: unknown) => {
      if (!storageState.values.has(key)) {
        storageState.values.set(key, initialValue)
      }
      const [value, setValue] = React.useState(storageState.values.get(key))
      const updateValue = (next: unknown | ((prev: unknown) => unknown)) => {
        setValue((prev) => {
          const resolved = typeof next === 'function' ? (next as (p: unknown) => unknown)(prev) : next
          storageState.values.set(key, resolved)
          return resolved
        })
      }
      return [value, updateValue] as const
    }
  }
})

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    listMediaIngestJobs: mocks.listMediaIngestJobs
  }
}))

vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }))
vi.mock("@/services/settings/registry", async (importOriginal) => ({
  ...(await importOriginal<any>()),
  setSetting: mocks.setSetting
}))
vi.mock("@/services/tldw/quick-ingest-authority", () => ({
  useQuickIngestAuthority: () => mocks.owner,
  quickIngestAuthority: {
    capture: () => {
      if (mocks.captureBlocked)
        throw new Error("Authority changed before capture")
      const owner = mocks.owner
      return {
        authorityKey: owner,
        signal: new AbortController().signal,
        requestScope: {
          config: { serverUrl: "https://fixture.test", authMode: "multi-user" },
          userId: owner
        },
        isCurrent: () => mocks.owner === owner
      }
    }
  }
}))

const begin = (id = "run") => {
  store.getState().createDraftSession({
    id,
    queueItems: [
      {
        id: "source",
        kind: "url",
        url: "https://example.com/report",
        detectedType: "web",
        icon: "Globe",
        fileSize: 0,
        validation: { valid: true }
      }
    ]
  })
  store
    .getState()
    .markProcessingTracking({ mode: "webui-direct", batchId: "batch-123" })
}
describe("MediaIngestJobsPanel", () => {
  beforeEach(() => {
    mocks.owner = "verified-A"
    mocks.captureBlocked = false
    store.getState().setAuthority(null)
    store.getState().setAuthority(mocks.owner)
    mocks.setSetting.mockReset()
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.navigate.mockReset()
    begin()
    storageState.values.clear()
    storageState.values.set('media:ingest:panelCollapsed', false)
    storageState.values.set('media:ingest:lastBatchId', 'batch-123')
    storageState.values.set('media:ingest:autoRefresh', false)
    mocks.listMediaIngestJobs.mockReset()
  })

  it('loads and renders ingest job rows for the active batch', async () => {
    mocks.listMediaIngestJobs.mockResolvedValue({
      jobs: [
        {
          id: 17,
          status: 'running',
          source: 'https://example.com/doc',
          source_kind: 'url',
          progress_percent: 42,
          progress_message: 'Extracting content'
        }
      ]
    })

    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))

    await waitFor(() =>
      expect(mocks.listMediaIngestJobs).toHaveBeenCalledWith(
        {
        batch_id: 'batch-123',
        limit: 50
      },
        expect.objectContaining({
          requestScope: expect.any(Object),
          signal: expect.any(AbortSignal)
        })
      )
    )

    expect(screen.getByTestId('media-ingest-job-row-17')).toBeInTheDocument()
    expect(screen.getByTestId('media-ingest-job-status-17')).toHaveTextContent('running')
    expect(screen.getByText('42% • Extracting content')).toBeInTheDocument()
  })

  it('applies a new batch id and shows an empty-state message when no jobs are returned', async () => {
    store.getState().setAuthority(null)
    store.getState().setAuthority(mocks.owner)
    storageState.values.set('media:ingest:lastBatchId', '')
    mocks.listMediaIngestJobs.mockResolvedValue({ jobs: [] })

    render(<MediaIngestJobsPanel />)

    expect(screen.getByTestId('media-ingest-jobs-empty-batch')).toBeInTheDocument()

    fireEvent.change(screen.getByTestId('media-ingest-batch-input'), {
      target: { value: 'new-batch' }
    })
    fireEvent.click(screen.getByTestId('media-ingest-batch-apply'))

    await waitFor(() =>
      expect(mocks.listMediaIngestJobs).toHaveBeenCalledWith(
        {
        batch_id: 'new-batch',
        limit: 50
      },
        expect.objectContaining({
          requestScope: expect.any(Object),
          signal: expect.any(AbortSignal)
        })
      )
    )

    expect(screen.getByTestId('media-ingest-jobs-empty')).toBeInTheDocument()
  })

  it('shows an inline error and retries successfully', async () => {
    mocks.listMediaIngestJobs.mockRejectedValueOnce(new Error('boom'))
    mocks.listMediaIngestJobs.mockResolvedValueOnce({
      jobs: [
        {
          id: 23,
          status: 'completed',
          source: 'report.pdf',
          source_kind: 'file'
        }
      ]
    })

    render(<MediaIngestJobsPanel />)

    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    expect(await screen.findByTestId('media-ingest-jobs-error')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId('media-ingest-jobs-retry'))

    await waitFor(() => {
      expect(mocks.listMediaIngestJobs).toHaveBeenCalledTimes(2)
    })
    expect(await screen.findByTestId('media-ingest-job-row-23')).toBeInTheDocument()
  })
  it("shows active sources from submission, resumes current results and reviews exact historical saved IDs", async () => {
    store.getState().upsertSession({
      lifecycle: "completed",
      results: [
        { id: "one", status: "ok", mediaId: 41, type: "web" },
        { id: "two", status: "ok", mediaId: 42, type: "web" },
        { id: "bad", status: "error", mediaId: 43, type: "web" }
      ]
    })
    store.getState().replaceWithNewDraft()
    const view = render(<MediaIngestJobsPanel />)
    expect(screen.getByText("example.com")).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Review 2 saved items" })
    )
    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith("/media-multi")
    )
    expect(mocks.setSetting.mock.calls[0][1]).toEqual({
      version: 1,
      authorityKey: mocks.owner,
      selectedIds: [41, 42]
    })
    act(() => {
      mocks.owner = "verified-B"
      store.getState().setAuthority(mocks.owner)
    })
    view.rerender(<MediaIngestJobsPanel />)
    expect(screen.queryByText("example.com")).not.toBeInTheDocument()
  })
  it("rejects a late historical refresh after authority changes", async () => {
    let finish!: (response: unknown) => void
    mocks.listMediaIngestJobs.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const view = render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await waitFor(() =>
      expect(mocks.listMediaIngestJobs).toHaveBeenCalledTimes(1)
    )
    act(() => {
      mocks.owner = "verified-B"
      store.getState().setAuthority(mocks.owner)
    })
    view.rerender(<MediaIngestJobsPanel />)
    await act(async () => {
      finish({
        jobs: [
          {
            id: 77,
            source: "private-source",
            status: "completed",
            result: { media_id: 77 }
          }
        ]
      })
    })
    expect(screen.queryByText("private-source")).not.toBeInTheDocument()
    expect(store.getState().recentImports).toEqual([])
  })

  it("refreshes all canonical known-batch pages before publishing saved IDs", async () => {
    mocks.listMediaIngestJobs.mockResolvedValueOnce({
        jobs: [{ id: 1, status: "completed", result: { media_id: 41 } }],
        has_more: true,
        next_offset: 50
      })
      .mockResolvedValueOnce({
        jobs: [{ id: 2, status: "completed", result: { media_id: 42 } }],
        has_more: false,
        next_offset: null
      })
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await waitFor(() =>
      expect(store.getState().recentImports[0].savedMediaIds).toEqual([41, 42])
    )
    expect(mocks.listMediaIngestJobs.mock.calls[1][1]).toBeDefined()
    expect(mocks.listMediaIngestJobs.mock.calls[1][1].requestScope).toBe(mocks.listMediaIngestJobs.mock.calls[0][1].requestScope)
    expect(mocks.listMediaIngestJobs.mock.calls[1][1].signal).toBe(mocks.listMediaIngestJobs.mock.calls[0][1].signal)
    expect(mocks.listMediaIngestJobs.mock.calls[1][0]).toEqual({
      batch_id: "batch-123",
      limit: 50,
      offset: 50
    })
  })

  it("does not hand off a completed extraction that explicitly was not saved", async () => {
    mocks.listMediaIngestJobs.mockResolvedValueOnce({
      jobs: [
        {
          id: 1,
          status: "completed",
          result: { media_id: 41, persisted: false }
        }
      ],
      has_more: false
    })
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await waitFor(() =>
      expect(screen.getByTestId("media-ingest-job-row-1")).toBeInTheDocument()
    )
    expect(store.getState().recentImports[0].savedMediaIds).toEqual([])
  })
  it("does not send or leak a rejection when authority changes just before refresh capture", async () => {
    mocks.captureBlocked = true
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.listMediaIngestJobs).not.toHaveBeenCalled()
  })
  it.each(["Review 1 saved item", "Refresh import", "Resume import"])(
    "rejects retained %s after a replacement owner is verified before capture",
    async (action) => {
      store
        .getState()
        .upsertSession({
          lifecycle: "completed",
          results: [{ id: "source", status: "ok", type: "web", mediaId: 41 }]
        })
      mocks.listMediaIngestJobs.mockResolvedValue({ jobs: [] })
      render(<MediaIngestJobsPanel />)
      if (action === "Refresh import") {
        fireEvent.click(screen.getByRole("button", { name: action }))
        await waitFor(() =>
          expect(mocks.listMediaIngestJobs).toHaveBeenCalledTimes(1)
        )
        mocks.listMediaIngestJobs.mockClear()
      }
      const retained = screen.getByRole("button", { name: action })
      act(() => {
        mocks.owner = "verified-B"
        store.getState().setAuthority(mocks.owner)
        store
          .getState()
          .createDraftSession({ id: "owner-B-draft", visibility: "hidden" })
        fireEvent.click(retained)
      })
      await act(async () => {
        await Promise.resolve()
      })
      expect(mocks.setSetting).not.toHaveBeenCalled()
      expect(mocks.navigate).not.toHaveBeenCalled()
      expect(mocks.listMediaIngestJobs).not.toHaveBeenCalled()
      expect(store.getState().session?.visibility).toBe("hidden")
    }
  )

  it.each([
    [
      "logical failure",
      { status: "Error", error: "Failed to save", media_id: 99 },
      [],
      "partial_failure"
    ],
    [
      "saved warning",
      { status: "Warning", warnings: ["Analysis unavailable"], db_id: 41 },
      [41],
      "completed"
    ],
    [
      "unsaved warning",
      { status: "Warning", warnings: ["Failed to save"] },
      [],
      "partial_failure"
    ],
    ["db alias", { status: "Success", db_id: 41 }, [41], "completed"],
    ["camel alias", { status: "Success", mediaId: 41 }, [41], "completed"],
    [
      "nested media/add",
      { results: [{ status: "Success", db_id: 41 }] },
      [41],
      "completed"
    ],
    ["direct media/add", [{ status: "Success", db_id: 41 }], [41], "completed"],
    [
      "unsaved alias",
      { status: "Success", db_id: 41, persisted: false },
      [],
      "completed"
    ],
    [
      "nested unsaved",
      { results: [{ status: "Success", db_id: 41, persisted: false }] },
      [],
      "completed"
    ],
    [
      "mixed logical outcomes",
      {
        results: [
          { status: "Error", media_id: 99, error: "Failed to save" },
          { status: "Success", db_id: 41 }
        ]
      },
      [41],
      "partial_failure"
    ]
  ])(
    "refresh interprets %s through canonical result semantics",
    async (_label, result, saved, lifecycle) => {
      mocks.listMediaIngestJobs.mockResolvedValue({
        jobs: [{ id: 7, status: "completed", result }]
      })
      render(<MediaIngestJobsPanel />)
      fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
      await screen.findByTestId("media-ingest-job-row-7")
      expect(store.getState().recentImports[0]).toMatchObject({
        savedMediaIds: saved,
        lifecycle
      })
    }
  )

  it("preserves a successful durable retry after replacement, reload and refresh of both attempts", async () => {
    store
      .getState()
      .markProcessingTracking({
        mode: "webui-direct",
        batchId: "failed-file-batch",
        jobIds: [10]
      })
    store
      .getState()
      .upsertSession({
        lifecycle: "partial_failure",
        completedAt: 1,
        results: [{ id: "source", type: "document", status: "error" }]
      })
    store.getState().upsertSession({ lifecycle: "processing", completedAt: null })
    store
      .getState()
      .markProcessingTracking({
        mode: "webui-direct",
        batchId: "retry-file-batch",
        jobIds: [20]
      })
    store
      .getState()
      .upsertSession({
        lifecycle: "completed",
        completedAt: 2,
        results: [{ id: "source", type: "document", status: "ok", mediaId: 41 }]
      })
    store.getState().replaceWithNewDraft()
    const raw = sessionStorage.getItem("tldw-quick-ingest-session")!
    store.setState({ recentImports: [], session: null })
    sessionStorage.setItem("tldw-quick-ingest-session", raw)
    await store.persist.rehydrate()
    mocks.listMediaIngestJobs.mockImplementation(async ({ batch_id }) => ({
      jobs:
        batch_id === "failed-file-batch"
          ? [{ id: 10, status: "failed", error_message: "Old failed attempt" }]
          : batch_id === "retry-file-batch"
            ? [{ id: 20, status: "completed", result: { media_id: 41 } }]
            : []
    }))
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await screen.findByTestId("media-ingest-job-row-20")
    expect(store.getState().recentImports[0]).toMatchObject({
      lifecycle: "completed",
      completedAt: 2,
      savedMediaIds: [41],
      jobIds: [10, 20]
    })
    expect(
      mocks.listMediaIngestJobs.mock.calls.map(([request]) => request.batch_id)
    ).toEqual(["batch-123", "failed-file-batch", "retry-file-batch"])
    fireEvent.click(screen.getByRole("button", { name: "Review 1 saved item" }))
    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith("/media-multi")
    )
    expect(mocks.setSetting.mock.calls[0][1]).toEqual({
      version: 1,
      authorityKey: "verified-A",
      selectedIds: [41]
    })
  })

  it("retains the initiating review owner across the snapshot write", async () => {
    store
      .getState()
      .upsertSession({
        lifecycle: "completed",
        results: [{ id: "source", status: "ok", type: "web", mediaId: 41 }]
      })
    let finish!: () => void
    mocks.setSetting.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve
        })
    )
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Review 1 saved item" }))
    await waitFor(() => expect(mocks.setSetting).toHaveBeenCalledTimes(1))
    act(() => {
      mocks.owner = "verified-B"
      store.getState().setAuthority(mocks.owner)
    })
    await act(async () => {
      finish()
      await Promise.resolve()
    })
    expect(mocks.setSetting).toHaveBeenCalledTimes(1)
    expect(mocks.setSetting.mock.calls[0][1]).toEqual({
      version: 1,
      authorityKey: "verified-A",
      selectedIds: [41]
    })
    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it("abandons the next known-batch page after the initiating owner changes", async () => {
    let finish!: (value: unknown) => void
    mocks.listMediaIngestJobs.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await waitFor(() =>
      expect(mocks.listMediaIngestJobs).toHaveBeenCalledTimes(1)
    )
    const originalScope = mocks.listMediaIngestJobs.mock.calls[0][1].requestScope
    act(() => {
      mocks.owner = "verified-B"
      store.getState().setAuthority(mocks.owner)
    })
    await act(async () => {
      finish({
        jobs: [{ id: 10, status: "completed", result: { media_id: 41 } }],
        has_more: true,
        next_offset: 50
      })
    })
    expect(originalScope.userId).toBe("verified-A")
    expect(mocks.listMediaIngestJobs).toHaveBeenCalledTimes(1)
    expect(store.getState().recentImports).toEqual([])
  })

  it("keeps a wholly cancelled batch cancelled when using logical result interpretation", async () => {
    mocks.listMediaIngestJobs.mockResolvedValue({
      jobs: [
        {
          id: 7,
          status: "cancelled",
          cancellation_reason: "User requested cancellation"
        }
      ]
    })
    render(<MediaIngestJobsPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Refresh import" }))
    await screen.findByTestId("media-ingest-job-row-7")
    expect(store.getState().recentImports[0]).toMatchObject({
      lifecycle: "cancelled",
      savedMediaIds: []
    })
  })

})
