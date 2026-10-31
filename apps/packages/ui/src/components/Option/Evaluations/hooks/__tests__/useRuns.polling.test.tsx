import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { useRunDetail, useRunsList } from "../useRuns"
import { getRun, listRuns } from "@/services/evaluations"
import { useEvaluationsStore } from "@/store/evaluations"

vi.mock("@/services/evaluations", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/evaluations")>(),
  listRuns: vi.fn(),
  getRun: vi.fn(async () => ({ ok: true, data: { id: "run_owned", status: "pending" } })),
}))

let owner = "owner-A"
let ready = true
vi.mock("@/hooks/usePromptLocalAuthority", () => ({
  usePromptLocalAuthority: () => ({
    ownerId: ready ? owner : null, ready, boundaryKey: "server-A"
  })
}))
vi.mock("@/services/prompt-local-owner", () => ({
  capturePromptOwnerScope: vi.fn(async (expected: string) => {
    const captured = owner
    if (captured !== expected) throw new Error("Account changed")
    return {
      assertCurrent: () => { if (owner !== captured || !ready) throw new Error("Account changed") },
      release: vi.fn()
    }
  })
}))

let client: QueryClient
beforeEach(() => {
  owner = "owner-A"
  ready = true
  vi.mocked(getRun).mockReset().mockResolvedValue({ ok: true, data: { id: "run_owned", status: "pending" } } as Awaited<ReturnType<typeof getRun>>)
  vi.mocked(listRuns).mockReset()
  useEvaluationsStore.getState().resetStore()
  client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
})
afterEach(() => { cleanup(); client.clear() })

function Wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

// RunsTab consumes an object selection while the hook consumes the same real store.
function useObservedRun(runId: string | null, enablePolling = true) {
  const state = useEvaluationsStore((s) => ({ isPolling: s.isPolling }))
  const query = useRunDetail(runId, { enablePolling })
  return { ...state, query }
}

it("mounts without a selected run without repeatedly publishing polling state", () => {
  const { result } = renderHook(() => useObservedRun(null), { wrapper: Wrapper })
  expect(result.current.isPolling).toBe(false)
  expect(result.current.query.fetchStatus).toBe("idle")
})

it("tracks active and completed results and clears polling when selection is removed", async () => {
  const { result, rerender } = renderHook(({ id }) => useObservedRun(id), {
    initialProps: { id: "run_owned" as string | null }, wrapper: Wrapper,
  })
  await waitFor(() => expect(result.current.isPolling).toBe(true))
  act(() => client.setQueryData(["evaluations", "run", "run_owned", "server-A", "owner-A"], {
    ok: true, data: { id: "run_owned", status: "completed" },
  }))
  await waitFor(() => expect(result.current.isPolling).toBe(false))
  act(() => client.setQueryData(["evaluations", "run", "run_owned", "server-A", "owner-A"], {
    ok: true, data: { id: "run_owned", status: "running" },
  }))
  await waitFor(() => expect(result.current.isPolling).toBe(true))
  rerender({ id: null })
  expect(result.current.isPolling).toBe(false)
})

it("does not let a comparison query overwrite the active run polling indicator", async () => {
  useEvaluationsStore.setState({ isPolling: true })
  client.setQueryData(["evaluations", "run", "run_comparison", "server-A", "owner-A"], {
    ok: true, data: { id: "run_comparison", status: "completed" },
  })
  const { result } = renderHook(() => useObservedRun("run_comparison", false), { wrapper: Wrapper })
  await waitFor(() => expect(result.current.query.isSuccess).toBe(true))
  expect(result.current.isPolling).toBe(true)
})

it("hides cached private run results while a different owner read is pending and denied", async () => {
  const privateRun = { ok: true, data: { id: "run_owned", status: "completed", results: { input: "A private input" } } }
  vi.mocked(getRun).mockResolvedValueOnce(privateRun as Awaited<ReturnType<typeof getRun>>)
  const { result, rerender } = renderHook(() => useObservedRun("run_owned"), { wrapper: Wrapper })
  await waitFor(() => expect(result.current.query.data?.data).toEqual(privateRun.data))

  let deny!: (value: Awaited<ReturnType<typeof getRun>>) => void
  vi.mocked(getRun).mockImplementationOnce(() => new Promise(resolve => { deny = resolve }))
  owner = "owner-B"
  rerender()
  expect(result.current.query.data?.data).toBeUndefined()
  await waitFor(() => expect(getRun).toHaveBeenCalledTimes(2))
  await act(async () => { deny({ ok: false, status: 404, data: null }) })
  await waitFor(() => expect(result.current.query.fetchStatus).toBe("idle"))
  expect(result.current.query.data?.ok).toBe(false)
  expect(result.current.query.data?.data).toBeNull()
})

it("retires private detail immediately while the current account is unresolved", async () => {
  vi.mocked(getRun).mockResolvedValueOnce({
    ok: true, data: { id: "run_owned", status: "completed", results: { input: "private input" } }
  } as Awaited<ReturnType<typeof getRun>>)
  const { result, rerender } = renderHook(() => useObservedRun("run_owned"), { wrapper: Wrapper })
  await waitFor(() => expect(result.current.query.data?.data?.results).toEqual({ input: "private input" }))
  ready = false
  rerender()
  expect(result.current.query.data?.data).toBeUndefined()
  expect(result.current.query.fetchStatus).toBe("idle")
  expect(getRun).toHaveBeenCalledTimes(1)
})

it("discards a late former-owner response before publishing its quota", async () => {
  let complete!: (value: Awaited<ReturnType<typeof getRun>>) => void
  vi.mocked(getRun).mockImplementationOnce(() => new Promise(resolve => { complete = resolve }))
  const { result, rerender } = renderHook(() => useObservedRun("run_owned"), { wrapper: Wrapper })
  await waitFor(() => expect(getRun).toHaveBeenCalledTimes(1))
  owner = "owner-B"
  vi.mocked(getRun).mockResolvedValueOnce({ ok: false, status: 404, data: null } as Awaited<ReturnType<typeof getRun>>)
  rerender()
  await act(async () => { complete({
    ok: true, headers: { "x-ratelimit-limit-day": "99" },
    data: { id: "run_owned", status: "completed", results: { input: "A private input" } }
  }) })
  expect(useEvaluationsStore.getState().quotaSnapshot).toBeNull()
  expect(result.current.query.data?.data?.results).toBeUndefined()
})


it("refreshes the matching run list when the selected detail reaches failure", async () => {
  const run = { id: "run_owned", eval_id: "eval_owned", status: "pending" }
  vi.mocked(getRun).mockResolvedValue({ ok: true, data: run } as Awaited<ReturnType<typeof getRun>>)
  vi.mocked(listRuns).mockResolvedValue({ ok: true, data: { data: [run] } } as Awaited<ReturnType<typeof listRuns>>)
  const { result } = renderHook(() => ({
    list: useRunsList("eval_owned"), detail: useRunDetail("run_owned"),
  }), { wrapper: Wrapper })
  await waitFor(() => {
    expect(result.current.detail.data?.data?.status).toBe("pending")
    expect(result.current.list.data?.data?.data?.[0]?.status).toBe("pending")
    expect(result.current.list.fetchStatus).toBe("idle")
  })
  const failed = { ...run, status: "failed", error_message: "Provider timed out" }
  vi.mocked(getRun).mockResolvedValue({ ok: true, data: failed } as Awaited<ReturnType<typeof getRun>>)
  vi.mocked(listRuns).mockResolvedValue({ ok: true, data: { data: [failed] } } as Awaited<ReturnType<typeof listRuns>>)
  await act(async () => { await result.current.detail.refetch() })
  await waitFor(() => expect(result.current.list.data?.data?.data?.[0]?.status).toBe("failed"))
  expect(result.current.detail.data?.data?.status).toBe("failed")
})
