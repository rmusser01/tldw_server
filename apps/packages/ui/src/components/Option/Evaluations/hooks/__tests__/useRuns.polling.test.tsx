import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { useRunDetail } from "../useRuns"
import { useEvaluationsStore } from "@/store/evaluations"

vi.mock("@/services/evaluations", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/evaluations")>(),
  getRun: vi.fn(async () => ({ ok: true, data: { id: "run_owned", status: "pending" } })),
}))

let client: QueryClient
beforeEach(() => {
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
  act(() => client.setQueryData(["evaluations", "run", "run_owned"], {
    ok: true, data: { id: "run_owned", status: "completed" },
  }))
  await waitFor(() => expect(result.current.isPolling).toBe(false))
  act(() => client.setQueryData(["evaluations", "run", "run_owned"], {
    ok: true, data: { id: "run_owned", status: "running" },
  }))
  await waitFor(() => expect(result.current.isPolling).toBe(true))
  rerender({ id: null })
  expect(result.current.isPolling).toBe(false)
})

it("does not let a comparison query overwrite the active run polling indicator", async () => {
  useEvaluationsStore.setState({ isPolling: true })
  client.setQueryData(["evaluations", "run", "run_comparison"], {
    ok: true, data: { id: "run_comparison", status: "completed" },
  })
  const { result } = renderHook(() => useObservedRun("run_comparison", false), { wrapper: Wrapper })
  await waitFor(() => expect(result.current.query.isSuccess).toBe(true))
  expect(result.current.isPolling).toBe(true)
})
