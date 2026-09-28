import { act, renderHook } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { useIngestResults, type UseIngestResultsDeps } from "../useIngestResults"

vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: {} }))
vi.mock("@/db/dexie/drafts", () => ({}))
vi.mock("@/services/settings/registry", () => ({}))
vi.mock("@/services/settings/ui-settings", () => ({}))
vi.mock("@/store/quick-ingest", () => ({ useQuickIngestStore: () => ({}) }))

afterEach(() => vi.useRealTimers())

it("updates elapsed time from the clock and captures the final partial tick", () => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date("2026-09-27T12:00:00Z"))
  const deps = { open: true, running: true, rows: [] } as unknown as UseIngestResultsDeps
  const { result, rerender, unmount } = renderHook(
    (props) => useIngestResults(props),
    { initialProps: deps }
  )
  act(() => result.current.setRunStartedAt(Date.now()))
  act(() => vi.advanceTimersByTime(2000))
  expect(result.current.progressMeta.elapsedLabel).toBe("0:02")
  vi.setSystemTime(new Date("2026-09-27T12:01:02.500Z"))
  act(() => vi.advanceTimersByTime(1000))
  expect(result.current.progressMeta.elapsedLabel).toBe("1:03")
  vi.setSystemTime(new Date("2026-09-27T12:01:04Z"))
  rerender({ ...deps, running: false })
  expect(result.current.progressMeta.elapsedLabel).toBe("1:04")
  unmount()
  expect(vi.getTimerCount()).toBe(0)
})
