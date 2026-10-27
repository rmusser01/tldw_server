import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { usePromptSearch } from "../usePromptSearch"
const state = vi.hoisted(() => ({ owner: "alice", ready: true, local: vi.fn(), server: vi.fn() }))
vi.mock("@/hooks/usePromptLocalAuthority", () => ({ usePromptLocalAuthority: () => ({ ownerId: state.owner, ready: state.ready, boundaryKey: state.owner }) }))
vi.mock("@/db/dexie/helpers", () => ({ getAllPrompts: (...args: any[]) => state.local(...args) }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { initialize: async () => {}, searchPrompts: (...args: any[]) => state.server(...args) } }))
beforeEach(() => { state.owner = "alice"; state.ready = true; state.local.mockReset().mockResolvedValue([{ id: "a", title: "Alice private", content: "Alice" }]); state.server.mockReset().mockResolvedValue([]) })
it("hides settled old-account search results when the same mounted view changes account", async () => {
  const view = renderHook(usePromptSearch)
  await act(async () => { await view.result.current.search("Alice") })
  expect(view.result.current.results[0].title).toBe("Alice private")
  state.owner = "bob"; view.rerender()
  expect(view.result.current.results).toEqual([])
})
it("rejects late local results and passes captured ownership to both supported searches", async () => {
  let release!: (value: any) => void
  state.local.mockReturnValue(new Promise(resolve => { release = resolve }))
  const view = renderHook(usePromptSearch)
  let pending!: Promise<void>
  act(() => { pending = view.result.current.search("Alice") })
  state.owner = "bob"; view.rerender()
  await act(async () => { release([{ title: "Alice private", content: "Alice" }]); await pending })
  expect(view.result.current.results).toEqual([])
  expect(state.local).toHaveBeenCalledWith("alice")
  expect(state.server).not.toHaveBeenCalled()
})
it("sends current-account server search with its expected owner", async () => {
  const view = renderHook(usePromptSearch)
  await act(async () => { await view.result.current.search("Alice") })
  expect(state.server).toHaveBeenCalledWith("Alice", { expectedOwnerId: "alice" })
})
