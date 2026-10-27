import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Character } from "@/types/character"
import { useCharacterGreeting } from "../useCharacterGreeting"

const mocks = vi.hoisted(() => ({ getCharacter: vi.fn(), snapshot: vi.fn(), authority: new AbortController(), release: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getCharacter: mocks.getCharacter } }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: mocks.snapshot }))
const card = { id: "card-1", name: "Guide", first_message: "{{user}} meets {{char}}", alternate_greetings: ["Alternate"] } as Character
const options = () => ({ playgroundReady: true, selectedCharacter: card, selectedCharacterMode: "tracked" as const, serverChatId: null, historyId: null, messagesLength: 0, viewKey: "fresh-1" })
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }
beforeEach(() => {
  mocks.authority = new AbortController()
  mocks.release.mockReset()
  mocks.getCharacter.mockReset().mockResolvedValue(card)
  mocks.snapshot.mockReset().mockImplementation(async (_ids, { signal }: { signal: AbortSignal }) => ({
    scopeSignal: signal, scopeInvalidatedSignal: mocks.authority.signal, requestScope: { config: { serverUrl: "http://localhost:8000", authMode: "single-user" } }, release: mocks.release
  }))
})

describe("read-only character greeting preview", () => {
  it("keeps preview out of transcript and request history before explicit startup", async () => {
    const setMessages = vi.fn()
    const setHistory = vi.fn()
    const params = { ...options(), setMessages, setHistory }
    const view = renderHook(() => useCharacterGreeting(params))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(view.result.current.character).toBe(card)
    expect(setMessages).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
  })
  it("captures the exact card owner scope and keeps canonical raw fields", async () => {
    const scope = { type: "workspace" as const, workspaceId: "workspace-a" }
    const view = renderHook(() => useCharacterGreeting({ ...options(), scope }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(mocks.getCharacter).toHaveBeenCalledWith("card-1", expect.objectContaining({ requestScope: expect.any(Object), signal: expect.any(AbortSignal) }))
    expect(view.result.current.character?.first_message).toBe("{{user}} meets {{char}}")
  })
  it.each([
    { selectedCharacterMode: "overlay" as const }, { serverChatId: "saved" },
    { historyId: "local-saved" }, { messagesLength: 1 }, { playgroundReady: false }
  ])("does not read a fresh greeting for incompatible state %j", patch => {
    const view = renderHook(() => useCharacterGreeting({ ...options(), ...patch }))
    expect(view.result.current.fresh).toBe(false)
    expect(mocks.snapshot).not.toHaveBeenCalled()
    expect(mocks.getCharacter).not.toHaveBeenCalled()
  })
  it("hydrates current name and avatar without rewriting raw greeting text", async () => {
    const hydrated = { ...card, name: "Renamed Guide", avatar_url: "https://example.com/card.png" }
    mocks.getCharacter.mockResolvedValue(hydrated)
    const view = renderHook(() => useCharacterGreeting(options()))
    await waitFor(() => expect(view.result.current.character).toBe(hydrated))
    view.rerender()
    expect(mocks.getCharacter).toHaveBeenCalledOnce()
  })
  it.each(["card", "view", "account", "unmount"])("rejects a held card read after %s retirement", async boundary => {
    const pending = deferred<Character>()
    mocks.getCharacter.mockReturnValueOnce(pending.promise)
    const initial = options()
    const view = renderHook(({ params }) => useCharacterGreeting(params), { initialProps: { params: initial } })
    await waitFor(() => expect(mocks.getCharacter).toHaveBeenCalledOnce())
    if (boundary === "card") view.rerender({ params: { ...initial, selectedCharacter: { ...card, id: "card-2" } } })
    else if (boundary === "view") view.rerender({ params: { ...initial, viewKey: "fresh-2" } })
    else if (boundary === "account") mocks.authority.abort()
    else view.unmount()
    await act(async () => pending.resolve({ ...card, name: "Stale name" }))
    expect(view.result.current.character?.name).not.toBe("Stale name")
  })
  it("rejects another card returned by the server without a ready opening", async () => {
    mocks.getCharacter.mockResolvedValue({ ...card, id: "foreign-card" })
    const view = renderHook(() => useCharacterGreeting(options()))
    await waitFor(() => expect(view.result.current.unavailable).toBe(true))
    expect(view.result.current.character).toBe(card)
  })
  it("keeps readable preview on a failed card refresh and forbids a saved-opening claim", async () => {
    mocks.getCharacter.mockRejectedValue(new Error("offline"))
    const view = renderHook(() => useCharacterGreeting(options()))
    await waitFor(() => expect(view.result.current.unavailable).toBe(true))
    expect(view.result.current.character).toBe(card)
    expect(mocks.release).toHaveBeenCalled()
  })
  it("does not refetch when a parent supplies the same card as a new object", async () => {
    const initial = options()
    const view = renderHook(({ params }) => useCharacterGreeting(params), { initialProps: { params: initial } })
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    view.rerender({ params: { ...initial, selectedCharacter: { ...card } } })
    expect(mocks.getCharacter).toHaveBeenCalledOnce()
  })
  it("a deliberate retry recovers a failed preview without writing chat rows", async () => {
    mocks.getCharacter.mockRejectedValueOnce(new Error("offline"))
    const view = renderHook(() => useCharacterGreeting(options()))
    await waitFor(() => expect(view.result.current.unavailable).toBe(true))
    act(() => view.result.current.retry())
    await waitFor(() => expect(view.result.current.unavailable).toBe(false))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(mocks.getCharacter).toHaveBeenCalledTimes(2)
  })

})
