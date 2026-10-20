import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useConnectionsDisclosure, type ConnectionsMemory } from "../notes-connections"

const key = (authority: string, noteId: string) => JSON.stringify([authority, noteId])

describe("useConnectionsDisclosure (NE-10)", () => {
  it("starts closed and requests connections only when opened", () => {
    const onRequest = vi.fn()
    const { result } = renderHook(
      ({ requestKey }) => useConnectionsDisclosure(requestKey, onRequest),
      { initialProps: { requestKey: key("alice", "n1") } }
    )
    expect(result.current.initiallyOpen).toBe(false)
    expect(onRequest).not.toHaveBeenCalled()

    act(() => result.current.onOpenChange(true))
    expect(onRequest).toHaveBeenCalledTimes(1)
  })

  it("reopens and loads the next note after the user left it open", () => {
    const onRequest = vi.fn()
    const { result, rerender } = renderHook(
      ({ requestKey }) => useConnectionsDisclosure(requestKey, onRequest),
      { initialProps: { requestKey: key("alice", "n1") } }
    )
    act(() => result.current.onOpenChange(true))
    onRequest.mockClear()

    rerender({ requestKey: key("alice", "n2") })
    expect(result.current.initiallyOpen).toBe(true)
    expect(onRequest).toHaveBeenCalledTimes(1)
  })

  it("stays closed on the next note after the user closed it", () => {
    const onRequest = vi.fn()
    const { result, rerender } = renderHook(
      ({ requestKey }) => useConnectionsDisclosure(requestKey, onRequest),
      { initialProps: { requestKey: key("alice", "n1") } }
    )
    act(() => result.current.onOpenChange(true))
    act(() => result.current.onOpenChange(false))
    onRequest.mockClear()

    rerender({ requestKey: key("alice", "n2") })
    expect(result.current.initiallyOpen).toBe(false)
    expect(onRequest).not.toHaveBeenCalled()
  })

  it("keeps the choice in page memory when the pane remounts", () => {
    const onRequest = vi.fn()
    const memory: ConnectionsMemory = { current: null }
    const first = renderHook(() => useConnectionsDisclosure(key("alice", "n1"), onRequest, memory))
    act(() => first.result.current.onOpenChange(true))
    // The Graph view unmounts the editor pane.
    first.unmount()
    onRequest.mockClear()

    const second = renderHook(() => useConnectionsDisclosure(key("alice", "n1"), onRequest, memory))
    expect(second.result.current.initiallyOpen).toBe(true)
    expect(onRequest).toHaveBeenCalledTimes(1)
  })

  it("forgets the choice when the account changes", () => {
    const onRequest = vi.fn()
    const { result, rerender } = renderHook(
      ({ requestKey }) => useConnectionsDisclosure(requestKey, onRequest),
      { initialProps: { requestKey: key("alice", "n1") } }
    )
    act(() => result.current.onOpenChange(true))
    onRequest.mockClear()

    rerender({ requestKey: key("bob", "n1") })
    expect(result.current.initiallyOpen).toBe(false)
    rerender({ requestKey: key("alice", "n2") })
    expect(result.current.initiallyOpen).toBe(false)
    expect(onRequest).not.toHaveBeenCalled()
  })
})
