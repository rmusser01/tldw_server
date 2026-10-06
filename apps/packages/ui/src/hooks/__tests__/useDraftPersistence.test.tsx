import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type DraftValue = string | { content: string; metadata?: Record<string, unknown> }
type DraftRecord = { value: DraftValue; updatedAt: number }

const bucketMocks = vi.hoisted(() => {
  let draftRecord: DraftRecord | null = null
  const getMock = vi.fn(async () => draftRecord)
  const setMock = vi.fn(async (_key: string, value: DraftValue) => {
    draftRecord = { value, updatedAt: Date.now() }
  })
  const removeMock = vi.fn(async () => {
    draftRecord = null
  })
  const cleanupMock = vi.fn(async () => 0)
  const reset = () => {
    draftRecord = null
  }
  return { getMock, setMock, removeMock, cleanupMock, reset }
})

vi.mock("@/services/settings/local-bucket", () => ({
  createLocalRegistryBucket: vi.fn(() => ({
    get: bucketMocks.getMock,
    set: bucketMocks.setMock,
    remove: bucketMocks.removeMock,
    cleanup: bucketMocks.cleanupMock,
    buildKey: (key: string) => `registry:draft:${key}`
  }))
}))
import { useDraftPersistence } from "@/hooks/useDraftPersistence"

describe("useDraftPersistence", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    bucketMocks.reset()
    bucketMocks.getMock.mockClear()
    bucketMocks.setMock.mockClear()
    bucketMocks.removeMock.mockClear()
    bucketMocks.cleanupMock.mockClear()
    localStorage.clear()
  })

  afterEach(async () => {
    await vi.runOnlyPendingTimersAsync()
    vi.useRealTimers()
  })

  it("does not hydrate older text over a newly owned image-only input", async () => {
    await bucketMocks.setMock("chat:first", "OLDER OWNED TEXT")
    let value = ""
    const restore = vi.fn((next: string) => { value = next })
    const { rerender } = renderHook(({ enabled }) => useDraftPersistence({
      storageKey: "chat:first", enabled, hasUnpersistedInput: true,
      getValue: () => value, setValue: restore
    }), { initialProps: { enabled: false } })
    rerender({ enabled: true })
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    expect(restore).not.toHaveBeenCalled()
    expect(value).toBe("")
  })

  it.each([true, false])("qualifies a source-switch edit against the previous owner: %s", async previousOwnerCurrent => {
    let value = ""
    const firstOwner = () => previousOwnerCurrent
    const nextOwner = () => true
    const restore = vi.fn((next: string) => { value = next })
    const { rerender } = renderHook(({ storageKey, isCurrent }) => useDraftPersistence({
      storageKey, isCurrent, getValue: () => value, setValue: restore
    }), { initialProps: { storageKey: "first", isCurrent: firstOwner } })
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    restore.mockClear()
    value = "NEW OWNED IDEA"
    rerender({ storageKey: "new", isCurrent: nextOwner })
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    if (previousOwnerCurrent) expect(restore).not.toHaveBeenCalled()
    else expect(restore).toHaveBeenCalledWith("")
  })

  it("does not persist again when value is unchanged and metadata callback identity changes", async () => {
    let value = "persistent draft"

    const { rerender } = renderHook(
      ({ nonce }) =>
        useDraftPersistence({
          storageKey: "chat:1",
          getValue: () => value,
          setValue: () => {},
          getMetadata: () => ({
            source: "composer",
            stable: true,
            _nonceForIdentityOnly: Boolean(nonce)
          }),
          enabled: true
        }),
      { initialProps: { nonce: 1 } }
    )

    await act(async () => {
      await Promise.resolve()
    })

    rerender({ nonce: 2 })
    await act(async () => {
      await Promise.resolve()
      await vi.runAllTimersAsync()
    })
    expect(bucketMocks.setMock).toHaveBeenCalledTimes(1)

    rerender({ nonce: 3 })
    await act(async () => {
      await Promise.resolve()
      await vi.runAllTimersAsync()
    })
    expect(bucketMocks.setMock).toHaveBeenCalledTimes(1)

    value = "persistent draft updated"
    rerender({ nonce: 4 })
    await act(async () => {
      await Promise.resolve()
      await vi.runAllTimersAsync()
    })
    expect(bucketMocks.setMock).toHaveBeenCalledTimes(2)
  })
})
