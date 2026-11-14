import { act, renderHook } from "@testing-library/react"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type DraftValue = string | { content: string; metadata?: Record<string, unknown> }
type DraftRecord = { value: DraftValue; updatedAt: number }

const bucketMocks = vi.hoisted(() => {
  const records = new Map<string, DraftRecord>()
  const getMock = vi.fn(async (key: string) => records.get(key) ?? null)
  const setMock = vi.fn(async (key: string, value: DraftValue) => {
    records.set(key, { value, updatedAt: Date.now() })
    return true
  })
  const removeMock = vi.fn(async (key: string) => {
    records.delete(key)
  })
  const cleanupMock = vi.fn(async () => 0)
  const reset = () => {
    records.clear()
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

  // Actual persistence hook and lifecycle; registry reads/writes are declared
  // keyed I/O doubles. The companion composer/privacy suite uses the real registry.
  it.each([false, true])(
    "retires an acknowledged receipt across a disabled interval without reviving it before a new save (tabScoped=%s)",
    async (tabScoped) => {
      let value = ""
      const view = renderHook(
        ({ enabled, revision }) => {
          void revision
          return useDraftPersistence({
            storageKey: "receipt:A",
            tabScoped,
            enabled,
            getValue: () => value,
            setValue: (next) => {
              value = next
            }
          })
        },
        { initialProps: { enabled: true, revision: 0 } }
      )
      await act(async () => {})
      value = "ACKNOWLEDGED A"
      view.rerender({ enabled: true, revision: 1 })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(
          COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
        )
      })
      expect(view.result.current.draftSaved).toBe(true)
      expect(bucketMocks.setMock).toHaveBeenCalledOnce()
      view.rerender({ enabled: false, revision: 2 })
      expect(view.result.current.draftReady).toBe(false)
      expect(view.result.current.draftSaved).toBe(false)
      view.rerender({ enabled: true, revision: 3 })
      await act(async () => {})
      await act(async () => {
        await vi.advanceTimersByTimeAsync(
          COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
        )
      })
      expect(view.result.current.draftReady).toBe(true)
      expect(view.result.current.draftSaved).toBe(false)
      expect(value).toBe("ACKNOWLEDGED A")
      expect(bucketMocks.setMock).toHaveBeenCalledOnce()
      value = "CURRENT NEW ACKNOWLEDGEMENT"
      view.rerender({ enabled: true, revision: 4 })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(
          COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
        )
      })
      expect(view.result.current.draftSaved).toBe(true)
      expect(bucketMocks.setMock).toHaveBeenCalledTimes(2)
    }
  )

  it.each([false, true])(
    "retires a saved receipt when current authority fails and does not revive it on the unchanged identity (tabScoped=%s)",
    async (tabScoped) => {
      let value = ""
      let current = true
      const isCurrent = () => current
      const view = renderHook(
        ({ revision }) => {
          void revision
          return useDraftPersistence({
            storageKey: "receipt:authority",
            tabScoped,
            isCurrent,
            getValue: () => value,
            setValue: (next) => {
              value = next
            }
          })
        },
        { initialProps: { revision: 0 } }
      )
      await act(async () => {})
      value = "ACKNOWLEDGED CURRENT SOURCE"
      view.rerender({ revision: 1 })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(
          COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
        )
      })
      expect(view.result.current.draftSaved).toBe(true)
      current = false
      view.rerender({ revision: 2 })
      expect(view.result.current.draftReady).toBe(false)
      expect(view.result.current.draftSaved).toBe(false)
      current = true
      view.rerender({ revision: 3 })
      expect(view.result.current.draftReady).toBe(true)
      expect(view.result.current.draftSaved).toBe(false)
      expect(bucketMocks.setMock).toHaveBeenCalledOnce()
      value = "NEXT ACKNOWLEDGED CURRENT INPUT"
      view.rerender({ revision: 4 })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(
          COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
        )
      })
      expect(view.result.current.draftSaved).toBe(true)
      expect(bucketMocks.setMock).toHaveBeenCalledTimes(2)
    }
  )

  it.each([false, true])(
    "never adopts an earlier receipt while another owner's read is held or after stored text is hydrated (tabScoped=%s)",
    async (tabScoped) => {
      let value = ""
      const view = renderHook(
        ({ storageKey, revision }) => {
          void revision
          return useDraftPersistence({
            storageKey,
            tabScoped,
            getValue: () => value,
            setValue: (next) => {
              value = next
            }
          })
        },
        { initialProps: { storageKey: "receipt:A", revision: 0 } }
      )
      await act(async () => {})
      value = "A SAVED"
      view.rerender({ storageKey: "receipt:A", revision: 1 })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(
          COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
        )
      })
      expect(view.result.current.draftSaved).toBe(true)
      let resolve!: (record: DraftRecord) => void
      bucketMocks.getMock.mockImplementationOnce(
        () =>
          new Promise<DraftRecord>((finish) => {
            resolve = finish
          })
      )
      view.rerender({ storageKey: "receipt:B", revision: 2 })
      expect(view.result.current.draftReady).toBe(false)
      expect(view.result.current.draftSaved).toBe(false)
      await act(async () => {
        resolve({ value: "B STORED TEXT", updatedAt: Date.now() })
      })
      expect(view.result.current.draftReady).toBe(true)
      expect(view.result.current.draftSaved).toBe(false)
      expect(value).toBe("B STORED TEXT")
      expect(bucketMocks.setMock).toHaveBeenCalledOnce()
    }
  )

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
