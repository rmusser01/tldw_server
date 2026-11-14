import React from "react"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => {
  const values = new Map<string, unknown>()
  const storage = {
    get: vi.fn(async (key: string) => (values.has(key) ? values.get(key) : undefined)),
    set: vi.fn(async (key: string, value: unknown) => {
      values.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      values.delete(key)
    })
  }

  return {
    values,
    storage
  }
})

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => mocks.storage
}))

vi.mock("@plasmohq/storage/hook", async () => {
  const ReactModule =
    await vi.importActual<typeof import("react")>("react")

  return {
    useStorage: (
      config: string | { key: string; instance?: unknown },
      defaultValue: unknown
    ) => {
      const key = typeof config === "string" ? config : config.key
      const [value, setRenderValue] = ReactModule.useState(defaultValue)
      const [isLoading, setIsLoading] = ReactModule.useState(true)

      ReactModule.useEffect(() => {
        let cancelled = false

        Promise.resolve().then(() => {
          if (cancelled) return
          setRenderValue(
            mocks.values.has(key) ? mocks.values.get(key) : defaultValue
          )
          setIsLoading(false)
        })

        return () => {
          cancelled = true
        }
      }, [defaultValue, key])

      const setValue = async (next: unknown) => {
        const previous =
          mocks.values.has(key) ? mocks.values.get(key) : defaultValue
        const resolved =
          typeof next === "function"
            ? (next as (value: unknown) => unknown)(previous)
            : next
        mocks.values.set(key, resolved)
        setRenderValue(resolved)
        setIsLoading(false)
      }

      return [value, setValue, { isLoading, setRenderValue }] as const
    }
  }
})

import {
  CHAT_MOOD_BADGE_DEFAULT,
  CHAT_MOOD_BADGE_MIGRATION_STORAGE_KEY,
  CHAT_MOOD_BADGE_STORAGE_KEY,
  useChatMoodBadgePreference
} from "../useChatMoodBadgePreference"

describe("useChatMoodBadgePreference", () => {
  beforeEach(() => {
    mocks.values.clear()
    mocks.storage.get.mockClear()
    mocks.storage.set.mockClear()
    mocks.storage.remove.mockClear()
  })

  it("preserves a stored enabled preference during migration without flashing the hydrated value", async () => {
    mocks.values.set(CHAT_MOOD_BADGE_STORAGE_KEY, true)

    const seenValues: boolean[] = []
    const { result, unmount } = renderHook(() => {
      const state = useChatMoodBadgePreference()
      React.useEffect(() => {
        seenValues.push(state[0])
      }, [state[0]])
      return state
    })

    expect(result.current[0]).toBe(CHAT_MOOD_BADGE_DEFAULT)

    await waitFor(() => {
      expect(mocks.values.get(CHAT_MOOD_BADGE_STORAGE_KEY)).toBe(true)
      expect(mocks.values.get(CHAT_MOOD_BADGE_MIGRATION_STORAGE_KEY)).toBe(true)
      expect(result.current[0]).toBe(false)
    })

    expect(seenValues).not.toContain(true)
    unmount()
    const remounted = renderHook(() => useChatMoodBadgePreference())
    await waitFor(() => expect(remounted.result.current[0]).toBe(true))
    expect(mocks.storage.set).not.toHaveBeenCalledWith(CHAT_MOOD_BADGE_STORAGE_KEY, false)
  })

  it("defaults to hidden when the preference has never been stored", async () => {
    const { result } = renderHook(() => useChatMoodBadgePreference())

    expect(result.current[0]).toBe(false)

    await waitFor(() => {
      expect(mocks.values.get(CHAT_MOOD_BADGE_STORAGE_KEY)).toBe(false)
      expect(mocks.values.get(CHAT_MOOD_BADGE_MIGRATION_STORAGE_KEY)).toBe(true)
    })
  })

  it("respects an enabled preference after the migration has already completed", async () => {
    mocks.values.set(CHAT_MOOD_BADGE_STORAGE_KEY, true)
    mocks.values.set(CHAT_MOOD_BADGE_MIGRATION_STORAGE_KEY, true)

    const { result } = renderHook(() => useChatMoodBadgePreference())

    await waitFor(() => {
      expect(result.current[0]).toBe(true)
    })

    expect(mocks.storage.set).not.toHaveBeenCalledWith(
      CHAT_MOOD_BADGE_STORAGE_KEY,
      false
    )
  })
})
