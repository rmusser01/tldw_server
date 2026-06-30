import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getCurrentUserProfile: vi.fn(),
  updateCurrentUserProfile: vi.fn(),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getCurrentUserProfile: (...args: unknown[]) =>
      mocks.getCurrentUserProfile(...args),
    updateCurrentUserProfile: (...args: unknown[]) =>
      mocks.updateCurrentUserProfile(...args),
  },
}))

import {
  COMPOSER_ENABLED_PREFERENCE_KEY,
  COMPOSER_ENABLED_PROFILE_KEY,
  useComposerEnabledPreference,
} from "../hooks/useComposerEnabledPreference"

describe("useComposerEnabledPreference", () => {
  beforeEach(() => {
    window.localStorage.clear()
    mocks.getCurrentUserProfile.mockReset()
    mocks.updateCurrentUserProfile.mockReset()
    mocks.getCurrentUserProfile.mockResolvedValue({ preferences: {} })
    mocks.updateCurrentUserProfile.mockResolvedValue({})
  })

  it("defaults to false when no preference is stored", () => {
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe(false)
  })

  it("reads '1' as true from localStorage", () => {
    window.localStorage.setItem(COMPOSER_ENABLED_PREFERENCE_KEY, "1")
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe(true)
  })

  it("anything other than '1' reads as false", () => {
    window.localStorage.setItem(COMPOSER_ENABLED_PREFERENCE_KEY, "0")
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe(false)
  })

  it("setEnabled(true) updates state + writes '1' to localStorage", () => {
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    act(() => {
      result.current[1](true)
    })
    expect(result.current[0]).toBe(true)
    expect(
      window.localStorage.getItem(COMPOSER_ENABLED_PREFERENCE_KEY)
    ).toBe("1")
  })

  it("setEnabled(false) writes '0' to localStorage", () => {
    window.localStorage.setItem(COMPOSER_ENABLED_PREFERENCE_KEY, "1")
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    act(() => {
      result.current[1](false)
    })
    expect(result.current[0]).toBe(false)
    expect(
      window.localStorage.getItem(COMPOSER_ENABLED_PREFERENCE_KEY)
    ).toBe("0")
  })

  it("hydrates true from server profile on mount", async () => {
    mocks.getCurrentUserProfile.mockResolvedValue({
      preferences: { [COMPOSER_ENABLED_PROFILE_KEY]: true },
    })
    const { result } = renderHook(() => useComposerEnabledPreference())
    await waitFor(() => {
      expect(result.current[0]).toBe(true)
    })
    expect(
      window.localStorage.getItem(COMPOSER_ENABLED_PREFERENCE_KEY)
    ).toBe("1")
  })

  it("ignores non-boolean server values", async () => {
    mocks.getCurrentUserProfile.mockResolvedValue({
      preferences: { [COMPOSER_ENABLED_PROFILE_KEY]: "yes" },
    })
    const { result } = renderHook(() => useComposerEnabledPreference())
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })
    expect(result.current[0]).toBe(false)
  })

  it("survives a server fetch failure", async () => {
    mocks.getCurrentUserProfile.mockRejectedValue(new Error("network down"))
    window.localStorage.setItem(COMPOSER_ENABLED_PREFERENCE_KEY, "1")
    const { result } = renderHook(() => useComposerEnabledPreference())
    expect(result.current[0]).toBe(true)
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })
    expect(result.current[0]).toBe(true)
  })

  it("setEnabled fires PATCH with the catalog key + boolean value", async () => {
    const { result } = renderHook(() => useComposerEnabledPreference())
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })
    act(() => {
      result.current[1](true)
    })
    expect(mocks.updateCurrentUserProfile).toHaveBeenCalledWith({
      updates: [{ key: COMPOSER_ENABLED_PROFILE_KEY, value: true }],
    })
  })

  it("setEnabled does not throw when PATCH rejects", async () => {
    mocks.updateCurrentUserProfile.mockRejectedValue(
      new Error("server angry")
    )
    const { result } = renderHook(() => useComposerEnabledPreference())
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })
    expect(() =>
      act(() => {
        result.current[1](true)
      })
    ).not.toThrow()
    expect(result.current[0]).toBe(true)
  })

  it("disableServerSync skips both GET and PATCH calls", () => {
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    act(() => {
      result.current[1](true)
    })
    expect(mocks.getCurrentUserProfile).not.toHaveBeenCalled()
    expect(mocks.updateCurrentUserProfile).not.toHaveBeenCalled()
  })

  it("does not let a late server hydrate overwrite a fresh local toggle", async () => {
    let resolveProfile:
      | ((value: { preferences: Record<string, unknown> }) => void)
      | undefined
    const pendingProfile = new Promise<{ preferences: Record<string, unknown> }>(
      (resolve) => {
        resolveProfile = resolve
      }
    )
    mocks.getCurrentUserProfile.mockReturnValue(pendingProfile)

    const { result } = renderHook(() => useComposerEnabledPreference())

    act(() => {
      result.current[1](true)
    })

    await act(async () => {
      resolveProfile?.({
        preferences: { [COMPOSER_ENABLED_PROFILE_KEY]: false },
      })
      await pendingProfile
    })

    expect(result.current[0]).toBe(true)
    expect(
      window.localStorage.getItem(COMPOSER_ENABLED_PREFERENCE_KEY)
    ).toBe("1")
  })

  it("picks up cross-tab storage events", () => {
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe(false)

    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: COMPOSER_ENABLED_PREFERENCE_KEY,
          newValue: "1",
          oldValue: null,
          storageArea: window.localStorage,
        })
      )
    })
    expect(result.current[0]).toBe(true)
  })

  it("ignores storage events for unrelated keys", () => {
    const { result } = renderHook(() =>
      useComposerEnabledPreference({ disableServerSync: true })
    )
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "some-other-key",
          newValue: "1",
          oldValue: null,
          storageArea: window.localStorage,
        })
      )
    })
    expect(result.current[0]).toBe(false)
  })
})
