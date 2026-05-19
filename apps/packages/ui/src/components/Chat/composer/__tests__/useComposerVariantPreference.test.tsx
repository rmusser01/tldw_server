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
  COMPOSER_VARIANT_PREFERENCE_KEY,
  COMPOSER_VARIANT_PROFILE_KEY,
  useComposerVariantPreference,
} from "../hooks/useComposerVariantPreference"

describe("useComposerVariantPreference", () => {
  beforeEach(() => {
    window.localStorage.clear()
    mocks.getCurrentUserProfile.mockReset()
    mocks.updateCurrentUserProfile.mockReset()
    mocks.getCurrentUserProfile.mockResolvedValue({ preferences: {} })
    mocks.updateCurrentUserProfile.mockResolvedValue({})
  })

  it("defaults to v1 when no preference is stored", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe("v1")
  })

  it("reads a valid stored preference", () => {
    window.localStorage.setItem(COMPOSER_VARIANT_PREFERENCE_KEY, "v3")
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe("v3")
  })

  it("falls back to v1 when the stored value is unrecognized", () => {
    window.localStorage.setItem(COMPOSER_VARIANT_PREFERENCE_KEY, "v99")
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe("v1")
  })

  it("setVariant updates state and persists to localStorage", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )

    act(() => {
      result.current[1]("v5")
    })

    expect(result.current[0]).toBe("v5")
    expect(window.localStorage.getItem(COMPOSER_VARIANT_PREFERENCE_KEY)).toBe(
      "v5"
    )
  })

  it("ignores attempts to set an unknown variant", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )

    act(() => {
      // @ts-expect-error — testing runtime guard
      result.current[1]("v99")
    })

    expect(result.current[0]).toBe("v1")
    expect(window.localStorage.getItem(COMPOSER_VARIANT_PREFERENCE_KEY)).toBeNull()
  })

  it("accepts an explicit default override", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({
        defaultVariant: "v3",
        disableServerSync: true,
      })
    )
    expect(result.current[0]).toBe("v3")
  })

  it("does not throw when localStorage.setItem rejects", () => {
    const originalSetItem = window.localStorage.setItem
    window.localStorage.setItem = () => {
      throw new Error("QuotaExceededError")
    }

    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )

    expect(() =>
      act(() => {
        result.current[1]("v5")
      })
    ).not.toThrow()

    window.localStorage.setItem = originalSetItem
  })

  // --- Server-sync layer ---

  it("hydrates from server profile on mount when sync is enabled", async () => {
    mocks.getCurrentUserProfile.mockResolvedValue({
      preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v5" },
    })

    const { result } = renderHook(() => useComposerVariantPreference())

    await waitFor(() => {
      expect(result.current[0]).toBe("v5")
    })
    expect(window.localStorage.getItem(COMPOSER_VARIANT_PREFERENCE_KEY)).toBe(
      "v5"
    )
  })

  it("ignores server values that are not a valid variant", async () => {
    mocks.getCurrentUserProfile.mockResolvedValue({
      preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "garbage" },
    })

    const { result } = renderHook(() => useComposerVariantPreference())

    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })
    expect(result.current[0]).toBe("v1")
  })

  it("survives a server fetch failure without crashing", async () => {
    mocks.getCurrentUserProfile.mockRejectedValue(new Error("network down"))
    window.localStorage.setItem(COMPOSER_VARIANT_PREFERENCE_KEY, "v3")

    const { result } = renderHook(() => useComposerVariantPreference())

    expect(result.current[0]).toBe("v3")
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })
    expect(result.current[0]).toBe("v3")
  })

  it("setVariant fires PATCH with the catalog key + value", async () => {
    const { result } = renderHook(() => useComposerVariantPreference())
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })

    act(() => {
      result.current[1]("v5")
    })

    expect(mocks.updateCurrentUserProfile).toHaveBeenCalledWith({
      updates: [{ key: COMPOSER_VARIANT_PROFILE_KEY, value: "v5" }],
    })
  })

  it("setVariant does not throw when the PATCH rejects", async () => {
    mocks.updateCurrentUserProfile.mockRejectedValue(
      new Error("server angry")
    )
    const { result } = renderHook(() => useComposerVariantPreference())
    await waitFor(() => {
      expect(mocks.getCurrentUserProfile).toHaveBeenCalled()
    })

    expect(() =>
      act(() => {
        result.current[1]("v5")
      })
    ).not.toThrow()
    expect(result.current[0]).toBe("v5")
  })

  it("disableServerSync skips both GET and PATCH calls", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    act(() => {
      result.current[1]("v3")
    })
    expect(mocks.getCurrentUserProfile).not.toHaveBeenCalled()
    expect(mocks.updateCurrentUserProfile).not.toHaveBeenCalled()
  })

  it("does not let a late server hydrate overwrite a fresh local variant pick", async () => {
    let resolveProfile:
      | ((value: { preferences: Record<string, unknown> }) => void)
      | undefined
    const pendingProfile = new Promise<{ preferences: Record<string, unknown> }>(
      (resolve) => {
        resolveProfile = resolve
      }
    )
    mocks.getCurrentUserProfile.mockReturnValue(pendingProfile)

    const { result } = renderHook(() => useComposerVariantPreference())

    act(() => {
      result.current[1]("v5")
    })

    await act(async () => {
      resolveProfile?.({
        preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v3" },
      })
      await pendingProfile
    })

    expect(result.current[0]).toBe("v5")
    expect(window.localStorage.getItem(COMPOSER_VARIANT_PREFERENCE_KEY)).toBe(
      "v5"
    )
  })

  // --- Cross-tab live sync ---

  it("picks up variant changes from another tab via storage event", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    expect(result.current[0]).toBe("v1")

    // Simulate another tab writing a new value to localStorage.
    act(() => {
      window.localStorage.setItem(COMPOSER_VARIANT_PREFERENCE_KEY, "v3")
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: COMPOSER_VARIANT_PREFERENCE_KEY,
          newValue: "v3",
          oldValue: null,
          storageArea: window.localStorage,
        })
      )
    })

    expect(result.current[0]).toBe("v3")
  })

  it("ignores storage events for unrelated keys", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "some-other-key",
          newValue: "v5",
          oldValue: null,
          storageArea: window.localStorage,
        })
      )
    })
    expect(result.current[0]).toBe("v1")
  })

  it("ignores storage events with invalid variant values", () => {
    const { result } = renderHook(() =>
      useComposerVariantPreference({ disableServerSync: true })
    )
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: COMPOSER_VARIANT_PREFERENCE_KEY,
          newValue: "garbage",
          oldValue: null,
          storageArea: window.localStorage,
        })
      )
    })
    expect(result.current[0]).toBe("v1")
  })
})
