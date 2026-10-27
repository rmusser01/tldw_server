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
  COMPOSER_ENABLED_PROFILE_KEY,
  useComposerEnabledPreference,
} from "../hooks/useComposerEnabledPreference"
import {
  COMPOSER_VARIANT_PROFILE_KEY,
  useComposerVariantPreference,
} from "../hooks/useComposerVariantPreference"

type Profile = { preferences: Record<string, unknown> }

const deferredProfile = () => {
  let resolve: (value: Profile) => void = () => {}
  const promise = new Promise<Profile>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

// The chat composer mounts both preference hooks together.
const useComposerPreferences = () => ({
  enabled: useComposerEnabledPreference()[0],
  variant: useComposerVariantPreference()[0],
})

describe("composer profile preference reads (XP-16)", () => {
  beforeEach(() => {
    window.localStorage.clear()
    mocks.getCurrentUserProfile.mockReset()
    mocks.updateCurrentUserProfile.mockReset()
    mocks.updateCurrentUserProfile.mockResolvedValue({})
  })

  it("reads the profile once when the composer mounts both preference hooks", async () => {
    const read = deferredProfile()
    mocks.getCurrentUserProfile.mockReturnValue(read.promise)

    const { result } = renderHook(() => useComposerPreferences())

    await act(async () => {
      read.resolve({
        preferences: {
          [COMPOSER_ENABLED_PROFILE_KEY]: true,
          [COMPOSER_VARIANT_PROFILE_KEY]: "v5",
        },
      })
      await read.promise
    })

    expect(result.current).toEqual({ enabled: true, variant: "v5" })
    expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(1)
  })

  it("reads the profile again for a composer that mounts after the first read finished", async () => {
    mocks.getCurrentUserProfile.mockResolvedValue({ preferences: {} })
    const first = renderHook(() => useComposerPreferences())
    await waitFor(() =>
      expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(1)
    )
    await act(async () => {})
    first.unmount()

    mocks.getCurrentUserProfile.mockResolvedValue({
      preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v3" },
    })
    const second = renderHook(() => useComposerPreferences())

    await waitFor(() => expect(second.result.current.variant).toBe("v3"))
    expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(2)
  })

  it("does not hand a read from before an account change to a composer for the new account", async () => {
    const oldAccount = deferredProfile()
    mocks.getCurrentUserProfile.mockReturnValueOnce(oldAccount.promise)
    renderHook(() => useComposerVariantPreference())

    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true },
        })
      )
    })

    mocks.getCurrentUserProfile.mockResolvedValueOnce({
      preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v3" },
    })
    const newAccount = renderHook(() => useComposerVariantPreference())

    await waitFor(() => expect(newAccount.result.current[0]).toBe("v3"))
    expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(2)

    await act(async () => {
      oldAccount.resolve({
        preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v5" },
      })
      await oldAccount.promise
    })
    expect(newAccount.result.current[0]).toBe("v3")
  })

  it("does not hand a read from before a preference change to a composer that mounts after it", async () => {
    const beforeChange = deferredProfile()
    mocks.getCurrentUserProfile.mockReturnValueOnce(beforeChange.promise)
    const first = renderHook(() => useComposerVariantPreference())

    act(() => {
      first.result.current[1]("v5")
    })

    mocks.getCurrentUserProfile.mockResolvedValueOnce({
      preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v5" },
    })
    const later = renderHook(() => useComposerVariantPreference())

    await waitFor(() =>
      expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(2)
    )
    await act(async () => {
      beforeChange.resolve({
        preferences: { [COMPOSER_VARIANT_PROFILE_KEY]: "v1" },
      })
      await beforeChange.promise
    })
    expect(later.result.current[0]).toBe("v5")
  })
})
