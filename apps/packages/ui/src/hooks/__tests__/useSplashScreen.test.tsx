import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"

import { clearSetting, setSetting } from "@/services/settings/registry"
import {
  SPLASH_DISABLED_SETTING,
  SPLASH_ENABLED_CARD_NAMES_SETTING,
  SPLASH_DURATION_SECONDS_SETTING
} from "@/services/settings/ui-settings"
import { useSplashScreen } from "../useSplashScreen"

describe("useSplashScreen", () => {
  beforeEach(async () => {
    await clearSetting(SPLASH_DISABLED_SETTING)
    await clearSetting(SPLASH_ENABLED_CARD_NAMES_SETTING)
    await clearSetting(SPLASH_DURATION_SECONDS_SETTING)
    localStorage.clear()
  })

  it("does not auto-show on mount", () => {
    const { result } = renderHook(() => useSplashScreen())

    expect(result.current.visible).toBe(false)
    expect(result.current.card).toBeNull()
  })

  it("shows a splash only when explicitly triggered", async () => {
    const { result } = renderHook(() => useSplashScreen())

    act(() => {
      result.current.show()
    })

    await waitFor(() => expect(result.current.visible).toBe(true))
    expect(result.current.card).not.toBeNull()
    expect(result.current.message.length).toBeGreaterThan(0)
  })

  it("blocks non-forced splash when disabled at runtime", () => {
    const { result } = renderHook(() => useSplashScreen())

    act(() => {
      result.current.setDisabled(true)
    })

    act(() => {
      result.current.show()
    })

    expect(result.current.visible).toBe(false)
    expect(result.current.card).toBeNull()
  })

  it("allows explicit forced splash when disabled", async () => {
    const { result } = renderHook(() => useSplashScreen())

    act(() => {
      result.current.setDisabled(true)
    })

    act(() => {
      result.current.show({ force: true })
    })

    await waitFor(() => expect(result.current.visible).toBe(true))
    expect(result.current.card).not.toBeNull()
  })

  it("persists disable toggle and hides an active splash", async () => {
    const { result } = renderHook(() => useSplashScreen())

    act(() => {
      result.current.show()
    })
    await waitFor(() => expect(result.current.visible).toBe(true))

    act(() => {
      result.current.setDisabled(true)
    })

    expect(result.current.disabled).toBe(true)
    expect(result.current.visible).toBe(false)
    await waitFor(() => {
      expect(localStorage.getItem("tldw_splash_disabled")).toBe("true")
    })
  })

  it("still shows splash in reduced-motion mode (static overlay path)", async () => {
    const originalMatchMedia = window.matchMedia
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      writable: true,
      value: ((query: string) => ({
        matches: query === "(prefers-reduced-motion: reduce)",
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false
      })) as typeof window.matchMedia
    })

    try {
      const { result } = renderHook(() => useSplashScreen())

      act(() => {
        result.current.show()
      })

      await waitFor(() => expect(result.current.visible).toBe(true))
      expect(result.current.card).not.toBeNull()
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        writable: true,
        value: originalMatchMedia
      })
    }
  })

  it("applies configured splash duration in seconds to card duration", async () => {
    await setSetting(SPLASH_DURATION_SECONDS_SETTING, 7)
    const { result } = renderHook(() => useSplashScreen())

    act(() => result.current.show())
    await waitFor(() => expect(result.current.card?.duration).toBe(7000))
  })

  it("clamps configured splash duration to supported range", async () => {
    await setSetting(SPLASH_DURATION_SECONDS_SETTING, 99 as unknown as number)
    const { result } = renderHook(() => useSplashScreen())

    act(() => result.current.show())
    await waitFor(() => expect(result.current.card?.duration).toBe(10_000))
  })
})
