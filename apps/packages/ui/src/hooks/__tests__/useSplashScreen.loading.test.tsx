// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { SplashCard } from "@/components/Common/SplashScreen/engine/types"

const card: SplashCard = {
  name: "default",
  effect: null,
  subtitle: "Original message"
}
let cardLoads = 0,
  messageLoads = 0
let contentReady: Promise<void> = Promise.resolve()
let disabled = false
let enabledNames: string[] | undefined
let duration: number | undefined

beforeEach(() => {
  vi.resetModules()
  cardLoads = messageLoads = 0
  contentReady = Promise.resolve()
  disabled = false
  enabledNames = undefined
  duration = undefined
  vi.doMock("@/hooks/useSetting", () => ({
    useSetting: (setting: { key: string; defaultValue: unknown }) => [
      setting.key === "tldw:splash:disabled"
        ? disabled
        : setting.key === "tldw:splash:enabledCards"
          ? enabledNames ?? setting.defaultValue
          : setting.key === "tldw:splash:durationSeconds"
            ? duration ?? setting.defaultValue
            : setting.defaultValue,
      async (value: boolean) => {
        disabled = value
      }
    ]
  }))
  vi.doMock("@/data/splash-cards", async () => {
    cardLoads++
    await contentReady
    return {
      DEFAULT_SPLASH_CARD_NAMES: ["default"],
      randomSplashCard: () => card
    }
  })
  vi.doMock("@/data/splash-messages", async () => {
    messageLoads++
    await contentReady
    return { randomSplashMessage: () => "Fallback message" }
  })
})
afterEach(() => {
  vi.restoreAllMocks()
})

const held = () => {
  let release!: () => void
  contentReady = new Promise<void>((resolve) => {
    release = resolve
  })
  return release
}

describe("optional splash content loading", () => {
  it("does not import content when hidden or disabled", async () => {
    const { useSplashScreen } = await import("../useSplashScreen")
    const hook = renderHook(() => useSplashScreen())
    expect([cardLoads, messageLoads]).toEqual([0, 0])
    act(() => hook.result.current.setDisabled(true))
    hook.rerender()
    act(() => hook.result.current.show())
    expect([cardLoads, messageLoads]).toEqual([0, 0])
  })

  it.each(["dismiss", "disable", "unmount"])(
    "retires a held request on %s",
    async (action) => {
      const { useSplashScreen } = await import("../useSplashScreen")
      const hook = renderHook(() => useSplashScreen())
      const release = held()
      act(() => hook.result.current.show())
      await waitFor(() => expect([cardLoads, messageLoads]).toEqual([1, 1]))
      expect(hook.result.current.visible).toBe(false)
      if (action === "dismiss") act(() => hook.result.current.dismiss())
      if (action === "disable") act(() => hook.result.current.setDisabled(true))
      if (action === "unmount") hook.unmount()
      await act(async () => {
        release()
        await Promise.all([
          import("@/data/splash-cards"),
          import("@/data/splash-messages")
        ])
      })
      expect(hook.result.current.visible).toBe(false)
    }
  )

  it("keeps the later forced request when an older request was dismissed", async () => {
    const { useSplashScreen } = await import("../useSplashScreen")
    const hook = renderHook(() => useSplashScreen())
    const release = held()
    act(() => hook.result.current.show())
    await waitFor(() => expect(cardLoads).toBe(1))
    act(() => {
      hook.result.current.dismiss()
      hook.result.current.setDisabled(true)
    })
    hook.rerender()
    act(() => hook.result.current.show({ force: true }))
    await act(async () => {
      release()
      await Promise.all([
        import("@/data/splash-cards"),
        import("@/data/splash-messages")
      ])
    })
    await waitFor(() => expect(hook.result.current.visible).toBe(true))
    expect(hook.result.current.card?.name).toBe("default")
    expect(hook.result.current.message).toBe("Original message")
  })

  it.each(["equivalent", "cards", "duration-undo", "disable-undo"])(
    "holds original preference ownership through %s",
    async (change) => {
      const { useSplashScreen } = await import("../useSplashScreen")
      const hook = renderHook(() => useSplashScreen())
      const release = held()
      act(() => hook.result.current.show({ force: true }))
      await waitFor(() => expect(cardLoads).toBe(1))
      if (change === "equivalent") {
        const { DEFAULT_SPLASH_CARD_NAMES } = await import(
          "@/data/splash-card-names"
        )
        enabledNames = [...DEFAULT_SPLASH_CARD_NAMES]
        hook.rerender()
      } else if (change === "cards") {
        enabledNames = ["matrix"]
        hook.rerender()
      } else if (change === "duration-undo") {
        duration = 7
        hook.rerender()
        duration = undefined
        hook.rerender()
      } else {
        disabled = true
        hook.rerender()
        disabled = false
        hook.rerender()
      }
      await act(async () => {
        release()
        await Promise.all([
          import("@/data/splash-cards"),
          import("@/data/splash-messages")
        ])
      })
      if (change === "equivalent")
        await waitFor(() => expect(hook.result.current.visible).toBe(true))
      else expect(hook.result.current.visible).toBe(false)
    }
  )

  it("records a content failure without displaying or rejecting the event", async () => {
    const { useSplashScreen } = await import("../useSplashScreen")
    const warning = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined)
    vi.doMock("@/data/splash-cards", () => {
      throw new Error("Optional chunk unavailable")
    })
    const hook = renderHook(() => useSplashScreen())
    act(() => hook.result.current.show())
    await waitFor(() => expect(warning).toHaveBeenCalled())
    expect(hook.result.current.visible).toBe(false)
  })
})
