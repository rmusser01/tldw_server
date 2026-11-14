import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import SplashOverlay from "../SplashOverlay"

function createCard(
  overrides?: Partial<{
    name: string
    effect: string | null
    asciiArt: string
    title: string
    duration: number
  }>
) {
  return {
    name: overrides?.name ?? "test-card",
    effect: overrides?.effect ?? null,
    asciiArt: overrides?.asciiArt ?? "default_splash",
    title: overrides?.title,
    duration: overrides?.duration
  }
}

function mockMatchMedia(reducedMotion: boolean): void {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches:
        query === "(prefers-reduced-motion: reduce)" ? reducedMotion : false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("Splash Stage 5 overlay behavior", () => {
  const originalMatchMedia = window.matchMedia

  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      writable: true,
      value: originalMatchMedia
    })
  })

  it("cancels the old fade callback when a newer card replaces it", () => {
    const dismiss = vi.fn()
    const view = render(
      <SplashOverlay
        card={createCard({ name: "same-card", duration: 10_000 })}
        message="Old splash"
        onDismiss={dismiss}
      />
    )
    fireEvent.click(screen.getByRole("dialog", { name: "Splash screen" }))
    view.rerender(
      <SplashOverlay
        card={createCard({ name: "same-card", duration: 10_000 })}
        message="New splash"
        onDismiss={dismiss}
      />
    )
    act(() => vi.advanceTimersByTime(300))
    expect(dismiss).not.toHaveBeenCalled()
    expect(
      screen.getByRole("dialog", { name: "Splash screen" }).style.opacity
    ).toBe("1")
    fireEvent.click(screen.getByRole("dialog", { name: "Splash screen" }))
    act(() => vi.advanceTimersByTime(300))
    expect(dismiss).toHaveBeenCalledOnce()
  })

  it("cancels a pending fade callback on unmount", () => {
    const dismiss = vi.fn()
    const view = render(
      <SplashOverlay
        card={createCard()}
        message="Original splash"
        onDismiss={dismiss}
      />
    )
    fireEvent.click(screen.getByRole("dialog", { name: "Splash screen" }))
    view.unmount()
    act(() => vi.advanceTimersByTime(300))
    expect(dismiss).not.toHaveBeenCalled()
  })

  it("auto-dismiss uses fade path before unmount callback", () => {
    const onDismiss = vi.fn()

    render(
      <SplashOverlay
        card={createCard({ duration: 1000 })}
        message="Loading..."
        onDismiss={onDismiss}
      />
    )

    const dialog = screen.getByRole("dialog", { name: "Splash screen" })

    act(() => {
      vi.advanceTimersByTime(1000)
    })

    expect(dialog.style.opacity).toBe("0")
    expect(onDismiss).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(299)
    })
    expect(onDismiss).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(onDismiss).toHaveBeenCalledTimes(1)
  })

  it("dismiss remains deterministic with rapid click + keypress", () => {
    const onDismiss = vi.fn()

    render(
      <SplashOverlay
        card={createCard({ duration: 10_000 })}
        message="Loading..."
        onDismiss={onDismiss}
      />
    )

    fireEvent.click(screen.getByRole("dialog", { name: "Splash screen" }))
    fireEvent.keyDown(window, { key: "Escape" })

    act(() => {
      vi.advanceTimersByTime(300)
    })

    expect(onDismiss).toHaveBeenCalledTimes(1)
  })

  it("stops intercepting pointer events as soon as fade-out begins", () => {
    const onDismiss = vi.fn()

    render(
      <SplashOverlay
        card={createCard({ duration: 10_000 })}
        message="Dismiss me"
        onDismiss={onDismiss}
      />
    )

    const dialog = screen.getByRole("dialog", { name: "Splash screen" })
    fireEvent.click(dialog)

    expect(dialog.style.opacity).toBe("0")
    expect(dialog.style.pointerEvents).toBe("none")
    expect(onDismiss).not.toHaveBeenCalled()
  })

  it("reduced-motion mode renders static HTML overlay without canvas", () => {
    mockMatchMedia(true)

    const { container } = render(
      <SplashOverlay
        card={createCard({ effect: "matrix_rain" })}
        message="Reduced motion mode"
        onDismiss={vi.fn()}
      />
    )

    expect(container.querySelector("canvas")).toBeNull()
    expect(screen.getByText("Reduced motion mode")).toBeInTheDocument()
    expect(
      screen.getByLabelText("TLDW Chatbook splash art")
    ).toBeInTheDocument()
  })

  it("uses theme-aware CSS variables for overlay readability", () => {
    const { container } = render(
      <SplashOverlay
        card={createCard({ title: "Theme Test" })}
        message="Theme message"
        onDismiss={vi.fn()}
      />
    )

    const dialogStyle =
      screen
        .getByRole("dialog", { name: "Splash screen" })
        .getAttribute("style") ?? ""
    expect(dialogStyle).toContain("var(--color-bg)")

    const artStyle = container.querySelector("pre")?.getAttribute("style") ?? ""
    expect(artStyle).toContain("var(--color-text)")

    const titleStyle =
      screen
        .getByRole("heading", { name: "Theme Test" })
        .getAttribute("style") ?? ""
    expect(titleStyle).toContain("var(--color-primary)")

    const messageStyle =
      screen.getByText("Theme message").getAttribute("style") ?? ""
    expect(messageStyle).toContain("var(--color-text-muted)")
  })
})
