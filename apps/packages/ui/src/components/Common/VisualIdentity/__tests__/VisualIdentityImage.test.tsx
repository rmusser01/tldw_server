import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { VisualIdentityImage } from "../VisualIdentityImage"

const mockMatchMedia = (matches: boolean) => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("VisualIdentityImage", () => {
  beforeEach(() => {
    mockMatchMedia(false)
  })

  it("renders the animated asset when reduced motion is not enabled", () => {
    render(
      <VisualIdentityImage
        assetUrl="/animated.webp"
        previewUrl="/preview.png"
        isAnimated
        alt="Happy sprite"
      />
    )

    expect(screen.getByRole("img", { name: "Happy sprite" })).toHaveAttribute(
      "src",
      "/animated.webp"
    )
  })

  it("renders still preview when reduced motion is enabled", () => {
    mockMatchMedia(true)

    render(
      <VisualIdentityImage
        assetUrl="/animated.webp"
        previewUrl="/preview.png"
        isAnimated
        alt="Animated sprite"
      />
    )

    expect(screen.getByRole("img", { name: "Animated sprite" })).toHaveAttribute(
      "src",
      "/preview.png"
    )
  })
})
