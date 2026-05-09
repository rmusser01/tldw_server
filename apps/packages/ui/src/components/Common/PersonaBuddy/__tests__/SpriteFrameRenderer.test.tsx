import React from "react"
import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { SpriteFrameRenderer } from "../SpriteFrameRenderer"
import type {
  PersonaVisualAsset,
  PersonaVisualManifest
} from "@/types/persona-visuals"

const assets: Record<string, PersonaVisualAsset> = {
  "idle-1": {
    id: "idle-1",
    url: "/assets/idle-1.png",
    mime_type: "image/png",
    asset_role: "frame",
    width: 32,
    height: 32
  },
  "idle-2": {
    id: "idle-2",
    url: "/assets/idle-2.png",
    mime_type: "image/png",
    asset_role: "frame",
    width: 32,
    height: 32
  },
  "sheet-1": {
    id: "sheet-1",
    url: "/assets/sheet.png",
    mime_type: "image/png",
    asset_role: "sprite_sheet",
    width: 64,
    height: 64
  }
}

const baseManifest = (
  overrides: Partial<PersonaVisualManifest> = {}
): PersonaVisualManifest => ({
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {
    idle: { animation_id: "idle" }
  },
  animations: {
    idle: {
      frames: [{ asset_id: "idle-1", duration_ms: 100 }],
      frame_rate: 1
    }
  },
  ...overrides
})

const currentFrame = () => screen.getByTestId("persona-visual-frame")

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("SpriteFrameRenderer", () => {
  it("renders the first frame for a state", () => {
    render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
      />
    )

    expect(currentFrame()).toHaveAttribute("src", expect.stringContaining("/assets/idle-1.png"))
    expect(currentFrame()).toHaveAttribute("data-visual-state", "idle")
  })

  it("uses preview_frame before the animation interval advances", () => {
    vi.useFakeTimers()
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: {
            idle: {
              preview_frame: 1,
              frames: [
                { asset_id: "idle-1", duration_ms: 100 },
                { asset_id: "idle-2", duration_ms: 100 }
              ]
            }
          }
        })}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
      />
    )

    expect(currentFrame()).toHaveAttribute("src", expect.stringContaining("/assets/idle-2.png"))
  })

  it("respects explicit frame order instead of asset id or upload order", () => {
    vi.useFakeTimers()
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: {
            idle: {
              frames: [
                { asset_id: "idle-2", duration_ms: 50 },
                { asset_id: "idle-1", duration_ms: 50 }
              ]
            }
          }
        })}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
      />
    )

    expect(currentFrame()).toHaveAttribute("src", expect.stringContaining("/assets/idle-2.png"))

    act(() => {
      vi.advanceTimersByTime(50)
    })

    expect(currentFrame()).toHaveAttribute("src", expect.stringContaining("/assets/idle-1.png"))
  })

  it("renders sprite-sheet region frames as cropped background regions", () => {
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: {
            idle: {
              frames: [
                {
                  asset_id: "sheet-1",
                  region: { x: 8, y: 12, width: 16, height: 24 },
                  duration_ms: 100
                }
              ]
            }
          }
        })}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
      />
    )

    expect(currentFrame()).toHaveStyle({
      backgroundImage: "url(/assets/sheet.png)",
      backgroundPosition: "-8px -12px",
      backgroundSize: "64px 64px",
      width: "16px",
      height: "24px"
    })
    expect(currentFrame()).toHaveAttribute("data-visual-state", "idle")
  })

  it("falls back to idle when the requested state is missing", () => {
    render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        state="speaking"
        fallbackLabel="Buddy"
      />
    )

    expect(currentFrame()).toHaveAttribute("src", expect.stringContaining("/assets/idle-1.png"))
    expect(currentFrame()).toHaveAttribute("data-visual-state", "speaking")
  })

  it("calls onRenderError when no state can resolve", () => {
    const onRenderError = vi.fn()
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          states: {},
          animations: {}
        })}
        assets={assets}
        state="speaking"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(onRenderError).toHaveBeenCalledWith("missing_animation")
  })
})
