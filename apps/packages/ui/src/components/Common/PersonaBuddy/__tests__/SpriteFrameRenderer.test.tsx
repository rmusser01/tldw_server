import React from "react"
import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { SpriteFrameRenderer } from "../SpriteFrameRenderer"
import type {
  PersonaVisualAsset,
  PersonaVisualManifest
} from "@/types/persona-visuals"

const buildAsset = (
  id: string,
  overrides: Partial<PersonaVisualAsset> = {}
): PersonaVisualAsset => ({
  id,
  pack_id: "pack-1",
  persona_id: "persona-1",
  asset_role: "frame",
  storage_key: `persona-1/pack-1/${id}`,
  url: `/assets/${id}.png`,
  original_filename: `${id}.png`,
  mime_type: "image/png",
  byte_size: 1,
  checksum_sha256: "0".repeat(64),
  width: null,
  height: null,
  duration_ms: null,
  provenance: "uploaded",
  created_at: "2026-08-23T00:00:00Z",
  last_modified: "2026-08-23T00:00:00Z",
  version: 1,
  ...overrides
})

const assets: Record<string, PersonaVisualAsset> = {
  "idle-1": buildAsset("idle-1", {
    width: 32,
    height: 32
  }),
  "idle-2": buildAsset("idle-2", {
    width: 32,
    height: 32
  }),
  "sheet-1": buildAsset("sheet-1", {
    asset_role: "sprite_sheet",
    url: "/assets/sheet.png",
    width: 64,
    height: 64
  })
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

  it("uses preview_frame for atlas animations that share one asset", () => {
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: {
            idle: {
              preview_frame: 1,
              frames: [
                {
                  asset_id: "sheet-1",
                  region: { x: 0, y: 0, width: 16, height: 16 },
                  duration_ms: 100
                },
                {
                  asset_id: "sheet-1",
                  region: { x: 16, y: 0, width: 16, height: 16 },
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
      backgroundPosition: "-16px 0px",
      width: "16px",
      height: "16px"
    })
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

  it("calls onRenderError when the resolved frame asset is missing", () => {
    const onRenderError = vi.fn()
    render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={{}}
        state="idle"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(onRenderError).toHaveBeenCalledWith("missing_asset")
  })

  it("clears onRenderError when a previous render failure becomes renderable", () => {
    const onRenderError = vi.fn()
    const view = render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={{}}
        state="idle"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(onRenderError).toHaveBeenLastCalledWith("missing_asset")

    view.rerender(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(onRenderError).toHaveBeenLastCalledWith(null)
    expect(currentFrame()).toHaveAttribute("src", expect.stringContaining("/assets/idle-1.png"))
  })

  it("reports unsupported regions before trying to render them", () => {
    const onRenderError = vi.fn()
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: {
            idle: {
              frames: [
                {
                  asset_id: "idle-1",
                  region: { x: 0, y: 0, width: 0, height: 32 }
                }
              ]
            }
          }
        })}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(onRenderError).toHaveBeenCalledWith("unsupported_region")
  })
})
