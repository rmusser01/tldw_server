import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SpriteFrameRenderer } from "../SpriteFrameRenderer"
import type {
  PersonaVisualAsset,
  PersonaVisualManifest
} from "@/types/persona-visuals"

const assetLoader = vi.hoisted(() => ({
  acquire: vi.fn()
}))

vi.mock("@/services/persona-visual-assets", () => ({
  acquirePersonaVisualAsset: assetLoader.acquire
}))

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

beforeEach(() => {
  assetLoader.acquire.mockReset()
  assetLoader.acquire.mockImplementation(async (asset: PersonaVisualAsset) => ({
    url: `blob:${asset.id}`,
    mimeType: asset.mime_type,
    release: vi.fn()
  }))
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe("SpriteFrameRenderer", () => {
  it("renders the first frame for a state through an authenticated Blob handle", async () => {
    render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        state="idle"
        fallbackLabel="Buddy"
      />
    )

    await act(async () => {})
    expect(assetLoader.acquire).toHaveBeenCalledWith(
      expect.objectContaining({ id: "idle-1" }),
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )
    expect(currentFrame()).toHaveAttribute("src", "blob:idle-1")
    expect(currentFrame()).toHaveAttribute("data-visual-state", "idle")
  })

  it("uses preview_frame before the animation interval advances", async () => {
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

    await act(async () => {})
    expect(currentFrame()).toHaveAttribute("src", "blob:idle-2")
  })

  it("respects explicit frame order instead of asset id or upload order", async () => {
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

    await act(async () => {})
    expect(currentFrame()).toHaveAttribute("src", "blob:idle-2")

    await act(async () => {
      vi.advanceTimersByTime(50)
    })

    expect(currentFrame()).toHaveAttribute("src", "blob:idle-1")
  })

  it("renders sprite-sheet region frames as cropped background regions", async () => {
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

    await screen.findByTestId("persona-visual-frame")
    expect(currentFrame()).toHaveStyle({
      backgroundImage: "url(blob:sheet-1)",
      backgroundPosition: "-8px -12px",
      backgroundSize: "64px 64px",
      width: "16px",
      height: "24px"
    })
    expect(currentFrame()).toHaveAttribute("data-visual-state", "idle")
  })

  it("uses preview_frame for atlas animations that share one asset", async () => {
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

    await screen.findByTestId("persona-visual-frame")
    expect(currentFrame()).toHaveStyle({
      backgroundPosition: "-16px 0px",
      width: "16px",
      height: "16px"
    })
  })

  it("falls back to idle when the requested state is missing", async () => {
    render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        state="speaking"
        fallbackLabel="Buddy"
      />
    )

    await screen.findByTestId("persona-visual-frame")
    expect(currentFrame()).toHaveAttribute("src", "blob:idle-1")
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

  it("clears onRenderError when a previous render failure becomes renderable", async () => {
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

    await screen.findByTestId("persona-visual-frame")
    expect(onRenderError).toHaveBeenLastCalledWith(null)
    expect(currentFrame()).toHaveAttribute("src", "blob:idle-1")
  })

  it("releases a presented Blob and falls back when the next state is structurally invalid", async () => {
    const release = vi.fn()
    assetLoader.acquire.mockResolvedValue({
      url: "blob:idle-1",
      mimeType: "image/png",
      release
    })
    const view = render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        requestedState="idle"
        generation={1}
        fallbackLabel="Buddy"
      />
    )
    await screen.findByTestId("persona-visual-frame")

    view.rerender(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={{}}
        requestedState="idle"
        generation={2}
        fallbackLabel="Buddy"
      />
    )

    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(release).toHaveBeenCalledTimes(1)
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

  it("selects a static PNG and allocates no sprite timer under reduced motion", async () => {
    vi.useFakeTimers()
    const setTimeoutSpy = vi.spyOn(window, "setTimeout")
    render(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: {
            idle: {
              preview_asset_id: "idle-2",
              frames: [
                { asset_id: "idle-1", duration_ms: 20 },
                { asset_id: "idle-2", duration_ms: 20 }
              ]
            }
          }
        })}
        assets={assets}
        requestedState="idle"
        generation={2}
        reducedMotion
        fallbackLabel="Persona Buddy"
      />
    )

    await act(async () => {})
    expect(assetLoader.acquire).toHaveBeenCalledWith(
      expect.objectContaining({ id: "idle-2", mime_type: "image/png" }),
      expect.any(Object)
    )
    expect(currentFrame()).toHaveAttribute("src", "blob:idle-2")
    expect(setTimeoutSpy).not.toHaveBeenCalled()
  })

  it("rejects an animated static fallback under reduced motion", async () => {
    const onFailure = vi.fn()
    render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={{ "idle-1": buildAsset("idle-1", { mime_type: "image/gif" }) }}
        requestedState="idle"
        generation={1}
        reducedMotion
        fallbackLabel="Buddy"
        onFailure={onFailure}
      />
    )

    expect(onFailure).toHaveBeenCalledWith("static_asset_unsupported")
    expect(assetLoader.acquire).not.toHaveBeenCalled()
  })

  it("releases and clears a presented animated Blob when reduced motion rejects it", async () => {
    vi.useFakeTimers()
    const release = vi.fn()
    const onFailure = vi.fn()
    const animatedAssets = {
      "idle-1": buildAsset("idle-1", { mime_type: "image/gif" }),
      "idle-2": buildAsset("idle-2", { mime_type: "image/gif" })
    }
    assetLoader.acquire.mockResolvedValue({
      url: "blob:animated",
      mimeType: "image/gif",
      release
    })
    const manifest = baseManifest({
      animations: {
        idle: {
          frames: [
            { asset_id: "idle-1", duration_ms: 100 },
            { asset_id: "idle-2", duration_ms: 100 }
          ]
        }
      }
    })
    const view = render(
      <SpriteFrameRenderer
        manifest={manifest}
        assets={animatedAssets}
        requestedState="idle"
        generation={1}
        fallbackLabel="Buddy"
        onFailure={onFailure}
      />
    )
    await act(async () => {})
    expect(currentFrame()).toHaveAttribute("src", "blob:animated")
    expect(vi.getTimerCount()).toBe(1)

    view.rerender(
      <SpriteFrameRenderer
        manifest={manifest}
        assets={animatedAssets}
        requestedState="idle"
        generation={2}
        reducedMotion
        fallbackLabel="Buddy"
        onFailure={onFailure}
      />
    )
    await act(async () => {})

    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(release).toHaveBeenCalledTimes(1)
    expect(onFailure).toHaveBeenCalledWith("static_asset_unsupported")
    expect(vi.getTimerCount()).toBe(0)
  })

  it("keeps the old Blob visible until the new generation is ready and releases stale handles", async () => {
    let resolveSecond: ((handle: { url: string; mimeType: string; release: () => void }) => void) | null = null
    const firstRelease = vi.fn()
    const staleRelease = vi.fn()
    assetLoader.acquire
      .mockResolvedValueOnce({ url: "blob:first", mimeType: "image/png", release: firstRelease })
      .mockImplementationOnce(() => new Promise((resolve) => { resolveSecond = resolve }))

    const view = render(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        requestedState="idle"
        generation={1}
        fallbackLabel="Buddy"
      />
    )
    await screen.findByTestId("persona-visual-frame")
    expect(currentFrame()).toHaveAttribute("src", "blob:first")

    view.rerender(
      <SpriteFrameRenderer
        manifest={baseManifest({
          animations: { idle: { frames: [{ asset_id: "idle-2" }] } }
        })}
        assets={assets}
        requestedState="idle"
        generation={2}
        fallbackLabel="Buddy"
      />
    )
    expect(currentFrame()).toHaveAttribute("src", "blob:first")
    expect(firstRelease).not.toHaveBeenCalled()

    view.rerender(
      <SpriteFrameRenderer
        manifest={baseManifest()}
        assets={assets}
        requestedState="idle"
        generation={3}
        fallbackLabel="Buddy"
      />
    )
    await act(async () => {
      resolveSecond?.({ url: "blob:stale", mimeType: "image/png", release: staleRelease })
    })
    expect(currentFrame()).not.toHaveAttribute("src", "blob:stale")
    expect(staleRelease).toHaveBeenCalledTimes(1)
    expect(firstRelease).toHaveBeenCalledTimes(1)
  })
})
