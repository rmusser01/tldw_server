import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type {
  PersonaVisualAsset,
  PersonaVisualPack
} from "@/types/persona-visuals"

import {
  canRenderPersonaVisualPack,
  getPersonaVisualRenderer,
  PersonaVisualRendererHost
} from "../personaVisualRenderers"

const assetLoader = vi.hoisted(() => vi.fn())

vi.mock("@/services/persona-visual-assets", () => ({
  acquirePersonaVisualAsset: assetLoader
}))

beforeEach(() => {
  assetLoader.mockReset().mockImplementation(async (asset: PersonaVisualAsset) => ({
    url: `blob:${asset.id}`,
    mimeType: asset.mime_type,
    release: vi.fn()
  }))
})

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

const buildPack = (
  overrides: Partial<PersonaVisualPack> = {}
): PersonaVisualPack => ({
  id: "pack-1",
  persona_id: "persona-1",
  user_id: "user-1",
  title: "Sprite Buddy",
  renderer_type: "sprite_frames",
  status: "active",
  manifest_version: 1,
  manifest: {
    manifest_version: 1,
    renderer_type: "sprite_frames",
    states: {
      idle: { animation_id: "idle" }
    },
    animations: {
      idle: {
        frames: [{ asset_id: "idle-1" }]
      }
    }
  },
  companion_behavior: null,
  review: null,
  parent_pack_id: null,
  revision_number: 1,
  provenance: "uploaded",
  active_at: "2026-08-23T00:00:00Z",
  assets: [],
  assets_by_id: {
    "idle-1": buildAsset("idle-1", { width: 32, height: 32 })
  },
  created_at: "2026-08-23T00:00:00Z",
  last_modified: "2026-08-23T00:00:00Z",
  version: 1,
  ...overrides
})

afterEach(() => {
  cleanup()
})

describe("persona visual renderer registry", () => {
  it("registers sprite frame rendering only", () => {
    expect(getPersonaVisualRenderer("sprite_frames")).toEqual(
      expect.objectContaining({ rendererType: "sprite_frames" })
    )
    expect(getPersonaVisualRenderer("live2d")).toBeNull()
    expect(getPersonaVisualRenderer("__proto__")).toBeNull()
  })

  it("reports renderability from the registered renderer", () => {
    expect(canRenderPersonaVisualPack(buildPack())).toBe(true)
    expect(
      canRenderPersonaVisualPack(
        buildPack({
          renderer_type: "live2d",
          manifest: {
            ...buildPack().manifest,
            renderer_type: "live2d"
          }
        })
      )
    ).toBe(false)
    expect(
      canRenderPersonaVisualPack(
        buildPack({
          assets_by_id: {
            "other-asset": buildAsset("other-asset")
          }
        })
      )
    ).toBe(false)
  })

  it("normalizes legacy asset_ids before checking renderability", () => {
    const basePack = buildPack()

    expect(
      canRenderPersonaVisualPack(
        buildPack({
          manifest: {
            ...basePack.manifest,
            animations: {
              idle: {
                asset_ids: [" idle-1 "]
              }
            }
          }
        })
      )
    ).toBe(true)
  })

  it("reports atlas-backed sprite frame packs as renderable", () => {
    const basePack = buildPack()

    expect(
      canRenderPersonaVisualPack({
        ...basePack,
        manifest: {
          ...basePack.manifest,
          animations: {
            idle: {
              frames: [
                {
                  asset_id: "idle-1",
                  region: { x: 0, y: 0, width: 16, height: 16 }
                }
              ]
            }
          }
        }
      })
    ).toBe(true)
  })

  it("keeps renderability coarse so atlas region errors can be reported", () => {
    const basePack = buildPack()

    expect(
      canRenderPersonaVisualPack({
        ...basePack,
        manifest: {
          ...basePack.manifest,
          animations: {
            idle: {
              frames: [
                {
                  asset_id: "idle-1",
                  region: { x: 0, y: 0, width: 0, height: 16 }
                }
              ]
            }
          }
        }
      })
    ).toBe(true)
  })

  it("renders sprite frame packs through the registered component", async () => {
    const onRenderError = vi.fn()

    render(
      <PersonaVisualRendererHost
        pack={buildPack()}
        requestedState="idle"
        generation={1}
        reducedMotion={false}
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(await screen.findByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:idle-1"
    )
    expect(onRenderError).toHaveBeenCalledWith(null)
  })

  it("renders sprite atlas regions through the registered sprite frame renderer", async () => {
    const onRenderError = vi.fn()

    render(
      <PersonaVisualRendererHost
        pack={buildPack({
          manifest: {
            ...buildPack().manifest,
            animations: {
              idle: {
                frames: [
                  {
                    asset_id: "sheet-1",
                    region: { x: 16, y: 8, width: 24, height: 32 }
                  }
                ]
              }
            }
          },
          assets_by_id: {
            "sheet-1": buildAsset("sheet-1", {
              asset_role: "sprite_sheet",
              url: "/assets/sheet.png",
              width: 96,
              height: 64
            })
          }
        })}
        requestedState="idle"
        generation={1}
        reducedMotion={false}
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(await screen.findByTestId("persona-visual-frame")).toHaveStyle({
      backgroundImage: "url(blob:sheet-1)",
      backgroundPosition: "-16px -8px",
      backgroundSize: "96px 64px",
      width: "24px",
      height: "32px"
    })
    expect(onRenderError).toHaveBeenCalledWith(null)
  })

  it("mounts supported renderers so they can report render errors", () => {
    const onRenderError = vi.fn()

    render(
      <PersonaVisualRendererHost
        pack={buildPack({
          assets_by_id: {
            "other-asset": buildAsset("other-asset")
          }
        })}
        requestedState="idle"
        generation={1}
        reducedMotion={false}
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
    expect(onRenderError).toHaveBeenCalledWith("missing_asset")
  })

  it("keeps unsupported sprite atlas regions fail-soft through the registered renderer", () => {
    const onRenderError = vi.fn()

    render(
      <PersonaVisualRendererHost
        pack={buildPack({
          manifest: {
            ...buildPack().manifest,
            animations: {
              idle: {
                frames: [
                  {
                    asset_id: "sheet-1",
                    region: { x: 80, y: 0, width: 32, height: 32 }
                  }
                ]
              }
            }
          },
          assets_by_id: {
            "sheet-1": buildAsset("sheet-1", {
              asset_role: "sprite_sheet",
              url: "/assets/sheet.png",
              width: 96,
              height: 64
            })
          }
        })}
        requestedState="idle"
        generation={1}
        reducedMotion={false}
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
    expect(onRenderError).toHaveBeenCalledWith("unsupported_region")
  })

  it("falls back for unsupported renderer packs", () => {
    const unsupportedPack = buildPack({
      renderer_type: "live2d",
      manifest: {
        ...buildPack().manifest,
        renderer_type: "live2d"
      }
    })

    render(
      <PersonaVisualRendererHost
        pack={unsupportedPack}
        requestedState="idle"
        generation={1}
        reducedMotion={false}
        fallbackLabel="Buddy"
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
  })
})
