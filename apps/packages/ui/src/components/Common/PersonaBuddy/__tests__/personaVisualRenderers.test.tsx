import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { PersonaVisualPack } from "@/types/persona-visuals"

import {
  canRenderPersonaVisualPack,
  getPersonaVisualRenderer,
  PersonaVisualRendererHost
} from "../personaVisualRenderers"

const buildPack = (
  overrides: Partial<PersonaVisualPack> = {}
): PersonaVisualPack => ({
  id: "pack-1",
  persona_id: "persona-1",
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
  assets_by_id: {
    "idle-1": {
      id: "idle-1",
      url: "/assets/idle-1.png",
      mime_type: "image/png",
      asset_role: "frame",
      width: 32,
      height: 32
    }
  },
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
            "other-asset": {
              id: "other-asset",
              url: "/assets/other.png",
              mime_type: "image/png",
              asset_role: "frame"
            }
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
    expect(
      canRenderPersonaVisualPack(
        buildPack({
          manifest: {
            ...buildPack().manifest,
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
      )
    ).toBe(true)
  })

  it("keeps renderability coarse so atlas region errors can be reported", () => {
    expect(
      canRenderPersonaVisualPack(
        buildPack({
          manifest: {
            ...buildPack().manifest,
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
      )
    ).toBe(true)
  })

  it("renders sprite frame packs through the registered component", () => {
    const onRenderError = vi.fn()

    render(
      <PersonaVisualRendererHost
        pack={buildPack()}
        state="idle"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      expect.stringContaining("/assets/idle-1.png")
    )
    expect(onRenderError).toHaveBeenCalledWith(null)
  })

  it("renders sprite atlas regions through the registered sprite frame renderer", () => {
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
            "sheet-1": {
              id: "sheet-1",
              url: "/assets/sheet.png",
              mime_type: "image/png",
              asset_role: "sprite_sheet",
              width: 96,
              height: 64
            }
          }
        })}
        state="idle"
        fallbackLabel="Buddy"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByTestId("persona-visual-frame")).toHaveStyle({
      backgroundImage: "url(/assets/sheet.png)",
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
            "other-asset": {
              id: "other-asset",
              url: "/assets/other.png",
              mime_type: "image/png",
              asset_role: "frame"
            }
          }
        })}
        state="idle"
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
            "sheet-1": {
              id: "sheet-1",
              url: "/assets/sheet.png",
              mime_type: "image/png",
              asset_role: "sprite_sheet",
              width: 96,
              height: 64
            }
          }
        })}
        state="idle"
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
        state="idle"
        fallbackLabel="Buddy"
      />
    )

    expect(screen.getByText("Buddy")).toBeInTheDocument()
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
  })
})
