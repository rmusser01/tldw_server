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
