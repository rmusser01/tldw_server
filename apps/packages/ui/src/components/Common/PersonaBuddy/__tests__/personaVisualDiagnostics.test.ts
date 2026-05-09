import { describe, expect, it } from "vitest"

import type { PersonaVisualPack } from "@/types/persona-visuals"

import {
  getPrimaryPersonaVisualDiagnostic,
  resolvePersonaVisualDiagnostics
} from "../personaVisualDiagnostics"

const buildPack = (
  overrides: Partial<PersonaVisualPack> = {}
): PersonaVisualPack => ({
  id: "pack-1",
  persona_id: "persona-1",
  title: "Reliable buddy",
  renderer_type: "sprite_frames",
  status: "active",
  manifest: {
    manifest_version: 1,
    renderer_type: "sprite_frames",
    states: {
      idle: { animation_id: "idle" },
      tool_running: { animation_id: "tool" }
    },
    animations: {
      idle: {
        frames: [{ asset_id: "idle-asset", duration_ms: 100 }]
      },
      tool: {
        frames: [{ asset_id: "tool-asset", duration_ms: 100 }]
      }
    },
    fallbacks: {
      thinking: ["idle"]
    }
  },
  assets_by_id: {
    "idle-asset": {
      id: "idle-asset",
      asset_role: "frame",
      url: "/assets/idle.png",
      mime_type: "image/png"
    },
    "tool-asset": {
      id: "tool-asset",
      asset_role: "frame",
      url: "/assets/tool.png",
      mime_type: "image/png"
    }
  },
  ...overrides
})

describe("resolvePersonaVisualDiagnostics", () => {
  it("reports no active pack when requested", () => {
    expect(
      resolvePersonaVisualDiagnostics({
        pack: null,
        includeNoActivePack: true
      })
    ).toEqual([
      expect.objectContaining({
        code: "no_active_pack",
        severity: "info"
      })
    ])
  })

  it("reports active pack load failures", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        loadStatus: "error",
        loadError: new Error("offline"),
        pack: null
      })
    ).toEqual(
      expect.objectContaining({
        code: "load_failed",
        severity: "warning",
        message: expect.stringContaining("offline")
      })
    )
  })

  it("reports unsupported renderer types", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        pack: buildPack({ renderer_type: "live2d" })
      })
    ).toEqual(
      expect.objectContaining({
        code: "unsupported_renderer",
        severity: "warning"
      })
    )
  })

  it("reports missing manifests", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        pack: buildPack({ manifest: null as unknown as PersonaVisualPack["manifest"] })
      })
    ).toEqual(
      expect.objectContaining({
        code: "missing_manifest",
        severity: "error"
      })
    )
  })

  it("reports missing asset maps", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        pack: buildPack({ assets_by_id: {}, assets: [] })
      })
    ).toEqual(
      expect.objectContaining({
        code: "missing_assets",
        severity: "error"
      })
    )
  })

  it("reports missing animations for the requested state", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        pack: buildPack({
          manifest: {
            ...buildPack().manifest,
            states: {},
            fallbacks: {}
          }
        }),
        visualState: "thinking"
      })
    ).toEqual(
      expect.objectContaining({
        code: "missing_animation",
        severity: "error"
      })
    )
  })

  it("reports missing assets referenced by resolved animations", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        pack: buildPack({
          assets_by_id: {
            "idle-asset": {
              id: "idle-asset",
              asset_role: "frame",
              url: "/assets/idle.png",
              mime_type: "image/png"
            }
          }
        }),
        visualState: "tool_running"
      })
    ).toEqual(
      expect.objectContaining({
        code: "missing_asset",
        severity: "error",
        message: expect.stringContaining("tool-asset")
      })
    )
  })

  it("returns no diagnostics for a renderable sprite-frame pack", () => {
    expect(resolvePersonaVisualDiagnostics({ pack: buildPack() })).toEqual([])
  })
})
