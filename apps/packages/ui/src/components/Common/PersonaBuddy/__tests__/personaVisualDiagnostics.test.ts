import { describe, expect, it } from "vitest"

import type {
  PersonaVisualAsset,
  PersonaVisualPack
} from "@/types/persona-visuals"

import {
  createPersonaCompanionDiagnostic,
  getPrimaryPersonaVisualDiagnostic,
  resolvePersonaVisualDiagnostics
} from "../personaVisualDiagnostics"

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
  title: "Reliable buddy",
  renderer_type: "sprite_frames",
  status: "active",
  manifest_version: 1,
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
  companion_behavior: null,
  review: null,
  parent_pack_id: null,
  revision_number: 1,
  provenance: "uploaded",
  active_at: "2026-08-23T00:00:00Z",
  assets: [],
  assets_by_id: {
    "idle-asset": buildAsset("idle-asset"),
    "tool-asset": buildAsset("tool-asset")
  },
  created_at: "2026-08-23T00:00:00Z",
  last_modified: "2026-08-23T00:00:00Z",
  version: 1,
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
            "idle-asset": buildAsset("idle-asset")
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

  it("reports unsupported sprite atlas regions as fail-soft warnings", () => {
    expect(
      getPrimaryPersonaVisualDiagnostic({
        pack: buildPack(),
        renderError: "unsupported_region"
      })
    ).toEqual(
      expect.objectContaining({
        code: "unsupported_region",
        severity: "warning"
      })
    )
  })

  it("returns no diagnostics for a renderable sprite-frame pack", () => {
    expect(resolvePersonaVisualDiagnostics({ pack: buildPack() })).toEqual([])
  })
})

describe("createPersonaCompanionDiagnostic", () => {
  it("keeps local diagnostics limited to safe identifiers and failure classes", () => {
    expect(
      createPersonaCompanionDiagnostic({
        event: "ambient_skipped",
        personaId: "persona-1",
        packId: "file:///Users/alice/private-pack",
        state: "ambient.look",
        failureClass: "empty_set"
      })
    ).toEqual({
      event: "ambient_skipped",
      personaId: "persona-1",
      state: "ambient.look",
      failureClass: "empty_set"
    })
  })
})
