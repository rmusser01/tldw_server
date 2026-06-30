import { createHash } from "node:crypto"

import JSZip from "jszip"

export const PERSONA_VISUAL_E2E_PERSONA_ID = "visual_persona_e2e"
export const PERSONA_VISUAL_E2E_SESSION_ID = "sess-visual-e2e-001"
export const PERSONA_VISUAL_E2E_FRAME_DATA_URI =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="

export const PERSONA_VISUAL_E2E_STARTER_PACK = {
  id: "research-buddy-starter",
  title: "Research Buddy Starter",
  description: "A deterministic bundled sprite-frame visual for Buddy setup.",
  renderer_type: "sprite_frames",
  manifest_version: 1,
  states_offered: ["idle", "listening", "thinking", "speaking", "error"],
  asset_count: 1,
  total_bytes: 96,
  tags: ["starter", "sprite_frames"],
  license_label: "bundled"
}

type BuildPersonaVisualPackOptions = {
  packId?: string
  title?: string
  status?: "draft" | "active"
  provenance?: string
}

export const buildPersonaVisualPackFixture = (
  personaId: string,
  options: BuildPersonaVisualPackOptions = {}
) => {
  const packId = options.packId ?? "visual-pack-e2e"
  const assetIds = {
    idle: `${packId}-frame-idle`,
    speaking: `${packId}-frame-speaking`,
    tool: `${packId}-frame-tool`,
    error: `${packId}-frame-error`
  }
  const buildAsset = (id: string) => ({
    id,
    pack_id: packId,
    persona_id: personaId,
    asset_role: "frame",
    storage_key: `persona-visuals/${personaId}/${id}.png`,
    url: PERSONA_VISUAL_E2E_FRAME_DATA_URI,
    original_filename: `${id}.png`,
    mime_type: "image/png",
    byte_size: 96,
    checksum_sha256: `${id}-checksum`,
    width: 1,
    height: 1,
    provenance: "e2e_fixture",
    created_at: "2026-05-08T00:00:00.000Z",
    last_modified: "2026-05-08T00:00:00.000Z",
    version: 1
  })
  const assetsById = {
    [assetIds.idle]: buildAsset(assetIds.idle),
    [assetIds.speaking]: buildAsset(assetIds.speaking),
    [assetIds.tool]: buildAsset(assetIds.tool),
    [assetIds.error]: buildAsset(assetIds.error)
  }

  return {
    id: packId,
    persona_id: personaId,
    user_id: "e2e-user",
    title: options.title ?? "Visual Runtime Pack",
    renderer_type: "sprite_frames",
    status: options.status ?? "active",
    manifest_version: 1,
    manifest: {
      manifest_version: 1,
      renderer_type: "sprite_frames",
      states: {
        idle: { animation_id: "idle-loop" },
        listening: { animation_id: "idle-loop" },
        thinking: { animation_id: "idle-loop" },
        speaking: { animation_id: "speaking-loop" },
        tool_running: { animation_id: "tool-loop" },
        error: { animation_id: "error-loop" }
      },
      animations: {
        "idle-loop": {
          frames: [{ asset_id: assetIds.idle, duration_ms: 250 }],
          loop: true
        },
        "speaking-loop": {
          frames: [{ asset_id: assetIds.speaking, duration_ms: 250 }],
          loop: true
        },
        "tool-loop": {
          frames: [{ asset_id: assetIds.tool, duration_ms: 250 }],
          loop: true
        },
        "error-loop": {
          frames: [{ asset_id: assetIds.error, duration_ms: 250 }],
          loop: false
        }
      },
      fallbacks: {
        speaking: ["idle"],
        tool_running: ["idle"],
        error: ["idle"]
      },
      authored_triggers: [
        {
          id: "mcp-runtime-override",
          source: "mcp_runtime",
          match: "persona_visuals.trigger_state",
          state: "speaking",
          duration_ms: 5000,
          priority: 10
        }
      ]
    },
    active_at:
      options.status === "draft" ? null : "2026-05-08T00:00:00.000Z",
    assets: Object.values(assetsById),
    assets_by_id: assetsById,
    provenance: options.provenance ?? "e2e_fixture",
    created_at: "2026-05-08T00:00:00.000Z",
    last_modified: "2026-05-08T00:00:00.000Z",
    version: 1
  }
}

const sortJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortJsonValue)
  if (!value || typeof value !== "object") return value
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nestedValue]) => [key, sortJsonValue(nestedValue)])
  )
}

const stableJson = (payload: unknown): Buffer =>
  Buffer.from(JSON.stringify(sortJsonValue(payload)), "utf8")

const sha256 = (payload: Buffer): string =>
  createHash("sha256").update(payload).digest("hex")

export const buildPortablePersonaVisualPackUpload = async () => {
  const frameBytes = Buffer.from(
    PERSONA_VISUAL_E2E_FRAME_DATA_URI.split(",")[1] || "",
    "base64"
  )
  const uploadedAssetId = "portable-upload-pack-frame-idle"
  const visualManifest = {
    manifest_version: 1,
    renderer_type: "sprite_frames",
    states: {
      idle: { animation_id: "idle-loop" }
    },
    animations: {
      "idle-loop": {
        frames: [{ asset_id: uploadedAssetId, duration_ms: 250 }],
        loop: true
      }
    },
    fallbacks: {}
  }
  const manifestBytes = stableJson({
    archive_profile: "backup",
    counts: { assets: 1, assets_with_bytes: 1 },
    pack_title: "Uploaded Visual Pack",
    renderer_type: "sprite_frames",
    schema_version: "tldw.persona_visual_pack.v1"
  })
  const packBytes = stableJson({
    pack: {
      renderer_type: "sprite_frames",
      source_persona_id: "portable-source",
      title: "Uploaded Visual Pack",
      visual_manifest: visualManifest
    }
  })
  const assetsBytes = stableJson({
    assets: [
      {
        asset_bytes_status: "present",
        asset_path: "assets/persona_visuals/uploaded-idle.png",
        asset_role: "frame",
        asset_sha256: sha256(frameBytes),
        mime_type: "image/png",
        original_filename: "uploaded-idle.png",
        source_asset_id: uploadedAssetId
      }
    ]
  })
  const entries = {
    "manifest.json": manifestBytes,
    "metadata/pack.json": packBytes,
    "metadata/assets.json": assetsBytes,
    "assets/persona_visuals/uploaded-idle.png": frameBytes
  }
  const checksumBytes = stableJson(
    Object.fromEntries(
      Object.entries(entries).map(([path, payload]) => [path, sha256(payload)])
    )
  )
  const archive = new JSZip()
  const fixedDate = new Date("2026-05-08T00:00:00.000Z")
  for (const [path, payload] of Object.entries(entries)) {
    archive.file(path, payload, { date: fixedDate })
  }
  archive.file("checksums/sha256.json", checksumBytes, { date: fixedDate })

  return {
    name: "uploaded-visual-pack.tldw-persona-vpack",
    mimeType: "application/vnd.tldw.persona.visual-pack+zip",
    buffer: await archive.generateAsync({
      type: "nodebuffer",
      compression: "DEFLATE",
      compressionOptions: { level: 9 }
    })
  }
}
