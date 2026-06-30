import { describe, expect, it } from "vitest"
import {
  RESEARCH_WORKSPACE_CAPABILITY_IDS,
  buildUnknownResearchWorkspaceCapabilities,
  getArtifactCapabilityId,
  getCapability,
  isResearchWorkspaceCapabilitiesStale,
  normalizeResearchWorkspaceCapabilities
} from "../research-workspace-capabilities"
import type { ArtifactType } from "@/types/workspace"

describe("Research Workspace capabilities", () => {
  it("builds an unknown warning fallback for every capability", () => {
    const fallback = buildUnknownResearchWorkspaceCapabilities("network_error")

    expect(Object.keys(fallback.capabilities)).toEqual([
      ...RESEARCH_WORKSPACE_CAPABILITY_IDS
    ])
    for (const id of RESEARCH_WORKSPACE_CAPABILITY_IDS) {
      expect(fallback.capabilities[id]).toMatchObject({
        status: "unknown",
        mode: "warn",
        reason_code: "network_error"
      })
    }
  })

  it("normalizes malformed and partial payloads without trusting missing keys", () => {
    const normalized = normalizeResearchWorkspaceCapabilities({
      status: "ready",
      ttl_seconds: 60,
      capabilities: {
        chat: {
          status: "ready",
          mode: "allow",
          dependencies: ["llm"]
        },
        slides_generation: {
          status: "unavailable",
          mode: "block",
          dependencies: ["slides"],
          reason_code: "slides_unavailable"
        },
        junk: {
          status: "ready",
          mode: "allow",
          dependencies: []
        }
      },
      timestamp: "2026-05-13T00:00:00Z"
    })

    expect(normalized.status).toBe("ready")
    expect(normalized.ttl_seconds).toBe(60)
    expect(normalized.capabilities.chat.mode).toBe("allow")
    expect(normalized.capabilities.slides_generation.mode).toBe("block")
    expect(normalized.capabilities.source_browse.mode).toBe("warn")
    expect("junk" in normalized.capabilities).toBe(false)
  })

  it("treats invalid payloads as unknown warning payloads", () => {
    const normalized = normalizeResearchWorkspaceCapabilities({
      status: "banana",
      capabilities: {
        chat: { status: "ready", mode: "banana", dependencies: [] }
      }
    })

    expect(normalized.status).toBe("unknown")
    expect(normalized.capabilities.chat.mode).toBe("warn")
    expect(normalized.capabilities.chat.status).toBe("unknown")
  })

  it("detects stale payloads from ttl seconds and fetch time", () => {
    const payload = normalizeResearchWorkspaceCapabilities({
      status: "ready",
      ttl_seconds: 30,
      capabilities: {}
    })

    expect(isResearchWorkspaceCapabilitiesStale(payload, 1_000, 30_999)).toBe(false)
    expect(isResearchWorkspaceCapabilitiesStale(payload, 1_000, 31_001)).toBe(true)
  })

  it("maps artifact types to their capability boundary", () => {
    const textTypes: ArtifactType[] = [
      "summary",
      "mindmap",
      "report",
      "compare_sources",
      "flashcards",
      "quiz",
      "timeline",
      "data_table"
    ]

    for (const type of textTypes) {
      expect(getArtifactCapabilityId(type)).toBe("artifact_text_generation")
    }
    expect(getArtifactCapabilityId("slides")).toBe("slides_generation")
    expect(getArtifactCapabilityId("audio_overview")).toBe("audio_summary")
  })

  it("returns a capability by id without applying unrelated blocks", () => {
    const normalized = normalizeResearchWorkspaceCapabilities({
      status: "degraded",
      capabilities: {
        artifact_text_generation: {
          status: "ready",
          mode: "allow",
          dependencies: ["llm"]
        },
        slides_generation: {
          status: "unavailable",
          mode: "block",
          dependencies: ["slides"],
          reason_code: "slides_unavailable"
        }
      }
    })

    expect(getCapability(normalized, "artifact_text_generation").mode).toBe("allow")
    expect(getCapability(normalized, "slides_generation").mode).toBe("block")
  })
})
