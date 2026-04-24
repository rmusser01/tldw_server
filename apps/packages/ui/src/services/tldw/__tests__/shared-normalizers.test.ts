import { describe, expect, it } from "vitest"

import {
  normalizeIngestionSourceListResponse,
  normalizeReadingDigestSchedule
} from "@/services/tldw/collections-normalizers"
import {
  buildPresentationVisualStyleSnapshot,
  clonePresentationVisualStyleSnapshot
} from "@/services/tldw/presentation-style"
import {
  normalizePersonaExemplar,
  normalizePersonaProfile
} from "@/services/tldw/persona-normalizers"

describe("shared tldw normalizers", () => {
  it("normalizes reading digest schedules with string ids and boolean flags", () => {
    expect(
      normalizeReadingDigestSchedule({
        id: 12,
        enabled: 1,
        require_online: 0,
        format: "html"
      })
    ).toMatchObject({
      id: "12",
      enabled: true,
      require_online: false,
      format: "html"
    })
  })

  it("drops undeclared reading digest fields and clones nested filters", () => {
    const payload = {
      id: 12,
      cron: "0 8 * * *",
      extra_field: "should-not-leak",
      filters: {
        tags: ["briefing"],
        suggestions: {
          enabled: true,
          exclude_tags: ["skip-me"]
        }
      }
    }

    const normalized = normalizeReadingDigestSchedule(payload)
    payload.filters.tags.push("mutated")
    payload.filters.suggestions.exclude_tags.push("later")

    expect(normalized).toEqual({
      id: "12",
      name: null,
      cron: "0 8 * * *",
      timezone: null,
      enabled: false,
      require_online: false,
      format: "md",
      template_id: null,
      template_name: null,
      retention_days: null,
      filters: {
        tags: ["briefing"],
        suggestions: {
          enabled: true,
          exclude_tags: ["skip-me"]
        }
      },
      last_run_at: null,
      next_run_at: null,
      last_status: null,
      created_at: null,
      updated_at: null
    })
    expect("extra_field" in normalized).toBe(false)
  })

  it("normalizes ingestion source list payload totals and nested ids", () => {
    expect(
      normalizeIngestionSourceListResponse({
        total: "2",
        sources: [
          {
            id: 4,
            user_id: "7",
            source_type: "archive_snapshot",
            sink_type: "notes",
            policy: "import_only"
          }
        ]
      })
    ).toEqual({
      total: 2,
      sources: [
        expect.objectContaining({
          id: "4",
          user_id: 7,
          source_type: "archive_snapshot",
          sink_type: "notes",
          policy: "import_only"
        })
      ]
    })
  })

  it("clones presentation style snapshots without reusing nested references", () => {
    const original = buildPresentationVisualStyleSnapshot({
      id: "style-1",
      scope: "builtin",
      name: "Blueprint",
      description: "Structured deck",
      category: "technical",
      guide_number: 7,
      tags: ["timeline"],
      best_for: ["systems explanation"],
      generation_rules: {
        pacing: {
          density: "tight"
        }
      },
      artifact_preferences: ["comparison"],
      appearance_defaults: {
        theme: "night"
      },
      fallback_policy: {
        mode: "reuse"
      },
      version: 3
    })

    const cloned = clonePresentationVisualStyleSnapshot(original)

    expect(cloned).toEqual(original)
    expect(cloned).not.toBe(original)
    expect(cloned?.tags).not.toBe(original.tags)
    expect(cloned?.generation_rules).not.toBe(original.generation_rules)
    expect(cloned?.appearance_defaults).not.toBe(original.appearance_defaults)
    expect(cloned?.fallback_policy).not.toBe(original.fallback_policy)
  })

  it("normalizes persona profile and exemplar ids, buddy summary, and tags", () => {
    const profile = normalizePersonaProfile({
      persona_id: 42,
      buddySummary: {
        personaName: " Garden Helper ",
        hasBuddy: "0",
        roleSummary: " research companion ",
        visual: {
          speciesId: "fox",
          silhouetteId: "slim",
          paletteId: "moss"
        }
      }
    })

    const exemplar = normalizePersonaExemplar({
      id: 9,
      personaId: "42",
      kind: "voice",
      scenarioTags: [" gardening ", "", "planning"],
      capabilityTags: [" research ", " "],
      priority: "4",
      notes: " keep this trimmed ",
      created_at: "2026-04-17T00:00:00Z",
      last_modified: "2026-04-17T01:00:00Z"
    })

    expect(profile).toMatchObject({
      id: "42",
      buddy_summary: {
        has_buddy: false,
        persona_name: "Garden Helper",
        role_summary: "research companion",
        visual: {
          species_id: "fox",
          silhouette_id: "slim",
          palette_id: "moss"
        }
      }
    })

    expect(exemplar).toMatchObject({
      id: "9",
      persona_id: "42",
      kind: "voice",
      scenario_tags: ["gardening", "planning"],
      capability_tags: ["research"],
      priority: 4,
      notes: " keep this trimmed ",
      created_at: "2026-04-17T00:00:00Z",
      last_modified: "2026-04-17T01:00:00Z"
    })
  })

  it("rejects array payloads when normalizing persona records", () => {
    const profile = normalizePersonaProfile(
      ["not", "a", "profile"] as unknown as Record<string, unknown>
    )
    const exemplar = normalizePersonaExemplar(
      ["not", "an", "exemplar"] as unknown as Record<string, unknown>
    )

    expect(profile).toMatchObject({
      id: "",
      buddy_summary: null
    })
    expect(Object.prototype.hasOwnProperty.call(profile, "0")).toBe(false)

    expect(exemplar).toMatchObject({
      id: "",
      persona_id: "",
      scenario_tags: [],
      capability_tags: []
    })
  })
})
