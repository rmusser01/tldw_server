import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  getTldwTTSModel: vi.fn(),
  getTldwTTSVoice: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args)
}))

vi.mock("@/services/tts", () => ({
  getTldwTTSModel: (...args: unknown[]) => mocks.getTldwTTSModel(...args),
  getTldwTTSVoice: (...args: unknown[]) => mocks.getTldwTTSVoice(...args)
}))

import {
  createWatchlistOutput,
  getWatchlistOutputEvidence,
  getWatchlistOutputReadiness
} from "../watchlists"
import type {
  WatchlistOutput,
  WatchlistOutputEvidenceResponse,
  WatchlistReportReadiness
} from "@/types/watchlists"

const output: WatchlistOutput = {
  id: 42,
  run_id: 7,
  job_id: 5,
  type: "briefing_markdown",
  format: "md",
  title: "CTI evidence report",
  metadata: {
    origin: "watchlists"
  },
  version: 1,
  expired: false,
  created_at: "2026-05-15T12:00:00Z"
}

const readiness: WatchlistReportReadiness = {
  state: "warning",
  score: 78,
  warnings: [
    {
      code: "single_source",
      severity: "warning",
      message: "Report evidence only includes one source.",
      affected_item_ids: [101, 102]
    }
  ]
}

const evidenceResponse: WatchlistOutputEvidenceResponse = {
  output_id: 42,
  immutable_snapshot: true,
  readiness,
  snapshot: {
    schema_version: 1,
    snapshot_id: "snapshot-42",
    generated_at: "2026-05-15T12:00:00Z",
    preset: "cti_osint",
    watchlist_id: 3,
    job_id: 5,
    run_id: 7,
    output_id: 42,
    included_items: [
      {
        id: 101,
        title: "Vendor advisory",
        url: "https://vendor.example/cve",
        source_id: 11,
        source_name: "Vendor feed",
        published_at: "2026-05-15T11:00:00Z",
        summary: "Active exploitation observed.",
        tags: ["cve"],
        reviewed: true,
        queued_for_briefing: true,
        alerts: [
          {
            id: 201,
            rule_id: 301,
            rule_name: "Active exploitation",
            severity: "critical",
            status: "unread",
            title: "Active exploitation observed",
            snippet: "Active exploitation observed.",
            matched_text: "active exploitation",
            evidence: { url: "https://vendor.example/cve" },
            created_at: "2026-05-15T11:05:00Z"
          }
        ]
      }
    ],
    excluded_items: [
      {
        id: 103,
        title: "Background update",
        url: "https://vendor.example/background",
        reason: "not_queued_for_report"
      }
    ],
    source_summary: {
      unique_source_count: 1,
      missing_source_count: 0
    },
    included_count: 1,
    excluded_count: 1,
    excluded_total_count: 1,
    excluded_items_truncated: false,
    alert_count: 1,
    critical_alert_count: 1,
    readiness
  }
}

describe("watchlists report evidence client contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("serializes Stage 5 report creation options on createWatchlistOutput", async () => {
    mocks.bgRequest.mockResolvedValue(output)

    await createWatchlistOutput({
      run_id: 7,
      item_ids: [101, 102],
      title: "CTI evidence report",
      format: "md",
      report_preset: "cti_osint",
      include_evidence_table: true,
      include_excluded_items: true,
      require_reviewed_items: true,
      allow_weak_evidence: false
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/watchlists/outputs",
      method: "POST",
      body: {
        run_id: 7,
        item_ids: [101, 102],
        title: "CTI evidence report",
        format: "md",
        report_preset: "cti_osint",
        include_evidence_table: true,
        include_excluded_items: true,
        require_reviewed_items: true,
        allow_weak_evidence: false
      }
    })
  })

  it("loads immutable output evidence from the output-scoped endpoint", async () => {
    mocks.bgRequest.mockResolvedValue(evidenceResponse)

    await expect(getWatchlistOutputEvidence(42)).resolves.toEqual(evidenceResponse)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/watchlists/outputs/42/evidence",
      method: "GET"
    })
  })

  it("loads report readiness from the output-scoped endpoint", async () => {
    mocks.bgRequest.mockResolvedValue(readiness)

    await expect(getWatchlistOutputReadiness(42)).resolves.toEqual(readiness)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/watchlists/outputs/42/readiness",
      method: "GET"
    })
  })
})
