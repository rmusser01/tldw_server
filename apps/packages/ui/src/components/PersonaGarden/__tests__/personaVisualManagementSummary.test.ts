import { describe, expect, it } from "vitest"

import type {
  PersonaVisualCandidate,
  PersonaVisualImportPreviewResponse,
  PersonaVisualLibraryItem,
  PersonaVisualManifest,
  PersonaVisualPack,
  PersonaVisualPackExportResponse,
  PersonaVisualPortabilityJobResponse
} from "@/types/persona-visuals"
import type { PersonaVisualGenerationReadinessView } from "../personaVisualGenerationReadiness"
import { buildPersonaVisualManagementSummary } from "../personaVisualManagementSummary"

const baseManifest: PersonaVisualManifest = {
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {},
  animations: {}
}

const makePack = (overrides: Partial<PersonaVisualPack>): PersonaVisualPack => ({
  id: "pack-1",
  persona_id: "persona-1",
  title: "Visual pack",
  renderer_type: "sprite_frames",
  status: "draft",
  manifest: baseManifest,
  ...overrides
})

const makeCandidate = (
  overrides: Partial<PersonaVisualCandidate>
): PersonaVisualCandidate => ({
  id: "candidate-1",
  pack_id: "pack-1",
  persona_id: "persona-1",
  status: "review",
  ...overrides
})

const makeLibraryItem = (
  overrides: Partial<PersonaVisualLibraryItem>
): PersonaVisualLibraryItem => ({
  id: "library-1",
  title: "Reusable pack",
  tags: [],
  source_available: true,
  source_changed: false,
  ...overrides
})

const makeImportPreview = (
  overrides: Partial<PersonaVisualImportPreviewResponse>
): PersonaVisualImportPreviewResponse => ({
  preview_id: "preview-1",
  job_id: "preview-job-1",
  portability_job_id: "portability-preview-1",
  operation: "import_preview",
  status: "completed",
  visual_status: "completed",
  stage: "completed",
  bundle_summary: {},
  validation_warnings: [],
  conflicts: [],
  proposed_plan: {},
  quota_estimate: {},
  required_choices: [],
  target_warnings: [],
  ...overrides
})

const makeExportJob = (
  overrides: Partial<PersonaVisualPackExportResponse>
): PersonaVisualPackExportResponse => ({
  job_id: "export-job-1",
  portability_job_id: "portability-export-1",
  operation: "export",
  persona_id: "persona-1",
  pack_id: "pack-1",
  status: "completed",
  stage: "completed",
  download_url: "/archive.tldw-persona-vpack",
  ...overrides
})

const makePortabilityJob = (
  overrides: Partial<PersonaVisualPortabilityJobResponse>
): PersonaVisualPortabilityJobResponse => ({
  job_id: "job-1",
  portability_job_id: "portability-job-1",
  operation: "import_commit",
  persona_id: "persona-1",
  pack_id: "pack-1",
  status: "processing",
  visual_status: "processing",
  stage: "processing",
  progress: {},
  warnings: [],
  ...overrides
})

const blockedReadiness: PersonaVisualGenerationReadinessView = {
  status: "jobs_unavailable",
  canQueue: false,
  blocking: true,
  enabledBackends: [],
  queue: "persona_visual_generation"
}

describe("buildPersonaVisualManagementSummary", () => {
  it("returns an empty management state when no visual packs exist", () => {
    const model = buildPersonaVisualManagementSummary({ packs: [] })

    expect(model.summary).toEqual({
      activePackId: null,
      activePackTitle: null,
      packCounts: {
        active: 0,
        draft: 0,
        review: 0,
        archived: 0,
        failed: 0
      },
      attentionCounts: {
        invalidPackCount: 0,
        reviewCandidates: 0,
        failedCandidates: 0,
        unavailableLibraryItems: 0,
        changedLibraryItems: 0,
        pendingJobs: 0,
        failedJobs: 0
      }
    })
    expect(model.attentionRows).toEqual([])
  })

  it("deduplicates the active pack returned outside the pack list", () => {
    const activePack = makePack({
      id: "active-pack",
      title: "Rendered now",
      status: "active"
    })

    const model = buildPersonaVisualManagementSummary({
      activePack,
      packs: [
        makePack({ id: "draft-pack", status: "draft" }),
        makePack({ id: "review-pack", status: "review" }),
        makePack({ id: "archived-pack", status: "archived" }),
        makePack({ id: "failed-pack", status: "failed" })
      ]
    })

    expect(model.summary.activePackId).toBe("active-pack")
    expect(model.summary.activePackTitle).toBe("Rendered now")
    expect(model.summary.packCounts).toEqual({
      active: 1,
      draft: 1,
      review: 1,
      archived: 1,
      failed: 1
    })
    expect(model.attentionRows.map((row) => row.kind)).toEqual(["failed_pack"])
  })

  it("surfaces selected-pack validation and generated candidate review state", () => {
    const model = buildPersonaVisualManagementSummary({
      packs: [makePack({ id: "pack-1", status: "draft" })],
      selectedPack: makePack({ id: "pack-1", status: "draft" }),
      validationErrors: ["Missing required state: speaking"],
      candidates: [
        makeCandidate({ id: "candidate-review", status: "review" }),
        makeCandidate({ id: "candidate-failed", status: "failed" }),
        makeCandidate({ id: "candidate-rejected", status: "rejected" })
      ]
    })

    expect(model.summary.attentionCounts.invalidPackCount).toBe(1)
    expect(model.summary.attentionCounts.reviewCandidates).toBe(1)
    expect(model.summary.attentionCounts.failedCandidates).toBe(1)
    expect(model.attentionRows.map((row) => row.kind)).toEqual([
      "invalid_manifest",
      "generated_candidates_review",
      "generated_candidates_failed"
    ])
  })

  it("tracks import/export job attention without activating completed drafts", () => {
    const model = buildPersonaVisualManagementSummary({
      packs: [makePack({ id: "pack-1", status: "draft" })],
      importPreview: makeImportPreview({ status: "completed" }),
      importCommitJob: makePortabilityJob({
        operation: "import_commit",
        status: "completed",
        visual_status: "completed",
        stage: "completed",
        pack_id: "imported-draft"
      }),
      exportJob: makeExportJob({ status: "completed" })
    })

    expect(model.summary.activePackId).toBeNull()
    expect(model.summary.attentionCounts.pendingJobs).toBe(0)
    expect(model.summary.attentionCounts.failedJobs).toBe(0)
    expect(model.attentionRows.map((row) => row.kind)).toEqual([
      "import_preview_ready",
      "import_commit_completed",
      "export_completed"
    ])
  })

  it("tracks stale library sources, unavailable generation, and failed jobs", () => {
    const model = buildPersonaVisualManagementSummary({
      packs: [makePack({ id: "pack-1", status: "draft" })],
      libraryItems: [
        makeLibraryItem({
          id: "unavailable-library-item",
          source_available: false,
          source_changed: false
        }),
        makeLibraryItem({
          id: "changed-library-item",
          source_available: true,
          source_changed: true
        })
      ],
      generationReadiness: blockedReadiness,
      exportJob: makePortabilityJob({
        operation: "export",
        status: "failed",
        visual_status: "failed",
        stage: "failed"
      }),
      importCommitJob: makePortabilityJob({
        operation: "import_commit",
        status: "processing",
        visual_status: "processing",
        stage: "processing"
      })
    })

    expect(model.summary.attentionCounts.unavailableLibraryItems).toBe(1)
    expect(model.summary.attentionCounts.changedLibraryItems).toBe(1)
    expect(model.summary.attentionCounts.pendingJobs).toBe(1)
    expect(model.summary.attentionCounts.failedJobs).toBe(1)
    expect(model.attentionRows.map((row) => row.kind)).toEqual([
      "library_source_unavailable",
      "library_source_changed",
      "generation_unavailable",
      "pending_job",
      "failed_job"
    ])
  })
})
