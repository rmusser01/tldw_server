import type {
  PersonaVisualCandidate,
  PersonaVisualImportCommitStartResponse,
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualLibraryItem,
  PersonaVisualPack,
  PersonaVisualPackExportResponse,
  PersonaVisualPackStatus,
  PersonaVisualPortabilityJobResponse
} from "@/types/persona-visuals"
import type { PersonaVisualGenerationReadinessView } from "./personaVisualGenerationReadiness"

/**
 * Derives post-setup Persona Visual management state from existing editor data.
 *
 * The helper is intentionally pure so the Visuals tab can render lifecycle
 * headers and attention queues without introducing new backend state.
 */

export type PersonaVisualManagementAttentionKind =
  | "failed_pack"
  | "invalid_manifest"
  | "generated_candidates_review"
  | "generated_candidates_failed"
  | "import_preview_ready"
  | "import_commit_completed"
  | "export_completed"
  | "library_source_unavailable"
  | "library_source_changed"
  | "generation_unavailable"
  | "pending_job"
  | "failed_job"

export type PersonaVisualManagementAttentionTarget =
  | "packs"
  | "validation"
  | "candidates"
  | "import"
  | "export"
  | "library"
  | "generation"
  | "jobs"

export interface PersonaVisualManagementAttentionRow {
  id: string
  kind: PersonaVisualManagementAttentionKind
  target: PersonaVisualManagementAttentionTarget
  count: number
}

export interface PersonaVisualManagementSummary {
  activePackId: string | null
  activePackTitle: string | null
  packCounts: Record<PersonaVisualPackStatus, number>
  attentionCounts: {
    invalidPackCount: number
    reviewCandidates: number
    failedCandidates: number
    unavailableLibraryItems: number
    changedLibraryItems: number
    pendingJobs: number
    failedJobs: number
  }
}

type PersonaVisualImportPreviewStatus =
  | PersonaVisualImportPreviewStartResponse
  | PersonaVisualImportPreviewResponse

type PersonaVisualImportCommitJob =
  | PersonaVisualImportCommitStartResponse
  | PersonaVisualPortabilityJobResponse

type PersonaVisualExportJob =
  | PersonaVisualPackExportResponse
  | PersonaVisualPortabilityJobResponse

type PersonaVisualManagementJob =
  | PersonaVisualImportPreviewStatus
  | PersonaVisualImportCommitJob
  | PersonaVisualExportJob

export interface PersonaVisualManagementSummaryInput {
  packs?: readonly PersonaVisualPack[]
  activePack?: PersonaVisualPack | null
  selectedPack?: PersonaVisualPack | null
  validationErrors?: readonly string[]
  candidates?: readonly PersonaVisualCandidate[]
  libraryItems?: readonly PersonaVisualLibraryItem[]
  importPreview?: PersonaVisualImportPreviewStatus | null
  importCommitJob?: PersonaVisualImportCommitJob | null
  exportJob?: PersonaVisualExportJob | null
  generationReadiness?: PersonaVisualGenerationReadinessView | null
}

export interface PersonaVisualManagementModel {
  summary: PersonaVisualManagementSummary
  attentionRows: PersonaVisualManagementAttentionRow[]
}

const PACK_STATUSES: PersonaVisualPackStatus[] = [
  "active",
  "draft",
  "review",
  "archived",
  "failed"
]

const PENDING_JOB_STATUSES = new Set(["queued", "pending", "processing", "running"])
const FAILED_JOB_STATUSES = new Set(["failed", "cancelled", "quarantined"])

const normalizeStatus = (status: string | null | undefined): string =>
  String(status || "").trim().toLowerCase()

const dedupePacks = (
  packs: readonly PersonaVisualPack[],
  activePack: PersonaVisualPack | null | undefined
): PersonaVisualPack[] => {
  const byId = new Map<string, PersonaVisualPack>()
  for (const pack of packs) {
    byId.set(pack.id, pack)
  }
  if (activePack) {
    byId.set(activePack.id, activePack)
  }
  return Array.from(byId.values())
}

const countByStatus = (
  packs: readonly PersonaVisualPack[]
): Record<PersonaVisualPackStatus, number> => {
  const counts = PACK_STATUSES.reduce<Record<PersonaVisualPackStatus, number>>(
    (nextCounts, status) => {
      nextCounts[status] = 0
      return nextCounts
    },
    {
      active: 0,
      draft: 0,
      review: 0,
      archived: 0,
      failed: 0
    }
  )

  for (const pack of packs) {
    counts[pack.status] += 1
  }
  return counts
}

const getActivePack = (
  packs: readonly PersonaVisualPack[],
  activePack: PersonaVisualPack | null | undefined
): PersonaVisualPack | null =>
  activePack ?? packs.find((pack) => pack.status === "active") ?? null

const getJobStatus = (job: PersonaVisualManagementJob | null | undefined): string =>
  normalizeStatus(job?.status)

const getJobVisualStatus = (
  job: PersonaVisualManagementJob | null | undefined
): string => {
  if (!job || !("visual_status" in job)) return ""
  return normalizeStatus(job.visual_status)
}

const isPendingJob = (job: PersonaVisualManagementJob | null | undefined): boolean =>
  PENDING_JOB_STATUSES.has(getJobStatus(job)) ||
  PENDING_JOB_STATUSES.has(getJobVisualStatus(job))

const isFailedJob = (job: PersonaVisualManagementJob | null | undefined): boolean =>
  FAILED_JOB_STATUSES.has(getJobStatus(job)) ||
  FAILED_JOB_STATUSES.has(getJobVisualStatus(job))

const isCompletedJob = (job: PersonaVisualManagementJob | null | undefined): boolean =>
  getJobStatus(job) === "completed" || getJobVisualStatus(job) === "completed"

const countJobs = (
  jobs: readonly (PersonaVisualManagementJob | null | undefined)[],
  predicate: (job: PersonaVisualManagementJob | null | undefined) => boolean
): number => jobs.filter(predicate).length

const isGenerationUnavailable = (
  readiness: PersonaVisualGenerationReadinessView | null | undefined
): boolean =>
  Boolean(
    readiness &&
      readiness.blocking &&
      readiness.status !== "loading" &&
      readiness.status !== "ready"
  )

const addAttentionRow = (
  rows: PersonaVisualManagementAttentionRow[],
  row: PersonaVisualManagementAttentionRow
): void => {
  if (row.count > 0) rows.push(row)
}

export function buildPersonaVisualManagementSummary(
  input: PersonaVisualManagementSummaryInput
): PersonaVisualManagementModel {
  const packs = dedupePacks(input.packs || [], input.activePack)
  const activePack = getActivePack(packs, input.activePack)
  const packCounts = countByStatus(packs)
  const candidates = input.candidates || []
  const libraryItems = input.libraryItems || []
  const jobs = [input.importPreview, input.importCommitJob, input.exportJob]

  const invalidPackCount =
    input.selectedPack && (input.validationErrors || []).length > 0 ? 1 : 0
  const reviewCandidates = candidates.filter(
    (candidate) => candidate.status === "review"
  ).length
  const failedCandidates = candidates.filter(
    (candidate) => candidate.status === "failed"
  ).length
  const unavailableLibraryItems = libraryItems.filter(
    (item) => !item.source_available
  ).length
  const changedLibraryItems = libraryItems.filter(
    (item) => item.source_available && item.source_changed
  ).length
  const pendingJobs = countJobs(jobs, isPendingJob)
  const failedJobs = countJobs(jobs, isFailedJob)

  const attentionRows: PersonaVisualManagementAttentionRow[] = []
  addAttentionRow(attentionRows, {
    id: "failed-pack",
    kind: "failed_pack",
    target: "packs",
    count: packCounts.failed
  })
  addAttentionRow(attentionRows, {
    id: "invalid-manifest",
    kind: "invalid_manifest",
    target: "validation",
    count: invalidPackCount
  })
  addAttentionRow(attentionRows, {
    id: "generated-candidates-review",
    kind: "generated_candidates_review",
    target: "candidates",
    count: reviewCandidates
  })
  addAttentionRow(attentionRows, {
    id: "generated-candidates-failed",
    kind: "generated_candidates_failed",
    target: "candidates",
    count: failedCandidates
  })
  addAttentionRow(attentionRows, {
    id: "import-preview-ready",
    kind: "import_preview_ready",
    target: "import",
    count:
      input.importPreview &&
      ["blocked", "completed"].includes(getJobStatus(input.importPreview))
        ? 1
        : 0
  })
  addAttentionRow(attentionRows, {
    id: "import-commit-completed",
    kind: "import_commit_completed",
    target: "import",
    count: isCompletedJob(input.importCommitJob) ? 1 : 0
  })
  addAttentionRow(attentionRows, {
    id: "export-completed",
    kind: "export_completed",
    target: "export",
    count: isCompletedJob(input.exportJob) ? 1 : 0
  })
  addAttentionRow(attentionRows, {
    id: "library-source-unavailable",
    kind: "library_source_unavailable",
    target: "library",
    count: unavailableLibraryItems
  })
  addAttentionRow(attentionRows, {
    id: "library-source-changed",
    kind: "library_source_changed",
    target: "library",
    count: changedLibraryItems
  })
  addAttentionRow(attentionRows, {
    id: "generation-unavailable",
    kind: "generation_unavailable",
    target: "generation",
    count: isGenerationUnavailable(input.generationReadiness) ? 1 : 0
  })
  addAttentionRow(attentionRows, {
    id: "pending-job",
    kind: "pending_job",
    target: "jobs",
    count: pendingJobs
  })
  addAttentionRow(attentionRows, {
    id: "failed-job",
    kind: "failed_job",
    target: "jobs",
    count: failedJobs
  })

  return {
    summary: {
      activePackId: activePack?.id || null,
      activePackTitle: activePack?.title || null,
      packCounts,
      attentionCounts: {
        invalidPackCount,
        reviewCandidates,
        failedCandidates,
        unavailableLibraryItems,
        changedLibraryItems,
        pendingJobs,
        failedJobs
      }
    },
    attentionRows
  }
}
