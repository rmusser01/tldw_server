import React from "react"
import { Button, Modal, Tag, Typography } from "antd"
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  Copy,
  Download,
  Edit3,
  FileSearch,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  Upload,
  XCircle
} from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  activatePersonaVisualPack,
  createPersonaVisualGenerationJob,
  createPersonaVisualImportPreview,
  createPersonaVisualPack,
  copyPersonaVisualStarterPack,
  deletePersonaVisualLibraryItem,
  deactivatePersonaVisualPack,
  downloadPersonaVisualPackExportArchive,
  duplicatePersonaVisualPack,
  getPersonaVisualGenerationReadiness,
  getPersonaVisualImportCommitStatus,
  getPersonaVisualImportPreview,
  getPersonaVisualPackExportJob,
  listPersonaVisualStarterPacks,
  listPersonaVisualLibraryItems,
  listPersonaVisualCandidates,
  listPersonaVisualDuplicateTargets,
  listPersonaVisualPacks,
  PersonaVisualApiError,
  reviewPersonaVisualCandidate,
  savePersonaVisualPackToLibrary,
  startPersonaVisualImportCommit,
  startPersonaVisualPackExport,
  updatePersonaVisualLibraryItem,
  updatePersonaVisualManifest,
  uploadPersonaVisualAsset,
  usePersonaVisualLibraryItem as copyPersonaVisualLibraryItem
} from "@/services/persona-visuals"
import { PERSONA_VISUAL_PACK_ACTIVATED_EVENT } from "@/types/persona-visuals"
import type {
  PersonaVisualAnimation,
  PersonaVisualAsset,
  PersonaVisualAssetRole,
  PersonaVisualAuthoredTrigger,
  PersonaVisualBuiltinStateId,
  PersonaVisualImportBundleAssetGroup,
  PersonaVisualImportBundleAssetSummary,
  PersonaVisualCandidate,
  PersonaVisualDuplicateTarget,
  PersonaVisualLibraryItem,
  PersonaVisualImportCommitStartResponse,
  PersonaVisualFrame,
  PersonaVisualGenerationReadinessResponse,
  PersonaVisualImportConflict,
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualImportRequiredChoice,
  PersonaVisualImportTargetMode,
  PersonaVisualManifest,
  PersonaVisualPack,
  PersonaVisualPackExportResponse,
  PersonaVisualPortabilityJobResponse,
  PersonaVisualRendererImportPreview,
  PersonaVisualStarterPackSummary,
  PersonaVisualStateId
} from "@/types/persona-visuals"
import {
  asPersonaVisualCustomStateId,
  asPersonaVisualStateId
} from "@/types/persona-visuals"
import { getDesignSystemState } from "@/design-system"
import {
  getPersonaVisualDiagnosticToneClassName,
  getPrimaryPersonaVisualDiagnostic,
  type PersonaVisualDiagnostic
} from "../Common/PersonaBuddy/personaVisualDiagnostics"
import {
  classifyPersonaVisualGenerationReadiness,
  type PersonaVisualGenerationReadinessView
} from "./personaVisualGenerationReadiness"
import {
  buildPersonaVisualManagementSummary,
  type PersonaVisualManagementAttentionRow,
  type PersonaVisualManagementModel
} from "./personaVisualManagementSummary"
import {
  formatStarterExpectedAssetGroups,
  getStarterComplexityTierLabel,
  getStarterProductionStatusLabel,
  VisualBuddySetupChoiceCard
} from "./VisualBuddySetupChoiceCard"
import { VisualPackReusePanel } from "./VisualPackReusePanel"
import {
  BUDDY_IMPORT_ARCHIVE_ACCEPT,
  getBuddyImportArchiveFileError,
  NATIVE_PERSONA_VISUAL_PACK_EXTENSION
} from "./buddyBuilderArchive"

type VisualPackEditorProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  isActive?: boolean
  onOpenPersonaVisuals?: (personaId: string) => void
}

type TriggerDraft = {
  source: PersonaVisualAuthoredTrigger["source"]
  match: string
  state: PersonaVisualStateId
  durationMs: string
  priority: string
}

type LibraryEditDraft = {
  title: string
  notes: string
  tags: string
}

type VisualPackTranslate = (
  key: string,
  options?: { defaultValue?: string; [key: string]: unknown }
) => string

const getManagementAttentionCopy = (
  row: PersonaVisualManagementAttentionRow | undefined,
  t: VisualPackTranslate
): { title: string; message: string } => {
  if (!row) {
    return {
      title: t("sidepanel:personaGarden.visuals.management.allClearTitle", {
        defaultValue: "No immediate attention needed"
      }),
      message: t("sidepanel:personaGarden.visuals.management.allClearMessage", {
        defaultValue: "Review and activation controls remain available below."
      })
    }
  }

  const countMessage = (
    key: string,
    singular: string,
    plural: string
  ): string =>
    t(key, {
      count: row.count,
      defaultValue: row.count === 1 ? `1 ${singular}` : `${row.count} ${plural}`,
      defaultValue_one: "{{count}} " + singular,
      defaultValue_other: "{{count}} " + plural
    })

  switch (row.kind) {
    case "failed_pack":
      return {
        title: t("sidepanel:personaGarden.visuals.management.failedPackTitle", {
          defaultValue: "Failed pack needs review"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.failedPackMessage",
          "failed pack needs cleanup or replacement.",
          "failed packs need cleanup or replacement."
        )
      }
    case "invalid_manifest":
      return {
        title: t("sidepanel:personaGarden.visuals.management.invalidManifestTitle", {
          defaultValue: "Activation is blocked"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.invalidManifestMessage",
          "selected manifest issue blocks activation.",
          "selected manifest issues block activation."
        )
      }
    case "generated_candidates_review":
      return {
        title: t("sidepanel:personaGarden.visuals.management.reviewRequiredTitle", {
          defaultValue: "Review required"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.reviewRequiredMessage",
          "generated candidate needs review.",
          "generated candidates need review."
        )
      }
    case "generated_candidates_failed":
      return {
        title: t("sidepanel:personaGarden.visuals.management.failedCandidateTitle", {
          defaultValue: "Generation failed"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.failedCandidateMessage",
          "generated candidate failed.",
          "generated candidates failed."
        )
      }
    case "import_preview_ready":
      return {
        title: t("sidepanel:personaGarden.visuals.management.importPreviewTitle", {
          defaultValue: "Import preview ready"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.importPreviewMessage",
          "import preview needs choices or review.",
          "import previews need choices or review."
        )
      }
    case "import_commit_completed":
      return {
        title: t("sidepanel:personaGarden.visuals.management.importCommitTitle", {
          defaultValue: "Imported draft ready"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.importCommitMessage",
          "import commit created a draft for review.",
          "import commits created drafts for review."
        )
      }
    case "export_completed":
      return {
        title: t("sidepanel:personaGarden.visuals.management.exportCompletedTitle", {
          defaultValue: "Export ready"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.exportCompletedMessage",
          "archive is ready to download.",
          "archives are ready to download."
        )
      }
    case "library_source_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.management.libraryUnavailableTitle", {
          defaultValue: "Library source unavailable"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.libraryUnavailableMessage",
          "library item points to a missing source pack.",
          "library items point to missing source packs."
        )
      }
    case "library_source_changed":
      return {
        title: t("sidepanel:personaGarden.visuals.management.libraryChangedTitle", {
          defaultValue: "Library source changed"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.libraryChangedMessage",
          "library item may be stale.",
          "library items may be stale."
        )
      }
    case "generation_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.management.generationUnavailableTitle", {
          defaultValue: "Generation unavailable"
        }),
        message: t("sidepanel:personaGarden.visuals.management.generationUnavailableMessage", {
          defaultValue: "Check generation readiness before queueing assets."
        })
      }
    case "pending_job":
      return {
        title: t("sidepanel:personaGarden.visuals.management.pendingJobTitle", {
          defaultValue: "Job in progress"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.pendingJobMessage",
          "visual job is still running.",
          "visual jobs are still running."
        )
      }
    case "failed_job":
    default:
      return {
        title: t("sidepanel:personaGarden.visuals.management.failedJobTitle", {
          defaultValue: "Job failed"
        }),
        message: countMessage(
          "sidepanel:personaGarden.visuals.management.failedJobMessage",
          "visual job needs recovery.",
          "visual jobs need recovery."
        )
      }
  }
}

const VisualManagementHeader: React.FC<{
  personaName: string
  model: PersonaVisualManagementModel
  t: VisualPackTranslate
}> = ({ personaName, model, t }) => {
  const topAttention = model.attentionRows[0]
  const attentionCopy = getManagementAttentionCopy(topAttention, t)
  const counts = model.summary.packCounts

  return (
    <section
      data-testid="persona-visual-management-header"
      className="border-b border-border pb-3"
      aria-label={t("sidepanel:personaGarden.visuals.management.ariaLabel", {
        defaultValue: "Persona Visual management summary"
      })}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("sidepanel:personaGarden.visuals.management.personaLabel", {
              defaultValue: "Selected persona"
            })}
          </div>
          <div className="mt-1 text-sm font-medium text-text">{personaName}</div>
          <div className="mt-1 text-xs text-text-muted">
            {model.summary.activePackTitle ||
              t("sidepanel:personaGarden.visuals.management.noActivePack", {
                defaultValue: "No active visual pack"
              })}
          </div>
        </div>
        <div
          className="flex flex-wrap gap-1 text-xs"
          aria-label={t("sidepanel:personaGarden.visuals.management.countsLabel", {
            defaultValue: "Pack lifecycle counts"
          })}
        >
          {(["active", "draft", "review", "archived", "failed"] as const).map(
            (status) => (
              <span
                key={status}
                className="rounded border border-border bg-surface px-2 py-1 text-text-muted"
              >
                <span>
                  {t(`sidepanel:personaGarden.visuals.management.status.${status}`, {
                    defaultValue: status
                  })}
                </span>{" "}
                <span className="font-medium text-text">{counts[status]}</span>
              </span>
            )
          )}
        </div>
      </div>
      <div className="mt-3 rounded-md border border-border bg-surface px-3 py-2 text-xs">
        <div className="font-medium text-text">{attentionCopy.title}</div>
        <div className="mt-1 text-text-muted">{attentionCopy.message}</div>
      </div>
    </section>
  )
}

const getGenerationReadinessCopy = (
  view: PersonaVisualGenerationReadinessView,
  t: VisualPackTranslate
): { title: string; message: string; toneClassName: string } => {
  switch (view.status) {
    case "ready":
      return {
        title: t("sidepanel:personaGarden.visuals.generationReadyTitle", {
          defaultValue: "Generation is ready."
        }),
        message: t("sidepanel:personaGarden.visuals.generationReadyMessage", {
          defaultValue:
            "Queued assets will appear here for review before they can be applied."
        }),
        toneClassName: "border-success/30 bg-success/5 text-success"
      }
    case "jobs_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.generationWorkerUnavailableTitle", {
          defaultValue: "Generation worker is not enabled."
        }),
        message: t("sidepanel:personaGarden.visuals.generationWorkerUnavailableMessage", {
          defaultValue:
            "Enable PERSONA_VISUAL_GENERATION_WORKER_ENABLED before queueing Persona Buddy visual generation jobs."
        }),
        toneClassName: "border-warning/40 bg-warning/10 text-warning"
      }
    case "image_provider_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.generationProviderUnavailableTitle", {
          defaultValue: "No image generation provider is configured."
        }),
        message: t("sidepanel:personaGarden.visuals.generationProviderUnavailableMessage", {
          defaultValue:
            "Enable an image backend before queueing a Persona Buddy visual generation job."
        }),
        toneClassName: "border-warning/40 bg-warning/10 text-warning"
      }
    case "image_adapter_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.generationAdapterUnavailableTitle", {
          defaultValue: "Configured image backend cannot be started."
        }),
        message: t("sidepanel:personaGarden.visuals.generationAdapterUnavailableMessage", {
          defaultValue:
            "Check the selected image backend installation and credentials before queueing a Persona Buddy visual generation job."
        }),
        toneClassName: "border-warning/40 bg-warning/10 text-warning"
      }
    case "dependency_check_failed":
      return {
        title: t("sidepanel:personaGarden.visuals.generationDependencyCheckFailedTitle", {
          defaultValue: "Generation readiness check failed."
        }),
        message: t("sidepanel:personaGarden.visuals.generationDependencyCheckFailedMessage", {
          defaultValue:
            "Check the server logs and image generation configuration before queueing a Persona Buddy visual generation job."
        }),
        toneClassName: "border-destructive/40 bg-destructive/10 text-destructive"
      }
    case "backend_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.generationBackendUnavailableTitle", {
          defaultValue: "Selected image backend is unavailable."
        }),
        message: t("sidepanel:personaGarden.visuals.generationBackendUnavailableMessage", {
          defaultValue:
            "Use an enabled backend name or leave the field blank to use the default backend."
        }),
        toneClassName: "border-warning/40 bg-warning/10 text-warning"
      }
    case "default_backend_unavailable":
      return {
        title: t("sidepanel:personaGarden.visuals.generationDefaultBackendUnavailableTitle", {
          defaultValue: "No default image backend is selected."
        }),
        message: t("sidepanel:personaGarden.visuals.generationDefaultBackendUnavailableMessage", {
          defaultValue:
            "Enter one enabled backend name before queueing this generation job."
        }),
        toneClassName: "border-warning/40 bg-warning/10 text-warning"
      }
    case "error":
      return {
        title: t("sidepanel:personaGarden.visuals.generationReadinessErrorTitle", {
          defaultValue: "Generation readiness could not be checked."
        }),
        message:
          view.errorMessage ||
          t("sidepanel:personaGarden.visuals.generationReadinessErrorMessage", {
            defaultValue:
              "Refresh the visual pack before queueing a generation job."
          }),
        toneClassName: "border-danger/30 bg-danger/5 text-danger"
      }
    case "loading":
    default:
      return {
        title: t("sidepanel:personaGarden.visuals.generationReadinessLoadingTitle", {
          defaultValue: "Checking generation readiness."
        }),
        message: t("sidepanel:personaGarden.visuals.generationReadinessLoadingMessage", {
          defaultValue:
            "Persona Buddy visual generation will be available after setup checks finish."
        }),
        toneClassName: "border-border bg-bg text-text-muted"
      }
  }
}

const REQUIRED_VISUAL_STATES: PersonaVisualBuiltinStateId[] = [
  "idle",
  "listening",
  "thinking",
  "speaking",
  "error"
]

const OPTIONAL_VISUAL_STATES: PersonaVisualBuiltinStateId[] = [
  "wake_armed",
  "tool_running",
  "approval_needed",
  "offline"
]

const VISUAL_STATES: PersonaVisualStateId[] = [
  ...REQUIRED_VISUAL_STATES,
  ...OPTIONAL_VISUAL_STATES
]

const ASSET_ROLES: PersonaVisualAssetRole[] = [
  "frame",
  "still_pose",
  "sprite_sheet",
  "preview",
  "generated_candidate"
]

const TRIGGER_SOURCES: PersonaVisualAuthoredTrigger["source"][] = [
  "live_state",
  "tool_category",
  "tool_name",
  "mcp_runtime"
]

const PORTABLE_VISUAL_PACK_EXTENSION = NATIVE_PERSONA_VISUAL_PACK_EXTENSION
const IMPORT_COMMIT_TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "quarantined"
])
const IMPORT_PREVIEW_TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "blocked",
  "deleted",
  "quarantined"
])

const getImportPreviewFileError = (
  file: File | null,
  t: VisualPackTranslate
): string | null => getBuddyImportArchiveFileError(file, t)

type ImportPreviewStatus =
  | PersonaVisualImportPreviewStartResponse
  | PersonaVisualImportPreviewResponse
  | null

const getImportPreviewJobCopy = (
  preview: ImportPreviewStatus,
  t: VisualPackTranslate
): string | null => {
  if (!preview) return null
  const statusValue = String(preview.status || "").trim()
  const stageValue = String(preview.stage || "").trim()
  const errorMessage =
    "error_message" in preview && typeof preview.error_message === "string"
      ? preview.error_message.trim()
      : ""
  if (
    IMPORT_PREVIEW_TERMINAL_STATUSES.has(statusValue) &&
    !["blocked", "completed"].includes(statusValue)
  ) {
    return (
      errorMessage ||
      t("sidepanel:personaGarden.visuals.importPreviewTerminalStatus", {
        stage: stageValue || "validation",
        status: statusValue,
        defaultValue: `Import preview ${statusValue} during ${stageValue || "validation"}.`
      })
    )
  }
  if (statusValue === "blocked") {
    return t("sidepanel:personaGarden.visuals.importPreviewBlocked", {
      defaultValue:
        "Import preview completed with blockers. Review diagnostics before committing."
    })
  }
  if (statusValue === "completed") {
    return t("sidepanel:personaGarden.visuals.importPreviewCompleted", {
      defaultValue: "Import preview completed. Review the draft plan before committing."
    })
  }
  if (statusValue === "queued") {
    return t("sidepanel:personaGarden.visuals.importPreviewQueued", {
      defaultValue: "Import preview queued. Refresh to check validation status."
    })
  }
  if (statusValue === "processing") {
    return t("sidepanel:personaGarden.visuals.importPreviewProcessing", {
      stage: stageValue,
      stageSuffix: stageValue ? `: ${stageValue}` : ".",
      defaultValue: `Import preview is processing${stageValue ? `: ${stageValue}` : "."}`
    })
  }
  return stageValue
    ? t("sidepanel:personaGarden.visuals.importPreviewStatus", {
        stage: stageValue,
        status: statusValue,
        defaultValue: `Import preview ${statusValue}: ${stageValue}`
      })
    : null
}

const getImportCommitJobCopy = (
  job: PersonaVisualImportCommitStartResponse | PersonaVisualPortabilityJobResponse | null,
  t: VisualPackTranslate
): string | null => {
  if (!job) return null
  const statusValue = String(job.status || "").trim()
  const stageValue = String(job.stage || "").trim()
  const errorMessage =
    "error_message" in job && typeof job.error_message === "string"
      ? job.error_message.trim()
      : ""
  if (["failed", "cancelled", "quarantined"].includes(statusValue)) {
    return (
      errorMessage ||
      t("sidepanel:personaGarden.visuals.importCommitTerminalStatus", {
        stage: stageValue || "commit",
        status: statusValue,
        defaultValue: `Import commit ${statusValue} during ${stageValue || "commit"}.`
      })
    )
  }
  if (statusValue === "completed") {
    return t("sidepanel:personaGarden.visuals.importCommitCompleted", {
      defaultValue: "Import commit completed. Review and activate the new draft when ready."
    })
  }
  if (statusValue === "queued") {
    return t("sidepanel:personaGarden.visuals.importCommitQueued", {
      defaultValue: "Import commit queued. Refresh to check draft creation status."
    })
  }
  if (statusValue === "processing") {
    return t("sidepanel:personaGarden.visuals.importCommitProcessing", {
      stage: stageValue,
      stageSuffix: stageValue ? `: ${stageValue}` : ".",
      defaultValue: `Import commit is processing${stageValue ? `: ${stageValue}` : "."}`
    })
  }
  return stageValue
    ? t("sidepanel:personaGarden.visuals.importCommitStatus", {
        stage: stageValue,
        status: statusValue,
        defaultValue: `Import commit ${statusValue}: ${stageValue}`
      })
    : null
}

const DEFAULT_MANIFEST: PersonaVisualManifest = {
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {},
  animations: {},
  fallbacks: {},
  state_catalog: {},
  authored_triggers: []
}

const DEFAULT_TRIGGER_DRAFT: TriggerDraft = {
  source: "tool_category",
  match: "",
  state: "tool_running",
  durationMs: "2500",
  priority: "20"
}

const cloneJson = <T,>(value: T): T => JSON.parse(JSON.stringify(value))

const normalizeManifest = (
  manifest: PersonaVisualPack["manifest"] | Record<string, unknown> | null | undefined
): PersonaVisualManifest => {
  const source =
    manifest && typeof manifest === "object"
      ? (cloneJson(manifest) as Partial<PersonaVisualManifest>)
      : {}
  return {
    manifest_version: 1,
    renderer_type: "sprite_frames",
    states: {
      ...(source.states || {})
    },
    animations: {
      ...(source.animations || {})
    },
    fallbacks: {
      ...(source.fallbacks || {})
    },
    state_catalog: {
      ...(source.state_catalog || {})
    },
    authored_triggers: Array.isArray(source.authored_triggers)
      ? source.authored_triggers
      : []
  }
}

const normalizeFrames = (
  animation: PersonaVisualAnimation | null | undefined
): PersonaVisualFrame[] => {
  if (!animation) return []
  if (Array.isArray(animation.frames) && animation.frames.length > 0) {
    return animation.frames.map((frame) => ({ ...frame }))
  }
  return (animation.asset_ids || [])
    .filter((assetId) => String(assetId || "").trim())
    .map((assetId) => ({ asset_id: String(assetId) }))
}

const formatStateLabel = (state: PersonaVisualStateId): string =>
  state.replace(/_/g, " ")

const resolveGenerationTargetState = (
  current: PersonaVisualStateId,
  availableStates: PersonaVisualStateId[]
): PersonaVisualStateId => {
  if (availableStates.includes(current)) return current
  return availableStates.includes("thinking")
    ? "thinking"
    : availableStates[0] ?? "thinking"
}

const stringifyPreviewValue = (value: unknown): string => {
  if (value == null) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

const formatPreviewList = (items: unknown[] | null | undefined): string =>
  (items || [])
    .map(stringifyPreviewValue)
    .filter(Boolean)
    .join(" ")

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value && typeof value === "object" && !Array.isArray(value))

const previewString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed || null
}

const previewNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null

const previewBoolean = (value: unknown): boolean | null =>
  typeof value === "boolean" ? value : null

const previewStringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .map(previewString)
        .filter((item): item is string => Boolean(item))
    : []

const previewRoleCategories = (
  value: unknown
): Record<string, string[]> | undefined => {
  if (!isRecord(value)) return undefined
  const categories = Object.entries(value).reduce<Record<string, string[]>>(
    (nextCategories, [category, sourceAssetIds]) => {
      const normalizedCategory = previewString(category)
      const normalizedIds = previewStringList(sourceAssetIds)
      if (normalizedCategory && normalizedIds.length) {
        nextCategories[normalizedCategory] = normalizedIds
      }
      return nextCategories
    },
    {}
  )
  return Object.keys(categories).length ? categories : undefined
}

const BUDDY_PIPELINE_ASSET_GROUP_LABELS: Record<
  PersonaVisualImportBundleAssetGroup,
  string
> = {
  neutral_anchor: "Neutral anchor",
  static_talking_sheet: "Static talking sheet",
  static_reaction_sheet: "Static reaction sheet",
  animation_strips: "Animation strips",
  animation_atlas: "Animation atlas"
}

type BuddyPipelineAssetKind = "source" | "runtime"

const BUDDY_PIPELINE_ASSET_KIND = {
  neutral_anchor: "source",
  static_talking_sheet: "source",
  static_reaction_sheet: "source",
  animation_strips: "runtime",
  animation_atlas: "runtime"
} satisfies Record<PersonaVisualImportBundleAssetGroup, BuddyPipelineAssetKind>

interface BuddyPipelinePacketAssetDiagnostic {
  sourceAssetId: string | null
  assetRole: string | null
  assetGroup: PersonaVisualImportBundleAssetGroup
  assetKind: BuddyPipelineAssetKind
  assetBytesStatus: string | null
  mimeType: string | null
  dimensions: string | null
  manifestReferenced: boolean
}

interface BuddyPipelinePacketDiagnostics {
  assets: BuddyPipelinePacketAssetDiagnostic[]
  manifestAssetReferences: string[]
}

const getBuddyPipelineAssetGroup = (
  value: unknown
): PersonaVisualImportBundleAssetGroup | null => {
  const group = previewString(value)
  if (!group) return null
  return Object.prototype.hasOwnProperty.call(
    BUDDY_PIPELINE_ASSET_GROUP_LABELS,
    group
  )
    ? (group as PersonaVisualImportBundleAssetGroup)
    : null
}

const getBuddyPipelineAssetKind = (
  group: PersonaVisualImportBundleAssetGroup
): BuddyPipelineAssetKind =>
  BUDDY_PIPELINE_ASSET_KIND[group]

const getBuddyPipelineAssetDimensions = (
  asset: PersonaVisualImportBundleAssetSummary
): string | null => {
  const width = previewNumber(asset.width)
  const height = previewNumber(asset.height)
  return width !== null && height !== null ? `${width}x${height}` : null
}

const getBuddyPipelinePacketDiagnostics = (
  preview: PersonaVisualImportPreviewResponse | null
): BuddyPipelinePacketDiagnostics | null => {
  const summary = preview?.bundle_summary
  if (!summary) return null
  const manifestAssetReferences = previewStringList(
    summary.manifest_asset_references
  )
  const manifestReferenceSet = new Set(manifestAssetReferences)
  const rawAssets = Array.isArray(summary.assets) ? summary.assets : []
  const assets = rawAssets.reduce<BuddyPipelinePacketAssetDiagnostic[]>(
    (nextAssets, rawAsset) => {
      if (!isRecord(rawAsset)) return nextAssets
      const asset = rawAsset as PersonaVisualImportBundleAssetSummary
      const assetGroup = getBuddyPipelineAssetGroup(asset.asset_group)
      if (!assetGroup) return nextAssets
      const sourceAssetId = previewString(asset.source_asset_id)
      const manifestReferenced =
        previewBoolean(asset.manifest_referenced) === true ||
        Boolean(sourceAssetId && manifestReferenceSet.has(sourceAssetId))
      nextAssets.push({
        sourceAssetId,
        assetRole: previewString(asset.asset_role),
        assetGroup,
        assetKind: getBuddyPipelineAssetKind(assetGroup),
        assetBytesStatus: previewString(asset.asset_bytes_status),
        mimeType: previewString(asset.mime_type),
        dimensions: getBuddyPipelineAssetDimensions(asset),
        manifestReferenced
      })
      return nextAssets
    },
    []
  )
  return assets.length ? { assets, manifestAssetReferences } : null
}

const getBuddyPipelineAssetDetailText = (
  asset: BuddyPipelinePacketAssetDiagnostic,
  manifestReferencedLabel: string
): string =>
  [
    asset.sourceAssetId,
    asset.assetRole,
    asset.assetBytesStatus,
    asset.mimeType,
    asset.dimensions,
    asset.manifestReferenced ? manifestReferencedLabel : null
  ]
    .filter(Boolean)
    .join(" / ")

const getRendererImportPreview = (
  preview: PersonaVisualImportPreviewResponse | null
): PersonaVisualRendererImportPreview | null => {
  const rawPreview = preview?.proposed_plan?.renderer_import_preview
  if (!isRecord(rawPreview)) return null
  return {
    status: previewString(rawPreview.status),
    renderer_type: previewString(rawPreview.renderer_type),
    manifest_version: previewNumber(rawPreview.manifest_version),
    renderer_contract_version: previewNumber(rawPreview.renderer_contract_version),
    can_commit: previewBoolean(rawPreview.can_commit),
    activation_eligible: previewBoolean(rawPreview.activation_eligible),
    blockers: previewStringList(rawPreview.blockers),
    warnings: previewStringList(rawPreview.warnings),
    normalized_role_categories: previewRoleCategories(
      rawPreview.normalized_role_categories
    ),
    setup_status: previewString(rawPreview.setup_status),
    setup_blockers: previewStringList(rawPreview.setup_blockers),
    disabled_reason: previewString(rawPreview.disabled_reason)
  }
}

const getRendererImportPreviewRoleSummary = (
  rendererPreview: PersonaVisualRendererImportPreview | null
): string => {
  const categories = rendererPreview?.normalized_role_categories || {}
  return Object.entries(categories)
    .map(([category, sourceAssetIds]) => `${category}: ${sourceAssetIds.join(", ")}`)
    .join(" ")
}

const getImportCommitBlockers = (
  preview: PersonaVisualImportPreviewResponse | null,
  rendererPreview: PersonaVisualRendererImportPreview | null
): string[] => {
  const planBlockers = previewStringList(preview?.proposed_plan?.commit_blockers)
  const rendererBlockers = [
    ...(rendererPreview?.blockers || []),
    ...(rendererPreview?.setup_blockers || []),
    rendererPreview?.disabled_reason || null
  ].filter((item): item is string => Boolean(item))
  return Array.from(new Set([...planBlockers, ...rendererBlockers]))
}

const isImportPreviewCommitEligible = (
  preview: PersonaVisualImportPreviewResponse | null,
  rendererPreview: PersonaVisualRendererImportPreview | null
): boolean => {
  if (!preview || preview.status !== "completed") return false
  if (preview.proposed_plan?.commit_eligible === false) return false
  if (rendererPreview?.can_commit === false) return false
  return true
}

const isFullImportPreview = (
  preview: PersonaVisualImportPreviewStartResponse | PersonaVisualImportPreviewResponse | null
): preview is PersonaVisualImportPreviewResponse =>
  Boolean(preview && "bundle_summary" in preview)

const getImportTargetChoice = (
  preview: PersonaVisualImportPreviewResponse | null
): PersonaVisualImportRequiredChoice | null =>
  preview?.required_choices.find(
    (choice) => choice.choice_id === "import_target_mode"
  ) || null

const getAllowedImportTargetModes = (
  choice: PersonaVisualImportRequiredChoice | null
): PersonaVisualImportTargetMode[] => {
  const modes = choice?.allowed_target_modes || []
  return modes.length ? modes : ["create_new"]
}

const getReplaceableImportConflicts = (
  conflicts: PersonaVisualImportConflict[]
): PersonaVisualImportConflict[] =>
  conflicts.filter(
    (conflict) =>
      Boolean(conflict.pack_id) &&
      (conflict.allowed_choices || []).includes("replace_draft")
  )

const getImportConflictLabel = (conflict: PersonaVisualImportConflict): string => {
  const title = conflict.pack_title || conflict.pack_id || "Draft pack"
  const status = conflict.pack_status ? ` (${conflict.pack_status})` : ""
  return `${title}${status}`
}

const getDefaultImportDraftTitle = (
  preview: PersonaVisualImportPreviewResponse | null
): string => {
  const title = preview?.bundle_summary?.pack_title
  return typeof title === "string" ? title : ""
}

const formatImportPreviewSummary = (
  preview: PersonaVisualImportPreviewStartResponse | PersonaVisualImportPreviewResponse
): string => {
  if (!isFullImportPreview(preview)) return `${preview.status} ${preview.stage}`
  const summary = preview.bundle_summary || {}
  const title =
    typeof summary.pack_title === "string" && summary.pack_title.trim()
      ? summary.pack_title
      : "Portable visual pack"
  const assetCount =
    typeof summary.asset_count === "number" ? `${summary.asset_count} assets` : null
  const assetsWithBytes =
    typeof summary.assets_with_bytes === "number"
      ? `${summary.assets_with_bytes} with bytes`
      : null
  return [title, assetCount, assetsWithBytes].filter(Boolean).join(" / ")
}

const buildExportFilename = (pack: PersonaVisualPack): string => {
  const slug = (pack.title || pack.id || "persona-visual-pack")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return `${slug || "persona-visual-pack"}${PORTABLE_VISUAL_PACK_EXTENSION}`
}

const parseLibraryTagsInput = (value: string): string[] => {
  const seen = new Set<string>()
  const tags: string[] = []
  for (const rawTag of value.split(",")) {
    const tag = rawTag.trim()
    if (!tag || seen.has(tag)) continue
    seen.add(tag)
    tags.push(tag)
  }
  return tags
}

const getLibrarySourcePersonaName = (item: PersonaVisualLibraryItem): string =>
  item.source_persona_name ||
  item.source_persona_id ||
  "Unavailable persona"

const getLibrarySourcePackTitle = (item: PersonaVisualLibraryItem): string =>
  item.source_pack_title ||
  item.source_pack_id ||
  "Unavailable pack"

const upsertLibraryItem = (
  items: PersonaVisualLibraryItem[],
  item: PersonaVisualLibraryItem
): PersonaVisualLibraryItem[] => {
  const found = items.some((current) => current.id === item.id)
  if (!found) return [item, ...items]
  return items.map((current) => (current.id === item.id ? item : current))
}

const getAnimationIds = (manifest: PersonaVisualManifest): string[] =>
  Object.keys(manifest.animations || {}).sort((a, b) => a.localeCompare(b))

const getPackAssets = (pack: PersonaVisualPack | null): PersonaVisualAsset[] => {
  if (!pack) return []
  if (Array.isArray(pack.assets)) return pack.assets
  return Object.values(pack.assets_by_id || {})
}

const mergePack = (
  packs: PersonaVisualPack[],
  nextPack: PersonaVisualPack
): PersonaVisualPack[] => {
  const found = packs.some((pack) => pack.id === nextPack.id)
  if (!found) return [nextPack, ...packs]
  return packs.map((pack) => (pack.id === nextPack.id ? nextPack : pack))
}

const resolveAnimationForState = (
  manifest: PersonaVisualManifest,
  state: PersonaVisualStateId,
  seen = new Set<PersonaVisualStateId>()
): string | null => {
  if (seen.has(state)) return null
  seen.add(state)
  const direct = manifest.states?.[state]?.animation_id
  if (direct && manifest.animations?.[direct]) return direct
  for (const fallback of manifest.fallbacks?.[state] || []) {
    const resolved = resolveAnimationForState(manifest, fallback, seen)
    if (resolved) return resolved
  }
  return null
}

const validateManifestForActivation = (
  manifest: PersonaVisualManifest
): string[] => {
  const missing = REQUIRED_VISUAL_STATES.filter(
    (state) => !resolveAnimationForState(manifest, state)
  )
  return missing.length
    ? [`Missing required visual states: ${missing.map(formatStateLabel).join(", ")}`]
    : []
}

const parseNumberOrUndefined = (value: string): number | undefined => {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

const replaceAnimation = (
  manifest: PersonaVisualManifest,
  animationId: string,
  updater: (animation: PersonaVisualAnimation) => PersonaVisualAnimation
): PersonaVisualManifest => ({
  ...manifest,
  animations: {
    ...manifest.animations,
    [animationId]: updater(manifest.animations[animationId] || {})
  }
})

const sourceBadgeClass =
  "inline-flex min-h-[22px] items-center rounded border px-1.5 py-0.5 text-xs font-medium"
const sourceAvailableBadgeClass = `${sourceBadgeClass} border-state-ready/30 bg-state-ready/10 text-state-ready`
const sourceUnavailableBadgeClass = `${sourceBadgeClass} border-state-unavailable/30 bg-state-unavailable/10 text-state-unavailable`
const LOADING_STATE_LABEL = getDesignSystemState("loading").label

export const VisualPackEditor: React.FC<VisualPackEditorProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  isActive = false,
  onOpenPersonaVisuals
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const loadingLabel = t("common:loading.title", {
    defaultValue: LOADING_STATE_LABEL
  })
  const refreshLabel = t("common:refresh", "Refresh")
  const unknownLabel = t("common:unknown", { defaultValue: "unknown" })
  const [packs, setPacks] = React.useState<PersonaVisualPack[]>([])
  const [selectedPackId, setSelectedPackId] = React.useState("")
  const [activePackId, setActivePackId] = React.useState("")
  const [packsLoaded, setPacksLoaded] = React.useState(false)
  const [packsLoadedPersonaId, setPacksLoadedPersonaId] = React.useState("")
  const [draftTitle, setDraftTitle] = React.useState("")
  const [draftManifest, setDraftManifest] =
    React.useState<PersonaVisualManifest>(DEFAULT_MANIFEST)
  const [selectedAnimationId, setSelectedAnimationId] = React.useState("")
  const [newAnimationId, setNewAnimationId] = React.useState("")
  const [selectedAddFrameAssetId, setSelectedAddFrameAssetId] = React.useState("")
  const [uploadRole, setUploadRole] = React.useState<PersonaVisualAssetRole>("frame")
  const [selectedUploadFile, setSelectedUploadFile] = React.useState<File | null>(null)
  const [exportJob, setExportJob] = React.useState<
    PersonaVisualPackExportResponse | PersonaVisualPortabilityJobResponse | null
  >(null)
  const [exportingPack, setExportingPack] = React.useState(false)
  const [refreshingExport, setRefreshingExport] = React.useState(false)
  const [downloadingExport, setDownloadingExport] = React.useState(false)
  const [selectedImportPreviewFile, setSelectedImportPreviewFile] =
    React.useState<File | null>(null)
  const [importPreview, setImportPreview] = React.useState<
    PersonaVisualImportPreviewStartResponse | PersonaVisualImportPreviewResponse | null
  >(null)
  const [importTargetMode, setImportTargetMode] =
    React.useState<PersonaVisualImportTargetMode | "">("create_new")
  const [importTargetChoicePreviewId, setImportTargetChoicePreviewId] =
    React.useState("")
  const [importReplacePackId, setImportReplacePackId] = React.useState("")
  const [importDraftTitle, setImportDraftTitle] = React.useState("")
  const [importCommitJob, setImportCommitJob] = React.useState<
    PersonaVisualImportCommitStartResponse | PersonaVisualPortabilityJobResponse | null
  >(null)
  const [duplicateTargets, setDuplicateTargets] = React.useState<
    PersonaVisualDuplicateTarget[]
  >([])
  const [duplicateTargetsLoading, setDuplicateTargetsLoading] = React.useState(false)
  const [duplicateTargetId, setDuplicateTargetId] = React.useState("")
  const [duplicateTitle, setDuplicateTitle] = React.useState("")
  const [duplicatingPack, setDuplicatingPack] = React.useState(false)
  const [lastDuplicatedPersonaId, setLastDuplicatedPersonaId] = React.useState("")
  const [libraryItems, setLibraryItems] = React.useState<PersonaVisualLibraryItem[]>([])
  const [libraryLoading, setLibraryLoading] = React.useState(false)
  const [savingToLibrary, setSavingToLibrary] = React.useState(false)
  const [libraryMutatingItemId, setLibraryMutatingItemId] = React.useState("")
  const [libraryTargetByItemId, setLibraryTargetByItemId] = React.useState<
    Record<string, string>
  >({})
  const [libraryEditingItemId, setLibraryEditingItemId] = React.useState("")
  const [libraryEditDraft, setLibraryEditDraft] =
    React.useState<LibraryEditDraft>({
      title: "",
      notes: "",
      tags: ""
    })
  const [starterPacks, setStarterPacks] = React.useState<
    PersonaVisualStarterPackSummary[]
  >([])
  const [starterCatalogLoading, setStarterCatalogLoading] = React.useState(false)
  const [starterCatalogError, setStarterCatalogError] =
    React.useState<string | null>(null)
  const [copyingStarterId, setCopyingStarterId] = React.useState("")
  const [starterPickerOpen, setStarterPickerOpen] = React.useState(false)
  const [previewingImport, setPreviewingImport] = React.useState(false)
  const [refreshingImportPreview, setRefreshingImportPreview] = React.useState(false)
  const [committingImport, setCommittingImport] = React.useState(false)
  const [refreshingImportCommit, setRefreshingImportCommit] = React.useState(false)
  const [triggerDraft, setTriggerDraft] =
    React.useState<TriggerDraft>(DEFAULT_TRIGGER_DRAFT)
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [uploading, setUploading] = React.useState(false)
  const [activating, setActivating] = React.useState(false)
  const [deactivating, setDeactivating] = React.useState(false)
  const [candidates, setCandidates] = React.useState<PersonaVisualCandidate[]>([])
  const [candidatesLoading, setCandidatesLoading] = React.useState(false)
  const [generationPrompt, setGenerationPrompt] = React.useState("")
  const [generationTargetState, setGenerationTargetState] =
    React.useState<PersonaVisualStateId>("thinking")
  const [generationBackend, setGenerationBackend] = React.useState("")
  const [generationReadiness, setGenerationReadiness] =
    React.useState<PersonaVisualGenerationReadinessResponse | null>(null)
  const [generationReadinessLoading, setGenerationReadinessLoading] =
    React.useState(false)
  const [generationReadinessError, setGenerationReadinessError] =
    React.useState<string | null>(null)
  const [enqueueingGeneration, setEnqueueingGeneration] = React.useState(false)
  const [reviewingCandidateId, setReviewingCandidateId] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const [statusMessage, setStatusMessage] = React.useState<string | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement | null>(null)
  const importPreviewInputRef = React.useRef<HTMLInputElement | null>(null)
  const draftTitleInputRef = React.useRef<HTMLInputElement | null>(null)
  const duplicateTargetSelectRef = React.useRef<HTMLSelectElement | null>(null)
  const libraryPanelRef = React.useRef<HTMLDivElement | null>(null)
  const generationReadinessRequestIdRef = React.useRef(0)
  const duplicateTargetsRequestIdRef = React.useRef(0)
  const libraryRequestIdRef = React.useRef(0)
  const packListRequestIdRef = React.useRef(0)
  const candidatesRequestIdRef = React.useRef(0)
  const candidateReviewRequestIdRef = React.useRef(0)
  const draftCreateRequestIdRef = React.useRef(0)
  const assetUploadRequestIdRef = React.useRef(0)
  const manifestSaveRequestIdRef = React.useRef(0)
  const starterCatalogRequestIdRef = React.useRef(0)
  const starterCopyRequestIdRef = React.useRef(0)
  const importPreviewRequestIdRef = React.useRef(0)
  const importCommitRequestIdRef = React.useRef(0)
  const importCommitInFlightRef = React.useRef(false)
  const importCommitRefreshInFlightRef = React.useRef(false)
  const selectedPersonaIdRef = React.useRef(selectedPersonaId)
  const selectedPackIdRef = React.useRef("")

  selectedPersonaIdRef.current = selectedPersonaId

  const selectPackId = React.useCallback((packId: string) => {
    selectedPackIdRef.current = packId
    setSelectedPackId(packId)
  }, [])

  React.useEffect(() => {
    selectedPackIdRef.current = selectedPackId
  }, [selectedPackId])

  const packStateMatchesSelectedPersona =
    Boolean(selectedPersonaId) && packsLoadedPersonaId === selectedPersonaId
  const visiblePacks = React.useMemo(
    () =>
      packStateMatchesSelectedPersona
        ? packs.filter((pack) => pack.persona_id === selectedPersonaId)
        : [],
    [packStateMatchesSelectedPersona, packs, selectedPersonaId]
  )
  const selectedPack = React.useMemo(
    () =>
      visiblePacks.find((pack) => pack.id === selectedPackId) ??
      visiblePacks[0] ??
      null,
    [selectedPackId, visiblePacks]
  )
  const activePack = React.useMemo(
    () => visiblePacks.find((pack) => pack.id === activePackId) ?? null,
    [activePackId, visiblePacks]
  )
  const availableDuplicateTargets = React.useMemo(
    () => duplicateTargets.filter((target) => target.id !== selectedPersonaId),
    [duplicateTargets, selectedPersonaId]
  )
  const selectedDuplicateTarget =
    availableDuplicateTargets.find((target) => target.id === duplicateTargetId) ?? null
  const selectedPackLibraryItem = React.useMemo(
    () =>
      selectedPack
        ? libraryItems.find(
            (item) =>
              item.source_persona_id === selectedPack.persona_id &&
              item.source_pack_id === selectedPack.id
          ) ?? null
        : null,
    [libraryItems, selectedPack]
  )
  const getLibraryTargetOptions = React.useCallback(
    (item: PersonaVisualLibraryItem): PersonaVisualDuplicateTarget[] =>
      duplicateTargets.filter((target) => target.id !== item.source_persona_id),
    [duplicateTargets]
  )
  const assets = React.useMemo(() => getPackAssets(selectedPack), [selectedPack])
  const animationIds = React.useMemo(
    () => getAnimationIds(draftManifest),
    [draftManifest]
  )
  const customVisualStates = React.useMemo(
    () =>
      Object.keys(draftManifest.state_catalog || {}).sort((a, b) =>
        a.localeCompare(b)
      ).map(asPersonaVisualCustomStateId),
    [draftManifest.state_catalog]
  )
  const activeVisualStates: PersonaVisualStateId[] = React.useMemo(
    () => [...VISUAL_STATES, ...customVisualStates],
    [customVisualStates]
  )
  const editableFallbackStates: PersonaVisualStateId[] = React.useMemo(
    () => [...OPTIONAL_VISUAL_STATES, ...customVisualStates],
    [customVisualStates]
  )
  const normalizedGenerationTargetState = React.useMemo(
    () => resolveGenerationTargetState(generationTargetState, activeVisualStates),
    [activeVisualStates, generationTargetState]
  )
  const selectedAnimation =
    selectedAnimationId && draftManifest.animations[selectedAnimationId]
      ? draftManifest.animations[selectedAnimationId]
      : null
  const selectedFrames = React.useMemo(
    () => normalizeFrames(selectedAnimation),
    [selectedAnimation]
  )
  const validationErrors = React.useMemo(
    () => validateManifestForActivation(draftManifest),
    [draftManifest]
  )
  const packHealthDiagnostic: PersonaVisualDiagnostic | null = React.useMemo(
    () =>
      selectedPack
        ? getPrimaryPersonaVisualDiagnostic({
            pack: {
              ...selectedPack,
              manifest: draftManifest
            },
            visualState: "idle"
          })
        : null,
    [draftManifest, selectedPack]
  )
  const generationReadinessView = React.useMemo(
    () =>
      classifyPersonaVisualGenerationReadiness(generationReadiness, generationBackend, {
        isLoading: generationReadinessLoading,
        errorMessage: generationReadinessError
      }),
    [generationBackend, generationReadiness, generationReadinessError, generationReadinessLoading]
  )
  const generationReadinessCopy = React.useMemo(
    () => getGenerationReadinessCopy(generationReadinessView, t),
    [generationReadinessView, t]
  )
  const managementModel = React.useMemo(
    () =>
      buildPersonaVisualManagementSummary({
        packs: visiblePacks,
        activePack,
        selectedPack,
        validationErrors,
        candidates,
        libraryItems,
        importPreview,
        importCommitJob,
        exportJob,
        generationReadiness: generationReadinessView
      }),
    [
      activePack,
      candidates,
      exportJob,
      generationReadinessView,
      importCommitJob,
      importPreview,
      libraryItems,
      selectedPack,
      validationErrors,
      visiblePacks
    ]
  )

  const loadPacks = React.useCallback(
    async (
      options: {
        preferredPackId?: string
        fallbackPack?: PersonaVisualPack
      } = {}
    ): Promise<boolean> => {
      if (!isActive || !selectedPersonaId) {
        packListRequestIdRef.current += 1
        setPacks([])
        setActivePackId("")
        setLoading(false)
        setError(null)
        setPacksLoaded(false)
        setPacksLoadedPersonaId("")
        selectPackId("")
        setDraftManifest(DEFAULT_MANIFEST)
        setCandidates([])
        return false
      }
      const targetPersonaId = selectedPersonaId
      const requestId = packListRequestIdRef.current + 1
      packListRequestIdRef.current = requestId
      const isLatestRequest = () =>
        packListRequestIdRef.current === requestId &&
        selectedPersonaIdRef.current === targetPersonaId
      setLoading(true)
      setPacksLoaded(false)
      setError(null)
      try {
        const response = await listPersonaVisualPacks(targetPersonaId)
        if (!isLatestRequest()) return false
        const listedPacks = options.fallbackPack
          ? mergePack(response.packs || [], options.fallbackPack)
          : response.packs || []
        const activePack =
          response.active_pack ??
          listedPacks.find((pack) => pack.status === "active") ??
          null
        const nextPacks = activePack ? mergePack(listedPacks, activePack) : listedPacks
        setActivePackId(activePack?.id || "")
        setPacks(nextPacks)
        const preferred =
          (options.preferredPackId
            ? nextPacks.find((pack) => pack.id === options.preferredPackId)
            : null) ??
          nextPacks.find((pack) => pack.id === selectedPackIdRef.current) ??
          activePack ??
          nextPacks[0] ??
          null
        selectPackId(preferred?.id || "")
        if (!preferred) {
          setDraftManifest(DEFAULT_MANIFEST)
          setSelectedAnimationId("")
        }
        setPacksLoaded(true)
        setPacksLoadedPersonaId(targetPersonaId)
        return true
      } catch (loadError) {
        if (!isLatestRequest()) return false
        if (options.fallbackPack) {
          const fallbackPack = options.fallbackPack
          setPacks((current) => mergePack(current, fallbackPack))
          setActivePackId((current) =>
            fallbackPack.status === "active" ? fallbackPack.id : current
          )
          setPacksLoaded(true)
          setPacksLoadedPersonaId(targetPersonaId)
          selectPackId(options.preferredPackId || fallbackPack.id)
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load visual packs."
        )
        return false
      } finally {
        if (isLatestRequest()) setLoading(false)
      }
    },
    [isActive, selectedPersonaId, selectPackId]
  )

  const loadStarterCatalog = React.useCallback(async () => {
    if (!isActive || !selectedPersonaId) {
      starterCatalogRequestIdRef.current += 1
      setStarterPacks([])
      setStarterCatalogError(null)
      setStarterCatalogLoading(false)
      return
    }
    const requestId = starterCatalogRequestIdRef.current + 1
    starterCatalogRequestIdRef.current = requestId
    const isLatestRequest = () => starterCatalogRequestIdRef.current === requestId
    setStarterCatalogLoading(true)
    setStarterCatalogError(null)
    try {
      const response = await listPersonaVisualStarterPacks()
      if (isLatestRequest()) setStarterPacks(response.starter_packs || [])
    } catch (starterError) {
      if (isLatestRequest()) {
        setStarterPacks([])
        setStarterCatalogError(
          starterError instanceof Error
            ? starterError.message
            : "Failed to load starter packs."
        )
      }
    } finally {
      if (isLatestRequest()) setStarterCatalogLoading(false)
    }
  }, [isActive, selectedPersonaId])

  const loadCandidates = React.useCallback(async () => {
    const packId = selectedPack?.id || ""
    if (!isActive || !selectedPersonaId || !packId) {
      candidatesRequestIdRef.current += 1
      setCandidates([])
      setCandidatesLoading(false)
      return
    }
    const targetPersonaId = selectedPersonaId
    const targetPackId = packId
    const requestId = candidatesRequestIdRef.current + 1
    candidatesRequestIdRef.current = requestId
    const isCurrentCandidatesRequest = () =>
      candidatesRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId &&
      selectedPackIdRef.current === targetPackId
    setCandidatesLoading(true)
    try {
      const response = await listPersonaVisualCandidates(targetPersonaId, targetPackId)
      if (isCurrentCandidatesRequest()) setCandidates(response.candidates || [])
    } catch (loadError) {
      if (isCurrentCandidatesRequest()) {
        setCandidates([])
        if (!(loadError instanceof PersonaVisualApiError && loadError.status === 404)) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Failed to load generated candidates."
          )
        }
      }
    } finally {
      if (isCurrentCandidatesRequest()) setCandidatesLoading(false)
    }
  }, [isActive, selectedPersonaId, selectedPack?.id])

  const loadGenerationReadiness = React.useCallback(async () => {
    const packId = selectedPack?.id || ""
    if (!isActive || !selectedPersonaId || !packId) {
      generationReadinessRequestIdRef.current += 1
      setGenerationReadiness(null)
      setGenerationReadinessError(null)
      setGenerationReadinessLoading(false)
      return
    }
    const targetPersonaId = selectedPersonaId
    const targetPackId = packId
    const requestId = generationReadinessRequestIdRef.current + 1
    generationReadinessRequestIdRef.current = requestId
    const isLatestRequest = () =>
      generationReadinessRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId &&
      selectedPackIdRef.current === targetPackId
    setGenerationReadinessLoading(true)
    setGenerationReadinessError(null)
    try {
      const response = await getPersonaVisualGenerationReadiness(
        targetPersonaId,
        targetPackId
      )
      if (isLatestRequest()) setGenerationReadiness(response)
    } catch (loadError) {
      if (isLatestRequest()) {
        setGenerationReadiness(null)
        setGenerationReadinessError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to check generation readiness."
        )
      }
    } finally {
      if (isLatestRequest()) setGenerationReadinessLoading(false)
    }
  }, [isActive, selectedPersonaId, selectedPack?.id])

  const loadDuplicateTargets = React.useCallback(async () => {
    if (!isActive || !selectedPersonaId) {
      duplicateTargetsRequestIdRef.current += 1
      setDuplicateTargets([])
      setDuplicateTargetId("")
      setDuplicateTargetsLoading(false)
      return
    }
    const requestId = duplicateTargetsRequestIdRef.current + 1
    duplicateTargetsRequestIdRef.current = requestId
    const targetPersonaId = selectedPersonaId
    const isLatestRequest = () =>
      duplicateTargetsRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId
    setDuplicateTargets([])
    setDuplicateTargetId("")
    setDuplicateTargetsLoading(true)
    try {
      const targets = await listPersonaVisualDuplicateTargets()
      if (!isLatestRequest()) return
      const available = targets.filter((target) => target.id !== targetPersonaId)
      setDuplicateTargets(targets)
      setDuplicateTargetId(available[0]?.id ?? "")
    } catch (loadError) {
      if (isLatestRequest()) {
        setDuplicateTargets([])
        setDuplicateTargetId("")
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load persona duplicate targets."
        )
      }
    } finally {
      if (isLatestRequest()) setDuplicateTargetsLoading(false)
    }
  }, [isActive, selectedPersonaId])

  const loadLibrary = React.useCallback(async () => {
    if (!isActive || !selectedPersonaId) {
      libraryRequestIdRef.current += 1
      setLibraryItems([])
      setLibraryLoading(false)
      return
    }
    const requestId = libraryRequestIdRef.current + 1
    libraryRequestIdRef.current = requestId
    const isLatestRequest = () => libraryRequestIdRef.current === requestId
    setLibraryLoading(true)
    try {
      const response = await listPersonaVisualLibraryItems()
      if (isLatestRequest()) setLibraryItems(response.items || [])
    } catch (loadError) {
      if (!isLatestRequest()) return
      setLibraryItems([])
      if (!(loadError instanceof PersonaVisualApiError && loadError.status === 404)) {
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load personal visual library."
        )
      }
    } finally {
      if (isLatestRequest()) setLibraryLoading(false)
    }
  }, [isActive, selectedPersonaId])

  React.useEffect(() => {
    void loadPacks()
  }, [loadPacks])

  React.useEffect(() => {
    void loadStarterCatalog()
  }, [loadStarterCatalog])

  React.useEffect(() => {
    void loadCandidates()
  }, [loadCandidates])

  React.useEffect(() => {
    void loadGenerationReadiness()
  }, [loadGenerationReadiness])

  React.useEffect(() => {
    void loadDuplicateTargets()
  }, [loadDuplicateTargets])

  React.useEffect(() => {
    void loadLibrary()
  }, [loadLibrary])

  React.useEffect(() => {
    setLibraryTargetByItemId((current) => {
      const next: Record<string, string> = {}
      for (const item of libraryItems) {
        const targets = getLibraryTargetOptions(item)
        const currentTarget = current[item.id]
        next[item.id] =
          currentTarget && targets.some((target) => target.id === currentTarget)
            ? currentTarget
            : targets[0]?.id ?? ""
      }
      return next
    })
  }, [getLibraryTargetOptions, libraryItems])

  React.useEffect(() => {
    if (!selectedPack) {
      setDraftManifest(DEFAULT_MANIFEST)
      setSelectedAnimationId("")
      return
    }
    const nextManifest = normalizeManifest(selectedPack.manifest)
    setDraftManifest(nextManifest)
    const ids = getAnimationIds(nextManifest)
    setSelectedAnimationId((current) =>
      current && nextManifest.animations[current] ? current : ids[0] || ""
    )
  }, [selectedPack?.id, selectedPack?.version])

  React.useEffect(() => {
    if (!selectedAnimationId && animationIds.length > 0) {
      setSelectedAnimationId(animationIds[0])
    }
    if (selectedAnimationId && !animationIds.includes(selectedAnimationId)) {
      setSelectedAnimationId(animationIds[0] || "")
    }
  }, [animationIds, selectedAnimationId])

  React.useEffect(() => {
    if (generationTargetState !== normalizedGenerationTargetState) {
      setGenerationTargetState(normalizedGenerationTargetState)
    }
  }, [generationTargetState, normalizedGenerationTargetState])

  React.useEffect(() => {
    draftCreateRequestIdRef.current += 1
    assetUploadRequestIdRef.current += 1
    manifestSaveRequestIdRef.current += 1
    setExportJob(null)
    setLastDuplicatedPersonaId("")
    setLibraryEditingItemId("")
    setLibraryEditDraft({ title: "", notes: "", tags: "" })
    setSaving(false)
    setUploading(false)
    setExportingPack(false)
    setRefreshingExport(false)
    setDownloadingExport(false)
    setDuplicatingPack(false)
    setSavingToLibrary(false)
    setLibraryMutatingItemId("")
    setReviewingCandidateId("")
    setEnqueueingGeneration(false)
    setActivating(false)
    setDeactivating(false)
  }, [selectedPersonaId, selectedPack?.id])

  React.useEffect(() => {
    importPreviewRequestIdRef.current += 1
    importCommitRequestIdRef.current += 1
    importCommitInFlightRef.current = false
    importCommitRefreshInFlightRef.current = false
    setImportPreview(null)
    setImportCommitJob(null)
    setSelectedImportPreviewFile(null)
    setPreviewingImport(false)
    setRefreshingImportPreview(false)
    setCommittingImport(false)
    setRefreshingImportCommit(false)
    if (importPreviewInputRef.current) importPreviewInputRef.current.value = ""
  }, [selectedPersonaId])

  React.useEffect(() => {
    starterCopyRequestIdRef.current += 1
    setCopyingStarterId("")
    setStarterPickerOpen(false)
  }, [selectedPersonaId])

  React.useEffect(() => {
    setDuplicateTitle(selectedPack ? `Copy of ${selectedPack.title}` : "")
    setLastDuplicatedPersonaId("")
  }, [selectedPack?.id])

  React.useEffect(() => {
    candidateReviewRequestIdRef.current += 1
  }, [selectedPersonaId, selectedPack?.id])

  const updateManifest = React.useCallback(
    (updater: (manifest: PersonaVisualManifest) => PersonaVisualManifest) => {
      setDraftManifest((current) => normalizeManifest(updater(normalizeManifest(current))))
      setStatusMessage(null)
    },
    []
  )

  const focusDraftTitleInput = React.useCallback(() => {
    draftTitleInputRef.current?.scrollIntoView?.({ block: "center" })
    draftTitleInputRef.current?.focus()
  }, [])

  const focusLibraryPanel = React.useCallback(() => {
    libraryPanelRef.current?.scrollIntoView?.({ block: "start" })
    libraryPanelRef.current?.focus()
  }, [])

  const openImportArchivePicker = React.useCallback(() => {
    importPreviewInputRef.current?.scrollIntoView?.({ block: "center" })
    importPreviewInputRef.current?.click()
    importPreviewInputRef.current?.focus()
  }, [])

  const focusDuplicateControls = React.useCallback(() => {
    duplicateTargetSelectRef.current?.scrollIntoView?.({ block: "center" })
    duplicateTargetSelectRef.current?.focus()
  }, [])

  const handleCreateDraft = async () => {
    const title = draftTitle.trim()
    if (!selectedPersonaId || !title) return
    const targetPersonaId = selectedPersonaId
    const requestId = draftCreateRequestIdRef.current + 1
    draftCreateRequestIdRef.current = requestId
    const isCurrentDraftCreateRequest = () =>
      draftCreateRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId
    setSaving(true)
    setError(null)
    try {
      const created = await createPersonaVisualPack(targetPersonaId, {
        title,
        manifest: DEFAULT_MANIFEST
      })
      if (!isCurrentDraftCreateRequest()) return
      setPacks((current) => mergePack(current, created))
      selectPackId(created.id)
      setDraftTitle("")
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.created", {
          defaultValue: "Draft created."
        })
      )
    } catch (createError) {
      if (isCurrentDraftCreateRequest()) {
        setError(
          createError instanceof Error
            ? createError.message
            : t("sidepanel:personaGarden.visuals.createError", {
                defaultValue: "Failed to create visual pack."
              })
        )
      }
    } finally {
      if (isCurrentDraftCreateRequest()) setSaving(false)
    }
  }

  const handleCopyStarterPack = React.useCallback(
    async (starterPackId: string) => {
      const normalizedStarterId = String(starterPackId || "").trim()
      const targetPersonaId = selectedPersonaId
      if (!targetPersonaId || !normalizedStarterId) return
      const requestId = starterCopyRequestIdRef.current + 1
      starterCopyRequestIdRef.current = requestId
      const isCurrentCopyRequest = () =>
        starterCopyRequestIdRef.current === requestId &&
        selectedPersonaIdRef.current === targetPersonaId
      setCopyingStarterId(normalizedStarterId)
      setError(null)
      try {
        const copied = await copyPersonaVisualStarterPack(normalizedStarterId, {
          target_persona_id: targetPersonaId
        })
        if (!isCurrentCopyRequest()) return
        setPacks((current) => mergePack(current, copied))
        selectPackId(copied.id)
        await loadPacks({
          preferredPackId: copied.id,
          fallbackPack: copied
        })
        if (!isCurrentCopyRequest()) return
        setStatusMessage(
          "Default visual copied as an inactive draft. Review it, then activate when ready."
        )
        setStarterPickerOpen(false)
      } catch (copyError) {
        if (isCurrentCopyRequest()) {
          setError(
            copyError instanceof Error
              ? copyError.message
              : "Failed to copy starter pack."
          )
        }
      } finally {
        if (starterCopyRequestIdRef.current === requestId) {
          setCopyingStarterId("")
        }
      }
    },
    [loadPacks, selectedPersonaId, selectPackId]
  )

  const handleUploadAsset = async () => {
    if (!selectedPersonaId || !selectedPack || !selectedUploadFile) return
    const targetPersonaId = selectedPersonaId
    const targetPackId = selectedPack.id
    const requestId = assetUploadRequestIdRef.current + 1
    assetUploadRequestIdRef.current = requestId
    const isCurrentAssetUploadRequest = () =>
      assetUploadRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId &&
      selectedPackIdRef.current === targetPackId
    setUploading(true)
    setError(null)
    try {
      const asset = await uploadPersonaVisualAsset(
        targetPersonaId,
        targetPackId,
        selectedUploadFile,
        uploadRole
      )
      if (!isCurrentAssetUploadRequest()) return
      const nextPack = {
        ...selectedPack,
        assets: [...getPackAssets(selectedPack), asset]
      }
      setPacks((current) => mergePack(current, nextPack))
      setSelectedUploadFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ""
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.uploaded", {
          defaultValue: "Asset uploaded."
        })
      )
    } catch (uploadError) {
      if (isCurrentAssetUploadRequest()) {
        setError(
          uploadError instanceof Error
            ? uploadError.message
            : t("sidepanel:personaGarden.visuals.uploadError", {
                defaultValue: "Failed to upload asset."
              })
        )
      }
    } finally {
      if (isCurrentAssetUploadRequest()) setUploading(false)
    }
  }

  const handleStartExport = async () => {
    if (!selectedPersonaId || !selectedPack) return
    setExportingPack(true)
    setError(null)
    try {
      const job = await startPersonaVisualPackExport(selectedPersonaId, selectedPack.id)
      setExportJob(job)
      setStatusMessage("Export job queued.")
    } catch (exportError) {
      setError(
        exportError instanceof Error
          ? exportError.message
          : "Failed to queue visual pack export."
      )
    } finally {
      setExportingPack(false)
    }
  }

  const handleRefreshExport = async () => {
    if (!selectedPersonaId || !selectedPack || !exportJob?.job_id) return
    setRefreshingExport(true)
    setError(null)
    try {
      const job = await getPersonaVisualPackExportJob(
        selectedPersonaId,
        selectedPack.id,
        exportJob.job_id
      )
      setExportJob(job)
    } catch (refreshError) {
      setError(
        refreshError instanceof Error
          ? refreshError.message
          : "Failed to refresh visual pack export."
      )
    } finally {
      setRefreshingExport(false)
    }
  }

  const handleDownloadExport = async () => {
    if (!selectedPersonaId || !selectedPack || !exportJob?.job_id) return
    setDownloadingExport(true)
    setError(null)
    try {
      await downloadPersonaVisualPackExportArchive(
        selectedPersonaId,
        selectedPack.id,
        exportJob.job_id,
        buildExportFilename(selectedPack)
      )
      setStatusMessage("Export archive downloaded.")
    } catch (downloadError) {
      setError(
        downloadError instanceof Error
          ? downloadError.message
          : "Failed to download visual pack export."
      )
    } finally {
      setDownloadingExport(false)
    }
  }

  const handleDuplicatePack = async () => {
    if (!selectedPersonaId || !selectedPack || !duplicateTargetId) return
    setDuplicatingPack(true)
    setError(null)
    try {
      const duplicated = await duplicatePersonaVisualPack(
        selectedPersonaId,
        selectedPack.id,
        {
          target_persona_id: duplicateTargetId,
          title: duplicateTitle.trim() || `Copy of ${selectedPack.title}`
        }
      )
      setLastDuplicatedPersonaId(duplicated.persona_id)
      const targetName =
        selectedDuplicateTarget?.name ||
        selectedDuplicateTarget?.id ||
        duplicated.persona_id
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.duplicatedDraft", {
          defaultValue: `Duplicated as a draft for ${targetName}. Review and activate it from that persona's Visuals tab.`
        })
      )
    } catch (duplicateError) {
      setError(
        duplicateError instanceof Error
          ? duplicateError.message
          : t("sidepanel:personaGarden.visuals.duplicateError", {
              defaultValue: "Failed to duplicate visual pack."
            })
      )
    } finally {
      setDuplicatingPack(false)
    }
  }

  const handleSavePackToLibrary = async () => {
    if (!selectedPersonaId || !selectedPack) return
    setSavingToLibrary(true)
    setError(null)
    try {
      const item = await savePersonaVisualPackToLibrary(
        selectedPersonaId,
        selectedPack.id,
        selectedPackLibraryItem
          ? {
              title: selectedPackLibraryItem.title,
              notes: selectedPackLibraryItem.notes ?? null,
              tags: selectedPackLibraryItem.tags
            }
          : {
              title: selectedPack.title
            }
      )
      setLibraryItems((current) => upsertLibraryItem(current, item))
      setStatusMessage("Saved to personal library.")
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Failed to save pack to personal library."
      )
    } finally {
      setSavingToLibrary(false)
    }
  }

  const handleStartLibraryEdit = (item: PersonaVisualLibraryItem) => {
    setLibraryEditingItemId(item.id)
    setLibraryEditDraft({
      title: item.title,
      notes: item.notes || "",
      tags: (item.tags || []).join(", ")
    })
  }

  const handleSaveLibraryEdit = async (item: PersonaVisualLibraryItem) => {
    const title = libraryEditDraft.title.trim()
    if (!title) return
    setLibraryMutatingItemId(item.id)
    setError(null)
    try {
      const updated = await updatePersonaVisualLibraryItem(item.id, {
        title,
        notes: libraryEditDraft.notes.trim() || null,
        tags: parseLibraryTagsInput(libraryEditDraft.tags),
        expected_version: item.version ?? null
      })
      setLibraryItems((current) => upsertLibraryItem(current, updated))
      setLibraryEditingItemId("")
      setLibraryEditDraft({ title: "", notes: "", tags: "" })
      setStatusMessage("Library item updated.")
    } catch (updateError) {
      setError(
        updateError instanceof Error
          ? updateError.message
          : "Failed to update library item."
      )
    } finally {
      setLibraryMutatingItemId("")
    }
  }

  const handleRemoveLibraryItem = async (item: PersonaVisualLibraryItem) => {
    setLibraryMutatingItemId(item.id)
    setError(null)
    try {
      await deletePersonaVisualLibraryItem(item.id)
      setLibraryItems((current) => current.filter((currentItem) => currentItem.id !== item.id))
      setStatusMessage("Library item removed.")
      if (libraryEditingItemId === item.id) {
        setLibraryEditingItemId("")
        setLibraryEditDraft({ title: "", notes: "", tags: "" })
      }
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : "Failed to remove library item."
      )
    } finally {
      setLibraryMutatingItemId("")
    }
  }

  const handleUseLibraryItem = async (item: PersonaVisualLibraryItem) => {
    const targetPersonaId = libraryTargetByItemId[item.id] || ""
    if (!targetPersonaId || !item.source_available) return
    setLibraryMutatingItemId(item.id)
    setError(null)
    try {
      const duplicated = await copyPersonaVisualLibraryItem(item.id, {
        target_persona_id: targetPersonaId
      })
      if (duplicated.persona_id === selectedPersonaId) {
        setPacks((current) => mergePack(current, duplicated))
        selectPackId(duplicated.id)
      }
      setLastDuplicatedPersonaId(duplicated.persona_id)
      const target =
        duplicateTargets.find((candidate) => candidate.id === duplicated.persona_id) ??
        duplicateTargets.find((candidate) => candidate.id === targetPersonaId)
      const targetName = target?.name || target?.id || duplicated.persona_id
      setStatusMessage(
        `Library item copied as a draft for ${targetName}. Review and activate it from that persona's Visuals tab.`
      )
    } catch (useError) {
      setError(
        useError instanceof Error
          ? useError.message
          : "Failed to use library item."
      )
    } finally {
      setLibraryMutatingItemId("")
    }
  }

  const handleStartImportPreview = async () => {
    if (!selectedPersonaId || !selectedImportPreviewFile) return
    const fileError = getImportPreviewFileError(selectedImportPreviewFile, t)
    if (fileError) {
      setError(fileError)
      return
    }
    const targetPersonaId = selectedPersonaId
    const requestId = importPreviewRequestIdRef.current + 1
    importPreviewRequestIdRef.current = requestId
    const isCurrentImportPreviewRequest = () =>
      importPreviewRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId
    setPreviewingImport(true)
    setError(null)
    try {
      const preview = await createPersonaVisualImportPreview(
        targetPersonaId,
        selectedImportPreviewFile
      )
      if (!isCurrentImportPreviewRequest()) return
      setImportPreview(preview)
      setImportCommitJob(null)
      setImportTargetMode("create_new")
      setImportTargetChoicePreviewId("")
      setImportReplacePackId("")
      setImportDraftTitle("")
      setSelectedImportPreviewFile(null)
      if (importPreviewInputRef.current) importPreviewInputRef.current.value = ""
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.importPreviewQueuedStatus", {
          defaultValue: "Import preview queued."
        })
      )
    } catch (previewError) {
      if (isCurrentImportPreviewRequest()) {
        setError(
          previewError instanceof Error
            ? previewError.message
            : "Failed to queue visual pack import preview."
        )
      }
    } finally {
      if (isCurrentImportPreviewRequest()) setPreviewingImport(false)
    }
  }

  const handleRefreshImportPreview = async () => {
    if (!selectedPersonaId || !importPreview?.preview_id) return
    const targetPersonaId = selectedPersonaId
    const previewId = importPreview.preview_id
    const requestId = importPreviewRequestIdRef.current + 1
    importPreviewRequestIdRef.current = requestId
    const isCurrentImportPreviewRequest = () =>
      importPreviewRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId
    setRefreshingImportPreview(true)
    setError(null)
    try {
      const preview = await getPersonaVisualImportPreview(
        targetPersonaId,
        previewId
      )
      if (!isCurrentImportPreviewRequest()) return
      setImportPreview(preview)
    } catch (refreshError) {
      if (isCurrentImportPreviewRequest()) {
        setError(
          refreshError instanceof Error
            ? refreshError.message
            : "Failed to refresh visual pack import preview."
        )
      }
    } finally {
      if (isCurrentImportPreviewRequest()) setRefreshingImportPreview(false)
    }
  }

  const handleStartImportCommit = async () => {
    if (importCommitInFlightRef.current || committingImport) return
    if (!selectedPersonaId || !fullImportPreview?.preview_id) return
    if (fullImportPreview.status !== "completed") return
    if (!importPreviewCommitEligible) return
    if (importConflictChoiceRequired && !importConflictChoiceValid) return
    const targetPersonaId = selectedPersonaId
    const previewId = fullImportPreview.preview_id
    const requestId = importCommitRequestIdRef.current + 1
    importCommitRequestIdRef.current = requestId
    const isCurrentImportCommitRequest = () =>
      importCommitRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId
    importCommitInFlightRef.current = true
    setCommittingImport(true)
    setError(null)
    try {
      const targetMode = importTargetMode || "create_new"
      const title = importDraftTitle.trim()
      const job = await startPersonaVisualImportCommit(
        targetPersonaId,
        previewId,
        {
          trust_mode: "untrusted_import",
          target_mode: targetMode,
          target_pack_id:
            targetMode === "replace_draft" ? importReplacePackId : null,
          title: title || null
        }
      )
      if (!isCurrentImportCommitRequest()) return
      setImportCommitJob(job)
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.importCommitQueuedStatus", {
          defaultValue:
            "Import commit queued. Imported packs remain drafts until activated."
        })
      )
    } catch (commitError) {
      if (isCurrentImportCommitRequest()) {
        setError(
          commitError instanceof Error
            ? commitError.message
            : "Failed to queue visual pack import commit."
        )
      }
    } finally {
      if (isCurrentImportCommitRequest()) {
        importCommitInFlightRef.current = false
        setCommittingImport(false)
      }
    }
  }

  const handleRefreshImportCommit = async () => {
    if (!selectedPersonaId || !importCommitJob?.job_id) return
    if (importCommitInFlightRef.current) return
    if (importCommitRefreshInFlightRef.current || refreshingImportCommit) return
    const targetPersonaId = selectedPersonaId
    const jobId = importCommitJob.job_id
    const requestId = importCommitRequestIdRef.current + 1
    importCommitRequestIdRef.current = requestId
    const isCurrentImportCommitRequest = () =>
      importCommitRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId
    importCommitRefreshInFlightRef.current = true
    setRefreshingImportCommit(true)
    setError(null)
    try {
      const job = await getPersonaVisualImportCommitStatus(
        targetPersonaId,
        jobId
      )
      if (!isCurrentImportCommitRequest()) return
      setImportCommitJob(job)
      if (job.status === "completed" && job.pack_id) {
        const refreshed = await loadPacks({ preferredPackId: job.pack_id })
        if (!isCurrentImportCommitRequest()) return
        if (refreshed) {
          setStatusMessage(
            t("sidepanel:personaGarden.visuals.importCommitCompleted", {
              defaultValue:
                "Import commit completed. Review and activate the new draft when ready."
            })
          )
        } else {
          setStatusMessage(null)
        }
      }
    } catch (refreshError) {
      if (isCurrentImportCommitRequest()) {
        setError(
          refreshError instanceof Error
            ? refreshError.message
            : "Failed to refresh visual pack import commit."
        )
      }
    } finally {
      if (importCommitRequestIdRef.current === requestId) {
        importCommitRefreshInFlightRef.current = false
      }
      if (isCurrentImportCommitRequest()) setRefreshingImportCommit(false)
    }
  }

  const handleSaveManifest = async () => {
    if (!selectedPersonaId || !selectedPack) return
    const targetPersonaId = selectedPersonaId
    const targetPackId = selectedPack.id
    const requestId = manifestSaveRequestIdRef.current + 1
    manifestSaveRequestIdRef.current = requestId
    const isCurrentManifestSaveRequest = () =>
      manifestSaveRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId &&
      selectedPackIdRef.current === targetPackId
    setSaving(true)
    setError(null)
    try {
      const saved = await updatePersonaVisualManifest(
        targetPersonaId,
        targetPackId,
        {
          manifest: draftManifest,
          expected_version: selectedPack.version ?? null
        }
      )
      if (!isCurrentManifestSaveRequest()) return
      setPacks((current) =>
        mergePack(current, {
          ...saved,
          assets: getPackAssets(saved).length ? getPackAssets(saved) : assets
        })
      )
      selectPackId(saved.id)
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.saved", {
          defaultValue: "Manifest saved."
        })
      )
    } catch (saveError) {
      if (isCurrentManifestSaveRequest()) {
        setError(
          saveError instanceof Error
            ? saveError.message
            : t("sidepanel:personaGarden.visuals.saveError", {
                defaultValue: "Failed to save visual manifest."
              })
        )
      }
    } finally {
      if (isCurrentManifestSaveRequest()) setSaving(false)
    }
  }

  const handleActivate = async () => {
    if (!selectedPersonaId || !selectedPack || validationErrors.length) return
    const targetPersonaId = selectedPersonaId
    const targetPackId = selectedPack.id
    setActivating(true)
    setError(null)
    try {
      const active = await activatePersonaVisualPack(targetPersonaId, targetPackId)
      if (
        selectedPersonaIdRef.current !== targetPersonaId ||
        selectedPackIdRef.current !== targetPackId
      ) {
        return
      }
      setPacks((current) =>
        current.map((pack) =>
          pack.id === active.id
            ? { ...active, assets: getPackAssets(active).length ? getPackAssets(active) : assets }
            : pack.status === "active"
              ? { ...pack, status: "archived" }
              : pack
        )
      )
      setActivePackId(active.id)
      selectPackId(active.id)
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.activated", {
          defaultValue: "Visual pack activated."
        })
      )
      if (typeof window !== "undefined") {
        window.dispatchEvent(
          new CustomEvent(PERSONA_VISUAL_PACK_ACTIVATED_EVENT, {
            detail: {
              personaId: selectedPersonaId,
              packId: active.id
            }
          })
        )
      }
    } catch (activateError) {
      if (
        selectedPersonaIdRef.current === targetPersonaId &&
        selectedPackIdRef.current === targetPackId
      ) {
        setError(
          activateError instanceof Error
            ? activateError.message
            : t("sidepanel:personaGarden.visuals.activateError", {
                defaultValue: "Failed to activate visual pack."
              })
        )
      }
    } finally {
      if (
        selectedPersonaIdRef.current === targetPersonaId &&
        selectedPackIdRef.current === targetPackId
      ) {
        setActivating(false)
      }
    }
  }

  const handleDeactivate = async () => {
    if (!selectedPersonaId) return
    const targetPersonaId = selectedPersonaId
    setDeactivating(true)
    setError(null)
    try {
      await deactivatePersonaVisualPack(targetPersonaId)
      if (selectedPersonaIdRef.current !== targetPersonaId) return
      setPacks((current) =>
        current.map((pack) =>
          pack.status === "active" ? { ...pack, status: "archived" } : pack
        )
      )
      setActivePackId("")
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.deactivated", {
          defaultValue: "Active visual pack deactivated."
        })
      )
    } catch (deactivateError) {
      if (selectedPersonaIdRef.current === targetPersonaId) {
        setError(
          deactivateError instanceof Error
            ? deactivateError.message
            : t("sidepanel:personaGarden.visuals.deactivateError", {
                defaultValue: "Failed to deactivate visual pack."
              })
        )
      }
    } finally {
      if (selectedPersonaIdRef.current === targetPersonaId) setDeactivating(false)
    }
  }

  const handleEnqueueGeneration = async () => {
    if (
      !selectedPersonaId ||
      !selectedPack ||
      !generationPrompt.trim() ||
      !generationReadinessView.canQueue
    ) {
      return
    }
    const targetPersonaId = selectedPersonaId
    const targetPackId = selectedPack.id
    const isCurrentGenerationRequest = () =>
      selectedPersonaIdRef.current === targetPersonaId &&
      selectedPackIdRef.current === targetPackId
    setEnqueueingGeneration(true)
    setError(null)
    try {
      const targetState = resolveGenerationTargetState(
        generationTargetState,
        activeVisualStates
      )
      if (targetState !== generationTargetState) {
        setGenerationTargetState(targetState)
      }
      const job = await createPersonaVisualGenerationJob(
        targetPersonaId,
        targetPackId,
        {
          prompt: generationPrompt.trim(),
          target_state: targetState || null,
          backend: generationBackend.trim() || null
        }
      )
      if (!isCurrentGenerationRequest()) return
      setGenerationPrompt("")
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.generationQueued", {
          defaultValue: `Generation job ${job.job_id} queued.`
        })
      )
    } catch (generationError) {
      if (isCurrentGenerationRequest()) {
        setError(
          generationError instanceof Error
            ? generationError.message
            : t("sidepanel:personaGarden.visuals.generationError", {
                defaultValue: "Failed to queue generation job."
              })
        )
      }
    } finally {
      if (isCurrentGenerationRequest()) setEnqueueingGeneration(false)
    }
  }

  const handleReviewCandidate = async (
    candidateId: string,
    status: "accepted" | "rejected"
  ) => {
    if (!selectedPersonaId || !selectedPack) return
    const targetPersonaId = selectedPersonaId
    const targetPackId = selectedPack.id
    const requestId = candidateReviewRequestIdRef.current + 1
    candidateReviewRequestIdRef.current = requestId
    const isCurrentCandidateReview = () =>
      candidateReviewRequestIdRef.current === requestId &&
      selectedPersonaIdRef.current === targetPersonaId &&
      selectedPackIdRef.current === targetPackId
    setReviewingCandidateId(candidateId)
    setError(null)
    try {
      const updated = await reviewPersonaVisualCandidate(
        targetPersonaId,
        targetPackId,
        candidateId,
        {
          status,
          failure_reason: status === "rejected" ? "Rejected in editor." : null
        }
      )
      if (!isCurrentCandidateReview()) return
      setCandidates((current) =>
        current.map((candidate) =>
          candidate.id === candidateId ? { ...candidate, ...updated } : candidate
        )
      )
      if (status === "accepted") {
        void loadPacks()
      }
      setStatusMessage(
        status === "accepted"
          ? t("sidepanel:personaGarden.visuals.candidateAccepted", {
              defaultValue: "Candidate accepted."
            })
          : t("sidepanel:personaGarden.visuals.candidateRejected", {
              defaultValue: "Candidate rejected."
            })
      )
    } catch (reviewError) {
      if (isCurrentCandidateReview()) {
        setError(
          reviewError instanceof Error
            ? reviewError.message
            : t("sidepanel:personaGarden.visuals.reviewError", {
                defaultValue: "Failed to review candidate."
              })
        )
      }
    } finally {
      if (isCurrentCandidateReview()) setReviewingCandidateId("")
    }
  }

  const handleStateMappingChange = (
    state: PersonaVisualStateId,
    animationId: string
  ) => {
    updateManifest((manifest) => {
      const states = { ...manifest.states }
      if (animationId) {
        states[state] = { animation_id: animationId }
      } else {
        delete states[state]
      }
      return { ...manifest, states }
    })
  }

  const handleFallbackChange = (state: PersonaVisualStateId, value: string) => {
    updateManifest((manifest) => {
      const fallbacks = { ...(manifest.fallbacks || {}) }
      const nextValues = value
        .split(",")
        .map((item) => asPersonaVisualStateId(item.trim()))
        .filter((item) => activeVisualStates.includes(item))
      if (nextValues.length) {
        fallbacks[state] = nextValues
      } else {
        delete fallbacks[state]
      }
      return { ...manifest, fallbacks }
    })
  }

  const handleAddAnimation = () => {
    const animationId = newAnimationId.trim()
    if (!animationId || draftManifest.animations[animationId]) return
    updateManifest((manifest) => ({
      ...manifest,
      animations: {
        ...manifest.animations,
        [animationId]: {
          frames: assets[0] ? [{ asset_id: assets[0].id }] : [],
          frame_rate: 1,
          loop: true,
          alignment: { x: 0.5, y: 1 },
          preview_frame: 0
        }
      }
    }))
    setSelectedAnimationId(animationId)
    setNewAnimationId("")
  }

  const handleAnimationFieldChange = (
    field: keyof PersonaVisualAnimation,
    value: unknown
  ) => {
    if (!selectedAnimationId) return
    updateManifest((manifest) =>
      replaceAnimation(manifest, selectedAnimationId, (animation) => ({
        ...animation,
        frames: normalizeFrames(animation),
        [field]: value
      }))
    )
  }

  const handleMoveFrame = (index: number, direction: -1 | 1) => {
    if (!selectedAnimationId) return
    updateManifest((manifest) =>
      replaceAnimation(manifest, selectedAnimationId, (animation) => {
        const frames = normalizeFrames(animation)
        const target = index + direction
        if (target < 0 || target >= frames.length) return { ...animation, frames }
        const nextFrames = [...frames]
        const [frame] = nextFrames.splice(index, 1)
        nextFrames.splice(target, 0, frame)
        return { ...animation, frames: nextFrames }
      })
    )
  }

  const handleFrameChange = (
    index: number,
    updater: (frame: PersonaVisualFrame) => PersonaVisualFrame
  ) => {
    if (!selectedAnimationId) return
    updateManifest((manifest) =>
      replaceAnimation(manifest, selectedAnimationId, (animation) => {
        const frames = normalizeFrames(animation)
        if (!frames[index]) return { ...animation, frames }
        const nextFrames = frames.map((frame, frameIndex) =>
          frameIndex === index ? updater(frame) : frame
        )
        return { ...animation, frames: nextFrames }
      })
    )
  }

  const handleFrameRegionChange = (
    index: number,
    key: "x" | "y" | "width" | "height",
    value: string
  ) => {
    handleFrameChange(index, (frame) => {
      const currentRegion = frame.region || {
        x: 0,
        y: 0,
        width: 0,
        height: 0
      }
      const parsed = parseNumberOrUndefined(value) ?? 0
      return {
        ...frame,
        region: {
          ...currentRegion,
          [key]: parsed
        }
      }
    })
  }

  const handleAddFrame = () => {
    if (!selectedAnimationId || !selectedAddFrameAssetId) return
    updateManifest((manifest) =>
      replaceAnimation(manifest, selectedAnimationId, (animation) => ({
        ...animation,
        frames: [
          ...normalizeFrames(animation),
          { asset_id: selectedAddFrameAssetId }
        ]
      }))
    )
  }

  const handleAddTrigger = () => {
    const match = triggerDraft.match.trim()
    if (!match) return
    updateManifest((manifest) => ({
      ...manifest,
      authored_triggers: [
        ...(manifest.authored_triggers || []),
        {
          id: `trigger-${Date.now()}`,
          source: triggerDraft.source,
          match,
          state: triggerDraft.state,
          duration_ms: parseNumberOrUndefined(triggerDraft.durationMs) ?? 2500,
          priority: parseNumberOrUndefined(triggerDraft.priority) ?? 20
        }
      ]
    }))
    setTriggerDraft(DEFAULT_TRIGGER_DRAFT)
  }

  const renderAnimationOptions = () => (
    <>
      <option value="">None</option>
      {animationIds.map((animationId) => (
        <option key={animationId} value={animationId}>
          {animationId}
        </option>
      ))}
    </>
  )

  const renderLibraryItem = (item: PersonaVisualLibraryItem) => {
    const isEditing = libraryEditingItemId === item.id
    const targetOptions = getLibraryTargetOptions(item)
    const targetId = libraryTargetByItemId[item.id] || ""
    const isMutating = libraryMutatingItemId === item.id

    return (
      <div
        key={item.id}
        data-testid={`persona-visual-library-item-${item.id}`}
        className="rounded border border-border bg-bg p-2 text-xs"
      >
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="font-medium text-text">{item.title}</div>
            <div className="mt-1 text-text-muted">
              {`${getLibrarySourcePersonaName(item)} / ${getLibrarySourcePackTitle(item)}`}
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            {item.source_available ? (
              <span className={sourceAvailableBadgeClass}>available</span>
            ) : (
              <span className={sourceUnavailableBadgeClass}>unavailable</span>
            )}
            {item.source_changed ? <Tag color="orange">source changed</Tag> : null}
            {item.tags.map((tag) => (
              <Tag key={tag}>{tag}</Tag>
            ))}
          </div>
        </div>
        {item.notes ? <div className="mt-2 text-text-muted">{item.notes}</div> : null}
        {isEditing ? (
          <div className="mt-3 grid gap-2 md:grid-cols-[minmax(160px,1fr)_minmax(160px,1fr)_minmax(120px,1fr)_auto]">
            <label className="text-xs text-text-muted">
              <span className="mb-1 block">Title</span>
              <input
                data-testid={`persona-visual-library-edit-title-${item.id}`}
                className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                value={libraryEditDraft.title}
                onChange={(event) =>
                  setLibraryEditDraft((current) => ({
                    ...current,
                    title: event.target.value
                  }))
                }
              />
            </label>
            <label className="text-xs text-text-muted">
              <span className="mb-1 block">Notes</span>
              <input
                data-testid={`persona-visual-library-edit-notes-${item.id}`}
                className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                value={libraryEditDraft.notes}
                onChange={(event) =>
                  setLibraryEditDraft((current) => ({
                    ...current,
                    notes: event.target.value
                  }))
                }
              />
            </label>
            <label className="text-xs text-text-muted">
              <span className="mb-1 block">Tags</span>
              <input
                data-testid={`persona-visual-library-edit-tags-${item.id}`}
                className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                value={libraryEditDraft.tags}
                onChange={(event) =>
                  setLibraryEditDraft((current) => ({
                    ...current,
                    tags: event.target.value
                  }))
                }
              />
            </label>
            <div className="flex items-end gap-1">
              <Button
                data-testid={`persona-visual-library-save-edit-${item.id}`}
                size="small"
                type="primary"
                loading={isMutating}
                disabled={!libraryEditDraft.title.trim()}
                onClick={() => void handleSaveLibraryEdit(item)}
              >
                Save
              </Button>
              <Button
                size="small"
                onClick={() => {
                  setLibraryEditingItemId("")
                  setLibraryEditDraft({ title: "", notes: "", tags: "" })
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <div className="mt-3 grid gap-2 md:grid-cols-[minmax(160px,1fr)_auto_auto_auto]">
            <label className="text-xs text-text-muted">
              <span className="mb-1 block">Use for persona</span>
              <select
                data-testid={`persona-visual-library-target-${item.id}`}
                className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                value={targetId}
                disabled={!item.source_available || !targetOptions.length}
                onChange={(event) =>
                  setLibraryTargetByItemId((current) => ({
                    ...current,
                    [item.id]: event.target.value
                  }))
                }
              >
                {targetOptions.map((target) => (
                  <option key={target.id} value={target.id}>
                    {target.name || target.id}
                  </option>
                ))}
              </select>
            </label>
            <Button
              data-testid={`persona-visual-library-use-${item.id}`}
              className="self-end"
              size="small"
              icon={<Copy className="h-3.5 w-3.5" />}
              loading={isMutating}
              disabled={!item.source_available || !targetId}
              onClick={() => void handleUseLibraryItem(item)}
            >
              Use as draft
            </Button>
            <Button
              data-testid={`persona-visual-library-edit-${item.id}`}
              className="self-end"
              size="small"
              icon={<Edit3 className="h-3.5 w-3.5" />}
              onClick={() => handleStartLibraryEdit(item)}
            >
              Edit details
            </Button>
            <Button
              data-testid={`persona-visual-library-remove-${item.id}`}
              className="self-end"
              size="small"
              danger
              icon={<Trash2 className="h-3.5 w-3.5" />}
              loading={isMutating}
              onClick={() => void handleRemoveLibraryItem(item)}
            >
              Remove
            </Button>
          </div>
        )}
      </div>
    )
  }

  const exportWarnings =
    exportJob && "warnings" in exportJob && Array.isArray(exportJob.warnings)
      ? exportJob.warnings
      : []
  const fullImportPreview = isFullImportPreview(importPreview) ? importPreview : null
  const importPreviewFileError = getImportPreviewFileError(selectedImportPreviewFile, t)
  const importPreviewJobCopy = getImportPreviewJobCopy(importPreview, t)
  const importCommitJobCopy = getImportCommitJobCopy(importCommitJob, t)
  const importCommitJobPackIsSelected =
    !importCommitJob ||
    !("pack_id" in importCommitJob) ||
    !importCommitJob.pack_id ||
    selectedPack?.id === importCommitJob.pack_id
  const importPreviewWarnings = fullImportPreview
    ? [
        ...(fullImportPreview.validation_warnings || []),
        ...(fullImportPreview.target_warnings || [])
      ]
    : []
  const importPreviewConflicts = fullImportPreview?.conflicts || []
  const importPreviewPlan = fullImportPreview?.proposed_plan || null
  const rendererImportPreview = getRendererImportPreview(fullImportPreview)
  const rendererImportRoleSummary =
    getRendererImportPreviewRoleSummary(rendererImportPreview)
  const buddyPipelinePacketDiagnostics =
    getBuddyPipelinePacketDiagnostics(fullImportPreview)
  const buddyPipelineSourceAssets =
    buddyPipelinePacketDiagnostics?.assets.filter(
      (asset) => asset.assetKind === "source"
    ) || []
  const buddyPipelineRuntimeAssets =
    buddyPipelinePacketDiagnostics?.assets.filter(
      (asset) => asset.assetKind === "runtime"
    ) || []
  const importCommitBlockers = getImportCommitBlockers(
    fullImportPreview,
    rendererImportPreview
  )
  const importTargetChoice = getImportTargetChoice(fullImportPreview)
  const importAllowedTargetModes = getAllowedImportTargetModes(importTargetChoice)
  const replaceableImportConflicts = getReplaceableImportConflicts(importPreviewConflicts)
  const importConflictChoiceRequired = Boolean(importTargetChoice)
  const importConflictChoiceSelected =
    !importConflictChoiceRequired ||
    (Boolean(importTargetMode) &&
      importTargetChoicePreviewId === fullImportPreview?.preview_id)
  const importConflictChoiceValid =
    !importConflictChoiceRequired ||
    (importConflictChoiceSelected &&
      importAllowedTargetModes.includes(
        importTargetMode as PersonaVisualImportTargetMode
      ) &&
      (importTargetMode !== "replace_draft" || Boolean(importReplacePackId)))
  const canCommitImportPreview = fullImportPreview?.status === "completed"
  const importPreviewCommitEligible = isImportPreviewCommitEligible(
    fullImportPreview,
    rendererImportPreview
  )
  const importCommitStatus = importCommitJob?.status || null
  const importCommitIsTerminal = importCommitStatus
    ? IMPORT_COMMIT_TERMINAL_STATUSES.has(importCommitStatus)
    : false
  const visibleCandidates = selectedPack
    ? candidates.filter(
        (candidate) =>
          candidate.persona_id === selectedPersonaId &&
          candidate.pack_id === selectedPack.id
      )
    : []
  const selectedActivePack = visiblePacks.find(
    (pack) => pack.id === activePackId && pack.status === "active"
  )
  const canStartImportCommit =
    importPreviewCommitEligible &&
    importConflictChoiceValid &&
    (!importCommitJob?.job_id || importCommitStatus === "failed")
  const canRefreshImportCommit =
    Boolean(importCommitJob?.job_id) && !importCommitIsTerminal
  const hasActiveVisual =
    packStateMatchesSelectedPersona &&
    Boolean(
      selectedActivePack || visiblePacks.some((pack) => pack.status === "active")
    )
  const showSetupChoices =
    isActive &&
    Boolean(selectedPersonaId) &&
    packStateMatchesSelectedPersona &&
    packsLoaded &&
    !loading &&
    !hasActiveVisual &&
    !importPreview &&
    !importCommitJob
  const showManagementHeader =
    isActive &&
    Boolean(selectedPersonaId) &&
    packStateMatchesSelectedPersona &&
    packsLoaded &&
    !showSetupChoices
  const recommendedStarter = starterPacks[0] ?? null
  const importPreviewPanel = (
    <div className="rounded border border-border bg-bg p-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Typography.Text strong>Import preview</Typography.Text>
        <Tag>review only</Tag>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          ref={importPreviewInputRef}
          data-testid="persona-visual-import-preview-input"
          type="file"
          accept={BUDDY_IMPORT_ARCHIVE_ACCEPT}
          className="text-xs text-text"
          onChange={(event) =>
            setSelectedImportPreviewFile(event.target.files?.[0] ?? null)
          }
        />
        <Button
          data-testid="persona-visual-import-preview-button"
          size="small"
          icon={<FileSearch className="h-3.5 w-3.5" />}
          loading={previewingImport}
          disabled={!selectedImportPreviewFile || Boolean(importPreviewFileError)}
          onClick={() => void handleStartImportPreview()}
        >
          Preview
        </Button>
        <Button
          data-testid="persona-visual-import-preview-refresh-button"
          size="small"
          icon={<RefreshCw className="h-3.5 w-3.5" />}
          loading={refreshingImportPreview}
          disabled={!importPreview?.preview_id}
          onClick={() => void handleRefreshImportPreview()}
        >
          Refresh
        </Button>
      </div>
      {importPreviewFileError ? (
        <div
          data-testid="persona-visual-import-preview-file-error"
          className="mt-2 text-xs text-state-error"
        >
          {importPreviewFileError}
        </div>
      ) : null}
      {importPreview ? (
        <div className="mt-2 space-y-1 text-xs text-text-muted">
          <div className="flex flex-wrap items-center gap-2">
            <Tag data-testid="persona-visual-import-preview-status">
              {importPreview.status}
            </Tag>
            <span>{importPreview.stage}</span>
            <span>{importPreview.preview_id}</span>
          </div>
          <div data-testid="persona-visual-import-preview-summary">
            {formatImportPreviewSummary(importPreview)}
          </div>
          {importPreviewJobCopy ? (
            <div data-testid="persona-visual-import-preview-job-copy">
              {importPreviewJobCopy}
            </div>
          ) : null}
          {buddyPipelinePacketDiagnostics ? (
            <div
              data-testid="persona-visual-import-buddy-packet-diagnostics"
              className="rounded border border-border bg-bg p-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-text">
                  {t("sidepanel:personaGarden.visuals.buddyPipelinePacketTitle", {
                    defaultValue: "Buddy pipeline packet"
                  })}
                </span>
                <Tag>
                  {t("sidepanel:personaGarden.visuals.buddyPipelineAssetCount", {
                    defaultValue: "{{count}} assets",
                    count: buddyPipelinePacketDiagnostics.assets.length
                  })}
                </Tag>
              </div>
              {buddyPipelinePacketDiagnostics.manifestAssetReferences.length ? (
                <div className="mt-1">
                  {t("sidepanel:personaGarden.visuals.buddyPipelineManifestReferences", {
                    defaultValue: "Manifest references"
                  })}
                  :{" "}
                  {buddyPipelinePacketDiagnostics.manifestAssetReferences.join(
                    ", "
                  )}
                </div>
              ) : null}
              {buddyPipelineSourceAssets.length ? (
                <div className="mt-2">
                  <div className="font-medium text-text">
                    {t("sidepanel:personaGarden.visuals.buddyPipelineSourceMaterial", {
                      defaultValue: "Source material"
                    })}
                  </div>
                  <ul className="mt-1 list-disc space-y-1 pl-5">
                    {buddyPipelineSourceAssets.map((asset, index) => (
                      <li
                        key={`${asset.assetGroup}-${asset.sourceAssetId ?? "unknown"}-${index}`}
                      >
                        <span className="text-text">
                          {t(
                            `sidepanel:personaGarden.visuals.buddyPipelineAssetGroup.${asset.assetGroup}`,
                            {
                              defaultValue:
                                BUDDY_PIPELINE_ASSET_GROUP_LABELS[asset.assetGroup]
                            }
                          )}
                        </span>
                        {": "}
                        {getBuddyPipelineAssetDetailText(
                          asset,
                          t(
                            "sidepanel:personaGarden.visuals.buddyPipelineManifestReferenced",
                            { defaultValue: "manifest referenced" }
                          )
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {buddyPipelineRuntimeAssets.length ? (
                <div className="mt-2">
                  <div className="font-medium text-text">
                    {t("sidepanel:personaGarden.visuals.buddyPipelineRuntimeOutput", {
                      defaultValue: "Runtime output"
                    })}
                  </div>
                  <ul className="mt-1 list-disc space-y-1 pl-5">
                    {buddyPipelineRuntimeAssets.map((asset, index) => (
                      <li
                        key={`${asset.assetGroup}-${asset.sourceAssetId ?? "unknown"}-${index}`}
                      >
                        <span className="text-text">
                          {t(
                            `sidepanel:personaGarden.visuals.buddyPipelineAssetGroup.${asset.assetGroup}`,
                            {
                              defaultValue:
                                BUDDY_PIPELINE_ASSET_GROUP_LABELS[asset.assetGroup]
                            }
                          )}
                        </span>
                        {": "}
                        {getBuddyPipelineAssetDetailText(
                          asset,
                          t(
                            "sidepanel:personaGarden.visuals.buddyPipelineManifestReferenced",
                            { defaultValue: "manifest referenced" }
                          )
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}
          {importPreviewWarnings.length ? (
            <div data-testid="persona-visual-import-preview-warnings">
              {formatPreviewList(importPreviewWarnings)}
            </div>
          ) : null}
          {importPreviewConflicts.length ? (
            <div data-testid="persona-visual-import-preview-conflicts">
              {formatPreviewList(importPreviewConflicts)}
            </div>
          ) : null}
          {importPreviewPlan ? (
            <div data-testid="persona-visual-import-preview-plan">
              {stringifyPreviewValue(importPreviewPlan)}
            </div>
          ) : null}
          {rendererImportPreview ? (
            <div
              data-testid="persona-visual-import-renderer-diagnostics"
              className="rounded border border-border bg-bg p-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-text">
                  {t("sidepanel:personaGarden.visuals.rendererDiagnosticsTitle", {
                    defaultValue: "Renderer diagnostics"
                  })}
                </span>
                <Tag>{rendererImportPreview.renderer_type || unknownLabel}</Tag>
                <Tag>{rendererImportPreview.status || unknownLabel}</Tag>
                {rendererImportPreview.setup_status ? (
                  <Tag>{rendererImportPreview.setup_status}</Tag>
                ) : null}
              </div>
              <div className="mt-1">
                {t("sidepanel:personaGarden.visuals.manifestVersion", {
                  defaultValue: "Manifest v"
                })}
                {rendererImportPreview.manifest_version ?? unknownLabel}
                {" / "}
                {t("sidepanel:personaGarden.visuals.contractVersion", {
                  defaultValue: "Contract v"
                })}
                {rendererImportPreview.renderer_contract_version ?? unknownLabel}
              </div>
              <div className="mt-1">
                {rendererImportPreview.activation_eligible
                  ? t("sidepanel:personaGarden.visuals.activationEligible", {
                      defaultValue: "Activation eligible"
                    })
                  : t("sidepanel:personaGarden.visuals.activationUnavailable", {
                      defaultValue: "Activation unavailable"
                    })}
              </div>
              {importCommitBlockers.length ? (
                <div className="mt-1">
                  {t("sidepanel:personaGarden.visuals.commitBlockers", {
                    defaultValue: "Commit blockers"
                  })}
                  : {formatPreviewList(importCommitBlockers)}
                </div>
              ) : null}
              {rendererImportPreview.warnings?.length ? (
                <div className="mt-1">
                  {t("sidepanel:personaGarden.visuals.rendererWarnings", {
                    defaultValue: "Warnings"
                  })}
                  : {formatPreviewList(rendererImportPreview.warnings)}
                </div>
              ) : null}
              {rendererImportRoleSummary ? (
                <div className="mt-1">
                  {t("sidepanel:personaGarden.visuals.assetRoles", {
                    defaultValue: "Asset roles"
                  })}
                  : {rendererImportRoleSummary}
                </div>
              ) : null}
            </div>
          ) : null}
          {canCommitImportPreview ? (
            <div className="mt-2 border-t border-border pt-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="font-medium text-text">Commit reviewed import</div>
                <Tag>creates draft</Tag>
              </div>
              <div className="mt-1 text-text-muted">
                Commit creates a new draft pack. Activation remains separate.
              </div>
              {!importPreviewCommitEligible ? (
                <div
                  data-testid="persona-visual-import-commit-blocked"
                  className="mt-2 rounded border border-border bg-bg p-2 text-text-muted"
                >
                  {t("sidepanel:personaGarden.visuals.importCommitBlocked", {
                    defaultValue:
                      "Commit unavailable until preview blockers are resolved"
                  })}
                  {importCommitBlockers.length
                    ? `: ${formatPreviewList(importCommitBlockers)}`
                    : "."}
                </div>
              ) : null}
              {importConflictChoiceRequired ? (
                <div
                  data-testid="persona-visual-import-conflict-choice"
                  className="mt-2 grid gap-2 md:grid-cols-[minmax(140px,180px)_minmax(180px,1fr)_minmax(180px,1fr)]"
                >
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Target mode</span>
                    <select
                      data-testid="persona-visual-import-target-mode"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={importTargetMode}
                      onChange={(event) => {
                        const nextMode = event.target
                          .value as PersonaVisualImportTargetMode | ""
                        setImportTargetMode(nextMode)
                        setImportTargetChoicePreviewId(
                          nextMode ? fullImportPreview?.preview_id || "" : ""
                        )
                        if (nextMode !== "replace_draft") {
                          setImportReplacePackId("")
                        } else if (!importReplacePackId) {
                          setImportReplacePackId(
                            replaceableImportConflicts[0]?.pack_id || ""
                          )
                        }
                      }}
                    >
                      <option value="">Choose</option>
                      {importAllowedTargetModes.map((mode) => (
                        <option key={mode} value={mode}>
                          {mode === "replace_draft"
                            ? "Replace draft"
                            : "Create new draft"}
                        </option>
                      ))}
                    </select>
                  </label>
                  {importTargetMode === "replace_draft" ? (
                    <label className="text-xs text-text-muted">
                      <span className="mb-1 block">Draft</span>
                      <select
                        data-testid="persona-visual-import-replace-pack"
                        className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                        value={importReplacePackId}
                        onChange={(event) =>
                          setImportReplacePackId(event.target.value)
                        }
                      >
                        <option value="">Select draft</option>
                        {replaceableImportConflicts.map((conflict) => (
                          <option key={conflict.pack_id} value={conflict.pack_id}>
                            {getImportConflictLabel(conflict)}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Draft title</span>
                    <input
                      data-testid="persona-visual-import-draft-title"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={importDraftTitle}
                      onChange={(event) => setImportDraftTitle(event.target.value)}
                    />
                  </label>
                </div>
              ) : null}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <Button
                  data-testid="persona-visual-import-commit-button"
                  size="small"
                  icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                  loading={committingImport}
                  disabled={!canStartImportCommit}
                  onClick={() => void handleStartImportCommit()}
                >
                  Commit as draft
                </Button>
                <Button
                  data-testid="persona-visual-import-commit-refresh-button"
                  size="small"
                  icon={<RefreshCw className="h-3.5 w-3.5" />}
                  loading={refreshingImportCommit}
                  disabled={!canRefreshImportCommit}
                  onClick={() => void handleRefreshImportCommit()}
                >
                  Refresh commit
                </Button>
              </div>
              {importCommitJob ? (
                <div className="mt-2 space-y-1 text-text-muted">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag data-testid="persona-visual-import-commit-status">
                      {importCommitJob.status}
                    </Tag>
                    <span data-testid="persona-visual-import-commit-stage">
                      {importCommitJob.stage}
                    </span>
                    <span data-testid="persona-visual-import-commit-job-id">
                      {importCommitJob.job_id}
                    </span>
                  </div>
                  {selectedPack && importCommitJobCopy && importCommitJobPackIsSelected ? (
                    <div data-testid="persona-visual-import-commit-job-copy">
                      {importCommitJobCopy}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="mt-2 text-text-muted">No import commit job.</div>
              )}
            </div>
          ) : null}
        </div>
      ) : (
        <div className="mt-2 text-xs text-text-muted">No import preview.</div>
      )}
    </div>
  )
  const firstRunImportPanel = (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            Import visual pack
          </div>
          <div className="mt-1 text-xs text-text-muted">
            Preview a portable archive, then commit it as an inactive draft.
          </div>
        </div>
        <Tag>creates draft</Tag>
      </div>
      <div className="mt-3">
        {importPreviewPanel}
      </div>
    </div>
  )

  React.useEffect(() => {
    if (!fullImportPreview) {
      setImportTargetMode("create_new")
      setImportTargetChoicePreviewId("")
      setImportReplacePackId("")
      setImportDraftTitle("")
      return
    }
    const choice = getImportTargetChoice(fullImportPreview)
    setImportTargetMode(choice ? "" : "create_new")
    setImportTargetChoicePreviewId(choice ? "" : fullImportPreview.preview_id)
    setImportReplacePackId("")
    setImportDraftTitle(getDefaultImportDraftTitle(fullImportPreview))
  }, [fullImportPreview?.preview_id])

  return (
    <div className="space-y-3" data-testid="persona-visual-pack-editor">
      {showSetupChoices ? (
        <VisualBuddySetupChoiceCard
          selectedPersonaId={selectedPersonaId}
          selectedPersonaName={selectedPersonaName || selectedPersonaId}
          hasActiveVisual={hasActiveVisual}
          packCount={visiblePacks.length}
          recommendedStarter={recommendedStarter}
          starterCount={starterPacks.length}
          starterCatalogLoading={starterCatalogLoading}
          starterCatalogError={starterCatalogError}
          copyingDefault={Boolean(copyingStarterId)}
          onUseDefault={
            recommendedStarter
              ? () => void handleCopyStarterPack(recommendedStarter.id)
              : undefined
          }
          onChooseDefault={
            starterPacks.length > 1 ? () => setStarterPickerOpen(true) : undefined
          }
          onImportPack={openImportArchivePicker}
          onStartBlank={focusDraftTitleInput}
        />
      ) : null}

      <Modal
        title="Choose bundled default"
        open={starterPickerOpen}
        footer={null}
        destroyOnHidden
        onCancel={() => setStarterPickerOpen(false)}
      >
        <div
          data-testid="persona-visual-starter-picker"
          className="space-y-2"
        >
          {starterPacks.map((starter) => {
            const productionStatus = getStarterProductionStatusLabel(
              starter.production_status,
              t
            )
            const complexityTier = getStarterComplexityTierLabel(
              starter.complexity_tier,
              t
            )
            const expectedAssetGroups = formatStarterExpectedAssetGroups(
              starter.expected_asset_groups
            )
            const animationCoverageNotes = (
              starter.animation_coverage_notes || []
            ).join("; ")
            return (
              <div
                key={starter.id}
                className="rounded border border-border bg-bg p-2 text-xs"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <div className="font-medium text-text">{starter.title}</div>
                    <div className="mt-1 text-text-muted">{starter.description}</div>
                  </div>
                  <Tag>{starter.renderer_type}</Tag>
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {productionStatus ? (
                    <Tag color={starter.production_status === "art_ready" ? "green" : "orange"}>
                      {productionStatus}
                    </Tag>
                  ) : null}
                  {complexityTier ? <Tag>{complexityTier}</Tag> : null}
                  {starter.neutral_anchor_required ? (
                    <Tag color="blue">
                      {t(
                        "sidepanel:personaGarden.visuals.setup.neutralAnchorRequired",
                        {
                          defaultValue: "Neutral anchor required"
                        }
                      )}
                    </Tag>
                  ) : null}
                  {starter.tags.map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                  <Tag>{starter.license_label}</Tag>
                </div>
                {expectedAssetGroups ? (
                  <div className="mt-2 text-[11px] leading-5 text-text-muted">
                    <span className="font-medium text-text">
                      {t(
                        "sidepanel:personaGarden.visuals.setup.expectedAssetsLabel",
                        {
                          defaultValue: "Expected assets:"
                        }
                      )}{" "}
                    </span>
                    {expectedAssetGroups}
                  </div>
                ) : null}
                {animationCoverageNotes ? (
                  <div className="mt-1 text-[11px] leading-5 text-text-muted">
                    <span className="font-medium text-text">
                      {t(
                        "sidepanel:personaGarden.visuals.setup.coverageLabel",
                        {
                          defaultValue: "Coverage:"
                        }
                      )}{" "}
                    </span>
                    {animationCoverageNotes}
                  </div>
                ) : null}
                <Button
                  data-testid={`persona-visual-copy-starter-${starter.id}`}
                  className="mt-2"
                  size="small"
                  type="primary"
                  icon={<Copy className="h-3.5 w-3.5" />}
                  loading={copyingStarterId === starter.id}
                  disabled={Boolean(copyingStarterId)}
                  onClick={() => void handleCopyStarterPack(starter.id)}
                >
                  Copy as draft
                </Button>
              </div>
            )
          })}
        </div>
      </Modal>

      <div className="rounded-lg border border-border bg-surface p-3">
        {showManagementHeader ? (
          <VisualManagementHeader
            personaName={selectedPersonaName || selectedPersonaId}
            model={managementModel}
            t={t}
          />
        ) : null}
        <div
          className={
            showManagementHeader
              ? "mt-3 flex flex-wrap items-start justify-end gap-3"
              : "flex flex-wrap items-start justify-between gap-3"
          }
        >
          {!showManagementHeader ? (
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
                {t("sidepanel:personaGarden.visuals.heading", {
                  defaultValue: "Visual Packs"
                })}
              </div>
              <div className="mt-1 text-sm font-medium text-text">
                {selectedPersonaName || selectedPersonaId}
              </div>
            </div>
          ) : null}
          <Button
            data-testid="persona-visual-pack-refresh-button"
            size="small"
            onClick={() =>
              void loadPacks({ preferredPackId: selectedPack?.id })
            }
            disabled={loading}
          >
            {loading ? loadingLabel : refreshLabel}
          </Button>
        </div>
        <div className="mt-3 grid gap-2 md:grid-cols-[minmax(180px,1fr)_minmax(180px,1fr)_auto]">
          <label className="text-xs text-text-muted">
            <span className="mb-1 block">Pack</span>
            <select
              data-testid="persona-visual-pack-select"
              className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
              value={selectedPack?.id || ""}
              onChange={(event) => selectPackId(event.target.value)}
              disabled={!visiblePacks.length}
            >
              {visiblePacks.map((pack) => (
                <option key={pack.id} value={pack.id}>
                  {pack.title}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-text-muted">
            <span className="mb-1 block">New draft title</span>
            <input
              ref={draftTitleInputRef}
              data-testid="persona-visual-pack-title-input"
              className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
              value={draftTitle}
              onChange={(event) => setDraftTitle(event.target.value)}
            />
          </label>
          <Button
            data-testid="persona-visual-create-pack"
            className="self-end"
            size="small"
            type="primary"
            icon={<Plus className="h-3.5 w-3.5" />}
            disabled={!draftTitle.trim() || saving}
            onClick={() => void handleCreateDraft()}
          >
            Create draft
          </Button>
        </div>
        {!loading && !visiblePacks.length ? (
          <div
            data-testid="persona-visual-pack-empty"
            className="mt-3 rounded border border-dashed border-border bg-bg px-3 py-3 text-xs text-text-muted"
          >
            <div className="font-medium text-text">
              {selectedPersonaName || selectedPersonaId}
              {"'s Persona Buddy does not have a visual pack yet."}
            </div>
            <div className="mt-1">
              Create a draft visual pack first.
            </div>
            <div className="mt-1">
              After a draft exists, upload frames, map states, import or export
              packs, queue generation, review candidates, and activate a valid pack.
            </div>
          </div>
        ) : null}
        {selectedPack ? (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <Tag data-testid="persona-visual-pack-status" color="blue">
              {selectedPack.status}
            </Tag>
            {selectedPackLibraryItem ? (
              <>
                <Tag data-testid="persona-visual-library-selected-status" color="green">
                  in library
                </Tag>
                {selectedPackLibraryItem.source_changed ? (
                  <Tag color="orange">source changed</Tag>
                ) : null}
              </>
            ) : null}
            <span className="font-medium text-text">{selectedPack.title}</span>
            <span className="text-text-muted">{`v${selectedPack.version ?? 1}`}</span>
            <Button
              data-testid="persona-visual-library-save-button"
              size="small"
              icon={<BookOpen className="h-3.5 w-3.5" />}
              loading={savingToLibrary}
              disabled={savingToLibrary}
              onClick={() => void handleSavePackToLibrary()}
            >
              Save to library
            </Button>
          </div>
        ) : null}
        {selectedPack ? (
          <div
            data-testid="persona-visual-ownership-copy"
            className="mt-3 rounded-md border border-border bg-bg px-3 py-2 text-xs leading-5 text-text-muted"
          >
            <div className="font-medium text-text">
              {t("sidepanel:personaGarden.visuals.ownershipTitle", {
                defaultValue: "How Persona Visual packs work"
              })}
            </div>
            <div className="mt-1">
              {t("sidepanel:personaGarden.visuals.ownershipDescription", {
                defaultValue: `Assets are user-owned and attached to ${selectedPersonaName || selectedPersonaId} by default. Packs are stored as manifests with referenced assets, so they can be edited, exported, imported, and later duplicated or shared without changing the core format.`
              })}
            </div>
            <div className="mt-1">
              {t("sidepanel:personaGarden.visuals.activePackDescription", {
                defaultValue:
                  "The active pack is the one Persona Buddy renders now; other packs stay available for editing or review."
              })}
            </div>
          </div>
        ) : null}
        {packHealthDiagnostic ? (
          <div
            data-testid="persona-visual-pack-health"
            data-severity={packHealthDiagnostic.severity}
            className={`mt-3 rounded-md border px-3 py-2 text-xs leading-5 ${getPersonaVisualDiagnosticToneClassName(packHealthDiagnostic.severity)}`}
          >
            <div className="font-medium text-inherit">
              {packHealthDiagnostic.title}
            </div>
            <div>{packHealthDiagnostic.message}</div>
          </div>
        ) : null}
      </div>

      <VisualPackReusePanel
        selectedPersonaName={selectedPersonaName || selectedPersonaId}
        hasSelectedPack={Boolean(selectedPack)}
        canImport
        libraryItemCount={libraryItems.length}
        hasDuplicateTargets={availableDuplicateTargets.length > 0}
        duplicateTargetsLoading={duplicateTargetsLoading}
        onCreateDraft={focusDraftTitleInput}
        onOpenLibrary={focusLibraryPanel}
        onOpenImport={openImportArchivePicker}
        onOpenDuplicate={focusDuplicateControls}
      />

      {error ? (
        <div className="rounded-md border border-danger/30 bg-danger/10 p-2 text-xs text-danger">
          {error}
        </div>
      ) : null}
      {statusMessage ? (
        <div className="rounded-md border border-success/30 bg-success/10 p-2 text-xs text-success">
          {statusMessage}
        </div>
      ) : null}

      {!selectedPack ? firstRunImportPanel : null}

      <div
        ref={libraryPanelRef}
        data-testid="persona-visual-library-panel"
        tabIndex={-1}
        className="rounded-lg border border-border bg-surface p-3"
      >
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
              Personal library
            </div>
            <div className="mt-1 text-xs text-text-muted">
              Save reusable Persona Buddy visual packs as user-owned references.
              Using one creates a draft on the target persona.
            </div>
          </div>
          <Button
            size="small"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            loading={libraryLoading}
            onClick={() => void loadLibrary()}
          >
            Refresh library
          </Button>
        </div>
        <div className="mt-3 space-y-2">
          {libraryItems.length ? (
            libraryItems.map(renderLibraryItem)
          ) : (
            <div className="rounded border border-dashed border-border bg-bg px-3 py-2 text-xs text-text-muted">
              No saved visual packs.
            </div>
          )}
        </div>
      </div>

      {selectedPack ? (
        <>
          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="flex flex-wrap items-end gap-2">
              <label className="text-xs text-text-muted">
                <span className="mb-1 block">Asset role</span>
                <select
                  data-testid="persona-visual-upload-role-select"
                  className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                  value={uploadRole}
                  onChange={(event) =>
                    setUploadRole(event.target.value as PersonaVisualAssetRole)
                  }
                >
                  {ASSET_ROLES.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </label>
              <input
                ref={fileInputRef}
                data-testid="persona-visual-upload-input"
                type="file"
                accept="image/png,image/jpeg,image/webp,image/gif"
                className="text-xs text-text"
                onChange={(event) =>
                  setSelectedUploadFile(event.target.files?.[0] ?? null)
                }
              />
              <Button
                data-testid="persona-visual-upload-button"
                size="small"
                icon={<Upload className="h-3.5 w-3.5" />}
                loading={uploading}
                disabled={!selectedUploadFile}
                onClick={() => void handleUploadAsset()}
              >
                Upload
              </Button>
            </div>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {assets.map((asset) => (
                <div
                  key={asset.id}
                  className="rounded border border-border bg-bg px-2 py-1.5 text-xs"
                >
                  <div className="flex flex-wrap items-center gap-1">
                    <Tag>{asset.asset_role}</Tag>
                    <span className="font-medium text-text">
                      {asset.original_filename || asset.id}
                    </span>
                  </div>
                  <div className="mt-1 text-text-muted">{asset.id}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
              Portability
            </div>
            <div
              data-testid="persona-visual-portability-copy"
              className="mt-2 rounded-md border border-border bg-bg px-3 py-2 text-xs leading-5 text-text-muted"
            >
              <div>
                {t("sidepanel:personaGarden.visuals.importPreviewHelp", {
                  defaultValue:
                    "Import preview validates a portable pack archive before it changes this persona."
                })}
              </div>
              <div>
                {t("sidepanel:personaGarden.visuals.importCommitHelp", {
                  defaultValue:
                    "Commit import creates a reviewed draft pack for this persona."
                })}
              </div>
              <div>
                {t("sidepanel:personaGarden.visuals.exportHelp", {
                  defaultValue:
                    "Export downloads a portable archive and does not publish to a shared library."
                })}
              </div>
            </div>
            <div className="mt-3 grid gap-3 xl:grid-cols-3">
              <div className="rounded border border-border bg-bg p-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Typography.Text strong>Duplicate to persona</Typography.Text>
                  <Tag>creates draft</Tag>
                </div>
                <div className="mt-1 text-xs text-text-muted">
                  Copy this pack to another persona. It stays a draft until reviewed and
                  activated.
                </div>
                <div className="mt-2 grid gap-2">
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Target persona</span>
                    <select
                      ref={duplicateTargetSelectRef}
                      data-testid="persona-visual-duplicate-target-select"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={duplicateTargetId}
                      disabled={
                        duplicateTargetsLoading || !availableDuplicateTargets.length
                      }
                      onChange={(event) => setDuplicateTargetId(event.target.value)}
                    >
                      {availableDuplicateTargets.map((target) => (
                        <option key={target.id} value={target.id}>
                          {target.name || target.id}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Draft title</span>
                    <input
                      data-testid="persona-visual-duplicate-title-input"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={duplicateTitle}
                      onChange={(event) => setDuplicateTitle(event.target.value)}
                    />
                  </label>
                  <Button
                    data-testid="persona-visual-duplicate-button"
                    size="small"
                    icon={<Copy className="h-3.5 w-3.5" />}
                    loading={duplicatingPack}
                    disabled={!duplicateTargetId || !selectedPack}
                    onClick={() => void handleDuplicatePack()}
                  >
                    Duplicate as draft
                  </Button>
                  {lastDuplicatedPersonaId && onOpenPersonaVisuals ? (
                    <Button
                      data-testid="persona-visual-duplicate-open-target"
                      size="small"
                      type="link"
                      onClick={() => onOpenPersonaVisuals(lastDuplicatedPersonaId)}
                    >
                      Open target Visuals
                    </Button>
                  ) : null}
                </div>
                {!availableDuplicateTargets.length ? (
                  <div className="mt-2 text-xs text-text-muted">
                    Add another persona before duplicating this pack.
                  </div>
                ) : null}
              </div>

              <div className="rounded border border-border bg-bg p-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Typography.Text strong>Export archive</Typography.Text>
                  <div className="flex flex-wrap gap-1">
                    <Button
                      data-testid="persona-visual-export-button"
                      size="small"
                      icon={<Upload className="h-3.5 w-3.5" />}
                      loading={exportingPack}
                      onClick={() => void handleStartExport()}
                    >
                      Export
                    </Button>
                    <Button
                      data-testid="persona-visual-export-refresh-button"
                      size="small"
                      icon={<RefreshCw className="h-3.5 w-3.5" />}
                      loading={refreshingExport}
                      disabled={!exportJob?.job_id}
                      onClick={() => void handleRefreshExport()}
                    >
                      Refresh
                    </Button>
                    <Button
                      data-testid="persona-visual-export-download-button"
                      size="small"
                      icon={<Download className="h-3.5 w-3.5" />}
                      loading={downloadingExport}
                      disabled={exportJob?.status !== "completed"}
                      onClick={() => void handleDownloadExport()}
                    >
                      Download
                    </Button>
                  </div>
                </div>
                {exportJob ? (
                  <div className="mt-2 space-y-1 text-xs text-text-muted">
                    <div className="flex flex-wrap items-center gap-2">
                      <Tag data-testid="persona-visual-export-status">
                        {exportJob.status}
                      </Tag>
                      <span data-testid="persona-visual-export-stage">
                        {exportJob.stage}
                      </span>
                      <span>{exportJob.job_id}</span>
                    </div>
                    {exportWarnings.length ? (
                      <div>{formatPreviewList(exportWarnings)}</div>
                    ) : null}
                  </div>
                ) : (
                  <div className="mt-2 text-xs text-text-muted">No export job.</div>
                )}
              </div>

              {importPreviewPanel}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
              State Mapping
            </div>
            <div className="grid gap-2 md:grid-cols-2">
              {[...REQUIRED_VISUAL_STATES, ...OPTIONAL_VISUAL_STATES].map((state) => (
                <label key={state} className="text-xs text-text-muted">
                  <span className="mb-1 flex items-center gap-1">
                    <span>{formatStateLabel(state)}</span>
                    {REQUIRED_VISUAL_STATES.includes(state) ? (
                      <Tag color="red">required</Tag>
                    ) : null}
                  </span>
                  <select
                    data-testid={`persona-visual-state-${state}-select`}
                    className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                    value={draftManifest.states?.[state]?.animation_id || ""}
                    onChange={(event) =>
                      handleStateMappingChange(state, event.target.value)
                    }
                  >
                    {renderAnimationOptions()}
                  </select>
                </label>
              ))}
            </div>
            {customVisualStates.length ? (
              <>
                <div className="mt-4 text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
                  Custom States
                </div>
                <div className="mt-2 grid gap-2 md:grid-cols-2">
                  {customVisualStates.map((state) => (
                    <label key={state} className="text-xs text-text-muted">
                      <span className="mb-1 flex items-center gap-1">
                        <span>{formatStateLabel(state)}</span>
                        <Tag color="blue">
                          {draftManifest.state_catalog?.[state]?.kind || "custom"}
                        </Tag>
                      </span>
                      <select
                        data-testid={`persona-visual-state-${state}-select`}
                        className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                        value={draftManifest.states?.[state]?.animation_id || ""}
                        onChange={(event) =>
                          handleStateMappingChange(state, event.target.value)
                        }
                      >
                        {renderAnimationOptions()}
                      </select>
                    </label>
                  ))}
                </div>
              </>
            ) : null}
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {editableFallbackStates.map((state) => (
                <label key={state} className="text-xs text-text-muted">
                  <span className="mb-1 block">{`${formatStateLabel(state)} fallbacks`}</span>
                  <input
                    data-testid={`persona-visual-fallback-${state}-input`}
                    className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                    value={(draftManifest.fallbacks?.[state] || []).join(",")}
                    onChange={(event) => handleFallbackChange(state, event.target.value)}
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex flex-wrap items-end gap-2">
              <label className="min-w-[180px] text-xs text-text-muted">
                <span className="mb-1 block">Animation</span>
                <select
                  data-testid="persona-visual-animation-select"
                  className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                  value={selectedAnimationId}
                  onChange={(event) => setSelectedAnimationId(event.target.value)}
                >
                  {animationIds.map((animationId) => (
                    <option key={animationId} value={animationId}>
                      {animationId}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-text-muted">
                <span className="mb-1 block">New animation</span>
                <input
                  data-testid="persona-visual-new-animation-input"
                  className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                  value={newAnimationId}
                  onChange={(event) => setNewAnimationId(event.target.value)}
                />
              </label>
              <Button
                size="small"
                icon={<Plus className="h-3.5 w-3.5" />}
                disabled={!newAnimationId.trim()}
                onClick={handleAddAnimation}
              >
                Add animation
              </Button>
            </div>

            {selectedAnimation ? (
              <>
                <div className="grid gap-2 md:grid-cols-5">
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Frame rate</span>
                    <input
                      data-testid="persona-visual-frame-rate-input"
                      type="number"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={selectedAnimation.frame_rate ?? ""}
                      onChange={(event) =>
                        handleAnimationFieldChange(
                          "frame_rate",
                          parseNumberOrUndefined(event.target.value)
                        )
                      }
                    />
                  </label>
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Preview frame</span>
                    <select
                      data-testid="persona-visual-preview-frame-select"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={selectedAnimation.preview_frame ?? 0}
                      onChange={(event) =>
                        handleAnimationFieldChange(
                          "preview_frame",
                          Number(event.target.value)
                        )
                      }
                    >
                      {selectedFrames.map((_, index) => (
                        <option key={index} value={index}>
                          {index}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Alignment X</span>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={selectedAnimation.alignment?.x ?? 0.5}
                      onChange={(event) =>
                        handleAnimationFieldChange("alignment", {
                          ...(selectedAnimation.alignment || { x: 0.5, y: 1 }),
                          x: Number(event.target.value)
                        })
                      }
                    />
                  </label>
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Alignment Y</span>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={selectedAnimation.alignment?.y ?? 1}
                      onChange={(event) =>
                        handleAnimationFieldChange("alignment", {
                          ...(selectedAnimation.alignment || { x: 0.5, y: 1 }),
                          y: Number(event.target.value)
                        })
                      }
                    />
                  </label>
                  <label className="flex items-end gap-2 text-xs text-text-muted">
                    <input
                      data-testid="persona-visual-loop-checkbox"
                      type="checkbox"
                      checked={selectedAnimation.loop !== false}
                      onChange={(event) =>
                        handleAnimationFieldChange("loop", event.target.checked)
                      }
                    />
                    <span>Loop</span>
                  </label>
                </div>

                <div className="mt-3 flex flex-wrap items-end gap-2">
                  <label className="text-xs text-text-muted">
                    <span className="mb-1 block">Frame asset</span>
                    <select
                      data-testid="persona-visual-add-frame-asset-select"
                      className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                      value={selectedAddFrameAssetId}
                      onChange={(event) => setSelectedAddFrameAssetId(event.target.value)}
                    >
                      <option value="">Select asset</option>
                      {assets.map((asset) => (
                        <option key={asset.id} value={asset.id}>
                          {asset.original_filename || asset.id}
                        </option>
                      ))}
                    </select>
                  </label>
                  <Button
                    size="small"
                    icon={<Plus className="h-3.5 w-3.5" />}
                    disabled={!selectedAddFrameAssetId}
                    onClick={handleAddFrame}
                  >
                    Add frame
                  </Button>
                </div>

                <div className="mt-3 space-y-2">
                  {selectedFrames.map((frame, index) => (
                    <div
                      key={`${frame.asset_id}-${index}`}
                      data-testid={`persona-visual-frame-row-${index}`}
                      className="grid gap-2 rounded border border-border bg-bg p-2 md:grid-cols-[minmax(140px,1fr)_repeat(5,minmax(64px,90px))_auto]"
                    >
                      <label className="text-xs text-text-muted">
                        <span className="mb-1 block">Asset</span>
                        <select
                          data-testid="persona-visual-frame-asset-select"
                          className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                          value={frame.asset_id}
                          onChange={(event) =>
                            handleFrameChange(index, (current) => ({
                              ...current,
                              asset_id: event.target.value
                            }))
                          }
                        >
                          {assets.map((asset) => (
                            <option key={asset.id} value={asset.id}>
                              {asset.original_filename || asset.id}
                            </option>
                          ))}
                        </select>
                      </label>
                      {(["x", "y", "width", "height"] as const).map((key) => (
                        <label key={key} className="text-xs text-text-muted">
                          <span className="mb-1 block">{key}</span>
                          <input
                            data-testid={`persona-visual-frame-region-${key}`}
                            type="number"
                            className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                            value={frame.region?.[key] ?? ""}
                            onChange={(event) =>
                              handleFrameRegionChange(index, key, event.target.value)
                            }
                          />
                        </label>
                      ))}
                      <label className="text-xs text-text-muted">
                        <span className="mb-1 block">ms</span>
                        <input
                          data-testid="persona-visual-frame-duration-input"
                          type="number"
                          className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                          value={frame.duration_ms ?? ""}
                          onChange={(event) =>
                            handleFrameChange(index, (current) => ({
                              ...current,
                              duration_ms: parseNumberOrUndefined(event.target.value)
                            }))
                          }
                        />
                      </label>
                      <div className="flex items-end gap-1">
                        <Button
                          data-testid={`persona-visual-frame-move-up-${index}`}
                          size="small"
                          icon={<ArrowUp className="h-3.5 w-3.5" />}
                          disabled={index === 0}
                          onClick={() => handleMoveFrame(index, -1)}
                        />
                        <Button
                          data-testid={`persona-visual-frame-move-down-${index}`}
                          size="small"
                          icon={<ArrowDown className="h-3.5 w-3.5" />}
                          disabled={index >= selectedFrames.length - 1}
                          onClick={() => handleMoveFrame(index, 1)}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="rounded border border-dashed border-border bg-bg px-3 py-2 text-xs text-text-muted">
                No animations in this pack.
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
              Authored Triggers
            </div>
            <div className="grid gap-2 md:grid-cols-[130px_minmax(160px,1fr)_130px_100px_90px_auto]">
              <select
                data-testid="persona-visual-trigger-source-select"
                className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                value={triggerDraft.source}
                onChange={(event) =>
                  setTriggerDraft((current) => ({
                    ...current,
                    source: event.target.value as TriggerDraft["source"]
                  }))
                }
              >
                {TRIGGER_SOURCES.map((source) => (
                  <option key={source} value={source}>
                    {source}
                  </option>
                ))}
              </select>
              <input
                data-testid="persona-visual-trigger-match-input"
                className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                value={triggerDraft.match}
                onChange={(event) =>
                  setTriggerDraft((current) => ({
                    ...current,
                    match: event.target.value
                  }))
                }
              />
              <select
                data-testid="persona-visual-trigger-state-select"
                className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                value={triggerDraft.state}
                onChange={(event) =>
                  setTriggerDraft((current) => ({
                    ...current,
                    state: asPersonaVisualStateId(event.target.value)
                  }))
                }
              >
                {activeVisualStates.map((state) => (
                  <option key={state} value={state}>
                    {state}
                  </option>
                ))}
              </select>
              <input
                data-testid="persona-visual-trigger-duration-input"
                type="number"
                className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                value={triggerDraft.durationMs}
                onChange={(event) =>
                  setTriggerDraft((current) => ({
                    ...current,
                    durationMs: event.target.value
                  }))
                }
              />
              <input
                data-testid="persona-visual-trigger-priority-input"
                type="number"
                className="rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                value={triggerDraft.priority}
                onChange={(event) =>
                  setTriggerDraft((current) => ({
                    ...current,
                    priority: event.target.value
                  }))
                }
              />
              <Button
                data-testid="persona-visual-add-trigger"
                size="small"
                icon={<Plus className="h-3.5 w-3.5" />}
                disabled={!triggerDraft.match.trim()}
                onClick={handleAddTrigger}
              >
                Add
              </Button>
            </div>
            {draftManifest.authored_triggers?.length ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {draftManifest.authored_triggers.map((trigger) => (
                  <Tag key={trigger.id} color="purple">
                    {`${trigger.source}:${trigger.match} -> ${trigger.state}`}
                  </Tag>
                ))}
              </div>
            ) : null}
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <Typography.Text strong>Validation</Typography.Text>
                {validationErrors.length ? (
                  <div
                    data-testid="persona-visual-validation-errors"
                    className="mt-1 text-xs text-danger"
                  >
                    {validationErrors.join(" ")}
                  </div>
                ) : (
                  <div className="mt-1 text-xs text-success">
                    Required states resolve.
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  data-testid="persona-visual-save-manifest"
                  size="small"
                  type="primary"
                  icon={<Save className="h-3.5 w-3.5" />}
                  loading={saving}
                  onClick={() => void handleSaveManifest()}
                >
                  Save manifest
                </Button>
                <Button
                  data-testid="persona-visual-activate-button"
                  size="small"
                  icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                  disabled={validationErrors.length > 0}
                  loading={activating}
                  onClick={() => void handleActivate()}
                >
                  Activate
                </Button>
                <Button
                  data-testid="persona-visual-deactivate-button"
                  size="small"
                  icon={<XCircle className="h-3.5 w-3.5" />}
                  loading={deactivating}
                  disabled={!visiblePacks.some((pack) => pack.status === "active")}
                  onClick={() => void handleDeactivate()}
                >
                  Deactivate
                </Button>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
                Generated Candidates
              </div>
              <Button
                data-testid="persona-visual-candidates-refresh-button"
                size="small"
                onClick={() => void loadCandidates()}
              >
                {candidatesLoading ? loadingLabel : refreshLabel}
              </Button>
            </div>
            <div
              data-testid="persona-visual-generation-review-copy"
              className="mt-2 text-xs leading-5 text-text-muted"
            >
              {t("sidepanel:personaGarden.visuals.generationReviewHelp", {
                defaultValue:
                  "Generated candidates stay in review until accepted. Accepting a candidate updates this pack's manifest and assets; activation remains the explicit pack-level action."
              })}
            </div>
            <div className="mt-3 grid gap-2 md:grid-cols-[minmax(180px,1fr)_150px_minmax(120px,160px)_auto]">
              <label className="text-xs text-text-muted">
                <span className="mb-1 block">Prompt</span>
                <input
                  data-testid="persona-visual-generation-prompt-input"
                  className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                  value={generationPrompt}
                  onChange={(event) => setGenerationPrompt(event.target.value)}
                />
              </label>
              <label className="text-xs text-text-muted">
                <span className="mb-1 block">Target state</span>
                <select
                  data-testid="persona-visual-generation-target-state-select"
                  className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                  value={normalizedGenerationTargetState}
                  onChange={(event) =>
                    setGenerationTargetState(asPersonaVisualStateId(event.target.value))
                  }
                >
                  {activeVisualStates.map((state) => (
                    <option key={state} value={state}>
                      {state}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-text-muted">
                <span className="mb-1 block">Backend</span>
                <input
                  data-testid="persona-visual-generation-backend-input"
                  className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
                  value={generationBackend}
                  onChange={(event) => setGenerationBackend(event.target.value)}
                  list="persona-visual-generation-backends"
                />
                {generationReadinessView.enabledBackends.length ? (
                  <datalist id="persona-visual-generation-backends">
                    {generationReadinessView.enabledBackends.map((backend) => (
                      <option key={backend} value={backend} />
                    ))}
                  </datalist>
                ) : null}
              </label>
              <Button
                data-testid="persona-visual-generation-enqueue-button"
                className="self-end"
                size="small"
                type="primary"
                loading={enqueueingGeneration}
                disabled={!generationPrompt.trim() || !generationReadinessView.canQueue}
                onClick={() => void handleEnqueueGeneration()}
              >
                Queue
              </Button>
            </div>
            <div
              data-testid="persona-visual-generation-readiness"
              className={`mt-3 rounded border px-3 py-2 text-xs ${generationReadinessCopy.toneClassName}`}
            >
              <div className="font-medium">{generationReadinessCopy.title}</div>
              <div className="mt-1 text-current/80">{generationReadinessCopy.message}</div>
              {generationReadinessView.queue ? (
                <div className="mt-1 text-current/70">
                  Jobs queue: {generationReadinessView.queue}
                </div>
              ) : null}
              {generationReadinessView.enabledBackends.length ? (
                <div className="mt-1 text-current/70">
                  Enabled image backends: {generationReadinessView.enabledBackends.join(", ")}
                </div>
              ) : null}
            </div>
            <div className="mt-3 space-y-2">
              {visibleCandidates.length ? (
                visibleCandidates.map((candidate) => (
                  <div
                    key={candidate.id}
                    className="rounded border border-border bg-bg p-2 text-xs"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <div className="font-medium text-text">
                          {candidate.prompt || candidate.id}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1 text-text-muted">
                          <Tag>{candidate.status}</Tag>
                          {candidate.job_id ? <span>{candidate.job_id}</span> : null}
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <Button
                          data-testid={`persona-visual-candidate-accept-${candidate.id}`}
                          size="small"
                          loading={reviewingCandidateId === candidate.id}
                          onClick={() => void handleReviewCandidate(candidate.id, "accepted")}
                        >
                          Accept
                        </Button>
                        <Button
                          data-testid={`persona-visual-candidate-reject-${candidate.id}`}
                          size="small"
                          danger
                          loading={reviewingCandidateId === candidate.id}
                          onClick={() => void handleReviewCandidate(candidate.id, "rejected")}
                        >
                          Reject
                        </Button>
                      </div>
                    </div>
                    {candidate.generated_assets?.length ? (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {candidate.generated_assets.map((asset) => (
                          <div
                            key={asset.id}
                            className="flex items-center gap-2 rounded border border-border bg-surface px-2 py-1"
                          >
                            <img
                              src={asset.url}
                              alt={asset.original_filename || asset.id}
                              className="h-10 w-10 rounded object-contain"
                            />
                            <div>
                              <div className="text-text">
                                {asset.original_filename || asset.id}
                              </div>
                              <div className="text-text-muted">{asset.id}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))
              ) : (
                <div className="rounded border border-dashed border-border bg-bg px-3 py-2 text-xs text-text-muted">
                  No generated candidates.
                </div>
              )}
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}

export default VisualPackEditor
