import React from "react"
import { Button, Tag, Typography } from "antd"
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
  deletePersonaVisualLibraryItem,
  deactivatePersonaVisualPack,
  downloadPersonaVisualPackExportArchive,
  duplicatePersonaVisualPack,
  getPersonaVisualGenerationReadiness,
  getPersonaVisualImportCommitStatus,
  getPersonaVisualImportPreview,
  getPersonaVisualPackExportJob,
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
  usePersonaVisualLibraryItem
} from "@/services/persona-visuals"
import type {
  PersonaVisualAnimation,
  PersonaVisualAsset,
  PersonaVisualAssetRole,
  PersonaVisualAuthoredTrigger,
  PersonaVisualCandidate,
  PersonaVisualDuplicateTarget,
  PersonaVisualLibraryItem,
  PersonaVisualImportCommitStartResponse,
  PersonaVisualFrame,
  PersonaVisualGenerationReadinessResponse,
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualManifest,
  PersonaVisualPack,
  PersonaVisualPackExportResponse,
  PersonaVisualPortabilityJobResponse,
  PersonaVisualStateId
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

const getGenerationReadinessCopy = (
  view: PersonaVisualGenerationReadinessView,
  t: (key: string, options?: { defaultValue?: string }) => string
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

const REQUIRED_VISUAL_STATES: PersonaVisualStateId[] = [
  "idle",
  "listening",
  "thinking",
  "speaking",
  "error"
]

const OPTIONAL_VISUAL_STATES: PersonaVisualStateId[] = [
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
  "mcp_runtime"
]

const PORTABLE_VISUAL_PACK_EXTENSION = ".tldw-persona-vpack"
const IMPORT_COMMIT_TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "quarantined"
])

const DEFAULT_MANIFEST: PersonaVisualManifest = {
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {},
  animations: {},
  fallbacks: {},
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

const isFullImportPreview = (
  preview: PersonaVisualImportPreviewStartResponse | PersonaVisualImportPreviewResponse | null
): preview is PersonaVisualImportPreviewResponse =>
  Boolean(preview && "bundle_summary" in preview)

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
  item.source_persona_name_snapshot ||
  item.source_persona_id ||
  "Unavailable persona"

const getLibrarySourcePackTitle = (item: PersonaVisualLibraryItem): string =>
  item.source_pack_title ||
  item.source_pack_title_snapshot ||
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

export const VisualPackEditor: React.FC<VisualPackEditorProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  isActive = false,
  onOpenPersonaVisuals
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const loadingLabel = t("common:loading", getDesignSystemState("loading").label)
  const refreshLabel = t("common:refresh", "Refresh")
  const [packs, setPacks] = React.useState<PersonaVisualPack[]>([])
  const [selectedPackId, setSelectedPackId] = React.useState("")
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
  const generationReadinessRequestIdRef = React.useRef(0)
  const duplicateTargetsRequestIdRef = React.useRef(0)
  const libraryRequestIdRef = React.useRef(0)

  const selectedPack =
    packs.find((pack) => pack.id === selectedPackId) ?? packs[0] ?? null
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

  const loadPacks = React.useCallback(async (): Promise<boolean> => {
    if (!isActive || !selectedPersonaId) {
      setPacks([])
      setSelectedPackId("")
      setDraftManifest(DEFAULT_MANIFEST)
      setCandidates([])
      return false
    }
    setLoading(true)
    setError(null)
    try {
      const response = await listPersonaVisualPacks(selectedPersonaId)
      const nextPacks = response.packs || []
      setPacks(nextPacks)
      const preferred =
        response.active_pack ??
        nextPacks.find((pack) => pack.id === selectedPackId) ??
        nextPacks[0] ??
        null
      setSelectedPackId(preferred?.id || "")
      if (!preferred) {
        setDraftManifest(DEFAULT_MANIFEST)
        setSelectedAnimationId("")
      }
      return true
    } catch (loadError) {
      setPacks([])
      setSelectedPackId("")
      setDraftManifest(DEFAULT_MANIFEST)
      setError(
        loadError instanceof Error
          ? loadError.message
            : "Failed to load visual packs."
      )
      return false
    } finally {
      setLoading(false)
    }
  }, [isActive, selectedPersonaId])

  const loadCandidates = React.useCallback(async () => {
    const packId = selectedPack?.id || ""
    if (!isActive || !selectedPersonaId || !packId) {
      setCandidates([])
      return
    }
    setCandidatesLoading(true)
    try {
      const response = await listPersonaVisualCandidates(selectedPersonaId, packId)
      setCandidates(response.candidates || [])
    } catch (loadError) {
      setCandidates([])
      if (!(loadError instanceof PersonaVisualApiError && loadError.status === 404)) {
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load generated candidates."
        )
      }
    } finally {
      setCandidatesLoading(false)
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
    const requestId = generationReadinessRequestIdRef.current + 1
    generationReadinessRequestIdRef.current = requestId
    const isLatestRequest = () => generationReadinessRequestIdRef.current === requestId
    setGenerationReadinessLoading(true)
    setGenerationReadinessError(null)
    try {
      const response = await getPersonaVisualGenerationReadiness(
        selectedPersonaId,
        packId
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
    const isLatestRequest = () => duplicateTargetsRequestIdRef.current === requestId
    setDuplicateTargets([])
    setDuplicateTargetId("")
    setDuplicateTargetsLoading(true)
    try {
      const targets = await listPersonaVisualDuplicateTargets()
      if (!isLatestRequest()) return
      const available = targets.filter((target) => target.id !== selectedPersonaId)
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
    setExportJob(null)
    setImportPreview(null)
    setImportCommitJob(null)
    setSelectedImportPreviewFile(null)
    setLastDuplicatedPersonaId("")
    setLibraryEditingItemId("")
    setLibraryEditDraft({ title: "", notes: "", tags: "" })
    if (importPreviewInputRef.current) importPreviewInputRef.current.value = ""
  }, [selectedPersonaId, selectedPack?.id])

  React.useEffect(() => {
    setDuplicateTitle(selectedPack ? `Copy of ${selectedPack.title}` : "")
    setLastDuplicatedPersonaId("")
  }, [selectedPack?.id])

  const updateManifest = React.useCallback(
    (updater: (manifest: PersonaVisualManifest) => PersonaVisualManifest) => {
      setDraftManifest((current) => normalizeManifest(updater(normalizeManifest(current))))
      setStatusMessage(null)
    },
    []
  )

  const handleCreateDraft = async () => {
    const title = draftTitle.trim()
    if (!selectedPersonaId || !title) return
    setSaving(true)
    setError(null)
    try {
      const created = await createPersonaVisualPack(selectedPersonaId, {
        title,
        manifest: DEFAULT_MANIFEST
      })
      setPacks((current) => mergePack(current, created))
      setSelectedPackId(created.id)
      setDraftTitle("")
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.created", {
          defaultValue: "Draft created."
        })
      )
    } catch (createError) {
      setError(
        createError instanceof Error
          ? createError.message
          : t("sidepanel:personaGarden.visuals.createError", {
              defaultValue: "Failed to create visual pack."
            })
      )
    } finally {
      setSaving(false)
    }
  }

  const handleUploadAsset = async () => {
    if (!selectedPersonaId || !selectedPack || !selectedUploadFile) return
    setUploading(true)
    setError(null)
    try {
      const asset = await uploadPersonaVisualAsset(
        selectedPersonaId,
        selectedPack.id,
        selectedUploadFile,
        uploadRole
      )
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
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : t("sidepanel:personaGarden.visuals.uploadError", {
              defaultValue: "Failed to upload asset."
            })
      )
    } finally {
      setUploading(false)
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
      const duplicated = await usePersonaVisualLibraryItem(item.id, {
        target_persona_id: targetPersonaId
      })
      if (duplicated.persona_id === selectedPersonaId) {
        setPacks((current) => mergePack(current, duplicated))
        setSelectedPackId(duplicated.id)
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
    setPreviewingImport(true)
    setError(null)
    try {
      const preview = await createPersonaVisualImportPreview(
        selectedPersonaId,
        selectedImportPreviewFile
      )
      setImportPreview(preview)
      setImportCommitJob(null)
      setSelectedImportPreviewFile(null)
      if (importPreviewInputRef.current) importPreviewInputRef.current.value = ""
      setStatusMessage("Import preview queued.")
    } catch (previewError) {
      setError(
        previewError instanceof Error
          ? previewError.message
          : "Failed to queue visual pack import preview."
      )
    } finally {
      setPreviewingImport(false)
    }
  }

  const handleRefreshImportPreview = async () => {
    if (!selectedPersonaId || !importPreview?.preview_id) return
    setRefreshingImportPreview(true)
    setError(null)
    try {
      const preview = await getPersonaVisualImportPreview(
        selectedPersonaId,
        importPreview.preview_id
      )
      setImportPreview(preview)
    } catch (refreshError) {
      setError(
        refreshError instanceof Error
          ? refreshError.message
          : "Failed to refresh visual pack import preview."
      )
    } finally {
      setRefreshingImportPreview(false)
    }
  }

  const handleStartImportCommit = async () => {
    if (!selectedPersonaId || !fullImportPreview?.preview_id) return
    if (fullImportPreview.status !== "completed") return
    setCommittingImport(true)
    setError(null)
    try {
      const job = await startPersonaVisualImportCommit(
        selectedPersonaId,
        fullImportPreview.preview_id,
        {
          trust_mode: "untrusted_import",
          target_mode: "create_new"
        }
      )
      setImportCommitJob(job)
      setStatusMessage(
        "Import commit queued. Imported packs remain drafts until activated."
      )
    } catch (commitError) {
      setError(
        commitError instanceof Error
          ? commitError.message
          : "Failed to queue visual pack import commit."
      )
    } finally {
      setCommittingImport(false)
    }
  }

  const handleRefreshImportCommit = async () => {
    if (!selectedPersonaId || !importCommitJob?.job_id) return
    setRefreshingImportCommit(true)
    setError(null)
    try {
      const job = await getPersonaVisualImportCommitStatus(
        selectedPersonaId,
        importCommitJob.job_id
      )
      setImportCommitJob(job)
      if (job.status === "completed" && job.pack_id) {
        const refreshed = await loadPacks()
        if (refreshed) {
          setStatusMessage(
            "Import commit completed. Review and activate the new draft when ready."
          )
        } else {
          setStatusMessage(null)
        }
      }
    } catch (refreshError) {
      setError(
        refreshError instanceof Error
          ? refreshError.message
          : "Failed to refresh visual pack import commit."
      )
    } finally {
      setRefreshingImportCommit(false)
    }
  }

  const handleSaveManifest = async () => {
    if (!selectedPersonaId || !selectedPack) return
    setSaving(true)
    setError(null)
    try {
      const saved = await updatePersonaVisualManifest(
        selectedPersonaId,
        selectedPack.id,
        {
          manifest: draftManifest,
          expected_version: selectedPack.version ?? null
        }
      )
      setPacks((current) =>
        mergePack(current, {
          ...saved,
          assets: getPackAssets(saved).length ? getPackAssets(saved) : assets
        })
      )
      setSelectedPackId(saved.id)
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.saved", {
          defaultValue: "Manifest saved."
        })
      )
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("sidepanel:personaGarden.visuals.saveError", {
              defaultValue: "Failed to save visual manifest."
            })
      )
    } finally {
      setSaving(false)
    }
  }

  const handleActivate = async () => {
    if (!selectedPersonaId || !selectedPack || validationErrors.length) return
    setActivating(true)
    setError(null)
    try {
      const active = await activatePersonaVisualPack(selectedPersonaId, selectedPack.id)
      setPacks((current) =>
        current.map((pack) =>
          pack.id === active.id
            ? { ...active, assets: getPackAssets(active).length ? getPackAssets(active) : assets }
            : pack.status === "active"
              ? { ...pack, status: "archived" }
              : pack
        )
      )
      setSelectedPackId(active.id)
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.activated", {
          defaultValue: "Visual pack activated."
        })
      )
    } catch (activateError) {
      setError(
        activateError instanceof Error
          ? activateError.message
          : t("sidepanel:personaGarden.visuals.activateError", {
              defaultValue: "Failed to activate visual pack."
            })
      )
    } finally {
      setActivating(false)
    }
  }

  const handleDeactivate = async () => {
    if (!selectedPersonaId) return
    setDeactivating(true)
    setError(null)
    try {
      await deactivatePersonaVisualPack(selectedPersonaId)
      setPacks((current) =>
        current.map((pack) =>
          pack.status === "active" ? { ...pack, status: "archived" } : pack
        )
      )
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.deactivated", {
          defaultValue: "Active visual pack deactivated."
        })
      )
    } catch (deactivateError) {
      setError(
        deactivateError instanceof Error
          ? deactivateError.message
          : t("sidepanel:personaGarden.visuals.deactivateError", {
              defaultValue: "Failed to deactivate visual pack."
            })
      )
    } finally {
      setDeactivating(false)
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
    setEnqueueingGeneration(true)
    setError(null)
    try {
      const job = await createPersonaVisualGenerationJob(
        selectedPersonaId,
        selectedPack.id,
        {
          prompt: generationPrompt.trim(),
          target_state: generationTargetState || null,
          backend: generationBackend.trim() || null
        }
      )
      setGenerationPrompt("")
      setStatusMessage(
        t("sidepanel:personaGarden.visuals.generationQueued", {
          defaultValue: `Generation job ${job.job_id} queued.`
        })
      )
    } catch (generationError) {
      setError(
        generationError instanceof Error
          ? generationError.message
          : t("sidepanel:personaGarden.visuals.generationError", {
              defaultValue: "Failed to queue generation job."
            })
      )
    } finally {
      setEnqueueingGeneration(false)
    }
  }

  const handleReviewCandidate = async (
    candidateId: string,
    status: "accepted" | "rejected"
  ) => {
    if (!selectedPersonaId || !selectedPack) return
    setReviewingCandidateId(candidateId)
    setError(null)
    try {
      const updated = await reviewPersonaVisualCandidate(
        selectedPersonaId,
        selectedPack.id,
        candidateId,
        {
          status,
          failure_reason: status === "rejected" ? "Rejected in editor." : null
        }
      )
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
      setError(
        reviewError instanceof Error
          ? reviewError.message
          : t("sidepanel:personaGarden.visuals.reviewError", {
              defaultValue: "Failed to review candidate."
            })
      )
    } finally {
      setReviewingCandidateId("")
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
        .map((item) => item.trim() as PersonaVisualStateId)
        .filter((item) => VISUAL_STATES.includes(item))
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
              <Tag color="green">available</Tag>
            ) : (
              <Tag color="red">unavailable</Tag>
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
  const importPreviewWarnings = fullImportPreview
    ? [
        ...(fullImportPreview.validation_warnings || []),
        ...(fullImportPreview.target_warnings || [])
      ]
    : []
  const importPreviewConflicts = fullImportPreview?.conflicts || []
  const importPreviewPlan = fullImportPreview?.proposed_plan || null
  const canCommitImportPreview = fullImportPreview?.status === "completed"
  const importCommitStatus = importCommitJob?.status || null
  const importCommitIsTerminal = importCommitStatus
    ? IMPORT_COMMIT_TERMINAL_STATUSES.has(importCommitStatus)
    : false
  const canStartImportCommit =
    !importCommitJob?.job_id || importCommitStatus === "failed"
  const canRefreshImportCommit =
    Boolean(importCommitJob?.job_id) && !importCommitIsTerminal

  return (
    <div className="space-y-3" data-testid="persona-visual-pack-editor">
      <div className="rounded-lg border border-border bg-surface p-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
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
          <Button
            size="small"
            onClick={() => void loadPacks()}
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
              onChange={(event) => setSelectedPackId(event.target.value)}
              disabled={!packs.length}
            >
              {packs.map((pack) => (
                <option key={pack.id} value={pack.id}>
                  {pack.title}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-text-muted">
            <span className="mb-1 block">New draft title</span>
            <input
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
        {!loading && !packs.length ? (
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

      <div
        data-testid="persona-visual-library-panel"
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
                    accept={`${PORTABLE_VISUAL_PACK_EXTENSION},application/zip,application/octet-stream`}
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
                    disabled={!selectedImportPreviewFile}
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
                    {canCommitImportPreview ? (
                      <div className="mt-2 border-t border-border pt-2">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="font-medium text-text">
                            Commit reviewed import
                          </div>
                          <Tag>creates draft</Tag>
                        </div>
                        <div className="mt-1 text-text-muted">
                          Commit creates a new draft pack. Activation remains separate.
                        </div>
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
                          <div className="mt-2 flex flex-wrap items-center gap-2 text-text-muted">
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
                        ) : (
                          <div className="mt-2 text-text-muted">
                            No import commit job.
                          </div>
                        )}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="mt-2 text-xs text-text-muted">No import preview.</div>
                )}
              </div>
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
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {OPTIONAL_VISUAL_STATES.map((state) => (
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
                    state: event.target.value as PersonaVisualStateId
                  }))
                }
              >
                {VISUAL_STATES.map((state) => (
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
                  disabled={!packs.some((pack) => pack.status === "active")}
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
              <Button size="small" onClick={() => void loadCandidates()}>
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
                  value={generationTargetState}
                  onChange={(event) =>
                    setGenerationTargetState(event.target.value as PersonaVisualStateId)
                  }
                >
                  {VISUAL_STATES.map((state) => (
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
              {candidates.length ? (
                candidates.map((candidate) => (
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
