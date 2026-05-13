import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { getDesignSystemState } from "@/design-system"
import type { RagPresetName, RagSource } from "@/services/rag/unified-rag"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { cn } from "@/libs/utils"
import { Popover, Tooltip } from "antd"
import {
  ChevronDown,
  Layers,
  Settings,
  Globe,
  Check,
  CircleHelp,
  Info,
  Search,
  LoaderCircle,
  Filter,
  Bookmark,
  Trash2,
} from "lucide-react"
import {
  ALL_RAG_SOURCES,
  getRagSourceDescription,
  getRagSourceLabel,
  isRagSource,
} from "@/services/rag/sourceMetadata"
import { AnswerModelMenu } from "./AnswerModelMenu"

// ---------------------------------------------------------------------------
// Saved search profiles
// ---------------------------------------------------------------------------

const PROFILES_STORAGE_KEY = "tldw:knowledge-qa:saved-profiles"
const MAX_SAVED_PROFILES = 5
const READY_STATE_LABEL = getDesignSystemState("ready").label
const ERROR_STATE_LABEL = getDesignSystemState("error").label
const UNAVAILABLE_STATE_LABEL = getDesignSystemState("unavailable").label

type SearchProfile = {
  name: string
  sources: RagSource[]
  includeMediaIds?: number[]
  includeNoteIds?: string[]
  preset: RagPresetName
  enableWebFallback: boolean
}

const VALID_PRESET_KEYS: ReadonlySet<RagPresetName> = new Set([
  "fast",
  "balanced",
  "thorough",
  "custom",
])

function isSearchProfile(value: unknown): value is SearchProfile {
  const record = value as Record<string, unknown>
  return (
    typeof value === "object" &&
    value !== null &&
    typeof record.name === "string" &&
    record.name.trim().length > 0 &&
    Array.isArray(record.sources) &&
    record.sources.every(
      (source): source is RagSource => isRagSource(source)
    ) &&
    (record.includeMediaIds === undefined ||
      (Array.isArray(record.includeMediaIds) && record.includeMediaIds.every(
        (id): id is number => typeof id === "number" && Number.isInteger(id) && id > 0
      ))) &&
    (record.includeNoteIds === undefined ||
      (Array.isArray(record.includeNoteIds) && record.includeNoteIds.every(
        (id): id is string => typeof id === "string" && id.trim().length > 0
      ))) &&
    typeof record.preset === "string" &&
    VALID_PRESET_KEYS.has(record.preset as RagPresetName) &&
    typeof record.enableWebFallback === "boolean"
  )
}

function loadSavedProfiles(): SearchProfile[] {
  try {
    const raw = localStorage.getItem(PROFILES_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isSearchProfile).slice(0, MAX_SAVED_PROFILES)
  } catch {
    return []
  }
}

function persistProfiles(profiles: SearchProfile[]): void {
  try {
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles.slice(0, MAX_SAVED_PROFILES)))
  } catch {
    // localStorage full or unavailable -- silently ignore
  }
}

type KnowledgeContextBarProps = {
  preset: RagPresetName
  onPresetChange: (preset: RagPresetName) => void
  sources: RagSource[]
  onSourcesChange: (sources: RagSource[]) => void
  includeMediaIds: number[]
  onIncludeMediaIdsChange: (ids: number[]) => void
  includeNoteIds: string[]
  onIncludeNoteIdsChange: (ids: string[]) => void
  webEnabled: boolean
  onToggleWeb: () => void
  generationProvider: string | null
  generationModel: string | null
  onGenerationProviderChange: (provider: string | null) => void
  onGenerationModelChange: (model: string | null) => void
  contextChangedSinceLastRun: boolean
  scopeChangeDetails?: string[]
  onOpenSettings: () => void
}

type PresetKey = Exclude<RagPresetName, "custom">

type PresetDetails = {
  label: string
  summary: string
  responseTime: string
  sourcesChecked: string
  bestFor: string
}

type GranularSourceOption<T extends string | number> = {
  id: T
  label: string
  meta?: string
  status: string
  recent: boolean
  workspaceId?: string
  workspaceName?: string
  hiddenByDefault: boolean
}

const PRESET_OPTIONS: Array<{ value: PresetKey; label: string }> = [
  { value: "fast", label: "Fast" },
  { value: "balanced", label: "Balanced" },
  { value: "thorough", label: "Deep" },
]

const PRESET_DETAILS: Record<PresetKey, PresetDetails> = {
  fast: {
    label: "Fast",
    summary: "Quick lookup with minimal retrieval and rerank depth.",
    responseTime: "Fastest",
    sourcesChecked: "Fewer",
    bestFor: "Fact checks and quick lookups",
  },
  balanced: {
    label: "Balanced",
    summary: "Balanced retrieval depth for quality and speed.",
    responseTime: "Moderate",
    sourcesChecked: "Moderate",
    bestFor: "Default day-to-day research",
  },
  thorough: {
    label: "Deep",
    summary: "Exhaustive retrieval plus extra verification steps.",
    responseTime: "Slower",
    sourcesChecked: "Most thorough",
    bestFor: "High-confidence synthesis",
  },
}

/** Short comparison tooltips shown on hover for each preset button. */
const PRESET_COMPARISON_TOOLTIPS: Record<PresetKey, string> = {
  fast: "Checks fewer sources, fastest response (~2-5s)",
  balanced: "Moderate depth, good for most queries (~5-15s)",
  thorough: "Checks most sources, slowest but most thorough (~15-30s)",
}

const SOURCE_OPTIONS: Array<{ key: RagSource; label: string }> =
  ALL_RAG_SOURCES.map((source) => ({
    key: source,
    label: getRagSourceLabel(source),
  }))

const MAX_VISIBLE_GRANULAR_RESULTS = 80

function summarizeSources(sources: RagSource[]): string {
  if (!Array.isArray(sources) || sources.length === 0) {
    return "None selected"
  }
  if (sources.length === 1) {
    return getRagSourceLabel(sources[0])
  }
  if (sources.length >= ALL_RAG_SOURCES.length) {
    return "All sources"
  }
  return `${sources.length} selected`
}

function summarizeSpecificSources(mediaIds: number[], noteIds: string[]): string {
  if (mediaIds.length === 0 && noteIds.length === 0) {
    return "All items"
  }

  const parts: string[] = []
  if (mediaIds.length > 0) {
    parts.push(`${mediaIds.length} doc${mediaIds.length === 1 ? "" : "s"}`)
  }
  if (noteIds.length > 0) {
    parts.push(`${noteIds.length} note${noteIds.length === 1 ? "" : "s"}`)
  }
  return parts.join(" • ")
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object") return null
  return value as Record<string, unknown>
}

function asString(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value)
  }
  return null
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.round(value)
  }
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return Math.round(parsed)
    }
  }
  return null
}

function asBoolean(value: unknown): boolean {
  if (typeof value === "boolean") return value
  if (typeof value === "number") return value !== 0
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    return normalized === "true" || normalized === "yes" || normalized === "1"
  }
  return false
}

function normalizeOptionStatus(record: Record<string, unknown>): string {
  return (
    asString(record.status) ??
    asString(record.processing_status) ??
    asString(record.indexing_status) ??
    asString(record.ingest_status) ??
    "ready"
  ).toLowerCase()
}

function getWorkspaceMetadata(record: Record<string, unknown>): {
  workspaceId?: string
  workspaceName?: string
} {
  const workspace = asRecord(record.workspace)
  const workspaceId =
    asString(record.workspace_id) ??
    asString(record.workspaceId) ??
    asString(workspace?.id)
  const workspaceName =
    asString(record.workspace_name) ??
    asString(record.workspaceName) ??
    asString(workspace?.name)

  return {
    workspaceId: workspaceId ?? undefined,
    workspaceName: workspaceName ?? workspaceId ?? undefined,
  }
}

function isRecentImport(record: Record<string, unknown>): boolean {
  if (
    asBoolean(record.recent_import) ||
    asBoolean(record.recently_imported) ||
    asBoolean(record.recently_ingested)
  ) {
    return true
  }

  const timestamp =
    asString(record.imported_at) ??
    asString(record.created_at) ??
    asString(record.updated_at)
  if (!timestamp) return false
  const parsed = Date.parse(timestamp)
  if (!Number.isFinite(parsed)) return false
  const sevenDaysMs = 7 * 24 * 60 * 60 * 1000
  return Date.now() - parsed <= sevenDaysMs
}

function isHiddenArtifact(record: Record<string, unknown>): boolean {
  const artifactKind = (
    asString(record.artifact_kind) ??
    asString(record.source_kind) ??
    asString(record.kind) ??
    ""
  ).toLowerCase()
  return (
    asBoolean(record.is_generated) ||
    asBoolean(record.generated) ||
    asBoolean(record.test_artifact) ||
    asBoolean(record.is_test) ||
    asBoolean(record.workspace_artifact) ||
    asBoolean(record.is_workspace_artifact) ||
    artifactKind.includes("generated") ||
    artifactKind.includes("test") ||
    artifactKind.includes("workspace_artifact")
  )
}

function extractResponseItems(payload: unknown): unknown[] {
  const record = asRecord(payload)
  if (!record) return []

  const candidates = [record.items, record.media, record.results, record.data]
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) return candidate
  }
  return []
}

function normalizeMediaOptions(payload: unknown): GranularSourceOption<number>[] {
  const seen = new Set<number>()
  const normalized: GranularSourceOption<number>[] = []

  for (const item of extractResponseItems(payload)) {
    const record = asRecord(item)
    if (!record) continue

    const id = asNumber(record.media_id ?? record.id)
    if (id === null || id <= 0 || seen.has(id)) continue

    const label =
      asString(record.title) ??
      asString(record.name) ??
      asString(record.filename) ??
      asString(record.url) ??
      `Media ${id}`

    const status = normalizeOptionStatus(record)
    const { workspaceId, workspaceName } = getWorkspaceMetadata(record)
    const metaParts = [
      asString(record.type) ?? asString(record.media_type),
      status,
      workspaceName,
    ].filter((part): part is string => Boolean(part))

    seen.add(id)
    normalized.push({
      id,
      label,
      meta: metaParts.join(" • ") || undefined,
      status,
      recent: isRecentImport(record),
      workspaceId,
      workspaceName,
      hiddenByDefault: isHiddenArtifact(record),
    })
  }

  return normalized.sort((left, right) => left.label.localeCompare(right.label))
}

function normalizeNoteOptions(payload: unknown): GranularSourceOption<string>[] {
  const seen = new Set<string>()
  const normalized: GranularSourceOption<string>[] = []

  for (const item of extractResponseItems(payload)) {
    const record = asRecord(item)
    if (!record) continue

    const id = asString(record.id ?? record.note_id)
    if (id === null || seen.has(id)) continue

    const contentPreview = asString(record.content)?.slice(0, 80)
    const label = asString(record.title) ?? asString(record.name) ?? contentPreview ?? `Note ${id}`

    const status = normalizeOptionStatus(record)
    const { workspaceId, workspaceName } = getWorkspaceMetadata(record)
    const metaParts = [
      asString(record.updated_at) ?? asString(record.created_at),
      status,
      workspaceName,
    ].filter((part): part is string => Boolean(part))

    seen.add(id)
    normalized.push({
      id,
      label,
      meta: metaParts.join(" • ") || undefined,
      status,
      recent: isRecentImport(record),
      workspaceId,
      workspaceName,
      hiddenByDefault: isHiddenArtifact(record),
    })
  }

  return normalized.sort((left, right) => left.label.localeCompare(right.label))
}

export function KnowledgeContextBar({
  preset,
  onPresetChange,
  sources,
  onSourcesChange,
  includeMediaIds,
  onIncludeMediaIdsChange,
  includeNoteIds,
  onIncludeNoteIdsChange,
  webEnabled,
  onToggleWeb,
  generationProvider,
  generationModel,
  onGenerationProviderChange,
  onGenerationModelChange,
  contextChangedSinceLastRun,
  scopeChangeDetails = [],
  onOpenSettings,
}: KnowledgeContextBarProps) {
  const { t } = useTranslation("knowledge")
  const [sourceMenuOpen, setSourceMenuOpen] = useState(false)
  const [granularMenuOpen, setGranularMenuOpen] = useState(false)
  const [granularTab, setGranularTab] = useState<"media" | "notes">("media")
  const [granularQuery, setGranularQuery] = useState("")
  const [granularStatusFilter, setGranularStatusFilter] = useState("all")
  const [granularRecentFilter, setGranularRecentFilter] = useState("all")
  const [granularWorkspaceFilter, setGranularWorkspaceFilter] = useState("")
  const [granularLoading, setGranularLoading] = useState(false)
  const [granularError, setGranularError] = useState<string | null>(null)
  const [granularLoaded, setGranularLoaded] = useState(false)
  const [mediaOptions, setMediaOptions] = useState<GranularSourceOption<number>[]>([])
  const [noteOptions, setNoteOptions] = useState<GranularSourceOption<string>[]>([])

  // Saved search profiles state
  const [savedProfiles, setSavedProfiles] = useState<SearchProfile[]>(loadSavedProfiles)
  const [profileMenuOpen, setProfileMenuOpen] = useState(false)
  const [profileSaveMode, setProfileSaveMode] = useState(false)
  const [profileNameInput, setProfileNameInput] = useState("")

  const sourceMenuRef = useRef<HTMLDivElement | null>(null)
  const granularMenuRef = useRef<HTMLDivElement | null>(null)
  const granularLoadRequestIdRef = useRef(0)

  const normalizedSources = useMemo(
    () =>
      Array.from(
        new Set(
          sources
            .filter(
              (value): value is RagSource =>
                typeof value === "string" &&
                SOURCE_OPTIONS.some((option) => option.key === value)
            )
        )
      ),
    [sources]
  )

  const normalizedMediaIds = useMemo(
    () =>
      Array.from(
        new Set(
          includeMediaIds
            .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
            .map((value) => Math.round(value))
            .filter((value) => value > 0)
        )
      ),
    [includeMediaIds]
  )

  const normalizedNoteIds = useMemo(
    () =>
      Array.from(
        new Set(
          includeNoteIds
            .filter((value): value is string => typeof value === "string")
            .map((value) => value.trim())
            .filter(Boolean)
        )
      ),
    [includeNoteIds]
  )

  const selectedMediaSet = useMemo(() => new Set(normalizedMediaIds), [normalizedMediaIds])
  const selectedNoteSet = useMemo(() => new Set(normalizedNoteIds), [normalizedNoteIds])

  const presetDetail = preset === "custom" ? null : PRESET_DETAILS[preset]
  const presetDescription =
    preset === "custom"
      ? "Custom: Using your manually configured retrieval and generation settings."
      : presetDetail.summary
  const countableScopeTotal =
    (normalizedSources.includes("media_db") ? mediaOptions.length : 0) +
    (normalizedSources.includes("notes") ? noteOptions.length : 0)
  const hasOnlyCountableSources = normalizedSources.every(
    (source) => source === "media_db" || source === "notes"
  )

  const workspaceOptions = useMemo(() => {
    const seen = new Set<string>()
    const options: Array<{ id: string; label: string }> = []
    ;[...mediaOptions, ...noteOptions].forEach((option) => {
      if (!option.workspaceId || seen.has(option.workspaceId)) return
      seen.add(option.workspaceId)
      options.push({
        id: option.workspaceId,
        label: option.workspaceName ?? option.workspaceId,
      })
    })
    return options.sort((left, right) => left.label.localeCompare(right.label))
  }, [mediaOptions, noteOptions])

  const filterGranularOptions = useCallback(
    <T extends string | number>(options: GranularSourceOption<T>[]) => {
      const query = granularQuery.trim().toLowerCase()
      return options
        .filter((option) => {
          if (!granularWorkspaceFilter && option.hiddenByDefault) return false
          if (granularWorkspaceFilter && option.workspaceId !== granularWorkspaceFilter) return false
          if (granularStatusFilter !== "all" && option.status !== granularStatusFilter) return false
          if (granularRecentFilter === "recent" && !option.recent) return false
          if (!query) return true
          const haystack = `${option.label} ${option.meta || ""}`.toLowerCase()
          return haystack.includes(query)
        })
        .slice(0, MAX_VISIBLE_GRANULAR_RESULTS)
    },
    [granularQuery, granularRecentFilter, granularStatusFilter, granularWorkspaceFilter]
  )

  const filteredMediaOptions = useMemo(() => {
    return filterGranularOptions(mediaOptions)
  }, [filterGranularOptions, mediaOptions])

  const filteredNoteOptions = useMemo(() => {
    return filterGranularOptions(noteOptions)
  }, [filterGranularOptions, noteOptions])

  const loadGranularOptions = useCallback(async () => {
    const requestId = granularLoadRequestIdRef.current + 1
    granularLoadRequestIdRef.current = requestId
    setGranularLoading(true)
    setGranularError(null)
    try {
      const [mediaResponse, notesResponse] = await Promise.all([
        tldwClient.listMedia({ page: 1, results_per_page: 200, include_keywords: false }),
        tldwClient.listNotes({ page: 1, results_per_page: 200, include_keywords: false }),
      ])

      if (granularLoadRequestIdRef.current !== requestId) {
        return
      }
      setMediaOptions(normalizeMediaOptions(mediaResponse))
      setNoteOptions(normalizeNoteOptions(notesResponse))
      setGranularLoaded(true)
    } catch (error) {
      if (granularLoadRequestIdRef.current !== requestId) {
        return
      }
      setGranularError(error instanceof Error ? error.message : "Failed to load source lists")
    } finally {
      if (granularLoadRequestIdRef.current !== requestId) {
        return
      }
      setGranularLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!sourceMenuOpen && !granularMenuOpen) return

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node
      if (sourceMenuRef.current?.contains(target)) return
      if (granularMenuRef.current?.contains(target)) return
      setSourceMenuOpen(false)
      setGranularMenuOpen(false)
    }
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return
      event.preventDefault()
      event.stopPropagation()
      setSourceMenuOpen(false)
      setGranularMenuOpen(false)
    }

    document.addEventListener("mousedown", handleClickOutside)
    document.addEventListener("keydown", handleEscape, true)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
      document.removeEventListener("keydown", handleEscape, true)
    }
  }, [sourceMenuOpen, granularMenuOpen])

  useEffect(() => {
    if (!granularMenuOpen || granularLoaded || granularLoading) return
    void loadGranularOptions()
  }, [granularMenuOpen, granularLoaded, granularLoading, loadGranularOptions])

  const toggleSource = (sourceKey: RagSource) => {
    const exists = normalizedSources.includes(sourceKey)
    const nextSources = exists
      ? normalizedSources.filter((value) => value !== sourceKey)
      : [...normalizedSources, sourceKey]
    onSourcesChange(nextSources)

    if (!exists) return

    if (sourceKey === "media_db" && normalizedMediaIds.length > 0) {
      onIncludeMediaIdsChange([])
    }
    if (sourceKey === "notes" && normalizedNoteIds.length > 0) {
      onIncludeNoteIdsChange([])
    }
  }

  const toggleMediaId = (id: number) => {
    const isRemoving = selectedMediaSet.has(id)
    const next = isRemoving
      ? normalizedMediaIds.filter((value) => value !== id)
      : [...normalizedMediaIds, id].sort((left, right) => left - right)
    if (!isRemoving && !normalizedSources.includes("media_db")) {
      onSourcesChange([...normalizedSources, "media_db"])
    }
    onIncludeMediaIdsChange(next)
  }

  const toggleNoteId = (id: string) => {
    const isRemoving = selectedNoteSet.has(id)
    const next = isRemoving
      ? normalizedNoteIds.filter((value) => value !== id)
      : [...normalizedNoteIds, id].sort((left, right) => left.localeCompare(right))
    if (!isRemoving && !normalizedSources.includes("notes")) {
      onSourcesChange([...normalizedSources, "notes"])
    }
    onIncludeNoteIdsChange(next)
  }

  const selectAllSources = () => {
    onSourcesChange(SOURCE_OPTIONS.map((option) => option.key))
  }

  const clearSources = () => {
    onSourcesChange([])
    if (normalizedMediaIds.length > 0) {
      onIncludeMediaIdsChange([])
    }
    if (normalizedNoteIds.length > 0) {
      onIncludeNoteIdsChange([])
    }
  }

  const clearSpecificSources = () => {
    onIncludeMediaIdsChange([])
    onIncludeNoteIdsChange([])
  }

  const selectVisibleSpecificSources = () => {
    if (granularTab === "media") {
      const visibleIds = filteredMediaOptions.map((option) => option.id)
      if (visibleIds.length === 0) return
      if (!normalizedSources.includes("media_db")) {
        onSourcesChange([...normalizedSources, "media_db"])
      }
      onIncludeMediaIdsChange(
        Array.from(new Set([...normalizedMediaIds, ...visibleIds])).sort((left, right) => left - right)
      )
      return
    }

    const visibleIds = filteredNoteOptions.map((option) => option.id)
    if (visibleIds.length === 0) return
    if (!normalizedSources.includes("notes")) {
      onSourcesChange([...normalizedSources, "notes"])
    }
    onIncludeNoteIdsChange(
      Array.from(new Set([...normalizedNoteIds, ...visibleIds])).sort((left, right) =>
        left.localeCompare(right)
      )
    )
  }

  const clearVisibleSpecificSources = () => {
    if (granularTab === "media") {
      const visibleIds = new Set(filteredMediaOptions.map((option) => option.id))
      onIncludeMediaIdsChange(normalizedMediaIds.filter((id) => !visibleIds.has(id)))
      return
    }

    const visibleIds = new Set(filteredNoteOptions.map((option) => option.id))
    onIncludeNoteIdsChange(normalizedNoteIds.filter((id) => !visibleIds.has(id)))
  }

  const selectRecentSpecificSources = () => {
    if (granularTab === "media") {
      const recentIds = filteredMediaOptions
        .filter((option) => option.recent)
        .map((option) => option.id)
      if (recentIds.length === 0) return
      if (!normalizedSources.includes("media_db")) {
        onSourcesChange([...normalizedSources, "media_db"])
      }
      onIncludeMediaIdsChange(recentIds)
      return
    }

    const recentIds = filteredNoteOptions
      .filter((option) => option.recent)
      .map((option) => option.id)
    if (recentIds.length === 0) return
    if (!normalizedSources.includes("notes")) {
      onSourcesChange([...normalizedSources, "notes"])
    }
    onIncludeNoteIdsChange(recentIds)
  }

  // ---- Saved search profile handlers ----

  const saveCurrentProfile = useCallback(() => {
    const trimmed = profileNameInput.trim()
    if (!trimmed) return
    const newProfile: SearchProfile = {
      name: trimmed,
      sources: [...normalizedSources],
      includeMediaIds: [...normalizedMediaIds],
      includeNoteIds: [...normalizedNoteIds],
      preset,
      enableWebFallback: webEnabled,
    }
    const updated = [newProfile, ...savedProfiles.filter((p) => p.name !== trimmed)].slice(
      0,
      MAX_SAVED_PROFILES
    )
    setSavedProfiles(updated)
    persistProfiles(updated)
    setProfileSaveMode(false)
    setProfileNameInput("")
  }, [
    profileNameInput,
    normalizedSources,
    normalizedMediaIds,
    normalizedNoteIds,
    preset,
    webEnabled,
    savedProfiles,
  ])

  const loadProfile = useCallback(
    (profile: SearchProfile) => {
      onSourcesChange(profile.sources)
      onIncludeMediaIdsChange(profile.includeMediaIds ?? [])
      onIncludeNoteIdsChange(profile.includeNoteIds ?? [])
      onPresetChange(profile.preset)
      // Only toggle web fallback if the profile value differs from current state.
      // Note: only a toggle callback is available (no direct setter).
      if (profile.enableWebFallback !== webEnabled) {
        onToggleWeb()
      }
      setProfileMenuOpen(false)
    },
    [onIncludeMediaIdsChange, onIncludeNoteIdsChange, onSourcesChange, onPresetChange, onToggleWeb, webEnabled]
  )

  const deleteProfile = useCallback(
    (name: string) => {
      const updated = savedProfiles.filter((p) => p.name !== name)
      setSavedProfiles(updated)
      persistProfiles(updated)
    },
    [savedProfiles]
  )

  const activeGranularOptions = granularTab === "media" ? filteredMediaOptions : filteredNoteOptions

  return (
    <div className="rounded-xl border border-border/80 bg-surface px-3 py-3.5 md:px-4">
      <div className="grid gap-3 lg:grid-cols-2">
        <section className="rounded-lg border border-border/80 bg-surface px-3 py-2.5">
          <div className="mb-2 flex items-center gap-2">
            <Layers className="h-4 w-4 text-primary" />
            <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
              {t("contextBar.sourceScope", "Source Scope")}
            </h3>
            <Tooltip title="Choose which types of content to include in your search. Select categories, then optionally pick specific items within each.">
              <Info className="h-3.5 w-3.5 text-text-subtle cursor-help" />
            </Tooltip>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="relative" ref={sourceMenuRef}>
              <button
                id="knowledge-source-selector-toggle"
                type="button"
                onClick={() => setSourceMenuOpen((previous) => !previous)}
                className="inline-flex h-7 items-center gap-1 rounded-md border border-border px-2 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                aria-expanded={sourceMenuOpen}
                aria-controls={sourceMenuOpen ? "knowledge-source-menu" : undefined}
              >
                <Layers className="h-3.5 w-3.5 text-text-muted" />
                Sources: {summarizeSources(normalizedSources)}
                <ChevronDown className="h-3.5 w-3.5 text-text-muted" />
              </button>
              {sourceMenuOpen ? (
                <div
                  id="knowledge-source-menu"
                  className="absolute left-0 z-30 mt-2 w-64 rounded-lg border border-border/80 bg-surface p-2 shadow-lg"
                  role="menu"
                  aria-label={t("contextBar.sourceSelector", "Source selector")}
                >
                  <div className="mb-2 flex items-center justify-between px-1">
                    <span className="text-xs font-semibold text-text-muted">
                      {t("contextBar.sourceCategories", "Source categories")}
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        className="rounded px-1.5 py-0.5 text-[11px] text-text-muted hover:bg-hover hover:text-text transition-colors"
                        onClick={selectAllSources}
                      >
                        {t("contextBar.all", "All")}
                      </button>
                      <button
                        type="button"
                        className="rounded px-1.5 py-0.5 text-[11px] text-text-muted hover:bg-hover hover:text-text transition-colors"
                        onClick={clearSources}
                      >
                        {t("contextBar.none", "None")}
                      </button>
                    </div>
                  </div>
                  <div className="space-y-1">
                    {SOURCE_OPTIONS.map((option) => {
                      const selected = normalizedSources.includes(option.key)
                      return (
                        <button
                          key={option.key}
                          type="button"
                          role="menuitemcheckbox"
                          aria-checked={selected}
                          onClick={() => toggleSource(option.key)}
                          className={cn(
                            "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-xs transition-colors",
                            selected ? "bg-primary/10 text-primaryStrong" : "hover:bg-surface2 text-text"
                          )}
                        >
                          <span className="flex flex-col">
                            <span>{option.label}</span>
                            <span className="text-[10px] text-text-muted font-normal leading-tight">
                              {getRagSourceDescription(option.key)}
                            </span>
                          </span>
                          <span className="h-4 w-4 shrink-0">
                            {selected ? <Check className="h-4 w-4" /> : null}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                  {normalizedSources.length === 0 ? (
                    <p className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-2 py-1.5 text-[11px] text-warn">
                      {t(
                        "contextBar.noSourcesSelected",
                        "No sources selected. Searches may return empty results."
                      )}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="relative" ref={granularMenuRef}>
              <button
                type="button"
                onClick={() => {
                  setGranularMenuOpen((previous) => !previous)
                  setGranularQuery("")
                }}
                className="inline-flex h-7 items-center gap-1 rounded-md border border-border px-2 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                aria-expanded={granularMenuOpen}
                aria-controls={granularMenuOpen ? "knowledge-granular-source-menu" : undefined}
                title="Limit retrieval to specific docs or notes"
              >
                <Filter className="h-3.5 w-3.5 text-text-muted" />
                Specific: {summarizeSpecificSources(normalizedMediaIds, normalizedNoteIds)}
                <ChevronDown className="h-3.5 w-3.5 text-text-muted" />
              </button>
              {granularMenuOpen ? (
                <div
                  id="knowledge-granular-source-menu"
                  role="dialog"
                  aria-label="Specific source selector"
                className="absolute left-0 z-30 mt-2 w-[28rem] max-w-[85vw] rounded-lg border border-border/80 bg-surface p-3 shadow-lg"
              >
                  <div className="mb-2 flex items-start justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                        {t("contextBar.specificSourceScope", "Specific source scope")}
                      </p>
                      <p className="text-xs text-text-muted">
                        {t(
                          "contextBar.specificSourceDescription",
                          "Choose exact docs or notes. Leave this empty to search all items inside selected categories."
                        )}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={clearSpecificSources}
                      className="inline-flex h-7 items-center rounded-md border border-border px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                    >
                      {t("contextBar.useAll", "Use all")}
                    </button>
                  </div>

                  <div className="mb-2 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setGranularTab("media")}
                      className={cn(
                        "inline-flex h-7 items-center rounded-md px-2 text-[11px] font-medium transition-colors",
                        granularTab === "media"
                          ? "bg-primary text-white"
                          : "text-text hover:bg-surface2"
                      )}
                    >
                      {t("contextBar.documentsAndMedia", "Documents & Media")} ({mediaOptions.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => setGranularTab("notes")}
                      className={cn(
                        "inline-flex h-7 items-center rounded-md px-2 text-[11px] font-medium transition-colors",
                        granularTab === "notes"
                          ? "bg-primary text-white"
                          : "text-text hover:bg-surface2"
                      )}
                    >
                      {t("contextBar.notes", "Notes")} ({noteOptions.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setGranularLoaded(false)
                        void loadGranularOptions()
                      }}
                      className="ml-auto inline-flex h-7 items-center rounded-md px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                    >
                      {t("contextBar.reload", "Reload")}
                    </button>
                  </div>

                  <label className="mb-2 flex h-9 items-center gap-2 rounded-md border border-border bg-surface2/70 px-2">
                    <Search className="h-3.5 w-3.5 text-text-muted" />
                    <input
                      type="text"
                      value={granularQuery}
                      onChange={(event) => setGranularQuery(event.target.value)}
                      placeholder={
                        granularTab === "media"
                          ? t("contextBar.filterDocsPlaceholder", "Filter docs by title")
                          : t("contextBar.filterNotesPlaceholder", "Filter notes by title")
                      }
                      className="w-full bg-transparent text-[11px] text-text outline-none placeholder:text-text-muted"
                    />
                  </label>

                  <div className="mb-2 grid gap-2 sm:grid-cols-3">
                    <label className="text-[11px] font-medium text-text-muted">
                      {t("contextBar.sourceStatus", "Source status")}
                      <select
                        aria-label={t("contextBar.sourceStatus", "Source status")}
                        value={granularStatusFilter}
                        onChange={(event) => setGranularStatusFilter(event.target.value)}
                        className="mt-1 h-8 w-full rounded-md border border-border bg-surface2 px-2 text-[11px] text-text outline-none focus:border-primary"
                      >
                        <option value="all">{t("contextBar.allStatuses", "All statuses")}</option>
                        <option value="ready">{READY_STATE_LABEL}</option>
                        <option value="indexing">{t("contextBar.indexing", "Indexing")}</option>
                        <option value="error">{ERROR_STATE_LABEL}</option>
                        <option value="unavailable">{UNAVAILABLE_STATE_LABEL}</option>
                      </select>
                    </label>

                    <label className="text-[11px] font-medium text-text-muted">
                      {t("contextBar.recentImports", "Recent imports")}
                      <select
                        aria-label={t("contextBar.recentImports", "Recent imports")}
                        value={granularRecentFilter}
                        onChange={(event) => setGranularRecentFilter(event.target.value)}
                        className="mt-1 h-8 w-full rounded-md border border-border bg-surface2 px-2 text-[11px] text-text outline-none focus:border-primary"
                      >
                        <option value="all">{t("contextBar.allDates", "All dates")}</option>
                        <option value="recent">{t("contextBar.recentImports", "Recent imports")}</option>
                      </select>
                    </label>

                    <label className="text-[11px] font-medium text-text-muted">
                      {t("contextBar.workspaceScope", "Workspace scope")}
                      <select
                        aria-label={t("contextBar.workspaceScope", "Workspace scope")}
                        value={granularWorkspaceFilter}
                        onChange={(event) => setGranularWorkspaceFilter(event.target.value)}
                        className="mt-1 h-8 w-full rounded-md border border-border bg-surface2 px-2 text-[11px] text-text outline-none focus:border-primary"
                      >
                        <option value="">{t("contextBar.noWorkspaceScope", "No workspace scope")}</option>
                        {workspaceOptions.map((workspace) => (
                          <option key={workspace.id} value={workspace.id}>
                            {workspace.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="mb-2 flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      onClick={selectVisibleSpecificSources}
                      className="inline-flex h-7 items-center rounded-md border border-border px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                    >
                      {t("contextBar.selectVisible", "Select visible")}
                    </button>
                    <button
                      type="button"
                      onClick={clearVisibleSpecificSources}
                      className="inline-flex h-7 items-center rounded-md border border-border px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                    >
                      {t("contextBar.clearVisible", "Clear visible")}
                    </button>
                    <button
                      type="button"
                      onClick={selectRecentSpecificSources}
                      className="inline-flex h-7 items-center rounded-md border border-border px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text transition-colors"
                    >
                      {t("contextBar.selectRecentImports", "Select recent imports")}
                    </button>
                  </div>

                  {granularLoading ? (
                    <div className="flex items-center justify-center gap-2 rounded-md border border-border/80 bg-surface2/60 py-8 text-xs text-text-muted">
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      {t("contextBar.loadingAvailableSources", "Loading available sources...")}
                    </div>
                  ) : null}

                  {!granularLoading && granularError ? (
                    <p className="rounded-md border border-danger/30 bg-danger/10 px-2 py-1.5 text-xs text-danger">
                      {granularError}
                    </p>
                  ) : null}

                  {!granularLoading && !granularError ? (
                    <div className="max-h-64 overflow-y-auto rounded-md border border-border/80 bg-surface2/40">
                      {activeGranularOptions.length === 0 ? (
                        <p className="px-3 py-6 text-center text-xs text-text-muted">
                          {granularTab === "media"
                            ? t("contextBar.noMatchingDocuments", "No matching documents.")
                            : t("contextBar.noMatchingNotes", "No matching notes.")}
                        </p>
                      ) : (
                        <ul className="divide-y divide-border" role="list">
                          {activeGranularOptions.map((option) => {
                            const selected =
                              granularTab === "media"
                                ? selectedMediaSet.has(option.id as number)
                                : selectedNoteSet.has(option.id as string)

                            return (
                              <li key={`${granularTab}-${option.id}`}>
                                <label
                                  className={cn(
                                    "flex cursor-pointer items-start gap-2 px-2.5 py-2 text-xs transition-colors",
                                    selected ? "bg-primary/10" : "hover:bg-surface2"
                                  )}
                                >
                                  <input
                                    type="checkbox"
                                    checked={selected}
                                    onChange={() => {
                                      if (granularTab === "media") {
                                        toggleMediaId(option.id as number)
                                      } else {
                                        toggleNoteId(option.id as string)
                                      }
                                    }}
                                    className="mt-0.5 rounded border-border"
                                  />
                                  <span className="min-w-0 flex-1">
                                    <span className="block truncate text-text">{option.label}</span>
                                    <span className="block text-[11px] text-text-muted">
                                      {[
                                        t("contextBar.itemId", {
                                          id: option.id,
                                          defaultValue: `ID: ${option.id}`,
                                        }),
                                        option.meta,
                                      ].filter(Boolean).join(" • ")}
                                    </span>
                                  </span>
                                </label>
                              </li>
                            )
                          })}
                        </ul>
                      )}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>

            <button
              type="button"
              onClick={onOpenSettings}
              className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border text-text-muted hover:bg-surface2 hover:text-text transition-colors"
              aria-label="Explain source categories"
              title="Documents & Media (files, transcripts, web pages), Notes, Story Characters, Conversations, and Task Boards can all be included in search."
            >
              <CircleHelp className="h-3.5 w-3.5" />
            </button>
          </div>

          <p className="mt-2 text-[11px] text-text-muted">
            Scope: {summarizeSources(normalizedSources)} • Specific filters:{" "}
            {summarizeSpecificSources(normalizedMediaIds, normalizedNoteIds)}
            {normalizedSources.length > 0 && (
              <>
                {" "}
                &mdash;{" "}
                {normalizedMediaIds.length > 0 || normalizedNoteIds.length > 0
                  ? `Searching ${normalizedMediaIds.length + normalizedNoteIds.length} item${
                      normalizedMediaIds.length + normalizedNoteIds.length === 1 ? "" : "s"
                    }`
                  : granularLoaded && hasOnlyCountableSources
                    ? `${countableScopeTotal} items in scope`
                    : `${normalizedSources.length} of ${SOURCE_OPTIONS.length} categories selected`}
              </>
            )}
          </p>
        </section>

        <section className="rounded-lg border border-border/80 bg-surface px-3 py-2.5">
          <div className="mb-2 flex items-center gap-2">
            <Settings className="h-4 w-4 text-primary" />
            <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
              Search Profile
            </h3>
            <Tooltip title="Controls how thorough the search is. Fast returns quickly with fewer sources checked. Deep searches more thoroughly but takes longer.">
              <Info className="h-3.5 w-3.5 text-text-subtle cursor-help" />
            </Tooltip>
          </div>

          <div
            role="group"
            aria-label="Preset selection"
            aria-describedby="knowledge-preset-description"
            className="grid grid-cols-3 gap-2"
          >
            {PRESET_OPTIONS.map((option) => (
              <Tooltip
                key={option.value}
                title={PRESET_COMPARISON_TOOLTIPS[option.value]}
                placement="bottom"
                mouseEnterDelay={0.3}
              >
                <button
                  type="button"
                  onClick={() => onPresetChange(option.value)}
                  className={cn(
                    "rounded-md border px-2 py-2 text-left text-[11px] transition-colors",
                    preset === option.value
                      ? "border-primary bg-primary text-white"
                      : "border-border bg-surface2/70 text-text hover:bg-surface2"
                  )}
                  aria-pressed={preset === option.value}
                >
                  <span className="block text-[11px] font-semibold">{option.label}</span>
                  <span className="block text-[10px] opacity-80">
                    {PRESET_DETAILS[option.value].responseTime}
                  </span>
                </button>
              </Tooltip>
            ))}
          </div>

          {preset === "custom" ? (
            <button
              type="button"
              onClick={onOpenSettings}
              className="mt-2 inline-flex h-7 items-center rounded-md border border-primary/40 bg-primary/10 px-2 text-[11px] font-medium text-primaryStrong hover:bg-primary/15 transition-colors"
              aria-label="Custom preset active. Open settings."
            >
              Custom preset active
            </button>
          ) : null}

          <p id="knowledge-preset-description" className="mt-2 text-[11px] text-text-muted">
            {presetDescription}
            {presetDetail
              ? ` Response time: ${presetDetail.responseTime}. Sources checked: ${presetDetail.sourcesChecked}. Best for: ${presetDetail.bestFor}.`
              : ""}
          </p>
        </section>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Tooltip title="When enabled, includes web search results alongside your documents. Useful when your documents don't cover the topic.">
          <button
            type="button"
            onClick={onToggleWeb}
            className={cn(
              "inline-flex h-7 items-center gap-1 rounded-full border px-2.5 text-[11px] font-medium transition-colors",
              webEnabled
                ? "border-primary/40 bg-primary/10 text-primaryStrong"
                : "border-border text-text-muted hover:bg-surface2 hover:text-text"
            )}
            aria-pressed={webEnabled}
            aria-label={`Web fallback is currently ${webEnabled ? "enabled" : "disabled"}. Click to toggle.`}
          >
            <Globe className={cn("h-3.5 w-3.5", webEnabled ? "fill-current" : "")} />
            Web
          </button>
        </Tooltip>

        <AnswerModelMenu
          generationProvider={generationProvider}
          generationModel={generationModel}
          onGenerationProviderChange={onGenerationProviderChange}
          onGenerationModelChange={onGenerationModelChange}
          menuAlign="right"
        />

        <p className="basis-full text-[11px] text-text-subtle">
          Queries stay on your tldw server unless web fallback is enabled.
          When enabled, fallback uses your configured server default provider.
        </p>

        <Popover
          trigger="click"
          open={profileMenuOpen}
          onOpenChange={(open) => {
            setProfileMenuOpen(open)
            if (!open) {
              setProfileSaveMode(false)
              setProfileNameInput("")
            }
          }}
          placement="topLeft"
          content={
            <div
              id="knowledge-profile-menu"
              role="menu"
              aria-label="Saved search profiles"
              className="w-60"
            >
              <div className="mb-1.5 px-1 text-xs font-semibold text-text-muted">
                Saved Profiles
              </div>
              {savedProfiles.length === 0 && !profileSaveMode ? (
                <p className="px-2 py-3 text-center text-[11px] text-text-muted">
                  No saved profiles yet.
                </p>
              ) : null}
              {savedProfiles.map((profile) => (
                <div
                  key={profile.name}
                  className="group flex items-center gap-1 rounded-md px-2 py-1.5 hover:bg-surface2 transition-colors"
                >
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => loadProfile(profile)}
                    className="flex min-w-0 flex-1 flex-col text-left text-xs text-text"
                  >
                    <span className="truncate font-medium">{profile.name}</span>
                    <span className="text-[10px] text-text-muted">
                      {profile.preset} &middot;{" "}
                      {profile.sources.length === 0
                        ? "no sources"
                        : profile.sources.length >= ALL_RAG_SOURCES.length
                          ? "all sources"
                          : `${profile.sources.length} source${profile.sources.length === 1 ? "" : "s"}`}
                      {profile.enableWebFallback ? " &middot; web" : ""}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteProfile(profile.name)}
                    className="invisible group-hover:visible group-focus-within:visible focus:visible focus-visible:visible shrink-0 rounded p-0.5 text-text-muted hover:bg-danger/10 hover:text-danger transition-colors"
                    aria-label={`Delete profile ${profile.name}`}
                    title="Delete profile"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <div className="mt-1 border-t border-border/60 pt-1.5">
                {profileSaveMode ? (
                  <div className="flex items-center gap-1.5 px-1">
                    <input
                      type="text"
                      value={profileNameInput}
                      onChange={(e) => setProfileNameInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveCurrentProfile()
                        if (e.key === "Escape") {
                          setProfileSaveMode(false)
                          setProfileNameInput("")
                        }
                      }}
                      placeholder="Profile name"
                      maxLength={40}
                      autoFocus
                      className="h-7 flex-1 rounded-md border border-border bg-surface2/70 px-2 text-[11px] text-text outline-none placeholder:text-text-muted focus:border-primary"
                    />
                    <button
                      type="button"
                      onClick={saveCurrentProfile}
                      disabled={!profileNameInput.trim()}
                      className="inline-flex h-7 items-center rounded-md bg-primary px-2 text-[11px] font-medium text-white disabled:opacity-50 hover:bg-primary/90 transition-colors"
                    >
                      Save
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      if (savedProfiles.length >= MAX_SAVED_PROFILES) return
                      setProfileSaveMode(true)
                    }}
                    disabled={savedProfiles.length >= MAX_SAVED_PROFILES}
                    className="w-full rounded-md px-2 py-1.5 text-left text-xs text-primary hover:bg-primary/10 disabled:text-text-muted disabled:hover:bg-transparent transition-colors"
                  >
                    {savedProfiles.length >= MAX_SAVED_PROFILES
                      ? `Limit reached (${MAX_SAVED_PROFILES})`
                      : "Save current settings..."}
                  </button>
                )}
              </div>
            </div>
          }
        >
          <Tooltip title="Save or load search profiles to quickly switch between common configurations">
            <button
              type="button"
              className={cn(
                "inline-flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] font-medium transition-colors",
                profileMenuOpen
                  ? "border-primary/40 bg-primary/10 text-primaryStrong"
                  : "border-border text-text-muted hover:bg-surface2 hover:text-text"
              )}
              aria-expanded={profileMenuOpen}
              aria-controls={profileMenuOpen ? "knowledge-profile-menu" : undefined}
            >
              <Bookmark className="h-3.5 w-3.5" />
              Profiles
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
        </Popover>

        <button
          type="button"
          onClick={onOpenSettings}
          className="inline-flex h-7 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text transition-colors"
        >
          <Settings className="h-3.5 w-3.5" />
          Settings
        </button>

        {contextChangedSinceLastRun ? (
          <Popover
            trigger="click"
            placement="bottomRight"
            title="Scope changed since last search"
            content={
              <div className="max-w-xs space-y-1.5">
                {scopeChangeDetails.length > 0 ? (
                  <ul className="list-disc pl-4 text-xs text-text-muted space-y-1">
                    {scopeChangeDetails.map((detail, index) => (
                      <li key={index}>{detail}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-text-muted">
                    Search settings have changed since your last query.
                  </p>
                )}
                <p className="text-xs text-text-muted pt-1 border-t border-border/60">
                  Run a new search to apply the updated settings.
                </p>
              </div>
            }
          >
            <button
              type="button"
              className="inline-flex items-center rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-[11px] font-medium text-primary hover:bg-primary/20 transition-colors cursor-pointer"
            >
              Scope changed
            </button>
          </Popover>
        ) : null}
      </div>
    </div>
  )
}
