import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Button,
  Form,
  Input,
  Table,
  Tag,
  Tooltip,
  Dropdown,
  Pagination,
  Modal,
  Switch,
  Popover,
  Select,
  Checkbox,
  Segmented
} from "antd"
import type { InputRef, MenuProps } from "antd"
import type { ColumnsType, TableProps } from "antd/es/table"
import type { SortOrder } from "antd/es/table/interface"
import React from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import {
  Plus,
  Trash2,
  Pen,
  Download,
  Upload as UploadIcon,
  Play,
  FileDown,
  FileText,
  Database,
  Copy,
  Eye,
  MessageSquare,
  MoreHorizontal,
  SlidersHorizontal,
  Settings2,
  FileArchive,
  RotateCcw,
  X
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { SkillDrawer } from "./SkillDrawer"
import { SkillPreview } from "./SkillPreview"
import { SkillDetailsDrawer } from "./SkillDetailsDrawer"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"
import type {
  SkillContext,
  SkillImportPreviewResponse,
  SkillListOrder,
  SkillListSort,
  SkillRuntimeMetadata,
  SkillSummary,
  SkillResponse,
  SkillsListResponse,
  SkillTrashItem,
  SkillsTrashListResponse
} from "@/types/skill"
import { getFirstVisibleFocusableElement } from "@/utils/focus-return"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useMobile } from "@/hooks/useMediaQuery"
import {
  parseSkillsQueryState,
  serializeSkillsQueryState
} from "./skills-query-state"
import type { SkillsView } from "./skills-query-state"
import {
  limitSkillSelection,
  MAX_SKILLS_BULK_SELECTION
} from "./skill-form-utils"

const SKILLS_SEARCH_DEBOUNCE_MS = 300
const SKILL_NAME_REGEX = /^[a-z][a-z0-9-]{0,63}$/
const SKILLS_TABLE_PREFERENCES_STORAGE_KEY = "tldw:skills-manager:table-preferences:v1"
const IMPORT_TEXT_DRAFT_STORAGE_PREFIX = "tldw:skills:import-text-draft:v1:"
const SKILL_TABLE_SORT_DIRECTIONS: SortOrder[] = ["ascend", "descend"]
const BULK_EXPORT_CONCURRENCY = 4

interface ImportTextFormValues {
  name?: string
  content: string
  overwrite?: boolean
}

interface ImportTextPreviewRequest {
  name?: string
  content: string
  revision: number
}

interface ImportTextDraft {
  name?: string
  content: string
}

const getImportTextPreviewKey = (
  values: Pick<ImportTextFormValues, "name" | "content">
): string => JSON.stringify([(values.name ?? "").trim(), values.content])

interface SkillsSuccessAction {
  title: string
  description: string
  skillName?: string
  testLabel?: string
  viewLabel?: string
}

interface SeedSkillsResult {
  count?: number
  seeded?: unknown
}

interface FileImportReview {
  file: File
  preview: SkillImportPreviewResponse
  overwrite: boolean
}

interface FileImportPreviewRequest {
  file: File
  revision: number
}

interface FocusReturnTarget {
  element: HTMLElement | null
  selector: string | null
}

interface ActiveFilterTagProps {
  label: string
  removeLabel: string
  onRemove: () => void
}

interface DeleteSkillPayload {
  name: string
  version?: number
  scopeRevision: number
}

interface TrashSkillPayload {
  name: string
  version?: number
}

type SkillContextFilter = "all" | SkillContext
type SkillVisibilityFilter = "visible" | "hidden" | "all"
type SkillToolsFilter = "any" | "with-tools" | "without-tools"
type SkillTableDensity = "comfortable" | "compact"
type SkillSortOption =
  | "default"
  | "name:asc"
  | "name:desc"
  | "context:asc"
  | "context:desc"
  | "created_at:asc"
  | "created_at:desc"
  | "last_modified:asc"
  | "last_modified:desc"
type SkillOptionalColumnKey =
  | "description"
  | "context"
  | "argument_hint"
  | "user_invocable"
  | "model_invocation"
  | "runtime"

interface SkillSortState {
  field?: SkillListSort
  order?: SkillListOrder
}

interface SkillsTablePreferences {
  density: SkillTableDensity
  visibleColumns: SkillOptionalColumnKey[]
}

const DEFAULT_SKILLS_TABLE_PREFERENCES: SkillsTablePreferences = {
  density: "comfortable",
  visibleColumns: ["description", "context"]
}

const SKILL_OPTIONAL_COLUMN_KEYS: SkillOptionalColumnKey[] = [
  "description",
  "context",
  "argument_hint",
  "user_invocable",
  "model_invocation",
  "runtime"
]

const getResponseSkillName = (result: unknown): string | undefined => {
  if (!result || typeof result !== "object") return undefined
  const name = (result as { name?: unknown }).name
  const trimmedName = typeof name === "string" ? name.trim() : ""
  return SKILL_NAME_REGEX.test(trimmedName) ? trimmedName : undefined
}

const getSeededSkillNames = (result: SeedSkillsResult | undefined): string[] => {
  if (!Array.isArray(result?.seeded)) return []
  return result.seeded
    .map((name) => (typeof name === "string" ? name.trim() : ""))
    .filter((name): name is string => SKILL_NAME_REGEX.test(name))
}

const getErrorDescription = (error: unknown): string | undefined => {
  if (!error) return undefined
  return sanitizeServerErrorMessage(error, "") || undefined
}

const isConflictError = (error: unknown): boolean => {
  const candidate = error as {
    status?: unknown
    statusCode?: unknown
    response?: { status?: unknown }
    message?: unknown
  } | null
  if (!candidate) return false
  const message = typeof candidate.message === "string" ? candidate.message : ""
  const hasConflictMessage = /\b(?:http|status(?:\s+code)?)\s*[:=]?\s*409\b/i.test(message)
    || /\b409\b\s*(?:[:=-]\s*)?(?:version\s+)?conflict\b/i.test(message)
  return candidate.status === 409
    || candidate.statusCode === 409
    || candidate.response?.status === 409
    || hasConflictMessage
}

const createSkillsAbortError = (): Error => {
  const error = new Error("Skills request was cancelled")
  error.name = "AbortError"
  return error
}

const isAbortError = (error: unknown): boolean =>
  Boolean(error && typeof error === "object" && (error as { name?: unknown }).name === "AbortError")

const throwIfAborted = (signal: AbortSignal): void => {
  if (signal.aborted) throw createSkillsAbortError()
}

const getKnownSkillVersion = (version: unknown): number | undefined =>
  typeof version === "number" && Number.isSafeInteger(version) && version > 0
    ? version
    : undefined

const buildSkillInvocation = (skillName: string) => `/skill ${skillName}`

const getNextKnownSkillVersion = (version: number | undefined): number | undefined =>
  version === undefined ? undefined : getKnownSkillVersion(version + 1)

const formatDeletedAt = (value: string): string => {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

const isSkillTableSortField = (value: React.Key | undefined): value is SkillListSort =>
  value === "name" || value === "context"

const getSkillTableSortOrder = (
  sortState: SkillSortState,
  field: SkillListSort
): SortOrder => {
  if (sortState.field !== field) return null
  return sortState.order === "asc" ? "ascend" : "descend"
}

const getSkillSortOption = (sortState: SkillSortState): SkillSortOption => {
  if (sortState.field === "name" && sortState.order === "asc") return "name:asc"
  if (sortState.field === "name" && sortState.order === "desc") return "name:desc"
  if (sortState.field === "context" && sortState.order === "asc") return "context:asc"
  if (sortState.field === "context" && sortState.order === "desc") return "context:desc"
  if (sortState.field === "created_at" && sortState.order === "asc") return "created_at:asc"
  if (sortState.field === "created_at" && sortState.order === "desc") return "created_at:desc"
  if (sortState.field === "last_modified" && sortState.order === "asc") {
    return "last_modified:asc"
  }
  if (sortState.field === "last_modified" && sortState.order === "desc") {
    return "last_modified:desc"
  }
  return "default"
}

const getTrashRestoreStatusId = (name: string): string =>
  `skills-trash-restore-status-${name}`

const normalizeSkillsPageSize = (value: number): 10 | 20 | 50 => {
  if (value === 20 || value === 50) return value
  return 10
}

const ActiveFilterTag: React.FC<ActiveFilterTagProps> = ({
  label,
  removeLabel,
  onRemove
}) => (
  <Tag
    closable
    closeIcon={(
      <button
        type="button"
        aria-label={removeLabel}
        className="-mr-1 inline-flex min-h-11 min-w-11 items-center justify-center border-0 bg-transparent p-0 text-current md:min-h-6 md:min-w-6"
      >
        <X aria-hidden="true" size={12} />
      </button>
    )}
    onClose={onRemove}
  >
    {label}
  </Tag>
)

const isSkillTableDensity = (value: unknown): value is SkillTableDensity =>
  value === "comfortable" || value === "compact"

const isSkillOptionalColumnKey = (value: unknown): value is SkillOptionalColumnKey =>
  typeof value === "string"
  && SKILL_OPTIONAL_COLUMN_KEYS.includes(value as SkillOptionalColumnKey)

const normalizeSkillsTablePreferences = (value: unknown): SkillsTablePreferences => {
  if (!value || typeof value !== "object") return DEFAULT_SKILLS_TABLE_PREFERENCES

  const raw = value as {
    density?: unknown
    visibleColumns?: unknown
  }
  const density = isSkillTableDensity(raw.density)
    ? raw.density
    : DEFAULT_SKILLS_TABLE_PREFERENCES.density
  const visibleColumnValues = Array.isArray(raw.visibleColumns)
    ? raw.visibleColumns
    : undefined
  const visibleColumns = visibleColumnValues
    ? SKILL_OPTIONAL_COLUMN_KEYS.filter((key) => visibleColumnValues.includes(key))
    : DEFAULT_SKILLS_TABLE_PREFERENCES.visibleColumns

  return {
    density,
    visibleColumns
  }
}

const loadSkillsTablePreferences = (): SkillsTablePreferences => {
  if (typeof window === "undefined") return DEFAULT_SKILLS_TABLE_PREFERENCES

  try {
    const raw = window.localStorage.getItem(SKILLS_TABLE_PREFERENCES_STORAGE_KEY)
    if (!raw) return DEFAULT_SKILLS_TABLE_PREFERENCES
    return normalizeSkillsTablePreferences(JSON.parse(raw))
  } catch {
    return DEFAULT_SKILLS_TABLE_PREFERENCES
  }
}

const saveSkillsTablePreferences = (preferences: SkillsTablePreferences) => {
  if (typeof window === "undefined") return

  try {
    window.localStorage.setItem(
      SKILLS_TABLE_PREFERENCES_STORAGE_KEY,
      JSON.stringify(preferences)
    )
  } catch {
    // Preference persistence is best-effort; rendering must not depend on storage.
  }
}

const getImportTextDraftStorageKey = (draftScope: string | null): string | null =>
  draftScope ? `${IMPORT_TEXT_DRAFT_STORAGE_PREFIX}${draftScope}` : null

const readImportTextDraft = (draftScope: string | null): ImportTextDraft | null => {
  const storageKey = getImportTextDraftStorageKey(draftScope)
  if (typeof window === "undefined" || !storageKey) return null
  try {
    const parsed = JSON.parse(
      window.sessionStorage.getItem(storageKey) ?? "null"
    ) as Partial<ImportTextDraft> | null
    if (!parsed || typeof parsed.content !== "string") return null
    return {
      name: typeof parsed.name === "string" ? parsed.name : undefined,
      content: parsed.content
    }
  } catch {
    return null
  }
}

const writeImportTextDraft = (
  draftScope: string | null,
  draft: ImportTextDraft | null
): void => {
  const storageKey = getImportTextDraftStorageKey(draftScope)
  if (typeof window === "undefined" || !storageKey) return
  try {
    if (draft) {
      window.sessionStorage.setItem(storageKey, JSON.stringify(draft))
    } else {
      window.sessionStorage.removeItem(storageKey)
    }
  } catch {
    // Session recovery is best effort and must not block imports.
  }
}

const resolveSkillsScope = async (): Promise<string | null> => {
  const config = await tldwClient.getConfig().catch(() => null)
  if (!config) return null
  if (config.authMode !== "multi-user") {
    return buildChatSurfaceScopeKeyFromConfig(config)
  }
  const user = await tldwAuth.getCurrentUser().catch(() => null)
  if (!user?.id) return null
  return buildChatSurfaceScopeKeyFromConfig(config, { userId: user.id })
}

const getSkillRuntimeMetadata = (skill: SkillSummary | null | undefined): SkillRuntimeMetadata | null => {
  if (!skill) return null

  const declaredToolCount = Array.isArray(skill.allowed_tools) ? skill.allowed_tools.length : 0
  const runtime = skill.runtime
  const fallbackContext = skill.context === "fork" ? "fork" : "inline"
  const runtimeExecutionMode = runtime?.execution_mode
  const executionMode = runtimeExecutionMode === "fork" || runtimeExecutionMode === "inline"
    ? runtimeExecutionMode
    : fallbackContext

  return {
    execution_mode: executionMode,
    test_run_may_call_model: runtime?.test_run_may_call_model ?? executionMode === "fork",
    declares_tools: runtime?.declares_tools ?? declaredToolCount > 0,
    declared_tool_count: runtime?.declared_tool_count ?? declaredToolCount,
    model_override: runtime?.model_override ?? skill.model ?? null,
    auto_invocation_enabled: runtime?.auto_invocation_enabled ?? !skill.disable_model_invocation
  }
}

export const SkillsManager: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const queryClient = useQueryClient()
  const notification = useAntdNotification()
  const navigate = useNavigate()
  const [urlSearchParams, setUrlSearchParams] = useSearchParams()
  const { setSelectedQuickPrompt } = useMessageOption()
  const isMobile = useMobile()
  const initialQueryStateRef = React.useRef(parseSkillsQueryState(urlSearchParams))
  const initialQueryState = initialQueryStateRef.current
  const lastUrlParamsRef = React.useRef(urlSearchParams.toString())
  const restoringFromUrlRef = React.useRef(false)
  const urlUpdateModeRef = React.useRef<"replace" | "push">("replace")

  const [activeView, setActiveView] = React.useState<SkillsView>(initialQueryState.view)
  const [page, setPage] = React.useState(initialQueryState.page)
  const [pageSize, setPageSize] = React.useState(initialQueryState.pageSize)
  const [search, setSearch] = React.useState(initialQueryState.search)
  const [debouncedSearch, setDebouncedSearch] = React.useState(initialQueryState.search)
  const [contextFilter, setContextFilter] =
    React.useState<SkillContextFilter>(initialQueryState.context)
  const [visibilityFilter, setVisibilityFilter] =
    React.useState<SkillVisibilityFilter>(initialQueryState.visibility)
  const [toolsFilter, setToolsFilter] = React.useState<SkillToolsFilter>(initialQueryState.tools)
  const [modelFilter, setModelFilter] = React.useState(initialQueryState.model)
  const [debouncedModelFilter, setDebouncedModelFilter] = React.useState(initialQueryState.model)
  const [tableDensity, setTableDensity] =
    React.useState<SkillTableDensity>(() => loadSkillsTablePreferences().density)
  const [visibleOptionalColumns, setVisibleOptionalColumns] = React.useState<
    SkillOptionalColumnKey[]
  >(() => loadSkillsTablePreferences().visibleColumns)
  const [sortState, setSortState] = React.useState<SkillSortState>({
    field: initialQueryState.sort,
    order: initialQueryState.order
  })
  const [skillsScope, setSkillsScope] = React.useState<string | null>(null)
  const [skillsQueryScope, setSkillsQueryScope] = React.useState<string | null>(null)
  const skillsManagerInstanceId = React.useId()
  const [drawerOpen, setDrawerOpen] = React.useState(false)
  const [importTextOpen, setImportTextOpen] = React.useState(false)
  const [importTextDraftRecovered, setImportTextDraftRecovered] = React.useState(false)
  const [importTextPreview, setImportTextPreview] =
    React.useState<SkillImportPreviewResponse | null>(null)
  const [importTextPreviewPendingRevision, setImportTextPreviewPendingRevision] =
    React.useState<number | null>(null)
  const [fileImportReview, setFileImportReview] =
    React.useState<FileImportReview | null>(null)
  const [editingSkill, setEditingSkill] = React.useState<SkillResponse | null>(null)
  const [duplicateSkill, setDuplicateSkill] = React.useState<SkillResponse | null>(null)
  const [previewSkill, setPreviewSkill] = React.useState<string | null>(null)
  const [detailsSkill, setDetailsSkill] = React.useState<string | null>(null)
  const [selectedSkillNames, setSelectedSkillNames] = React.useState<string[]>([])
  const [selectedSkillSnapshots, setSelectedSkillSnapshots] = React.useState<
    Map<string, SkillSummary>
  >(() => new Map())
  const [isBulkExporting, setIsBulkExporting] = React.useState(false)
  const [successAction, setSuccessAction] =
    React.useState<SkillsSuccessAction | null>(null)
  const [importTextForm] = Form.useForm<ImportTextFormValues>()
  const importTextOverwrite = Form.useWatch("overwrite", importTextForm)
  const managerRootRef = React.useRef<HTMLDivElement | null>(null)
  const drawerReturnFocusRef = React.useRef<FocusReturnTarget | null>(null)
  const previewReturnFocusRef = React.useRef<FocusReturnTarget | null>(null)
  const detailsReturnFocusRef = React.useRef<FocusReturnTarget | null>(null)
  const importFileInputRef = React.useRef<HTMLInputElement | null>(null)
  const searchInputRef = React.useRef<InputRef | null>(null)
  const importTextDirtyRef = React.useRef(false)
  const importTextPreviewRevisionRef = React.useRef(0)
  const importTextPreviewKeyRef = React.useRef<string | null>(null)
  const importTextPreviewAbortRef = React.useRef<AbortController | null>(null)
  const fileImportPreviewRevisionRef = React.useRef(0)
  const skillsScopeResolvedRef = React.useRef(false)
  const skillsScopeRevisionRef = React.useRef(0)
  const skillsRequestControllerRef = React.useRef<AbortController | null>(null)
  const skillsConfirmationsRef = React.useRef(
    new Set<ReturnType<typeof Modal.confirm>>()
  )

  const commitUrlHistory = React.useCallback(() => {
    urlUpdateModeRef.current = "push"
  }, [])

  const clearImportTextDraft = React.useCallback(() => {
    importTextDirtyRef.current = false
    setImportTextDraftRecovered(false)
    writeImportTextDraft(skillsScope, null)
  }, [skillsScope])

  const invalidateImportTextPreview = React.useCallback(() => {
    importTextPreviewRevisionRef.current += 1
    importTextPreviewAbortRef.current?.abort()
    importTextPreviewAbortRef.current = null
    importTextPreviewKeyRef.current = null
    setImportTextPreviewPendingRevision(null)
    setImportTextPreview(null)
  }, [])

  const destroySkillsConfirmations = React.useCallback(() => {
    const confirmations = Array.from(skillsConfirmationsRef.current)
    skillsConfirmationsRef.current.clear()
    confirmations.forEach((confirmation) => confirmation?.destroy?.())
  }, [])

  const runInCurrentSkillsScope = React.useCallback(async <T,>(
    request: (signal: AbortSignal) => Promise<T>
  ): Promise<T> => {
    const revision = skillsScopeRevisionRef.current
    const controller = skillsRequestControllerRef.current
    if (!skillsScopeResolvedRef.current || !controller) {
      throw createSkillsAbortError()
    }
    throwIfAborted(controller.signal)
    const result = await request(controller.signal)
    if (
      revision !== skillsScopeRevisionRef.current
      || controller !== skillsRequestControllerRef.current
      || !skillsScopeResolvedRef.current
    ) {
      throw createSkillsAbortError()
    }
    throwIfAborted(controller.signal)
    return result
  }, [])

  const showSkillsConfirmation = React.useCallback((
    config: Parameters<typeof Modal.confirm>[0]
  ) => {
    const revision = skillsScopeRevisionRef.current
    let confirmation: ReturnType<typeof Modal.confirm>
    const isCurrent = () => (
      revision === skillsScopeRevisionRef.current
      && !skillsRequestControllerRef.current?.signal.aborted
    )
    confirmation = Modal.confirm({
      ...config,
      onOk: () => (isCurrent() ? config.onOk?.() : undefined),
      onCancel: () => (isCurrent() ? config.onCancel?.() : undefined),
      afterClose: () => {
        skillsConfirmationsRef.current.delete(confirmation)
        config.afterClose?.()
      }
    })
    if (confirmation) skillsConfirmationsRef.current.add(confirmation)
    return confirmation
  }, [])

  React.useEffect(() => {
    let disposed = false
    let revision = 0

    const refreshSkillsScope = async () => {
      const requestRevision = ++revision
      const hadResolvedScope = skillsScopeResolvedRef.current
      skillsScopeRevisionRef.current += 1
      skillsScopeResolvedRef.current = false
      skillsRequestControllerRef.current?.abort()
      skillsRequestControllerRef.current = new AbortController()
      destroySkillsConfirmations()
      setImportTextOpen(false)
      invalidateImportTextPreview()
      fileImportPreviewRevisionRef.current += 1
      setFileImportReview(null)
      setDrawerOpen(false)
      setEditingSkill(null)
      setDuplicateSkill(null)
      setPreviewSkill(null)
      detailsReturnFocusRef.current = null
      setDetailsSkill(null)
      setSelectedSkillNames([])
      setSelectedSkillSnapshots(new Map())
      setIsBulkExporting(false)
      setSuccessAction(null)
      importTextDirtyRef.current = false
      setImportTextDraftRecovered(false)
      importTextForm.resetFields()
      setSkillsScope(null)
      setSkillsQueryScope(null)
      if (hadResolvedScope) {
        await Promise.all([
          queryClient.cancelQueries({ queryKey: ["skills"] }),
          queryClient.cancelQueries({ queryKey: ["skills-trash"] }),
          queryClient.cancelQueries({ queryKey: ["skill-details"] })
        ])
      }

      const nextScope = await resolveSkillsScope()
      if (disposed || requestRevision !== revision) return

      skillsScopeResolvedRef.current = true
      setSkillsScope(nextScope)
      setSkillsQueryScope(
        nextScope ?? `unresolved:${skillsManagerInstanceId}:${requestRevision}`
      )
    }

    void refreshSkillsScope()
    window.addEventListener("tldw:config-updated", refreshSkillsScope)
    return () => {
      disposed = true
      revision += 1
      skillsScopeRevisionRef.current += 1
      skillsScopeResolvedRef.current = false
      skillsRequestControllerRef.current?.abort()
      skillsRequestControllerRef.current = null
      destroySkillsConfirmations()
      window.removeEventListener("tldw:config-updated", refreshSkillsScope)
    }
  }, [
    destroySkillsConfirmations,
    importTextForm,
    invalidateImportTextPreview,
    queryClient,
    skillsManagerInstanceId
  ])

  React.useEffect(() => () => {
    importTextPreviewRevisionRef.current += 1
    importTextPreviewAbortRef.current?.abort()
    fileImportPreviewRevisionRef.current += 1
  }, [])

  const discardImportTextAndClose = React.useCallback(() => {
    clearImportTextDraft()
    setImportTextOpen(false)
    invalidateImportTextPreview()
    importTextForm.resetFields()
  }, [clearImportTextDraft, importTextForm, invalidateImportTextPreview])

  const requestImportTextClose = React.useCallback(() => {
    if (!importTextDirtyRef.current) {
      setImportTextOpen(false)
      invalidateImportTextPreview()
      return
    }

    showSkillsConfirmation({
      title: t("option:skills.discardImportTitle", {
        defaultValue: "Discard unfinished import?"
      }),
      content: t("option:skills.discardImportDescription", {
        defaultValue: "The imported text and review state will be removed."
      }),
      okText: t("option:skills.discardImportConfirm", { defaultValue: "Discard import" }),
      okButtonProps: { danger: true },
      cancelText: t("option:skills.discardImportCancel", { defaultValue: "Keep editing" }),
      onOk: discardImportTextAndClose
    })
  }, [discardImportTextAndClose, invalidateImportTextPreview, showSkillsConfirmation, t])

  const requestFileImportClose = React.useCallback(() => {
    if (!fileImportReview) return
    showSkillsConfirmation({
      title: t("option:skills.discardFileImportTitle", {
        defaultValue: "Discard reviewed file import?"
      }),
      content: t("option:skills.discardFileImportDescription", {
        defaultValue: "You will need to select and review the file again."
      }),
      okText: t("option:skills.discardImportConfirm", { defaultValue: "Discard import" }),
      okButtonProps: { danger: true },
      cancelText: t("option:skills.discardImportCancel", { defaultValue: "Keep editing" }),
      onOk: () => setFileImportReview(null)
    })
  }, [fileImportReview, showSkillsConfirmation, t])

  const getActiveFocusTarget = React.useCallback((): HTMLElement | null => {
    if (typeof document === "undefined" || typeof HTMLElement === "undefined") {
      return null
    }

    return document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
  }, [])

  const escapeAttributeSelectorValue = React.useCallback((value: string) => {
    if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
      return CSS.escape(value)
    }
    return value.replace(/\\/g, "\\\\").replace(/"/g, "\\\"")
  }, [])

  const getFocusTargetSelector = React.useCallback((element: HTMLElement | null) => {
    if (!element) return null

    const action = element.dataset.skillAction
    if (!action) return null

    const actionSelector =
      `[data-skill-action="${escapeAttributeSelectorValue(action)}"]`
    const skillName = element.dataset.skillName
    return skillName
      ? `${actionSelector}[data-skill-name="${escapeAttributeSelectorValue(skillName)}"]`
      : actionSelector
  }, [escapeAttributeSelectorValue])

  const getFocusReturnTarget = React.useCallback((element: HTMLElement | null): FocusReturnTarget => ({
    element,
    selector: getFocusTargetSelector(element)
  }), [getFocusTargetSelector])

  const getSkillActionElement = React.useCallback((action: string, skillName: string) => {
    const managerRoot = managerRootRef.current
    if (!managerRoot) return null
    const selector = `[data-skill-action="${escapeAttributeSelectorValue(action)}"]`
      + `[data-skill-name="${escapeAttributeSelectorValue(skillName)}"]`
    return managerRoot.querySelector<HTMLElement>(selector)
  }, [escapeAttributeSelectorValue])

  const restoreFocus = React.useCallback((returnTarget: FocusReturnTarget | null): boolean => {
    if (typeof document === "undefined" || typeof HTMLElement === "undefined") return false
    if (!returnTarget) return false

    const activeElement = document.activeElement
    const managerRoot = managerRootRef.current
    if (
      activeElement instanceof HTMLElement
      && activeElement !== document.body
      && activeElement.isConnected
      && managerRoot?.contains(activeElement)
    ) {
      return true
    }

    const target = returnTarget.element?.isConnected
      ? returnTarget.element
      : returnTarget.selector && managerRoot
        ? getFirstVisibleFocusableElement(returnTarget.selector, managerRoot)
        : null
    if (!target) return false

    target.focus({ preventScroll: true })
    return document.activeElement === target
  }, [])

  const openSkillPreview = React.useCallback(
    (skillName: string, triggerElement?: HTMLElement | null) => {
      const returnTarget = triggerElement ?? getActiveFocusTarget()
      previewReturnFocusRef.current = getFocusReturnTarget(returnTarget)
      setPreviewSkill(skillName)
    },
    [getActiveFocusTarget, getFocusReturnTarget]
  )

  const closeSkillPreview = React.useCallback(() => {
    setPreviewSkill(null)
  }, [])

  const handlePreviewAfterClose = React.useCallback(() => {
    const returnTarget = previewReturnFocusRef.current
    previewReturnFocusRef.current = null
    restoreFocus(returnTarget)
  }, [restoreFocus])

  const openSkillDetails = React.useCallback(
    (skillName: string, triggerElement?: HTMLElement | null) => {
      const returnTarget = triggerElement ?? getActiveFocusTarget()
      detailsReturnFocusRef.current = getFocusReturnTarget(returnTarget)
      setDetailsSkill(skillName)
    },
    [getActiveFocusTarget, getFocusReturnTarget]
  )

  const closeSkillDetails = React.useCallback(() => {
    setDetailsSkill(null)
  }, [])

  const restoreDetailsFocus = React.useCallback(() => {
    const returnTarget = detailsReturnFocusRef.current
    detailsReturnFocusRef.current = null
    restoreFocus(returnTarget)
  }, [restoreFocus])

  React.useEffect(() => {
    if (detailsSkill || !detailsReturnFocusRef.current) return

    if (typeof window !== "undefined" && window.requestAnimationFrame) {
      const animationFrame = window.requestAnimationFrame(() => {
        restoreDetailsFocus()
      })
      return () => window.cancelAnimationFrame(animationFrame)
    }

    const timeout = globalThis.setTimeout(() => {
      restoreDetailsFocus()
    }, 0)
    return () => globalThis.clearTimeout(timeout)
  }, [detailsSkill, restoreDetailsFocus])

  const offset = (page - 1) * pageSize
  const searchQuery = debouncedSearch.trim()
  const modelQuery = debouncedModelFilter.trim()
  const contextQuery = contextFilter === "all" ? undefined : contextFilter
  const includeHiddenQuery =
    visibilityFilter === "hidden" || visibilityFilter === "all" ? true : undefined
  const userInvocableQuery = visibilityFilter === "hidden" ? false : undefined
  const hasToolsQuery =
    toolsFilter === "with-tools"
      ? true
      : toolsFilter === "without-tools"
        ? false
        : undefined
  const hasActiveFilters =
    contextFilter !== "all"
    || visibilityFilter !== "visible"
    || toolsFilter !== "any"
    || modelQuery.length > 0
  const activeFilterCount = [
    contextFilter !== "all",
    visibilityFilter !== "visible",
    toolsFilter !== "any",
    modelQuery.length > 0
  ].filter(Boolean).length

  React.useEffect(() => {
    if (search === debouncedSearch) return

    const timer = window.setTimeout(() => {
      setDebouncedSearch(search)
      setPage(1)
    }, SKILLS_SEARCH_DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [debouncedSearch, search])

  React.useEffect(() => {
    if (modelFilter === debouncedModelFilter) return

    const timer = window.setTimeout(() => {
      setDebouncedModelFilter(modelFilter)
      setPage(1)
    }, SKILLS_SEARCH_DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [debouncedModelFilter, modelFilter])

  React.useEffect(() => {
    const incomingParams = urlSearchParams.toString()
    if (incomingParams === lastUrlParamsRef.current) return

    lastUrlParamsRef.current = incomingParams
    restoringFromUrlRef.current = true
    const next = parseSkillsQueryState(urlSearchParams)
    setActiveView(next.view)
    setSearch(next.search)
    setDebouncedSearch(next.search)
    setContextFilter(next.context)
    setVisibilityFilter(next.visibility)
    setToolsFilter(next.tools)
    setModelFilter(next.model)
    setDebouncedModelFilter(next.model)
    setSortState({ field: next.sort, order: next.order })
    setPage(next.page)
    setPageSize(next.pageSize)
  }, [urlSearchParams])

  React.useEffect(() => {
    if (restoringFromUrlRef.current) {
      restoringFromUrlRef.current = false
      urlUpdateModeRef.current = "replace"
      return
    }

    const nextParams = serializeSkillsQueryState({
      view: activeView,
      search: debouncedSearch,
      context: contextFilter,
      visibility: visibilityFilter,
      tools: toolsFilter,
      model: debouncedModelFilter,
      sort: sortState.field,
      order: sortState.order,
      page,
      pageSize: pageSize === 20 || pageSize === 50 ? pageSize : 10
    })
    const nextParamsString = nextParams.toString()
    if (nextParamsString !== urlSearchParams.toString()) {
      const replace = urlUpdateModeRef.current !== "push"
      urlUpdateModeRef.current = "replace"
      lastUrlParamsRef.current = nextParamsString
      setUrlSearchParams(nextParams, { replace })
    } else {
      urlUpdateModeRef.current = "replace"
    }
  }, [
    activeView,
    contextFilter,
    debouncedModelFilter,
    debouncedSearch,
    page,
    pageSize,
    setUrlSearchParams,
    sortState.field,
    sortState.order,
    toolsFilter,
    urlSearchParams,
    visibilityFilter
  ])

  React.useEffect(() => {
    saveSkillsTablePreferences({
      density: tableDensity,
      visibleColumns: visibleOptionalColumns
    })
  }, [tableDensity, visibleOptionalColumns])

  const {
    data,
    isLoading,
    isError,
    error,
    refetch
  } = useQuery<SkillsListResponse>({
    queryKey: [
      "skills",
      skillsQueryScope,
      page,
      pageSize,
      searchQuery,
      contextFilter,
      visibilityFilter,
      toolsFilter,
      modelQuery,
      sortState.field ?? "",
      sortState.order ?? ""
    ],
    queryFn: ({ signal }) =>
      tldwClient.listSkills({
        ...(searchQuery ? { q: searchQuery } : {}),
        ...(contextQuery ? { context: contextQuery } : {}),
        ...(includeHiddenQuery !== undefined ? { includeHidden: includeHiddenQuery } : {}),
        ...(userInvocableQuery !== undefined ? { userInvocable: userInvocableQuery } : {}),
        ...(hasToolsQuery !== undefined ? { hasTools: hasToolsQuery } : {}),
        ...(modelQuery ? { model: modelQuery } : {}),
        ...(sortState.field ? { sort: sortState.field } : {}),
        ...(sortState.order ? { order: sortState.order } : {}),
        limit: pageSize,
        offset,
        abortSignal: signal
      }),
    enabled: skillsQueryScope !== null && activeView === "library"
  })

  const {
    data: trashData,
    isLoading: isTrashLoading,
    isError: isTrashError,
    error: trashError,
    refetch: refetchTrash
  } = useQuery<SkillsTrashListResponse>({
    queryKey: ["skills-trash", skillsQueryScope, page, pageSize],
    queryFn: ({ signal }) =>
      tldwClient.listSkillTrash({
        limit: pageSize,
        offset,
        abortSignal: signal
      }),
    enabled: skillsQueryScope !== null && activeView === "trash"
  })

  const hasLoadedSkills = data != null && !isError
  const currentSkills = React.useMemo(() => data?.skills ?? [], [data?.skills])
  const hasLoadedTrash = trashData != null && !isTrashError
  const currentTrash = React.useMemo(() => trashData?.skills ?? [], [trashData?.skills])
  const selectedSkills = React.useMemo(() => {
    return selectedSkillNames
      .map((name) => selectedSkillSnapshots.get(name))
      .filter((skill): skill is SkillSummary => Boolean(skill))
  }, [selectedSkillNames, selectedSkillSnapshots])
  const previewSkillSummary = React.useMemo(
    () => currentSkills.find((skill) => skill.name === previewSkill) ?? null,
    [currentSkills, previewSkill]
  )
  const previewRuntime = React.useMemo(
    () => getSkillRuntimeMetadata(previewSkillSummary),
    [previewSkillSummary]
  )
  const selectedSkillCount = selectedSkillNames.length
  const totalSkills = data?.total ?? 0
  const totalTrash = trashData?.total ?? 0
  const hasSearch = searchQuery.length > 0
  const isLibraryEmpty =
    hasLoadedSkills && !isLoading && totalSkills === 0 && !hasSearch && !hasActiveFilters
  const skillCountLabel = isError
    ? t("option:skills.countUnavailable", {
        defaultValue: "Count unavailable"
      })
    : t("option:skills.countSummary", {
        defaultValue: `${totalSkills} ${totalSkills === 1 ? "skill" : "skills"}`,
        count: totalSkills
      })
  const trashCountLabel = isTrashError
    ? t("option:skills.countUnavailable", {
        defaultValue: "Count unavailable"
      })
    : t("option:skills.trashCountSummary", {
        defaultValue: `${totalTrash} in Trash`,
        count: totalTrash
      })
  const activeCountLabel = activeView === "trash" ? trashCountLabel : skillCountLabel
  const activeTotal = activeView === "trash" ? totalTrash : totalSkills
  const hasLoadedActiveView = activeView === "trash" ? hasLoadedTrash : hasLoadedSkills
  const listLoadRecoveryState = isError
    ? buildCapabilityState({
        featureName: "Skills",
        capabilityName: "Skills API",
        endpoint: "/api/v1/skills",
        method: "GET",
        error,
        title: t("option:skills.loadListError", {
          defaultValue: "Failed to load skills"
        }),
        message: t("option:skills.loadListErrorRecoveryDescription", {
          defaultValue:
            "The Skills list could not be loaded. Try again or open diagnostics."
        })
      })
    : null
  const trashLoadRecoveryState = isTrashError
    ? buildCapabilityState({
        featureName: "Skills Trash",
        capabilityName: "Skills Trash API",
        endpoint: "/api/v1/skills/trash",
        method: "GET",
        error: trashError,
        title: t("option:skills.trashLoadError", {
          defaultValue: "Failed to load Trash"
        }),
        message: t("option:skills.trashLoadErrorDescription", {
          defaultValue: "Deleted skills could not be loaded. Try again or open diagnostics."
        })
      })
    : null

  React.useEffect(() => {
    if (!hasLoadedActiveView) return

    const lastPage = Math.max(1, Math.ceil(activeTotal / pageSize))
    if (page > lastPage) {
      setPage(lastPage)
    }
  }, [activeTotal, hasLoadedActiveView, page, pageSize])

  React.useEffect(() => {
    if (!hasLoadedSkills || selectedSkillNames.length === 0) return
    const selectedNames = new Set(selectedSkillNames)
    setSelectedSkillSnapshots((current) => {
      const next = new Map(current)
      let changed = false
      for (const skill of currentSkills) {
        if (!selectedNames.has(skill.name)) continue
        if (next.get(skill.name) !== skill) {
          next.set(skill.name, skill)
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [currentSkills, hasLoadedSkills, selectedSkillNames])

  const handleContextFilterChange = (nextFilter: SkillContextFilter) => {
    commitUrlHistory()
    setContextFilter(nextFilter)
    setPage(1)
  }

  const handleVisibilityFilterChange = (nextFilter: SkillVisibilityFilter) => {
    commitUrlHistory()
    setVisibilityFilter(nextFilter)
    setPage(1)
  }

  const handleToolsFilterChange = (nextFilter: SkillToolsFilter) => {
    commitUrlHistory()
    setToolsFilter(nextFilter)
    setPage(1)
  }

  const handleModelFilterChange = (nextValue: string) => {
    setModelFilter(nextValue)
  }

  const handleDensityChange = (nextDensity: SkillTableDensity) => {
    setTableDensity(nextDensity)
  }

  const handleSortOptionChange = (nextSort: SkillSortOption) => {
    commitUrlHistory()
    switch (nextSort) {
      case "name:asc":
        setSortState({ field: "name", order: "asc" })
        break
      case "name:desc":
        setSortState({ field: "name", order: "desc" })
        break
      case "context:asc":
        setSortState({ field: "context", order: "asc" })
        break
      case "context:desc":
        setSortState({ field: "context", order: "desc" })
        break
      case "created_at:asc":
        setSortState({ field: "created_at", order: "asc" })
        break
      case "created_at:desc":
        setSortState({ field: "created_at", order: "desc" })
        break
      case "last_modified:asc":
        setSortState({ field: "last_modified", order: "asc" })
        break
      case "last_modified:desc":
        setSortState({ field: "last_modified", order: "desc" })
        break
      default:
        setSortState({})
    }
    setPage(1)
  }

  const handleTableChange: TableProps<SkillSummary>["onChange"] = (
    _pagination,
    _filters,
    sorter
  ) => {
    commitUrlHistory()
    const activeSorter = Array.isArray(sorter)
      ? sorter.find((entry) => entry.order)
      : sorter

    if (
      activeSorter?.order
      && isSkillTableSortField(activeSorter.columnKey)
    ) {
      setSortState({
        field: activeSorter.columnKey,
        order: activeSorter.order === "ascend" ? "asc" : "desc"
      })
    } else {
      setSortState({})
    }
    setPage(1)
  }

  const restoreMutation = useMutation({
    mutationFn: ({ name, version }: TrashSkillPayload) =>
      runInCurrentSkillsScope((signal) =>
        tldwClient.restoreSkill(name, version, { signal })
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      queryClient.invalidateQueries({ queryKey: ["skills-trash"] })
      notification.success({
        message: t("option:skills.restoreSuccess", { defaultValue: "Skill restored" })
      })
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      if (isConflictError(err)) {
        queryClient.invalidateQueries({ queryKey: ["skills-trash"] })
        notification.error({
          message: t("option:skills.trashConflict", {
            defaultValue: "Trash item changed elsewhere"
          }),
          description: t("option:skills.restoreConflictDescription", {
            defaultValue: "Reload Trash before restoring this version."
          }),
          btn: (
            <Button
              size="small"
              aria-label={t("option:skills.reloadTrash", { defaultValue: "Reload Trash" })}
              onClick={() => void refetchTrash()}
            >
              {t("option:skills.reloadTrash", { defaultValue: "Reload Trash" })}
            </Button>
          )
        })
        return
      }
      notification.error({
        message: t("option:skills.restoreError", { defaultValue: "Failed to restore skill" }),
        description: getErrorDescription(err)
      })
    }
  })

  const purgeMutation = useMutation({
    mutationFn: ({ name, version }: TrashSkillPayload) =>
      runInCurrentSkillsScope((signal) =>
        tldwClient.purgeSkill(name, version, { signal })
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills-trash"] })
      notification.success({
        message: t("option:skills.purgeSuccess", {
          defaultValue: "Skill permanently deleted"
        })
      })
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      if (isConflictError(err)) {
        queryClient.invalidateQueries({ queryKey: ["skills-trash"] })
        notification.error({
          message: t("option:skills.trashConflict", {
            defaultValue: "Trash item changed elsewhere"
          }),
          description: t("option:skills.purgeConflictDescription", {
            defaultValue: "Reload Trash before permanently deleting this version."
          }),
          btn: (
            <Button
              size="small"
              aria-label={t("option:skills.reloadTrash", { defaultValue: "Reload Trash" })}
              onClick={() => void refetchTrash()}
            >
              {t("option:skills.reloadTrash", { defaultValue: "Reload Trash" })}
            </Button>
          )
        })
        return
      }
      notification.error({
        message: t("option:skills.purgeError", {
          defaultValue: "Failed to permanently delete skill"
        }),
        description: getErrorDescription(err)
      })
    }
  })

  const deleteMutation = useMutation({
    mutationFn: ({ name, version }: DeleteSkillPayload) =>
      runInCurrentSkillsScope((signal) =>
        tldwClient.deleteSkill(name, version, { signal })
      ),
    onSuccess: (_result, variables) => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      queryClient.invalidateQueries({ queryKey: ["skills-trash"] })
      setSuccessAction(null)
      notification.success({
        message: t("option:skills.deleteSuccess", { defaultValue: "Skill moved to Trash" }),
        description: t("option:skills.deleteSuccessDescription", {
          defaultValue: "You can restore it now or later from Trash."
        }),
        btn: (
          <Button
            size="small"
            aria-label={t("option:skills.undoDeleteNamedSkill", {
              defaultValue: `Undo delete ${variables.name}`,
              name: variables.name
            })}
            onClick={() => {
              if (
                variables.scopeRevision !== skillsScopeRevisionRef.current
                || !skillsScopeResolvedRef.current
              ) return
              restoreMutation.mutate({
                name: variables.name,
                version: getNextKnownSkillVersion(variables.version)
              })
            }}
          >
            {t("common:undo", { defaultValue: "Undo" })}
          </Button>
        )
      })
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      if (isConflictError(err)) {
        queryClient.invalidateQueries({ queryKey: ["skills"] })
        notification.error({
          message: t("option:skills.deleteConflict", {
            defaultValue: "Skill changed elsewhere"
          }),
          description: t("option:skills.deleteConflictDesc", {
            defaultValue: "Reload skills before deleting this version."
          }),
          btn: (
            <Button
              size="small"
              aria-label={t("option:skills.reloadSkills", { defaultValue: "Reload skills" })}
              onClick={() => void refetch()}
            >
              {t("option:skills.reloadSkills", { defaultValue: "Reload skills" })}
            </Button>
          )
        })
        return
      }
      notification.error({
        message: t("option:skills.deleteError", { defaultValue: "Failed to delete skill" }),
        description: getErrorDescription(err)
      })
    }
  })

  const bulkDeleteMutation = useMutation({
    mutationFn: (skills: SkillSummary[]) =>
      runInCurrentSkillsScope((signal) =>
        tldwClient.bulkDeleteSkills(
          skills.map((skill) => {
            const version = getKnownSkillVersion(skill.version)
            return {
              name: skill.name,
              ...(version ? { version } : {})
            }
          }),
          { signal }
        )
      ),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      queryClient.invalidateQueries({ queryKey: ["skills-trash"] })
      setSelectedSkillNames([])
      setSelectedSkillSnapshots(new Map())
      setSuccessAction(null)
      const count = Number(result?.count ?? 0)
      notification.success({
        message: t("option:skills.bulkDeleteSuccess", {
          defaultValue: "Skills moved to Trash"
        }),
        description: t("option:skills.bulkDeleteSuccessDesc", {
          defaultValue: `${count} skill(s) can be restored from Trash.`,
          count
        })
      })
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      if (isConflictError(err)) {
        queryClient.invalidateQueries({ queryKey: ["skills"] })
        setSelectedSkillNames([])
        setSelectedSkillSnapshots(new Map())
        notification.error({
          message: t("option:skills.bulkDeleteConflict", {
            defaultValue: "Selected skills changed elsewhere"
          }),
          description: t("option:skills.bulkDeleteConflictDesc", {
            defaultValue: "The stale selection was cleared. Select current versions and try again."
          }),
          btn: (
            <Button
              size="small"
              aria-label={t("option:skills.reloadSkills", { defaultValue: "Reload skills" })}
              onClick={() => void refetch()}
            >
              {t("option:skills.reloadSkills", { defaultValue: "Reload skills" })}
            </Button>
          )
        })
        return
      }
      notification.error({
        message: t("option:skills.bulkDeleteError", {
          defaultValue: "Failed to delete selected skills"
        }),
        description: getErrorDescription(err)
      })
    }
  })

  const showImportSuccess = (result: unknown, fallbackName?: string) => {
    queryClient.invalidateQueries({ queryKey: ["skills"] })
    const skillName = getResponseSkillName(result) ?? fallbackName
    if (skillName) {
      setSuccessAction({
        title: t("option:skills.importSuccess", { defaultValue: "Skill imported" }),
        description: t("option:skills.importSuccessActionDesc", {
          defaultValue:
            "Next, test it here or open the skill to confirm the imported details."
        }),
        skillName,
        viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
      })
    }
    notification.success({
      message: t("option:skills.importSuccess", { defaultValue: "Skill imported" })
    })
  }

  const previewImportTextMutation = useMutation({
    mutationFn: async ({ revision: _revision, ...payload }: ImportTextPreviewRequest) =>
      runInCurrentSkillsScope(async (scopeSignal) => {
        const controller = new AbortController()
        const abortForScopeChange = () => controller.abort()
        importTextPreviewAbortRef.current?.abort()
        importTextPreviewAbortRef.current = controller
        scopeSignal.addEventListener("abort", abortForScopeChange, { once: true })
        try {
          throwIfAborted(scopeSignal)
          return await tldwClient.previewSkillImport(payload, { signal: controller.signal })
        } finally {
          scopeSignal.removeEventListener("abort", abortForScopeChange)
          if (importTextPreviewAbortRef.current === controller) {
            importTextPreviewAbortRef.current = null
          }
        }
      }),
    onMutate: (variables) => {
      setImportTextPreviewPendingRevision(variables.revision)
    },
    onSuccess: (result, variables) => {
      const currentValues = importTextForm.getFieldsValue(true) as ImportTextFormValues
      const previewKey = getImportTextPreviewKey(variables)
      if (
        variables.revision !== importTextPreviewRevisionRef.current
        || previewKey !== getImportTextPreviewKey(currentValues)
      ) {
        return
      }
      importTextPreviewKeyRef.current = previewKey
      setImportTextPreview(result)
      importTextForm.setFieldValue("overwrite", false)
    },
    onError: (err: unknown, variables) => {
      if (variables.revision !== importTextPreviewRevisionRef.current) return
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.importPreviewError", {
          defaultValue: "Failed to review skill import"
        }),
        description: getErrorDescription(err)
      })
    },
    onSettled: (_result, _error, variables) => {
      setImportTextPreviewPendingRevision((current) =>
        current === variables.revision ? null : current
      )
    }
  })

  const importTextMutation = useMutation({
    mutationFn: (payload: {
      name?: string
      content: string
      overwrite?: boolean
      expected_version?: number
    }) => runInCurrentSkillsScope((signal) =>
      tldwClient.importSkill(payload, { signal })
    ),
    onSuccess: (result, variables) => {
      clearImportTextDraft()
      setImportTextOpen(false)
      invalidateImportTextPreview()
      importTextForm.resetFields()
      showImportSuccess(result, variables.name)
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.importError", { defaultValue: "Failed to import skill" }),
        description: getErrorDescription(err)
      })
    }
  })

  const previewImportFileMutation = useMutation({
    mutationFn: ({ file }: FileImportPreviewRequest) =>
      runInCurrentSkillsScope((signal) =>
        tldwClient.previewSkillImportFile(file, { signal })
      ),
    onSuccess: (preview, { file, revision }) => {
      if (revision !== fileImportPreviewRevisionRef.current) return
      setSuccessAction(null)
      setFileImportReview({ file, preview, overwrite: false })
    },
    onError: (err: unknown, { revision }) => {
      if (revision !== fileImportPreviewRevisionRef.current) return
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.importPreviewError", {
          defaultValue: "Failed to review skill import"
        }),
        description: getErrorDescription(err)
      })
    }
  })

  const importFileMutation = useMutation({
    mutationFn: ({
      file,
      overwrite,
      expectedVersion
    }: {
      file: File
      overwrite: boolean
      expectedVersion?: number
    }) => runInCurrentSkillsScope((signal) =>
      tldwClient.importSkillFile(file, {
        overwrite,
        ...(expectedVersion !== undefined ? { expectedVersion } : {}),
        signal
      })
    ),
    onSuccess: (result) => {
      setFileImportReview(null)
      showImportSuccess(result)
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.importError", { defaultValue: "Failed to import skill" }),
        description: getErrorDescription(err)
      })
    }
  })

  const seedBuiltinsMutation = useMutation({
    mutationFn: (overwrite: boolean = false) =>
      runInCurrentSkillsScope((signal) =>
        tldwClient.seedSkills({ overwrite }, { signal })
      ),
    onSuccess: (result: SeedSkillsResult | undefined) => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      const count = Number(result?.count ?? 0)
      const seededSkillNames = getSeededSkillNames(result)
      const suggestedSkillName =
        seededSkillNames.includes("summarize") ? "summarize" : seededSkillNames[0]
      if (suggestedSkillName) {
        setSuccessAction({
          title: t("option:skills.seedSuccess", {
            defaultValue: "Built-in skills seeded"
          }),
          description: t("option:skills.seedSuccessActionDesc", {
            defaultValue:
              "Try a built-in skill now, or copy the chat invocation for later."
          }),
          skillName: suggestedSkillName,
          testLabel: t("option:skills.testSpecificSkill", {
            defaultValue: `Test ${suggestedSkillName}`,
            skillName: suggestedSkillName
          }),
          viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
        })
      }
      notification.success({
        message: t("option:skills.seedSuccess", { defaultValue: "Built-in skills seeded" }),
        description: t("option:skills.seedSuccessDesc", {
          defaultValue: `${count} built-in skill(s) seeded.`,
          count
        })
      })
    },
    onError: (err: unknown) => {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.seedError", { defaultValue: "Failed to seed built-in skills" }),
        description: getErrorDescription(err)
      })
    }
  })

  const confirmSeedOverwrite = () => {
    showSkillsConfirmation({
      title: t("option:skills.seedOverwriteConfirmTitle", {
        defaultValue: "Overwrite existing built-in skills?"
      }),
      content: t("option:skills.seedOverwriteConfirmContent", {
        defaultValue:
          "This replaces existing skill copies that match built-in skill names. Custom skills with other names are not changed."
      }),
      okText: t("option:skills.seedOverwriteConfirmOk", {
        defaultValue: "Overwrite built-ins"
      }),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => seedBuiltinsMutation.mutateAsync(true).catch((err: unknown) => {
        if (isAbortError(err)) return
        throw err
      })
    })
  }

  const handleNew = React.useCallback((triggerElement?: HTMLElement | null) => {
    const returnTarget = triggerElement ?? getActiveFocusTarget()
    drawerReturnFocusRef.current = getFocusReturnTarget(returnTarget)
    setSuccessAction(null)
    setEditingSkill(null)
    setDuplicateSkill(null)
    setDrawerOpen(true)
  }, [getActiveFocusTarget, getFocusReturnTarget])

  const handleEdit = async (name: string, triggerElement?: HTMLElement | null) => {
    const returnTarget = triggerElement ?? getActiveFocusTarget()
    drawerReturnFocusRef.current = getFocusReturnTarget(returnTarget)
    try {
      const skill = await runInCurrentSkillsScope((signal) =>
        tldwClient.getSkill(name, { signal })
      )
      setDuplicateSkill(null)
      setEditingSkill(skill)
      detailsReturnFocusRef.current = null
      setDetailsSkill(null)
      setDrawerOpen(true)
    } catch (err: unknown) {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.loadError", { defaultValue: "Failed to load skill" }),
        description: getErrorDescription(err)
      })
    }
  }

  const handleDuplicate = async (name: string, triggerElement?: HTMLElement | null) => {
    const returnTarget = triggerElement ?? getActiveFocusTarget()
    drawerReturnFocusRef.current = getFocusReturnTarget(returnTarget)
    try {
      const source = await runInCurrentSkillsScope((signal) =>
        tldwClient.getSkill(name, { signal })
      )
      setEditingSkill(null)
      setDuplicateSkill(source)
      detailsReturnFocusRef.current = null
      setDetailsSkill(null)
      setDrawerOpen(true)
    } catch (err: unknown) {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.loadDuplicateError", {
          defaultValue: "Failed to prepare skill duplicate"
        }),
        description: getErrorDescription(err)
      })
    }
  }

  const handleUseInChat = React.useCallback((name: string) => {
    detailsReturnFocusRef.current = null
    setSelectedQuickPrompt(buildSkillInvocation(name))
    navigate("/chat")
  }, [navigate, setSelectedQuickPrompt])

  const handleViewChange = (nextView: SkillsView) => {
    if (nextView === activeView) return
    commitUrlHistory()
    setActiveView(nextView)
    setPage(1)
    setSelectedSkillNames([])
    setSelectedSkillSnapshots(new Map())
    setSuccessAction(null)
  }

  const clearFilters = () => {
    commitUrlHistory()
    setContextFilter("all")
    setVisibilityFilter("visible")
    setToolsFilter("any")
    setModelFilter("")
    setDebouncedModelFilter("")
    setPage(1)
  }

  const applySkillSelection = React.useCallback((requestedNames: string[]) => {
    const { names: nextNames, limited } = limitSkillSelection(requestedNames)
    if (limited) {
      notification.warning({
        message: t("option:skills.selectionLimitTitle", {
          defaultValue: "Selection limited to 100 skills"
        }),
        description: t("option:skills.selectionLimitDescription", {
          defaultValue: "Bulk actions accept at most {{limit}} skills at a time.",
          limit: MAX_SKILLS_BULK_SELECTION
        })
      })
    }

    const nextNameSet = new Set(nextNames)
    setSelectedSkillNames(nextNames)
    setSelectedSkillSnapshots((current) => {
      const next = new Map(
        Array.from(current.entries()).filter(([name]) => nextNameSet.has(name))
      )
      for (const skill of currentSkills) {
        if (nextNameSet.has(skill.name)) next.set(skill.name, skill)
      }
      return next
    })
  }, [currentSkills, notification, t])

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (activeView !== "library") return
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return
      const target = event.target
      const isEditable = target instanceof HTMLElement
        && (target.isContentEditable
          || target.tagName === "INPUT"
          || target.tagName === "TEXTAREA"
          || target.tagName === "SELECT")
      if (isEditable) return

      if (event.key === "/") {
        event.preventDefault()
        searchInputRef.current?.focus()
      } else if (event.key.toLowerCase() === "n") {
        event.preventDefault()
        handleNew(null)
      }
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [activeView, handleNew])

  const handlePurge = (skill: SkillTrashItem) => {
    showSkillsConfirmation({
      title: t("option:skills.purgeConfirmTitle", {
        defaultValue: "Permanently delete {{name}}?",
        name: skill.name
      }),
      content: t("option:skills.purgeConfirmContent", {
        defaultValue: `Permanently delete "${skill.name}" and its archived files? This cannot be undone.`,
        name: skill.name
      }),
      okText: t("option:skills.purgeConfirmOk", {
        defaultValue: "Delete permanently"
      }),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => purgeMutation.mutateAsync({
        name: skill.name,
        version: getKnownSkillVersion(skill.version)
      }).catch((err: unknown) => {
        if (isAbortError(err) || isConflictError(err)) return
        throw err
      })
    })
  }

  const handleDelete = (
    skill: Pick<SkillSummary, "name"> & Partial<Pick<SkillSummary, "version">>
  ) => {
    showSkillsConfirmation({
      title: t("option:skills.deleteConfirmTitle", {
        defaultValue: "Delete {{name}}?",
        name: skill.name
      }),
      content: t("option:skills.deleteConfirmContent", {
        defaultValue: `Move "${skill.name}" to Trash? You can restore it later.`,
        name: skill.name
      }),
      okText: t("option:skills.moveToTrash", { defaultValue: "Move to Trash" }),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => deleteMutation.mutateAsync({
        name: skill.name,
        version: getKnownSkillVersion(skill.version),
        scopeRevision: skillsScopeRevisionRef.current
      }).catch((err: unknown) => {
        if (isAbortError(err) || isConflictError(err)) return
        throw err
      })
    })
  }

  const handleBulkDelete = () => {
    if (!selectedSkills.length) return

    const count = selectedSkills.length
    showSkillsConfirmation({
      title: t("option:skills.bulkDeleteConfirmTitle", {
        defaultValue: "Delete selected skills?"
      }),
      content: t("option:skills.bulkDeleteConfirmContent", {
        defaultValue: `Move ${count} selected skill(s) to Trash? You can restore them later.`,
        count
      }),
      okText: t("option:skills.bulkDeleteConfirmOk", {
        defaultValue: "Move selected to Trash"
      }),
      okButtonProps: {
        danger: true,
        loading: bulkDeleteMutation.isPending
      },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => bulkDeleteMutation.mutateAsync(selectedSkills).catch((err: unknown) => {
        if (isAbortError(err) || isConflictError(err)) return
        throw err
      })
    })
  }

  const handleExport = async (name: string) => {
    try {
      const filename = await runInCurrentSkillsScope(async (signal) => {
        const result = await tldwClient.exportSkill(name, { signal })
        throwIfAborted(signal)
        const url = URL.createObjectURL(result.blob)
        const a = document.createElement("a")
        a.href = url
        a.download = result.filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        return result.filename
      })
      notification.success({
        message: t("option:skills.exportStarted", { defaultValue: "Export started" }),
        description: t("option:skills.exportStartedDescription", {
          defaultValue: "{{filename}} download has started.",
          filename
        })
      })
    } catch (err: unknown) {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.exportError", { defaultValue: "Failed to export skill" }),
        description: getErrorDescription(err)
      })
    }
  }

  const handleBulkExport = async () => {
    if (selectedSkillNames.length === 0 || isBulkExporting) return
    setIsBulkExporting(true)
    try {
      const { exportedCount, failed, filename } = await runInCurrentSkillsScope(
        async (signal) => {
          const { default: JSZip } = await import("jszip")
          throwIfAborted(signal)
          const archive = new JSZip()
          const failedNames: string[] = []
          const names = [...selectedSkillNames]

          for (let index = 0; index < names.length; index += BULK_EXPORT_CONCURRENCY) {
            throwIfAborted(signal)
            const batch = names.slice(index, index + BULK_EXPORT_CONCURRENCY)
            await Promise.all(batch.map(async (skillName) => {
              try {
                const result = await tldwClient.exportSkill(skillName, { signal })
                throwIfAborted(signal)
                archive.file(result.filename || `${skillName}.zip`, result.blob)
              } catch (error: unknown) {
                if (isAbortError(error)) throw error
                failedNames.push(skillName)
              }
            }))
          }

          const completedCount = names.length - failedNames.length
          if (completedCount === 0) {
            throw new Error("No selected skills could be exported.")
          }

          const blob = await archive.generateAsync({ type: "blob" })
          throwIfAborted(signal)
          const archiveFilename = `skills-export-${new Date().toISOString().slice(0, 10)}.zip`
          const url = URL.createObjectURL(blob)
          const link = document.createElement("a")
          link.href = url
          link.download = archiveFilename
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)
          URL.revokeObjectURL(url)
          return {
            exportedCount: completedCount,
            failed: failedNames,
            filename: archiveFilename
          }
        }
      )

      notification.success({
        message: t("option:skills.bulkExportSuccess", { defaultValue: "Skills exported" }),
        description: t("option:skills.bulkExportSuccessDescription", {
          defaultValue: `${exportedCount} skill(s) were added to ${filename}.`,
          count: exportedCount,
          filename
        })
      })
      if (failed.length > 0) {
        notification.error({
          message: t("option:skills.bulkExportPartial", {
            defaultValue: "Some skills could not be exported"
          }),
          description: failed.join(", ")
        })
      }
    } catch (err: unknown) {
      if (isAbortError(err)) return
      notification.error({
        message: t("option:skills.bulkExportError", {
          defaultValue: "Failed to export selected skills"
        }),
        description: getErrorDescription(err)
      })
    } finally {
      setIsBulkExporting(false)
    }
  }

  const handleImportFile = async (file: File) => {
    const revision = fileImportPreviewRevisionRef.current + 1
    fileImportPreviewRevisionRef.current = revision
    previewImportFileMutation.mutate({ file, revision })
  }

  const handleImportFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (file) void handleImportFile(file)
  }

  const openImportTextModal = () => {
    setSuccessAction(null)
    invalidateImportTextPreview()
    importTextForm.resetFields()
    const draft = readImportTextDraft(skillsScope)
    const hasDraft = Boolean(draft && (draft.name?.trim() || draft.content.trim()))
    importTextDirtyRef.current = hasDraft
    setImportTextDraftRecovered(hasDraft)
    importTextForm.setFieldsValue({
      name: draft?.name,
      overwrite: false,
      content: draft?.content ?? ""
    })
    setImportTextOpen(true)
  }

  const handleImportTextSubmit = async () => {
    try {
      const values = await importTextForm.validateFields()
      const payload: {
        name?: string
        content: string
        overwrite?: boolean
        expected_version?: number
      } = {
        content: values.content
      }
      const trimmedName = values.name?.trim()
      if (trimmedName) {
        payload.name = trimmedName
      }
      const previewKey = getImportTextPreviewKey(payload)
      if (
        !importTextPreview?.valid
        || importTextPreviewKeyRef.current !== previewKey
      ) {
        if (importTextPreview || importTextPreviewKeyRef.current) {
          invalidateImportTextPreview()
        }
        const revision = importTextPreviewRevisionRef.current + 1
        importTextPreviewRevisionRef.current = revision
        await previewImportTextMutation.mutateAsync({
          name: payload.name,
          content: payload.content,
          revision
        })
        return
      }
      if (importTextPreview.conflict && !values.overwrite) {
        return
      }
      payload.overwrite = importTextPreview.conflict
        ? Boolean(values.overwrite)
        : false
      if (payload.overwrite) {
        const expectedVersion = getKnownSkillVersion(importTextPreview.existing_version)
        if (expectedVersion === undefined) return
        payload.expected_version = expectedVersion
      }
      await importTextMutation.mutateAsync(payload)
    } catch {
      // validation errors handled by antd
    }
  }

  const restoreDrawerFocus = React.useCallback((clearWhenDone = true) => {
    const returnTarget = drawerReturnFocusRef.current
    const restored = restoreFocus(returnTarget)
    if (clearWhenDone || restored) {
      drawerReturnFocusRef.current = null
    }
  }, [restoreFocus])

  const handleDrawerAfterClose = React.useCallback(() => {
    restoreDrawerFocus()
  }, [restoreDrawerFocus])

  React.useEffect(() => {
    if (drawerOpen || !drawerReturnFocusRef.current) return

    if (typeof window !== "undefined" && window.requestAnimationFrame) {
      const animationFrame = window.requestAnimationFrame(() => {
        restoreDrawerFocus(false)
      })
      return () => window.cancelAnimationFrame(animationFrame)
    }

    const timeout = globalThis.setTimeout(() => {
      restoreDrawerFocus(false)
    }, 0)
    return () => globalThis.clearTimeout(timeout)
  }, [drawerOpen, restoreDrawerFocus])

  const handleDrawerClose = () => {
    setDrawerOpen(false)
    setEditingSkill(null)
    setDuplicateSkill(null)
  }

  const handleDrawerSaved = (savedSkillName?: string) => {
    const wasCreating = !editingSkill
    queryClient.invalidateQueries({ queryKey: ["skills"] })
    handleDrawerClose()
    if (wasCreating && savedSkillName) {
      setSuccessAction({
        title: t("option:skills.createSuccess", { defaultValue: "Skill created" }),
        description: t("option:skills.createSuccessActionDesc", {
          defaultValue:
            "Next, test it here or copy the chat invocation for a conversation."
        }),
        skillName: savedSkillName,
        viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
      })
    }
  }

  const handleCopyInvocation = async (skillName: string) => {
    try {
      const writeText = navigator.clipboard?.writeText
      if (!writeText) {
        throw new Error(
          t("option:skills.copyInvocationUnavailable", {
            defaultValue: "Clipboard is not available in this browser context."
          })
        )
      }
      await writeText.call(navigator.clipboard, buildSkillInvocation(skillName))
      notification.success({
        message: t("option:skills.copyInvocationSuccess", {
          defaultValue: "Skill invocation copied"
        })
      })
    } catch (err: unknown) {
      notification.error({
        message: t("option:skills.copyInvocationError", {
          defaultValue: "Failed to copy skill invocation"
        }),
        description: getErrorDescription(err)
      })
    }
  }

  const isOptionalColumnVisible = (columnKey: SkillOptionalColumnKey) =>
    visibleOptionalColumns.includes(columnKey)

  const optionalColumnLabels: Record<SkillOptionalColumnKey, string> = {
    description: t("option:skills.colDescription", { defaultValue: "Description" }),
    context: t("option:skills.colContext", { defaultValue: "Mode" }),
    argument_hint: t("option:skills.colArgumentHint", { defaultValue: "Argument hint" }),
    user_invocable: t("option:skills.colVisibility", { defaultValue: "Visibility" }),
    model_invocation: t("option:skills.colModelUse", { defaultValue: "Model use" }),
    runtime: t("option:skills.colRuntime", { defaultValue: "Runtime" })
  }
  const contextFilterValueLabel = contextFilter === "fork"
    ? t("option:skills.filterFork", { defaultValue: "Fork" })
    : t("option:skills.filterInline", { defaultValue: "Inline" })
  const visibilityFilterValueLabel = visibilityFilter === "hidden"
    ? t("option:skills.filterHidden", { defaultValue: "Hidden" })
    : t("option:skills.filterAllVisibility", { defaultValue: "All visibility" })
  const toolsFilterValueLabel = toolsFilter === "with-tools"
    ? t("option:skills.filterHasTools", { defaultValue: "Has tools" })
    : t("option:skills.filterNoTools", { defaultValue: "No tools" })
  const activeContextFilterLabel = t("option:skills.activeModeFilter", {
    defaultValue: `Mode: ${contextFilterValueLabel}`,
    value: contextFilterValueLabel
  })
  const activeVisibilityFilterLabel = t("option:skills.activeVisibilityFilter", {
    defaultValue: `Visibility: ${visibilityFilterValueLabel}`,
    value: visibilityFilterValueLabel
  })
  const activeToolsFilterLabel = t("option:skills.activeToolsFilter", {
    defaultValue: `Tools: ${toolsFilterValueLabel}`,
    value: toolsFilterValueLabel
  })
  const activeModelFilterLabel = t("option:skills.activeModelFilter", {
    defaultValue: `Model: ${modelQuery}`,
    value: modelQuery
  })
  const getRemoveFilterLabel = (label: string) => t("option:skills.removeActiveFilter", {
    defaultValue: `Remove ${label} filter`,
    label
  })

  const getSkillMoreMenuItems = (record: SkillSummary): MenuProps["items"] => [
    {
      key: "edit",
      icon: <Pen size={14} />,
      label: t("common:edit", { defaultValue: "Edit" }),
      onClick: () => void handleEdit(
        record.name,
        getSkillActionElement("more", record.name)
      )
    },
    {
      key: "duplicate",
      icon: <Plus size={14} />,
      label: t("option:skills.duplicate", { defaultValue: "Duplicate" }),
      onClick: () => void handleDuplicate(
        record.name,
        getSkillActionElement("more", record.name)
      )
    },
    {
      key: "export",
      icon: <Download size={14} />,
      label: t("option:skills.export", { defaultValue: "Export" }),
      onClick: () => void handleExport(record.name)
    },
    { type: "divider" },
    {
      key: "delete",
      danger: true,
      icon: <Trash2 size={14} />,
      label: t("common:delete", { defaultValue: "Delete" }),
      onClick: () => handleDelete(record)
    }
  ]

  const renderSkillActions = (record: SkillSummary, mobile = false) => (
    <div className="flex items-center gap-1">
      <Tooltip title={t("option:skills.viewSkill", { defaultValue: "View skill" })}>
        <Button
          aria-label={t("option:skills.viewNamedSkill", {
            defaultValue: `View ${record.name}`,
            name: record.name
          })}
          type="text"
          size={mobile ? "middle" : "small"}
          className={mobile ? "min-h-11 min-w-11" : undefined}
          icon={<Eye size={14} />}
          data-skill-action="view"
          data-skill-name={record.name}
          onClick={(event) => openSkillDetails(record.name, event.currentTarget)}
        />
      </Tooltip>
      <Tooltip title={t("option:skills.useInChat", { defaultValue: "Use in chat" })}>
        <Button
          aria-label={t("option:skills.useNamedSkillInChat", {
            defaultValue: `Use ${record.name} in chat`,
            name: record.name
          })}
          type="text"
          size={mobile ? "middle" : "small"}
          className={mobile ? "min-h-11 min-w-11" : undefined}
          icon={<MessageSquare size={14} />}
          onClick={() => handleUseInChat(record.name)}
        />
      </Tooltip>
      <Tooltip title={t("option:skills.copyInvocationAction", { defaultValue: "Copy invocation" })}>
        <Button
          aria-label={t("option:skills.copyNamedInvocation", {
            defaultValue: `Copy invocation for ${record.name}`,
            name: record.name
          })}
          type="text"
          size={mobile ? "middle" : "small"}
          className={mobile ? "min-h-11 min-w-11" : undefined}
          icon={<Copy size={14} />}
          onClick={() => void handleCopyInvocation(record.name)}
        />
      </Tooltip>
      <Tooltip title={t("option:skills.testRun", { defaultValue: "Test run" })}>
        <Button
          aria-label={t("option:skills.testRunSkill", {
            defaultValue: `Test run ${record.name}`,
            name: record.name
          })}
          type="text"
          size={mobile ? "middle" : "small"}
          className={mobile ? "min-h-11 min-w-11" : undefined}
          icon={<Play size={14} />}
          data-skill-action="test-run"
          data-skill-name={record.name}
          onClick={(event) => openSkillPreview(record.name, event.currentTarget)}
        />
      </Tooltip>
      <Dropdown menu={{ items: getSkillMoreMenuItems(record) }} trigger={["click"]}>
        <Button
          aria-label={t("option:skills.moreActionsForSkill", {
            defaultValue: `More actions for ${record.name}`,
            name: record.name
          })}
          type="text"
          size={mobile ? "middle" : "small"}
          className={mobile ? "min-h-11 min-w-11" : undefined}
          icon={<MoreHorizontal size={14} />}
          data-skill-action="more"
          data-skill-name={record.name}
        />
      </Dropdown>
    </div>
  )

  const columns: ColumnsType<SkillSummary> = [
    {
      title: t("option:skills.colName", { defaultValue: "Name" }),
      dataIndex: "name",
      key: "name",
      sorter: true,
      sortDirections: SKILL_TABLE_SORT_DIRECTIONS,
      sortOrder: getSkillTableSortOrder(sortState, "name"),
      render: (name: string) => (
        <span className="font-mono text-sm">{name}</span>
      )
    },
    ...(isOptionalColumnVisible("description")
      ? [{
          title: optionalColumnLabels.description,
          dataIndex: "description",
          key: "description",
          ellipsis: true,
          render: (desc: string | null) => desc || "-"
        }]
      : []),
    ...(isOptionalColumnVisible("context")
      ? [{
          title: optionalColumnLabels.context,
          dataIndex: "context",
          key: "context",
          width: 100,
          sorter: true,
          sortDirections: SKILL_TABLE_SORT_DIRECTIONS,
          sortOrder: getSkillTableSortOrder(sortState, "context"),
          render: (ctx: string) => (
            <Tag color={ctx === "fork" ? "blue" : "green"}>
              {ctx}
            </Tag>
          )
        }]
      : []),
    ...(isOptionalColumnVisible("argument_hint")
      ? [{
          title: optionalColumnLabels.argument_hint,
          dataIndex: "argument_hint",
          key: "argument_hint",
          width: 160,
          ellipsis: true,
          render: (hint: string | null) => hint ? (
            <span className="font-mono text-sm">{hint}</span>
          ) : "-"
        }]
      : []),
    ...(isOptionalColumnVisible("user_invocable")
      ? [{
          title: optionalColumnLabels.user_invocable,
          dataIndex: "user_invocable",
          key: "user_invocable",
          width: 110,
          render: (userInvocable: boolean) => (
            <Tag color={userInvocable ? "green" : "default"}>
              {userInvocable
                ? t("option:skills.visibleState", { defaultValue: "Visible" })
                : t("option:skills.hiddenState", { defaultValue: "Hidden" })}
            </Tag>
          )
        }]
      : []),
    ...(isOptionalColumnVisible("model_invocation")
      ? [{
          title: optionalColumnLabels.model_invocation,
          dataIndex: "disable_model_invocation",
          key: "model_invocation",
          width: 130,
          render: (disableModelInvocation: boolean) => (
            <Tag color={disableModelInvocation ? "orange" : "green"}>
              {disableModelInvocation
                ? t("option:skills.modelDisabled", { defaultValue: "Model disabled" })
                : t("option:skills.modelAllowed", { defaultValue: "Model allowed" })}
            </Tag>
          )
        }]
      : []),
    ...(isOptionalColumnVisible("runtime")
      ? [{
          title: optionalColumnLabels.runtime,
          key: "runtime",
          width: 260,
          render: (_: unknown, record: SkillSummary) => {
            const runtime = getSkillRuntimeMetadata(record)
            if (!runtime) return "-"

            const toolLabel = t("option:skills.runtimeDeclaredTools", {
              defaultValue: `${runtime.declared_tool_count} tools declared`,
              count: runtime.declared_tool_count
            })

            return (
              <div className="flex flex-wrap gap-1">
                <Tag color={runtime.execution_mode === "fork" ? "blue" : "green"}>
                  {runtime.execution_mode === "fork"
                    ? t("option:skills.runtimeFork", { defaultValue: "Fork" })
                    : t("option:skills.runtimeInline", { defaultValue: "Inline" })}
                </Tag>
                <Tag color={runtime.test_run_may_call_model ? "orange" : "default"}>
                  {runtime.test_run_may_call_model
                    ? t("option:skills.runtimeMayCallModel", { defaultValue: "Test may call model" })
                    : t("option:skills.runtimePromptOnly", { defaultValue: "Prompt only by default" })}
                </Tag>
                {runtime.declares_tools && (
                  <Tag color="geekblue">{toolLabel}</Tag>
                )}
                {runtime.model_override && (
                  <Tag>
                    {t("option:skills.runtimeModelOverride", {
                      defaultValue: "Model override"
                    })}
                  </Tag>
                )}
                {!runtime.auto_invocation_enabled && (
                  <Tag color="warning">
                    {t("option:skills.runtimeAutoOff", {
                      defaultValue: "Auto invocation off"
                    })}
                  </Tag>
                )}
              </div>
            )
          }
        }]
      : []),
    {
      title: t("option:skills.colActions", { defaultValue: "Actions" }),
      key: "actions",
      width: 220,
      render: (_: unknown, record: SkillSummary) => renderSkillActions(record)
    }
  ]

  const renderTrashActions = (record: SkillTrashItem, mobile = false) => (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        aria-label={t("option:skills.restoreNamedSkill", {
          defaultValue: `Restore ${record.name}`,
          name: record.name
        })}
        aria-describedby={!record.restorable ? getTrashRestoreStatusId(record.name) : undefined}
        size={mobile ? "middle" : "small"}
        className={mobile ? "min-h-11" : undefined}
        icon={<RotateCcw size={14} />}
        disabled={
          !record.restorable || restoreMutation.isPending || purgeMutation.isPending
        }
        loading={restoreMutation.isPending && restoreMutation.variables?.name === record.name}
        onClick={() => restoreMutation.mutate({
          name: record.name,
          version: getKnownSkillVersion(record.version)
        })}
      >
        {t("option:skills.restore", { defaultValue: "Restore" })}
      </Button>
      <Button
        aria-label={t("option:skills.purgeNamedSkill", {
          defaultValue: `Permanently delete ${record.name}`,
          name: record.name
        })}
        size={mobile ? "middle" : "small"}
        className={mobile ? "min-h-11" : undefined}
        danger
        icon={<Trash2 size={14} />}
        disabled={restoreMutation.isPending || purgeMutation.isPending}
        loading={purgeMutation.isPending && purgeMutation.variables?.name === record.name}
        onClick={() => handlePurge(record)}
      >
        {t("option:skills.purge", { defaultValue: "Delete permanently" })}
      </Button>
    </div>
  )

  const trashColumns: ColumnsType<SkillTrashItem> = [
    {
      title: t("option:skills.colName", { defaultValue: "Name" }),
      dataIndex: "name",
      key: "name",
      render: (name: string) => <span className="font-mono text-sm">{name}</span>
    },
    {
      title: t("option:skills.colDescription", { defaultValue: "Description" }),
      dataIndex: "description",
      key: "description",
      ellipsis: true,
      render: (description: string | null) => description || "-"
    },
    {
      title: t("option:skills.deletedAt", { defaultValue: "Deleted" }),
      dataIndex: "deleted_at",
      key: "deleted_at",
      width: 190,
      render: (deletedAt: string) => formatDeletedAt(deletedAt)
    },
    {
      title: t("option:skills.restoreStatus", { defaultValue: "Restore status" }),
      key: "restore_status",
      width: 240,
      render: (_: unknown, record: SkillTrashItem) => record.restorable ? (
        <Tag color="success">
          {t("option:skills.readyToRestore", { defaultValue: "Ready to restore" })}
        </Tag>
      ) : (
        <div
          id={getTrashRestoreStatusId(record.name)}
          className="flex flex-col items-start gap-1"
        >
          <Tag color="warning">
            {t("option:skills.restoreUnavailable", { defaultValue: "Restore unavailable" })}
          </Tag>
          <span className="text-xs text-text-muted">
            {record.restore_unavailable_reason ?? t("option:skills.restoreUnavailableReason", {
              defaultValue: "Archived skill files are unavailable."
            })}
          </span>
        </div>
      )
    },
    {
      title: t("option:skills.colActions", { defaultValue: "Actions" }),
      key: "actions",
      width: 300,
      render: (_: unknown, record: SkillTrashItem) => renderTrashActions(record)
    }
  ]

  const importMenuItems = [
    {
      key: "text",
      label: (
        <span className="flex items-center gap-2">
          <FileText size={14} />
          {t("option:skills.importText", { defaultValue: "Import Text" })}
        </span>
      ),
      onClick: openImportTextModal
    },
    {
      key: "file",
      label: (
        <span className="flex items-center gap-2">
          <FileDown size={14} />
          {t("option:skills.importFile", { defaultValue: "Import File (.md/.zip)" })}
        </span>
      ),
      onClick: () => importFileInputRef.current?.click()
    }
  ]

  const seedMenuItems = [
    {
      key: "seed-missing-only",
      label: t("option:skills.seedBuiltinsMissingOnly", {
        defaultValue: "Seed Missing Only"
      }),
      onClick: () => seedBuiltinsMutation.mutate(false)
    },
    {
      key: "seed-overwrite-existing",
      label: t("option:skills.seedBuiltinsOverwrite", {
        defaultValue: "Seed and Overwrite Existing"
      }),
      onClick: confirmSeedOverwrite
    }
  ]

  const beginnerEmptyState = (
    <div
      className="mx-auto flex max-w-xl flex-col items-center gap-3 py-8 text-center"
      data-testid="skills-empty-state"
    >
      <div className="space-y-1">
        <h2 className="text-base font-semibold text-text">
          {t("option:skills.emptyTitle", {
            defaultValue: "Start with a reusable skill"
          })}
        </h2>
        <p className="m-0 text-sm text-text-muted">
          {t("option:skills.emptyDescription", {
            defaultValue:
              "Skills are reusable instructions that can be tested here and used from chat."
          })}
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button
          type="primary"
          icon={<Database size={14} />}
          loading={seedBuiltinsMutation.isPending}
          onClick={() => seedBuiltinsMutation.mutate(false)}
        >
          {t("option:skills.emptySeedBuiltins", {
            defaultValue: "Seed built-ins"
          })}
        </Button>
        <Button
          data-skill-action="new"
          icon={<Plus size={14} />}
          onClick={(event) => handleNew(event.currentTarget)}
        >
          {t("option:skills.emptyCreateFromTemplate", {
            defaultValue: "Create from template"
          })}
        </Button>
        <Button icon={<UploadIcon size={14} />} onClick={openImportTextModal}>
          {t("option:skills.emptyImportText", {
            defaultValue: "Import from text"
          })}
        </Button>
      </div>
    </div>
  )

  const trashEmptyState = (
    <div className="mx-auto flex max-w-xl flex-col items-center gap-1 py-8 text-center">
      <h2 className="m-0 text-base font-semibold text-text">
        {t("option:skills.trashEmptyTitle", { defaultValue: "Trash is empty" })}
      </h2>
      <p className="m-0 text-sm text-text-muted">
        {t("option:skills.trashEmptyDescription", {
          defaultValue: "Skills you move to Trash will appear here until permanently deleted."
        })}
      </p>
    </div>
  )

  let tableEmptyText: React.ReactNode = beginnerEmptyState
  if (!isLibraryEmpty) {
    let emptyText = t("option:skills.emptyTable", {
      defaultValue: "No skills yet."
    })
    if (isError) {
      emptyText = t("option:skills.emptyTableError", {
        defaultValue: "Unable to load skills."
      })
    } else if (hasActiveFilters) {
      emptyText = t("option:skills.noFilterMatches", {
        defaultValue: "No skills match these filters."
      })
    } else if (hasSearch) {
      emptyText = t("option:skills.noMatches", {
        defaultValue: "No skills match this search."
      })
    } else if (totalSkills > 0) {
      emptyText = t("option:skills.emptyCurrentPage", {
        defaultValue: "No skills on this page."
      })
    }

    tableEmptyText = emptyText
  }

  const tableSize = tableDensity === "compact" ? "small" : "middle"
  const importTextCanSubmit =
    Boolean(importTextPreview?.valid)
    && (
      !importTextPreview?.conflict
      || (
        Boolean(importTextOverwrite)
        && getKnownSkillVersion(importTextPreview.existing_version) !== undefined
      )
    )
  const importTextOkLabel = importTextPreview?.valid
    ? t("option:skills.importSkill", { defaultValue: "Import skill" })
    : t("option:skills.reviewImport", { defaultValue: "Review import" })
  const importFileOkLabel = t("option:skills.importSkill", { defaultValue: "Import skill" })

  const renderImportReview = (preview: SkillImportPreviewResponse) => {
    const statusLabel = !preview.valid
      ? t("option:skills.importReviewNeedsFixes", { defaultValue: "Needs fixes" })
      : preview.conflict
        ? t("option:skills.importReviewConflict", { defaultValue: "Conflict" })
        : t("option:skills.importReviewReady", { defaultValue: "Ready" })
    const statusColor = !preview.valid ? "error" : preview.conflict ? "warning" : "success"

    return (
      <div
        className="mt-3 rounded-md border border-border bg-surface p-3"
        aria-live="polite"
      >
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3 className="m-0 text-sm font-semibold text-text">
            {t("option:skills.importReviewTitle", { defaultValue: "Import review" })}
          </h3>
          <Tag color={statusColor}>{statusLabel}</Tag>
        </div>

        {!preview.valid ? (
          <div className="rounded-md border border-danger/30 bg-danger/10 p-2 text-sm text-text">
            <p className="m-0 font-medium">
              {t("option:skills.importReviewErrors", {
                defaultValue: "Fix these issues before importing."
              })}
            </p>
            <ul className="mb-0 mt-1 pl-5">
              {(preview.errors.length ? preview.errors : [
                t("option:skills.importReviewUnknownError", {
                  defaultValue: "The skill could not be validated."
                })
              ]).map((error, index) => (
                <li key={`${error}-${index}`}>{error}</li>
              ))}
            </ul>
          </div>
        ) : (
          <>
            <dl className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
              <div>
                <dt className="text-xs font-semibold uppercase text-text-muted">
                  {t("option:skills.nameLabel", { defaultValue: "Name" })}
                </dt>
                <dd className="m-0 break-all font-mono text-text">
                  {preview.name ?? t("option:skills.importReviewMissingName", {
                    defaultValue: "Not resolved"
                  })}
                </dd>
              </div>
              <div>
                <dt className="text-xs font-semibold uppercase text-text-muted">
                  {t("option:skills.mode", { defaultValue: "Mode" })}
                </dt>
                <dd className="m-0 text-text">{preview.context ?? "inline"}</dd>
              </div>
              {preview.description && (
                <div className="sm:col-span-2">
                  <dt className="text-xs font-semibold uppercase text-text-muted">
                    {t("option:skills.descriptionLabel", { defaultValue: "Description" })}
                  </dt>
                  <dd className="m-0 text-text">{preview.description}</dd>
                </div>
              )}
              {preview.argument_hint && (
                <div>
                  <dt className="text-xs font-semibold uppercase text-text-muted">
                    {t("option:skills.argumentHint", { defaultValue: "Argument hint" })}
                  </dt>
                  <dd className="m-0 font-mono text-text">{preview.argument_hint}</dd>
                </div>
              )}
              <div>
                <dt className="text-xs font-semibold uppercase text-text-muted">
                  {t("option:skills.supportingFiles", { defaultValue: "Supporting files" })}
                </dt>
                <dd className="m-0 text-text">{preview.supporting_file_count}</dd>
              </div>
              {preview.model && (
                <div>
                  <dt className="text-xs font-semibold uppercase text-text-muted">
                    {t("option:skills.model", { defaultValue: "Model" })}
                  </dt>
                  <dd className="m-0 font-mono text-text">{preview.model}</dd>
                </div>
              )}
              {preview.allowed_tools?.length ? (
                <div className="sm:col-span-2">
                  <dt className="text-xs font-semibold uppercase text-text-muted">
                    {t("option:skills.declaredTools", { defaultValue: "Declared tools" })}
                  </dt>
                  <dd className="m-0 text-text">{preview.allowed_tools.join(", ")}</dd>
                </div>
              ) : null}
            </dl>

            {preview.conflict && (
              <div className="mt-3 rounded-md border border-warn/40 bg-warn/10 p-2 text-sm text-text">
                <p className="m-0 font-semibold">
                  {t("option:skills.importConflictTitle", {
                    defaultValue: "Existing skill detected"
                  })}
                </p>
                <p className="m-0 text-text-muted">
                  {t("option:skills.importConflictDescription", {
                    defaultValue:
                      "Importing will replace the existing skill or Trash item only if overwrite is enabled."
                  })}
                </p>
                {preview.existing_version != null && (
                  <p className="m-0 mt-1 font-mono text-xs text-text-muted">
                    {t("option:skills.importConflictVersion", {
                      defaultValue: `Version ${preview.existing_version}`,
                      version: preview.existing_version
                    })}
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    )
  }

  return (
    <div ref={managerRootRef} className="flex flex-col gap-4">
      <input
        ref={importFileInputRef}
        type="file"
        accept=".md,.zip"
        className="sr-only"
        aria-label={t("option:skills.importFileInputLabel", {
          defaultValue: "Import skill file"
        })}
        onChange={handleImportFileInputChange}
      />
      <section
        aria-labelledby="skills-manager-title"
        className="flex flex-col gap-1"
      >
        <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
          <div>
            <h1
              id="skills-manager-title"
              className="m-0 text-xl font-semibold text-text"
            >
              {t("option:skills.title", { defaultValue: "Skills" })}
            </h1>
            <p className="m-0 max-w-2xl text-sm text-text-muted">
              {t("option:skills.description", {
                defaultValue:
                  "Discover, test, create, import, and manage reusable instructions."
              })}
            </p>
          </div>
          <p className="m-0 text-sm font-medium text-text-muted">
            {activeCountLabel}
          </p>
        </div>
      </section>

      <div>
        <Segmented
          aria-label={t("option:skills.libraryViewSelector", {
            defaultValue: "Skills view"
          })}
          value={activeView}
          onChange={(value) => handleViewChange(value as SkillsView)}
          options={[
            {
              value: "library",
              label: t("option:skills.libraryView", { defaultValue: "Library" })
            },
            {
              value: "trash",
              label: t("option:skills.trashView", { defaultValue: "Trash" })
            }
          ]}
        />
      </div>

      {activeView === "library" ? (
        <>
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex w-full max-w-[360px] flex-col gap-1">
          <label htmlFor="skills-search" className="text-xs font-medium text-text-muted">
            {t("option:skills.searchLabel", { defaultValue: "Search skills" })}
          </label>
          <Input.Search
            ref={searchInputRef}
            id="skills-search"
            placeholder={t("option:skills.searchPlaceholder", {
              defaultValue: "Search skills..."
            })}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            allowClear
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Dropdown menu={{ items: importMenuItems }} trigger={["click"]}>
            <Button icon={<UploadIcon size={14} />}>
              {t("option:skills.import", { defaultValue: "Import" })}
            </Button>
          </Dropdown>
          <Dropdown menu={{ items: seedMenuItems }} trigger={["click"]}>
            <Button
              icon={<Database size={14} />}
              loading={seedBuiltinsMutation.isPending}
            >
              {t("option:skills.seedBuiltins", { defaultValue: "Seed Built-ins" })}
            </Button>
          </Dropdown>
          <Button
            type="primary"
            icon={<Plus size={14} />}
            data-skill-action="new"
            onClick={(event) => handleNew(event.currentTarget)}
          >
            {t("option:skills.newSkill", { defaultValue: "New Skill" })}
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Popover
          trigger="click"
          placement="bottomLeft"
          title={t("option:skills.filters", { defaultValue: "Filters" })}
          content={(
            <div className="grid w-64 gap-3">
              <label className="grid gap-1 text-xs font-medium text-text-muted">
                {t("option:skills.mode", { defaultValue: "Mode" })}
                <Select
                  aria-label={t("option:skills.contextFilter", { defaultValue: "Skill mode filter" })}
                  value={contextFilter}
                  onChange={handleContextFilterChange}
                  options={[
                    { value: "all", label: t("option:skills.filterAllModes", { defaultValue: "All modes" }) },
                    { value: "inline", label: t("option:skills.filterInline", { defaultValue: "Inline" }) },
                    { value: "fork", label: t("option:skills.filterFork", { defaultValue: "Fork" }) }
                  ]}
                />
              </label>
              <label className="grid gap-1 text-xs font-medium text-text-muted">
                {t("option:skills.visibility", { defaultValue: "Visibility" })}
                <Select
                  aria-label={t("option:skills.visibilityFilter", { defaultValue: "Skill visibility filter" })}
                  value={visibilityFilter}
                  onChange={handleVisibilityFilterChange}
                  options={[
                    { value: "visible", label: t("option:skills.filterVisible", { defaultValue: "Visible" }) },
                    { value: "hidden", label: t("option:skills.filterHidden", { defaultValue: "Hidden" }) },
                    { value: "all", label: t("option:skills.filterAllVisibility", { defaultValue: "All visibility" }) }
                  ]}
                />
              </label>
              <label className="grid gap-1 text-xs font-medium text-text-muted">
                {t("option:skills.declaredTools", { defaultValue: "Declared tools" })}
                <Select
                  aria-label={t("option:skills.toolsFilter", { defaultValue: "Skill tools filter" })}
                  value={toolsFilter}
                  onChange={handleToolsFilterChange}
                  options={[
                    { value: "any", label: t("option:skills.filterAnyTools", { defaultValue: "Any tools" }) },
                    { value: "with-tools", label: t("option:skills.filterHasTools", { defaultValue: "Has tools" }) },
                    { value: "without-tools", label: t("option:skills.filterNoTools", { defaultValue: "No tools" }) }
                  ]}
                />
              </label>
              <label className="grid gap-1 text-xs font-medium text-text-muted">
                {t("option:skills.model", { defaultValue: "Model" })}
                <Input
                  aria-label={t("option:skills.modelFilter", { defaultValue: "Filter by model" })}
                  placeholder={t("option:skills.modelFilterPlaceholder", { defaultValue: "Model" })}
                  value={modelFilter}
                  onChange={(event) => handleModelFilterChange(event.target.value)}
                  allowClear
                />
              </label>
            </div>
          )}
        >
          <Button icon={<SlidersHorizontal size={14} />}>
            {hasActiveFilters
              ? t("option:skills.filtersCount", {
                  defaultValue: "Filters ({{count}})",
                  count: activeFilterCount
                })
              : t("option:skills.filters", { defaultValue: "Filters" })}
          </Button>
        </Popover>

        <Popover
          trigger="click"
          placement="bottomLeft"
          title={t("option:skills.viewOptions", { defaultValue: "View options" })}
          content={(
            <div className="grid w-64 gap-4">
              <label className="grid gap-1 text-xs font-medium text-text-muted">
                {t("option:skills.sortBy", { defaultValue: "Sort by" })}
                <Select<SkillSortOption>
                  aria-label={t("option:skills.sortBy", { defaultValue: "Sort by" })}
                  value={getSkillSortOption(sortState)}
                  onChange={handleSortOptionChange}
                  options={[
                    { value: "default", label: t("option:skills.sortDefault", { defaultValue: "Default order" }) },
                    { value: "name:asc", label: t("option:skills.sortNameAsc", { defaultValue: "Name (A-Z)" }) },
                    { value: "name:desc", label: t("option:skills.sortNameDesc", { defaultValue: "Name (Z-A)" }) },
                    { value: "context:asc", label: t("option:skills.sortModeAsc", { defaultValue: "Mode (A-Z)" }) },
                    { value: "context:desc", label: t("option:skills.sortModeDesc", { defaultValue: "Mode (Z-A)" }) },
                    { value: "created_at:asc", label: t("option:skills.sortCreatedAsc", { defaultValue: "Created (oldest)" }) },
                    { value: "created_at:desc", label: t("option:skills.sortCreatedDesc", { defaultValue: "Created (newest)" }) },
                    { value: "last_modified:asc", label: t("option:skills.sortModifiedAsc", { defaultValue: "Modified (oldest)" }) },
                    { value: "last_modified:desc", label: t("option:skills.sortModifiedDesc", { defaultValue: "Modified (newest)" }) }
                  ]}
                />
              </label>
              {!isMobile && (
                <>
                  <div className="grid gap-1">
                    <span className="text-xs font-medium text-text-muted">
                      {t("option:skills.tableDensity", { defaultValue: "Table density" })}
                    </span>
                    <Segmented
                      block
                      value={tableDensity}
                      onChange={(value) => handleDensityChange(value as SkillTableDensity)}
                      options={[
                        { value: "comfortable", label: t("option:skills.densityComfortable", { defaultValue: "Comfortable" }) },
                        { value: "compact", label: t("option:skills.densityCompact", { defaultValue: "Compact" }) }
                      ]}
                    />
                  </div>
                  <Checkbox.Group
                    aria-label={t("option:skills.columnVisibility", { defaultValue: "Column visibility" })}
                    className="grid gap-2"
                    value={visibleOptionalColumns}
                    options={SKILL_OPTIONAL_COLUMN_KEYS.map((key) => ({
                      label: optionalColumnLabels[key],
                      value: key
                    }))}
                    onChange={(values) => {
                      const next = values.filter(isSkillOptionalColumnKey)
                      if (!next.includes("context") && sortState.field === "context") {
                        commitUrlHistory()
                        setSortState({})
                        setPage(1)
                      }
                      setVisibleOptionalColumns(next)
                    }}
                  />
                </>
              )}
            </div>
          )}
        >
          <Button icon={<Settings2 size={14} />}>
            {t("option:skills.viewOptions", { defaultValue: "View options" })}
          </Button>
        </Popover>

        {contextFilter !== "all" && (
          <ActiveFilterTag
            label={activeContextFilterLabel}
            removeLabel={getRemoveFilterLabel(activeContextFilterLabel)}
            onRemove={() => handleContextFilterChange("all")}
          />
        )}
        {visibilityFilter !== "visible" && (
          <ActiveFilterTag
            label={activeVisibilityFilterLabel}
            removeLabel={getRemoveFilterLabel(activeVisibilityFilterLabel)}
            onRemove={() => handleVisibilityFilterChange("visible")}
          />
        )}
        {toolsFilter !== "any" && (
          <ActiveFilterTag
            label={activeToolsFilterLabel}
            removeLabel={getRemoveFilterLabel(activeToolsFilterLabel)}
            onRemove={() => handleToolsFilterChange("any")}
          />
        )}
        {modelQuery && (
          <ActiveFilterTag
            label={activeModelFilterLabel}
            removeLabel={getRemoveFilterLabel(activeModelFilterLabel)}
            onRemove={() => handleModelFilterChange("")}
          />
        )}
        <Button size="small" onClick={clearFilters} disabled={!hasActiveFilters}>
          {t("option:skills.clearFilters", { defaultValue: "Clear filters" })}
        </Button>
      </div>

      {listLoadRecoveryState && (
        <RecoveryCallout
          state={listLoadRecoveryState.state}
          title={listLoadRecoveryState.title}
          message={listLoadRecoveryState.message}
          diagnostics={listLoadRecoveryState.diagnostics}
          role="alert"
          primaryAction={{
            label: t("common:tryAgain", { defaultValue: "Try again" }),
            onClick: () => void refetch()
          }}
          data-testid="skills-list-recovery-state"
        />
      )}

      <div role="status" aria-live="polite" className="sr-only">
        {isLoading
          ? t("option:skills.loadingStatus", {
              defaultValue: "Loading skills"
            })
          : ""}
      </div>

      {successAction && (
        <DesignSystemAlert
          data-testid="skills-success-actions"
          variant="success"
          title={successAction.title}
          dismissible
          dismissLabel={t("common:close", { defaultValue: "Close" })}
          onDismiss={() => setSuccessAction(null)}
        >
          <p className="m-0">{successAction.description}</p>
          {(() => {
            const skillName = successAction.skillName
            if (!skillName) return null
            const invocation = buildSkillInvocation(skillName)
            return (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <Button
                  size="small"
                  icon={<Play size={14} />}
                  data-skill-action="success-test-run"
                  data-skill-name={skillName}
                  onClick={(event) => openSkillPreview(skillName, event.currentTarget)}
                >
                  {successAction.testLabel ??
                    t("option:skills.testRun", { defaultValue: "Test run" })}
                </Button>
                <Button
                  size="small"
                  icon={<Eye size={14} />}
                  data-skill-action="success-view"
                  data-skill-name={skillName}
                  onClick={(event) => openSkillDetails(skillName, event.currentTarget)}
                >
                  {successAction.viewLabel ??
                    t("option:skills.viewSkill", { defaultValue: "View skill" })}
                </Button>
                <Button
                  size="small"
                  icon={<Copy size={14} />}
                  onClick={() => void handleCopyInvocation(skillName)}
                >
                  {t("option:skills.copyInvocation", {
                    defaultValue: `Copy ${invocation}`,
                    invocation
                  })}
                </Button>
              </div>
            )
          })()}
        </DesignSystemAlert>
      )}

      {selectedSkillCount > 0 && (
        <div
          data-testid="skills-selection-actions"
          className="flex flex-wrap items-center justify-between gap-2 border-y border-border py-2"
        >
          <div className="grid gap-0.5" aria-live="polite">
            <span className="text-sm font-medium text-text">
              {t("option:skills.selectedCount", {
                defaultValue: `${selectedSkillCount} selected`,
                count: selectedSkillCount
              })}
            </span>
            {selectedSkillCount >= MAX_SKILLS_BULK_SELECTION && (
              <span className="text-xs text-text-muted">
                {t("option:skills.selectionLimitReached", {
                  defaultValue: "Selection limit reached ({{limit}} maximum).",
                  limit: MAX_SKILLS_BULK_SELECTION
                })}
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="small"
              onClick={() => {
                setSelectedSkillNames([])
                setSelectedSkillSnapshots(new Map())
              }}
            >
              {t("option:skills.clearSelection", {
                defaultValue: "Clear selection"
              })}
            </Button>
            <Button
              size="small"
              icon={<FileArchive size={14} />}
              loading={isBulkExporting}
              onClick={() => void handleBulkExport()}
            >
              {t("option:skills.bulkExportAction", {
                defaultValue: "Export selected"
              })}
            </Button>
            <Button
              size="small"
              danger
              icon={<Trash2 size={14} />}
              loading={bulkDeleteMutation.isPending}
              onClick={handleBulkDelete}
            >
              {t("option:skills.bulkDeleteAction", {
                defaultValue: "Delete selected"
              })}
            </Button>
          </div>
        </div>
      )}

      {isMobile ? (
        <div className="grid gap-2" data-testid="skills-mobile-list">
          {currentSkills.map((skill) => (
            <article key={skill.name} className="border-b border-border py-3">
              <div className="flex items-start gap-3">
                <Checkbox
                  className="flex min-h-11 min-w-11 items-center justify-center"
                  aria-label={t("option:skills.selectSkillRow", {
                    defaultValue: `Select ${skill.name}`,
                    name: skill.name
                  })}
                  checked={selectedSkillNames.includes(skill.name)}
                  disabled={
                    !selectedSkillNames.includes(skill.name)
                    && selectedSkillCount >= MAX_SKILLS_BULK_SELECTION
                  }
                  onChange={(event) => {
                    const nextNames = event.target.checked
                      ? Array.from(new Set([...selectedSkillNames, skill.name]))
                      : selectedSkillNames.filter((name) => name !== skill.name)
                    applySkillSelection(nextNames)
                  }}
                />
                <div className="min-w-0 flex-1">
                  <button
                    type="button"
                    className="inline-flex min-h-11 items-center break-all bg-transparent p-0 text-left font-mono text-sm font-medium text-text"
                    data-skill-action="mobile-view"
                    data-skill-name={skill.name}
                    onClick={(event) => openSkillDetails(skill.name, event.currentTarget)}
                  >
                    {skill.name}
                  </button>
                  <p className="mb-0 mt-1 line-clamp-2 text-sm text-text-muted">
                    {skill.description || t("option:skills.noDescription", { defaultValue: "No description" })}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-1">
                    <Tag>{skill.context}</Tag>
                    {!skill.user_invocable && (
                      <Tag>{t("option:skills.hiddenState", { defaultValue: "Hidden" })}</Tag>
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-2 flex justify-end">{renderSkillActions(skill, true)}</div>
            </article>
          ))}
          {!isLoading && currentSkills.length === 0 && (
            <div className="py-4 text-center text-sm text-text-muted">{tableEmptyText}</div>
          )}
        </div>
      ) : (
        <Table
          data-testid="skills-table"
          data-density={tableDensity}
          dataSource={currentSkills}
          columns={columns}
          rowKey="name"
          rowSelection={{
            selectedRowKeys: selectedSkillNames,
            preserveSelectedRowKeys: true,
            onChange: (selectedRowKeys) => {
              applySkillSelection(selectedRowKeys.map((key) => String(key)))
            },
            getCheckboxProps: (record) => {
              const checkboxProps: { disabled?: boolean } & React.AriaAttributes = {
                disabled:
                  !selectedSkillNames.includes(record.name)
                  && selectedSkillCount >= MAX_SKILLS_BULK_SELECTION,
                "aria-label": t("option:skills.selectSkillRow", {
                  defaultValue: `Select ${record.name}`,
                  name: record.name
                })
              }
              return checkboxProps
            }
          }}
          loading={isLoading}
          onChange={handleTableChange}
          pagination={false}
          size={tableSize}
          locale={{
            emptyText: tableEmptyText
          }}
        />
      )}

      {totalSkills > pageSize && (
        <div className="flex justify-end">
          <Pagination
            current={page}
            pageSize={pageSize}
            total={totalSkills}
            onChange={(p, ps) => {
              commitUrlHistory()
              setPage(p)
              setPageSize(normalizeSkillsPageSize(ps))
            }}
            showSizeChanger
            pageSizeOptions={["10", "20", "50"]}
          />
        </div>
      )}
        </>
      ) : (
        <>
          {trashLoadRecoveryState && (
            <RecoveryCallout
              state={trashLoadRecoveryState.state}
              title={trashLoadRecoveryState.title}
              message={trashLoadRecoveryState.message}
              diagnostics={trashLoadRecoveryState.diagnostics}
              role="alert"
              primaryAction={{
                label: t("common:tryAgain", { defaultValue: "Try again" }),
                onClick: () => void refetchTrash()
              }}
              data-testid="skills-trash-recovery-state"
            />
          )}

          <div role="status" aria-live="polite" className="sr-only">
            {isTrashLoading
              ? t("option:skills.trashLoadingStatus", {
                  defaultValue: "Loading deleted skills"
                })
              : ""}
          </div>

          {isMobile ? (
            <div className="grid gap-2" data-testid="skills-trash-mobile-list">
              {currentTrash.map((skill) => (
                <article key={skill.name} className="border-b border-border py-3">
                  <div className="min-w-0">
                    <p className="m-0 break-all font-mono text-sm font-medium text-text">
                      {skill.name}
                    </p>
                    <p className="mb-0 mt-1 line-clamp-2 text-sm text-text-muted">
                      {skill.description || t("option:skills.noDescription", {
                        defaultValue: "No description"
                      })}
                    </p>
                    <p className="mb-0 mt-2 text-xs text-text-muted">
                      {t("option:skills.deletedAtValue", {
                        defaultValue: `Deleted ${formatDeletedAt(skill.deleted_at)}`,
                        deletedAt: formatDeletedAt(skill.deleted_at)
                      })}
                    </p>
                    {!skill.restorable && (
                      <p
                        id={getTrashRestoreStatusId(skill.name)}
                        className="mb-0 mt-2 text-sm text-warn"
                      >
                        <span className="font-medium">
                          {t("option:skills.restoreUnavailable", {
                            defaultValue: "Restore unavailable"
                          })}.
                        </span>{" "}
                        {skill.restore_unavailable_reason ?? t(
                          "option:skills.restoreUnavailableReason",
                          { defaultValue: "Archived skill files are unavailable." }
                        )}
                      </p>
                    )}
                  </div>
                  <div className="mt-3">{renderTrashActions(skill, true)}</div>
                </article>
              ))}
              {!isTrashLoading && currentTrash.length === 0 && (
                <div className="py-4 text-center text-sm text-text-muted">
                  {isTrashError
                    ? t("option:skills.trashEmptyError", {
                        defaultValue: "Unable to load Trash."
                      })
                    : trashEmptyState}
                </div>
              )}
            </div>
          ) : (
            <Table
              data-testid="skills-trash-table"
              dataSource={currentTrash}
              columns={trashColumns}
              rowKey="name"
              loading={isTrashLoading}
              pagination={false}
              size={tableSize}
              locale={{
                emptyText: isTrashError
                  ? t("option:skills.trashEmptyError", {
                      defaultValue: "Unable to load Trash."
                    })
                  : trashEmptyState
              }}
            />
          )}

          {totalTrash > pageSize && (
            <div className="flex justify-end">
              <Pagination
                current={page}
                pageSize={pageSize}
                total={totalTrash}
                onChange={(nextPage, nextPageSize) => {
                  commitUrlHistory()
                  setPage(nextPage)
                  setPageSize(normalizeSkillsPageSize(nextPageSize))
                }}
                showSizeChanger
                pageSizeOptions={["10", "20", "50"]}
              />
            </div>
          )}
        </>
      )}

      <Modal
        title={t("option:skills.importTextTitle", {
          defaultValue: "Import Skill from Text"
        })}
        open={importTextOpen}
        onCancel={requestImportTextClose}
        onOk={handleImportTextSubmit}
        okText={importTextOkLabel}
        okButtonProps={{
          "aria-label": importTextOkLabel,
          loading:
            importTextMutation.isPending
            || importTextPreviewPendingRevision === importTextPreviewRevisionRef.current,
          disabled: Boolean(importTextPreview?.valid) && !importTextCanSubmit
        }}
        destroyOnHidden
      >
        {importTextDraftRecovered && (
          <div
            role="status"
            className="mb-4 flex items-center justify-between gap-3 rounded border border-border bg-surface p-3"
          >
            <span className="text-sm text-text">
              {t("option:skills.importDraftRecovered", {
                defaultValue: "Recovered your unfinished import from this session."
              })}
            </span>
            <Button
              size="small"
              onClick={() => {
                clearImportTextDraft()
                invalidateImportTextPreview()
                importTextForm.setFieldsValue({ name: undefined, content: "", overwrite: false })
              }}
            >
              {t("option:skills.discardRecoveredImport", {
                defaultValue: "Discard recovered import"
              })}
            </Button>
          </div>
        )}
        <Form
          form={importTextForm}
          layout="vertical"
          initialValues={{ overwrite: false }}
          autoComplete="off"
          onValuesChange={(changedValues, allValues: ImportTextFormValues) => {
            if ("name" in changedValues || "content" in changedValues) {
              invalidateImportTextPreview()
              importTextForm.setFieldValue("overwrite", false)
              const draft = {
                name: allValues.name,
                content: allValues.content ?? ""
              }
              const hasDraft = Boolean(draft.name?.trim() || draft.content.trim())
              importTextDirtyRef.current = hasDraft
              if (!hasDraft) setImportTextDraftRecovered(false)
              writeImportTextDraft(skillsScope, hasDraft ? draft : null)
            }
          }}
        >
          <Form.Item
            name="name"
            label={t("option:skills.nameLabel", { defaultValue: "Name" })}
            rules={[
              {
                validator: async (_, value: string | undefined) => {
                  const trimmed = (value ?? "").trim()
                  if (!trimmed) return
                  if (!SKILL_NAME_REGEX.test(trimmed)) {
                    throw new Error(
                      t("option:skills.nameInvalid", {
                        defaultValue:
                          "Must start with a letter, use only lowercase letters, numbers, and hyphens (max 64 chars)"
                      })
                    )
                  }
                }
              }
            ]}
            extra={t("option:skills.importNameOptional", {
              defaultValue: "Optional. If omitted, name from frontmatter will be used."
            })}
          >
            <Input placeholder="my-skill-name" maxLength={64} className="font-mono" />
          </Form.Item>

          <Form.Item
            name="content"
            label={t("option:skills.contentLabel", {
              defaultValue: "SKILL.md Content"
            })}
            rules={[
              {
                required: true,
                whitespace: true,
                message: t("option:skills.contentRequired", {
                  defaultValue: "Content is required"
                })
              }
            ]}
          >
            <Input.TextArea rows={14} className="font-mono text-xs" />
          </Form.Item>

          {importTextPreview && renderImportReview(importTextPreview)}

          {importTextPreview?.valid && importTextPreview.conflict && (
            <Form.Item
              name="overwrite"
              valuePropName="checked"
              label={t("option:skills.importOverwrite", {
                defaultValue: "Overwrite existing skill"
              })}
              extra={t("option:skills.importOverwriteRequired", {
                defaultValue:
                  "Required because this import matches an existing skill name."
              })}
            >
              <Switch
                aria-label={t("option:skills.importOverwrite", {
                  defaultValue: "Overwrite existing skill"
                })}
              />
            </Form.Item>
          )}
        </Form>
      </Modal>

      <Modal
        title={t("option:skills.importFileReviewTitle", {
          defaultValue: "Review Skill Import"
        })}
        open={Boolean(fileImportReview)}
        onCancel={requestFileImportClose}
        onOk={() => {
          if (!fileImportReview?.preview.valid) return
          if (fileImportReview.preview.conflict && !fileImportReview.overwrite) return
          const expectedVersion = fileImportReview.preview.conflict
            ? getKnownSkillVersion(fileImportReview.preview.existing_version)
            : undefined
          if (fileImportReview.preview.conflict && expectedVersion === undefined) return
          importFileMutation.mutate({
            file: fileImportReview.file,
            overwrite: fileImportReview.preview.conflict
              ? fileImportReview.overwrite
              : false,
            expectedVersion
          })
        }}
        okText={importFileOkLabel}
        okButtonProps={{
          "aria-label": importFileOkLabel,
          loading: importFileMutation.isPending,
          disabled:
            !fileImportReview?.preview.valid
            || Boolean(fileImportReview.preview.conflict && !fileImportReview.overwrite)
            || Boolean(
              fileImportReview?.preview.conflict
              && getKnownSkillVersion(fileImportReview.preview.existing_version) === undefined
            )
        }}
        destroyOnHidden
      >
        {fileImportReview && (
          <div className="flex flex-col gap-3">
            <p className="m-0 text-sm text-text-muted">
              {t("option:skills.importFileReviewSource", {
                defaultValue: "Selected file:"
              })}{" "}
              <span className="font-mono text-text">{fileImportReview.file.name}</span>
            </p>
            {renderImportReview(fileImportReview.preview)}
            {fileImportReview.preview.valid && fileImportReview.preview.conflict && (
              <div className="flex items-center gap-2">
                <Switch
                  aria-label={t("option:skills.importOverwrite", {
                    defaultValue: "Overwrite existing skill"
                  })}
                  checked={fileImportReview.overwrite}
                  onChange={(checked) => {
                    setFileImportReview((current) =>
                      current ? { ...current, overwrite: checked } : current
                    )
                  }}
                />
                <span className="text-sm text-text">
                  {t("option:skills.importOverwrite", {
                    defaultValue: "Overwrite existing skill"
                  })}
                </span>
              </div>
            )}
          </div>
        )}
      </Modal>

      <SkillDrawer
        open={drawerOpen}
        skill={editingSkill}
        duplicateFrom={duplicateSkill}
        draftScope={skillsScope}
        requestSignal={skillsRequestControllerRef.current?.signal}
        onClose={handleDrawerClose}
        onAfterClose={handleDrawerAfterClose}
        onSaved={handleDrawerSaved}
      />

      <SkillPreview
        skillName={previewSkill}
        runtime={previewRuntime}
        onClose={closeSkillPreview}
        onAfterClose={handlePreviewAfterClose}
      />

      <SkillDetailsDrawer
        scopeKey={skillsQueryScope}
        skillName={detailsSkill}
        onClose={closeSkillDetails}
        onTest={(name) => {
          const returnTarget = getSkillActionElement("view", name)
          detailsReturnFocusRef.current = null
          setDetailsSkill(null)
          openSkillPreview(name, returnTarget)
        }}
        onEdit={(name) => void handleEdit(name, getSkillActionElement("view", name))}
        onUseInChat={handleUseInChat}
        onCopyInvocation={(name) => void handleCopyInvocation(name)}
        onDuplicate={(name) => void handleDuplicate(
          name,
          getSkillActionElement("view", name)
        )}
      />
    </div>
  )
}
