import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Button,
  Form,
  Input,
  Table,
  Tag,
  Tooltip,
  Dropdown,
  Upload,
  Pagination,
  Modal,
  Switch
} from "antd"
import type { MenuProps } from "antd"
import type { ColumnsType, TableProps } from "antd/es/table"
import type { SortOrder } from "antd/es/table/interface"
import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
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
  Columns3,
  Rows3
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { SkillDrawer } from "./SkillDrawer"
import { SkillPreview } from "./SkillPreview"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"
import type {
  SkillContext,
  SkillImportPreviewResponse,
  SkillListOrder,
  SkillListSort,
  SkillSummary,
  SkillResponse,
  SkillsListResponse
} from "@/types/skill"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"

const DEFAULT_PAGE_SIZE = 10
const SKILLS_SEARCH_DEBOUNCE_MS = 300
const SKILL_NAME_REGEX = /^[a-z][a-z0-9-]{0,63}$/
const SKILLS_TABLE_PREFERENCES_STORAGE_KEY = "tldw:skills-manager:table-preferences:v1"
const SKILL_TABLE_SORT_DIRECTIONS: SortOrder[] = ["ascend", "descend"]

interface ImportTextFormValues {
  name?: string
  content: string
  overwrite?: boolean
}

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

interface DeleteSkillPayload {
  name: string
  version?: number
}

type SkillContextFilter = "all" | SkillContext
type SkillVisibilityFilter = "visible" | "hidden" | "all"
type SkillToolsFilter = "any" | "with-tools" | "without-tools"
type SkillTableDensity = "comfortable" | "compact"
type SkillOptionalColumnKey =
  | "description"
  | "context"
  | "argument_hint"
  | "user_invocable"
  | "model_invocation"

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
  "model_invocation"
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

const buildSkillInvocation = (skillName: string) => `/skill ${skillName}`

const isSkillTableSortField = (value: React.Key | undefined): value is SkillListSort =>
  value === "name" || value === "context"

const getSkillTableSortOrder = (
  sortState: SkillSortState,
  field: SkillListSort
): SortOrder => {
  if (sortState.field !== field) return null
  return sortState.order === "asc" ? "ascend" : "descend"
}

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

export const SkillsManager: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const queryClient = useQueryClient()
  const notification = useAntdNotification()

  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(DEFAULT_PAGE_SIZE)
  const [search, setSearch] = React.useState("")
  const [debouncedSearch, setDebouncedSearch] = React.useState("")
  const [contextFilter, setContextFilter] =
    React.useState<SkillContextFilter>("all")
  const [visibilityFilter, setVisibilityFilter] =
    React.useState<SkillVisibilityFilter>("visible")
  const [toolsFilter, setToolsFilter] = React.useState<SkillToolsFilter>("any")
  const [modelFilter, setModelFilter] = React.useState("")
  const [debouncedModelFilter, setDebouncedModelFilter] = React.useState("")
  const [tableDensity, setTableDensity] =
    React.useState<SkillTableDensity>(() => loadSkillsTablePreferences().density)
  const [visibleOptionalColumns, setVisibleOptionalColumns] = React.useState<
    SkillOptionalColumnKey[]
  >(() => loadSkillsTablePreferences().visibleColumns)
  const [sortState, setSortState] = React.useState<SkillSortState>({})
  const [drawerOpen, setDrawerOpen] = React.useState(false)
  const [importTextOpen, setImportTextOpen] = React.useState(false)
  const [importTextPreview, setImportTextPreview] =
    React.useState<SkillImportPreviewResponse | null>(null)
  const [fileImportReview, setFileImportReview] =
    React.useState<FileImportReview | null>(null)
  const [editingSkill, setEditingSkill] = React.useState<SkillResponse | null>(null)
  const [previewSkill, setPreviewSkill] = React.useState<string | null>(null)
  const [successAction, setSuccessAction] =
    React.useState<SkillsSuccessAction | null>(null)
  const [importTextForm] = Form.useForm<ImportTextFormValues>()
  const importTextOverwrite = Form.useWatch("overwrite", importTextForm)

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
      })
  })

  const hasLoadedSkills = data != null && !isError
  const totalSkills = data?.total ?? 0
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

  React.useEffect(() => {
    if (!hasLoadedSkills) return

    const lastPage = Math.max(1, Math.ceil(totalSkills / pageSize))
    if (page > lastPage) {
      setPage(lastPage)
    }
  }, [hasLoadedSkills, page, pageSize, totalSkills])

  const handleContextFilterChange = (nextFilter: SkillContextFilter) => {
    setContextFilter(nextFilter)
    setPage(1)
  }

  const handleVisibilityFilterChange = (nextFilter: SkillVisibilityFilter) => {
    setVisibilityFilter(nextFilter)
    setPage(1)
  }

  const handleToolsFilterChange = (nextFilter: SkillToolsFilter) => {
    setToolsFilter(nextFilter)
    setPage(1)
  }

  const handleModelFilterChange = (nextValue: string) => {
    setModelFilter(nextValue)
  }

  const handleDensityChange = (nextDensity: SkillTableDensity) => {
    setTableDensity(nextDensity)
  }

  const handleColumnVisibilityToggle = (columnKey: string) => {
    if (!isSkillOptionalColumnKey(columnKey)) return

    if (
      visibleOptionalColumns.includes(columnKey)
      && isSkillTableSortField(columnKey)
      && sortState.field === columnKey
    ) {
      setSortState({})
      setPage(1)
    }

    setVisibleOptionalColumns((current) => {
      return current.includes(columnKey)
        ? current.filter((key) => key !== columnKey)
        : SKILL_OPTIONAL_COLUMN_KEYS.filter((key) => current.includes(key) || key === columnKey)
    })
  }

  const handleTableChange: TableProps<SkillSummary>["onChange"] = (
    _pagination,
    _filters,
    sorter
  ) => {
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

  const deleteMutation = useMutation({
    mutationFn: ({ name, version }: DeleteSkillPayload) =>
      tldwClient.deleteSkill(name, version),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      setSuccessAction(null)
      notification.success({
        message: t("option:skills.deleteSuccess", { defaultValue: "Skill deleted" })
      })
    },
    onError: (err: unknown) => {
      if (isConflictError(err)) {
        queryClient.invalidateQueries({ queryKey: ["skills"] })
        notification.error({
          message: t("option:skills.deleteConflict", {
            defaultValue: "Skill changed elsewhere"
          }),
          description: t("option:skills.deleteConflictDesc", {
            defaultValue: "Reload skills before deleting this version."
          })
        })
        return
      }
      notification.error({
        message: t("option:skills.deleteError", { defaultValue: "Failed to delete skill" }),
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
    mutationFn: (payload: {
      name?: string
      content: string
    }) => tldwClient.previewSkillImport(payload),
    onSuccess: (result) => {
      setImportTextPreview(result)
      importTextForm.setFieldValue("overwrite", false)
    },
    onError: (err: unknown) => {
      notification.error({
        message: t("option:skills.importPreviewError", {
          defaultValue: "Failed to review skill import"
        }),
        description: getErrorDescription(err)
      })
    }
  })

  const importTextMutation = useMutation({
    mutationFn: (payload: {
      name?: string
      content: string
      overwrite?: boolean
    }) => tldwClient.importSkill(payload),
    onSuccess: (result, variables) => {
      setImportTextOpen(false)
      setImportTextPreview(null)
      importTextForm.resetFields()
      showImportSuccess(result, variables.name)
    },
    onError: (err: unknown) => {
      notification.error({
        message: t("option:skills.importError", { defaultValue: "Failed to import skill" }),
        description: getErrorDescription(err)
      })
    }
  })

  const previewImportFileMutation = useMutation({
    mutationFn: (file: File) => tldwClient.previewSkillImportFile(file),
    onSuccess: (preview, file) => {
      setSuccessAction(null)
      setFileImportReview({ file, preview, overwrite: false })
    },
    onError: (err: unknown) => {
      notification.error({
        message: t("option:skills.importPreviewError", {
          defaultValue: "Failed to review skill import"
        }),
        description: getErrorDescription(err)
      })
    }
  })

  const importFileMutation = useMutation({
    mutationFn: ({ file, overwrite }: { file: File; overwrite: boolean }) =>
      tldwClient.importSkillFile(file, { overwrite }),
    onSuccess: (result) => {
      setFileImportReview(null)
      showImportSuccess(result)
    },
    onError: (err: unknown) => {
      notification.error({
        message: t("option:skills.importError", { defaultValue: "Failed to import skill" }),
        description: getErrorDescription(err)
      })
    }
  })

  const seedBuiltinsMutation = useMutation({
    mutationFn: (overwrite: boolean = false) => tldwClient.seedSkills({ overwrite }),
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
      notification.error({
        message: t("option:skills.seedError", { defaultValue: "Failed to seed built-in skills" }),
        description: getErrorDescription(err)
      })
    }
  })

  const confirmSeedOverwrite = () => {
    Modal.confirm({
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
      onOk: () => seedBuiltinsMutation.mutateAsync(true)
    })
  }

  const handleNew = () => {
    setSuccessAction(null)
    setEditingSkill(null)
    setDrawerOpen(true)
  }

  const handleEdit = async (name: string) => {
    try {
      const skill = await tldwClient.getSkill(name)
      setEditingSkill(skill)
      setDrawerOpen(true)
    } catch (err: unknown) {
      notification.error({
        message: t("option:skills.loadError", { defaultValue: "Failed to load skill" }),
        description: getErrorDescription(err)
      })
    }
  }

  const handleDelete = (
    skill: Pick<SkillSummary, "name"> & Partial<Pick<SkillSummary, "version">>
  ) => {
    Modal.confirm({
      title: t("option:skills.deleteConfirmTitle", {
        defaultValue: "Delete skill?"
      }),
      content: t("option:skills.deleteConfirmContent", {
        defaultValue: `Are you sure you want to delete "${skill.name}"? This cannot be undone.`,
        name: skill.name
      }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => deleteMutation.mutateAsync({
        name: skill.name,
        version: Number.isSafeInteger(skill.version) && Number(skill.version) > 0
          ? skill.version
          : undefined
      }).catch((err: unknown) => {
        if (isConflictError(err)) return
        throw err
      })
    })
  }

  const handleExport = async (name: string) => {
    try {
      const blob = await tldwClient.exportSkill(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${name}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err: unknown) {
      notification.error({
        message: t("option:skills.exportError", { defaultValue: "Failed to export skill" }),
        description: getErrorDescription(err)
      })
    }
  }

  const handleImportFile = async (file: File) => {
    previewImportFileMutation.mutate(file)
    return false // prevent antd Upload default behavior
  }

  const openImportTextModal = () => {
    setSuccessAction(null)
    setImportTextPreview(null)
    importTextForm.resetFields()
    importTextForm.setFieldsValue({ overwrite: false, content: "" })
    setImportTextOpen(true)
  }

  const handleImportTextSubmit = async () => {
    try {
      const values = await importTextForm.validateFields()
      const payload: {
        name?: string
        content: string
        overwrite?: boolean
      } = {
        content: values.content
      }
      const trimmedName = values.name?.trim()
      if (trimmedName) {
        payload.name = trimmedName
      }
      if (!importTextPreview?.valid) {
        await previewImportTextMutation.mutateAsync(payload)
        return
      }
      if (importTextPreview.conflict && !values.overwrite) {
        return
      }
      payload.overwrite = importTextPreview.conflict
        ? Boolean(values.overwrite)
        : false
      await importTextMutation.mutateAsync(payload)
    } catch {
      // validation errors handled by antd
    }
  }

  const handleDrawerClose = () => {
    setDrawerOpen(false)
    setEditingSkill(null)
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
    model_invocation: t("option:skills.colModelUse", { defaultValue: "Model use" })
  }

  const columnVisibilityMenuItems: MenuProps["items"] = SKILL_OPTIONAL_COLUMN_KEYS.map(
    (key) => ({
      key,
      label: optionalColumnLabels[key]
    })
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
    {
      title: t("option:skills.colActions", { defaultValue: "Actions" }),
      key: "actions",
      width: 180,
      render: (_: unknown, record: SkillSummary) => (
        <div className="flex items-center gap-1">
          <Tooltip title={t("option:skills.testRun", { defaultValue: "Test run" })}>
            <Button
              aria-label={t("option:skills.testRunSkill", {
                defaultValue: `Test run ${record.name}`,
                name: record.name
              })}
              type="text"
              size="small"
              icon={<Play size={14} />}
              onClick={() => setPreviewSkill(record.name)}
            />
          </Tooltip>
          <Tooltip title={t("common:edit", { defaultValue: "Edit" })}>
            <Button
              aria-label={t("option:skills.editSkill", {
                defaultValue: `Edit ${record.name}`,
                name: record.name
              })}
              type="text"
              size="small"
              icon={<Pen size={14} />}
              onClick={() => handleEdit(record.name)}
            />
          </Tooltip>
          <Tooltip title={t("option:skills.export", { defaultValue: "Export" })}>
            <Button
              aria-label={t("option:skills.exportSkill", {
                defaultValue: `Export ${record.name}`,
                name: record.name
              })}
              type="text"
              size="small"
              icon={<Download size={14} />}
              onClick={() => handleExport(record.name)}
            />
          </Tooltip>
          <Tooltip title={t("common:delete", { defaultValue: "Delete" })}>
            <Button
              aria-label={t("option:skills.deleteSkill", {
                defaultValue: `Delete ${record.name}`,
                name: record.name
              })}
              type="text"
              size="small"
              danger
              icon={<Trash2 size={14} />}
              onClick={() => handleDelete(record)}
            />
          </Tooltip>
        </div>
      )
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
        <Upload
          accept=".md,.zip"
          showUploadList={false}
          beforeUpload={handleImportFile}
        >
          <span className="flex items-center gap-2">
            <FileDown size={14} />
            {t("option:skills.importFile", { defaultValue: "Import File (.md/.zip)" })}
          </span>
        </Upload>
      )
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
        <Button icon={<Plus size={14} />} onClick={handleNew}>
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
    && (!importTextPreview?.conflict || Boolean(importTextOverwrite))
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
                    {t("option:skills.description", { defaultValue: "Description" })}
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
                    {t("option:skills.allowedTools", { defaultValue: "Allowed tools" })}
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
                      "Importing will replace the active skill only if overwrite is enabled."
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
    <div className="flex flex-col gap-4">
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
            {skillCountLabel}
          </p>
        </div>
      </section>

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <Input.Search
          placeholder={t("option:skills.searchPlaceholder", {
            defaultValue: "Search skills..."
          })}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          allowClear
          style={{ maxWidth: 360 }}
        />
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
          <Button type="primary" icon={<Plus size={14} />} onClick={handleNew}>
            {t("option:skills.newSkill", { defaultValue: "New Skill" })}
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div
          role="group"
          aria-label={t("option:skills.contextFilter", {
            defaultValue: "Skill mode filter"
          })}
          className="flex flex-wrap items-center gap-1"
        >
          <Button
            size="small"
            type={contextFilter === "all" ? "primary" : "default"}
            aria-pressed={contextFilter === "all"}
            onClick={() => handleContextFilterChange("all")}
          >
            {t("option:skills.filterAllModes", { defaultValue: "All modes" })}
          </Button>
          <Button
            size="small"
            type={contextFilter === "inline" ? "primary" : "default"}
            aria-pressed={contextFilter === "inline"}
            onClick={() => handleContextFilterChange("inline")}
          >
            {t("option:skills.filterInline", { defaultValue: "Inline" })}
          </Button>
          <Button
            size="small"
            type={contextFilter === "fork" ? "primary" : "default"}
            aria-pressed={contextFilter === "fork"}
            onClick={() => handleContextFilterChange("fork")}
          >
            {t("option:skills.filterFork", { defaultValue: "Fork" })}
          </Button>
        </div>
        <div
          role="group"
          aria-label={t("option:skills.visibilityFilter", {
            defaultValue: "Skill visibility filter"
          })}
          className="flex flex-wrap items-center gap-1"
        >
          <Button
            size="small"
            type={visibilityFilter === "visible" ? "primary" : "default"}
            aria-pressed={visibilityFilter === "visible"}
            onClick={() => handleVisibilityFilterChange("visible")}
          >
            {t("option:skills.filterVisible", { defaultValue: "Visible" })}
          </Button>
          <Button
            size="small"
            type={visibilityFilter === "hidden" ? "primary" : "default"}
            aria-pressed={visibilityFilter === "hidden"}
            onClick={() => handleVisibilityFilterChange("hidden")}
          >
            {t("option:skills.filterHidden", { defaultValue: "Hidden" })}
          </Button>
          <Button
            size="small"
            type={visibilityFilter === "all" ? "primary" : "default"}
            aria-pressed={visibilityFilter === "all"}
            onClick={() => handleVisibilityFilterChange("all")}
          >
            {t("option:skills.filterAllVisibility", { defaultValue: "All visibility" })}
          </Button>
        </div>
        <div
          role="group"
          aria-label={t("option:skills.toolsFilter", {
            defaultValue: "Skill tools filter"
          })}
          className="flex flex-wrap items-center gap-1"
        >
          <Button
            size="small"
            type={toolsFilter === "any" ? "primary" : "default"}
            aria-pressed={toolsFilter === "any"}
            onClick={() => handleToolsFilterChange("any")}
          >
            {t("option:skills.filterAnyTools", { defaultValue: "Any tools" })}
          </Button>
          <Button
            size="small"
            type={toolsFilter === "with-tools" ? "primary" : "default"}
            aria-pressed={toolsFilter === "with-tools"}
            onClick={() => handleToolsFilterChange("with-tools")}
          >
            {t("option:skills.filterHasTools", { defaultValue: "Has tools" })}
          </Button>
          <Button
            size="small"
            type={toolsFilter === "without-tools" ? "primary" : "default"}
            aria-pressed={toolsFilter === "without-tools"}
            onClick={() => handleToolsFilterChange("without-tools")}
          >
            {t("option:skills.filterNoTools", { defaultValue: "No tools" })}
          </Button>
        </div>
        <Input
          aria-label={t("option:skills.modelFilter", {
            defaultValue: "Filter by model"
          })}
          placeholder={t("option:skills.modelFilterPlaceholder", {
            defaultValue: "Model"
          })}
          value={modelFilter}
          onChange={(event) => handleModelFilterChange(event.target.value)}
          allowClear
          size="small"
          style={{ width: 160 }}
        />
        <div
          role="group"
          aria-label={t("option:skills.tableDensity", {
            defaultValue: "Table density"
          })}
          className="flex flex-wrap items-center gap-1"
        >
          <Button
            size="small"
            type={tableDensity === "comfortable" ? "primary" : "default"}
            aria-label={t("option:skills.comfortableDensity", {
              defaultValue: "Comfortable density"
            })}
            aria-pressed={tableDensity === "comfortable"}
            icon={<Rows3 size={14} />}
            onClick={() => handleDensityChange("comfortable")}
          >
            {t("option:skills.densityComfortable", { defaultValue: "Comfortable" })}
          </Button>
          <Button
            size="small"
            type={tableDensity === "compact" ? "primary" : "default"}
            aria-label={t("option:skills.compactDensity", {
              defaultValue: "Compact density"
            })}
            aria-pressed={tableDensity === "compact"}
            icon={<Rows3 size={14} />}
            onClick={() => handleDensityChange("compact")}
          >
            {t("option:skills.densityCompact", { defaultValue: "Compact" })}
          </Button>
        </div>
        <Dropdown
          menu={{
            items: columnVisibilityMenuItems,
            selectable: true,
            multiple: true,
            selectedKeys: visibleOptionalColumns,
            onClick: ({ key }) => handleColumnVisibilityToggle(key)
          }}
          trigger={["click"]}
        >
          <Button
            size="small"
            aria-label={t("option:skills.columnVisibility", {
              defaultValue: "Column visibility"
            })}
            icon={<Columns3 size={14} />}
          >
            {t("option:skills.columns", { defaultValue: "Columns" })}
          </Button>
        </Dropdown>
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
                  onClick={() => setPreviewSkill(skillName)}
                >
                  {successAction.testLabel ??
                    t("option:skills.testRun", { defaultValue: "Test run" })}
                </Button>
                <Button
                  size="small"
                  icon={<Pen size={14} />}
                  onClick={() => void handleEdit(skillName)}
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

      <Table
        data-testid="skills-table"
        data-density={tableDensity}
        dataSource={data?.skills ?? []}
        columns={columns}
        rowKey="name"
        loading={isLoading}
        onChange={handleTableChange}
        pagination={false}
        size={tableSize}
        locale={{
          emptyText: tableEmptyText
        }}
      />

      {totalSkills > pageSize && (
        <div className="flex justify-end">
          <Pagination
            current={page}
            pageSize={pageSize}
            total={totalSkills}
            onChange={(p, ps) => {
              setPage(p)
              setPageSize(ps)
            }}
            showSizeChanger
            pageSizeOptions={["10", "20", "50"]}
          />
        </div>
      )}

      <Modal
        title={t("option:skills.importTextTitle", {
          defaultValue: "Import Skill from Text"
        })}
        open={importTextOpen}
        onCancel={() => {
          setImportTextOpen(false)
          setImportTextPreview(null)
        }}
        onOk={handleImportTextSubmit}
        okText={importTextOkLabel}
        okButtonProps={{
          "aria-label": importTextOkLabel,
          loading:
            importTextMutation.isPending
            || (!importTextPreview && previewImportTextMutation.isPending),
          disabled: Boolean(importTextPreview?.valid) && !importTextCanSubmit
        }}
        destroyOnHidden
      >
        <Form
          form={importTextForm}
          layout="vertical"
          initialValues={{ overwrite: false }}
          autoComplete="off"
          onValuesChange={(changedValues) => {
            if ("name" in changedValues || "content" in changedValues) {
              setImportTextPreview(null)
              importTextForm.setFieldValue("overwrite", false)
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
                  "Required because this import matches an active skill name."
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
        onCancel={() => setFileImportReview(null)}
        onOk={() => {
          if (!fileImportReview?.preview.valid) return
          if (fileImportReview.preview.conflict && !fileImportReview.overwrite) return
          importFileMutation.mutate({
            file: fileImportReview.file,
            overwrite: fileImportReview.preview.conflict
              ? fileImportReview.overwrite
              : false
          })
        }}
        okText={importFileOkLabel}
        okButtonProps={{
          "aria-label": importFileOkLabel,
          loading: importFileMutation.isPending,
          disabled:
            !fileImportReview?.preview.valid
            || Boolean(fileImportReview.preview.conflict && !fileImportReview.overwrite)
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
        onClose={handleDrawerClose}
        onSaved={handleDrawerSaved}
      />

      <SkillPreview
        skillName={previewSkill}
        onClose={() => setPreviewSkill(null)}
      />
    </div>
  )
}
