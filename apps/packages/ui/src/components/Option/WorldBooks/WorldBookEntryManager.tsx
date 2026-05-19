import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Button, Form, Input, InputNumber, Modal, Skeleton, Switch, Table, Tooltip, Tag, Select, Empty, Popover, Grid, Progress } from "antd"
import React from "react"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { Pen, Trash2 } from "lucide-react"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { parseBulkEntries, SUPPORTED_BULK_SEPARATORS } from "./entryParsers"
import {
  formatEntryContentStats,
  estimateEntryTokens,
  getPriorityBand,
  getPriorityTagColor,
  normalizeKeywordList,
  validateRegexKeywords
} from "./worldBookEntryUtils"
import {
  DEFAULT_BULK_ADD_CONCURRENCY,
  type BulkAddFailure,
  type BulkAddProgress,
  runBulkAddEntries
} from "./worldBookBulkUtils"
import {
  buildBulkSetPriorityPayload,
  clampBulkPriority,
  normalizeBulkEntryIds
} from "./worldBookBulkActionUtils"
// worldBookStatsUtils consumed transitively by WorldBookBudgetBar
import { buildReferencedBySignalMap } from "./worldBookRelationshipUtils"
import { WorldBookBudgetBar } from "./WorldBookBudgetBar"

const normalizeKeywords = (value: any): string[] => {
  return normalizeKeywordList(value)
}

const normalizeEntryGroup = (value: unknown): string | null => {
  const normalized = String(value ?? "").trim()
  return normalized.length > 0 ? normalized : null
}

const buildMatchPreview = (keywordsValue: any, opts: { caseSensitive?: boolean; regexMatch?: boolean; wholeWord?: boolean }) => {
  const keyword = normalizeKeywords(keywordsValue)[0]
  if (!keyword) return "Add a keyword to see a preview."
  if (opts.regexMatch) {
    return `Regex enabled. Example pattern: /${keyword}/`
  }
  const lower = keyword.toLowerCase()
  const upper = keyword.toUpperCase()
  const caseExample = opts.caseSensitive ? `'${keyword}' only` : `'${lower}', '${upper}'`
  const wordExample = opts.wholeWord ? "whole-word matches" : "partial matches"
  return `Preview: ${caseExample}; ${wordExample}.`
}

// Helper component for form field labels with tooltips
const LabelWithHelp: React.FC<{ label: string; help: string }> = ({ label, help }) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <Tooltip title={help}>
      <span className="w-4 h-4 text-text-muted cursor-help inline-flex items-center">?</span>
    </Tooltip>
  </span>
)

// Keyword preview component for real-time feedback
const KeywordPreview: React.FC<{ value?: unknown }> = ({ value }) => {
  const keywords = normalizeKeywordList(value)
  if (keywords.length === 0) return null
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {keywords.map((k, i) => <Tag key={i}>{k}</Tag>)}
    </div>
  )
}

const ACCESSIBLE_SWITCH_TEXT_PROPS = {
  checkedChildren: "On",
  unCheckedChildren: "Off"
} as const

const MODAL_BODY_SCROLL_STYLE: React.CSSProperties = {
  maxHeight: "80vh",
  overflowY: "auto"
}

const FALLBACK_AI_GENERATION_MODEL = "gpt-4o-mini"
const DEFAULT_AI_GENERATION_COUNT = 3
const MIN_AI_GENERATION_COUNT = 1
const MAX_AI_GENERATION_COUNT = 8

type WorldBookGeneratedEntryDraft = {
  id: string
  keywordsText: string
  content: string
  priority: number
  group: string
}

const BULK_ENTRY_FORMAT_EXAMPLES = [
  { separator: "=>", example: "keyword1, keyword2 => content" },
  { separator: "->", example: "keyword1, keyword2 -> content" },
  { separator: "|", example: "keyword1, keyword2 | content" },
  { separator: "\t", example: "keyword1, keyword2<TAB>content" }
] as const

const toWorldBookGenerationString = (value: unknown): string | null => {
  if (typeof value === "string") {
    const normalized = value.trim()
    return normalized.length > 0 ? normalized : null
  }
  if (typeof value === "number" && Number.isFinite(value)) return String(value)
  return null
}

const toWorldBookGenerationModel = (value: unknown): string | null => {
  const direct = toWorldBookGenerationString(value)
  if (direct) return direct
  if (!value || typeof value !== "object") return null
  const modelRecord = value as Record<string, unknown>
  return (
    toWorldBookGenerationString(modelRecord.name) ||
    toWorldBookGenerationString(modelRecord.id) ||
    toWorldBookGenerationString(modelRecord.model_id) ||
    toWorldBookGenerationString(modelRecord.display_name)
  )
}

const pickWorldBookGenerationModel = (provider: unknown): string | null => {
  if (!provider || typeof provider !== "object") return null
  const providerRecord = provider as Record<string, unknown>
  const modelsRaw = Array.isArray(providerRecord.models)
    ? providerRecord.models
    : Array.isArray(providerRecord.models_info)
      ? providerRecord.models_info
      : []
  for (const modelEntry of modelsRaw) {
    const model = toWorldBookGenerationModel(modelEntry)
    if (model) return model
  }
  return null
}

const resolveWorldBookGenerationProviderConfig = (
  payload: unknown
): { provider?: string; model: string } => {
  const result: { provider?: string; model: string } = {
    model: FALLBACK_AI_GENERATION_MODEL
  }
  if (!payload || typeof payload !== "object") return result

  const root = payload as Record<string, unknown>
  const providersRaw = Array.isArray(root.providers) ? root.providers : []
  const defaultProviderName = toWorldBookGenerationString(root.default_provider)

  const providers = providersRaw
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null
      const providerRecord = entry as Record<string, unknown>
      const name =
        toWorldBookGenerationString(providerRecord.name) ||
        toWorldBookGenerationString(providerRecord.provider)
      if (!name) return null
      return {
        name,
        model: pickWorldBookGenerationModel(providerRecord)
      }
    })
    .filter((entry): entry is { name: string; model: string | null } => entry !== null)

  const selected =
    providers.find((provider) => provider.name === defaultProviderName) || providers[0] || null
  if (!selected) return result

  result.provider = selected.name
  result.model = selected.model || result.model
  return result
}

const normalizeWorldBookCompletionText = (candidate: unknown): string => {
  if (typeof candidate === "string") return candidate.trim()
  if (Array.isArray(candidate)) {
    return candidate
      .map((part) => {
        if (typeof part === "string") return part
        if (!part || typeof part !== "object") return ""
        const partRecord = part as Record<string, unknown>
        return toWorldBookGenerationString(partRecord.text) || ""
      })
      .join("\n")
      .trim()
  }
  return ""
}

export type EntryFilterPreset = {
  enabledFilter: "all" | "enabled" | "disabled"
  matchFilter: "all" | "regex" | "plain"
  searchText: string
}

export const DEFAULT_ENTRY_FILTER_PRESET: EntryFilterPreset = {
  enabledFilter: "all",
  matchFilter: "all",
  searchText: ""
}

export const WorldBookEntryManager: React.FC<{
  worldBookId: number
  worldBookName?: string
  tokenBudget?: number | null
  worldBooks: Array<{ id?: number; name?: string }>
  entryFilterPreset?: EntryFilterPreset
  form: any
}> = ({
  worldBookId,
  worldBookName,
  tokenBudget,
  worldBooks,
  entryFilterPreset = DEFAULT_ENTRY_FILTER_PRESET,
  form
}) => {
  const readSessionBoolean = React.useCallback((key: string, fallback: boolean) => {
    try {
      const raw = sessionStorage.getItem(key)
      if (raw == null) return fallback
      return raw === "true"
    } catch {
      return fallback
    }
  }, [])
  const writeSessionBoolean = React.useCallback((key: string, value: boolean) => {
    try {
      sessionStorage.setItem(key, String(value))
    } catch {
      // noop
    }
  }, [])

  const qc = useQueryClient()
  const screens = Grid.useBreakpoint()
  const notification = useAntdNotification()
  const confirmDanger = useConfirmDanger()
  const isEntryManagerMobile = !screens.md
  const entryActionButtonSize = isEntryManagerMobile ? "middle" : "small"
  const entryActionButtonClassName = isEntryManagerMobile
    ? "min-h-11 min-w-11 px-2 inline-flex items-center justify-center"
    : undefined
  const [editingEntry, setEditingEntry] = React.useState<any | null>(null)
  const [editForm] = Form.useForm()
  const [bulkMode, setBulkMode] = React.useState(false)
  const [bulkText, setBulkText] = React.useState('')
  const [bulkAdding, setBulkAdding] = React.useState(false)
  const [bulkProgress, setBulkProgress] = React.useState<BulkAddProgress | null>(null)
  const [bulkFailures, setBulkFailures] = React.useState<BulkAddFailure[]>([])
  const [selectedRowKeys, setSelectedRowKeys] = React.useState<React.Key[]>([])
  const [bulkPriorityValue, setBulkPriorityValue] = React.useState(50)
  const [bulkPriorityPopoverOpen, setBulkPriorityPopoverOpen] = React.useState(false)
  const [bulkMoveTargetId, setBulkMoveTargetId] = React.useState<number | null>(null)
  const [bulkMoveStrategy, setBulkMoveStrategy] = React.useState<"skip_existing" | "duplicate">(
    "skip_existing"
  )
  const [bulkMovePopoverOpen, setBulkMovePopoverOpen] = React.useState(false)
  const [bulkMovePending, setBulkMovePending] = React.useState(false)
  const [entrySearch, setEntrySearch] = React.useState(entryFilterPreset.searchText)
  const [entryEnabledFilter, setEntryEnabledFilter] = React.useState<"all" | "enabled" | "disabled">(
    entryFilterPreset.enabledFilter
  )
  const [entryMatchFilter, setEntryMatchFilter] = React.useState<"all" | "regex" | "plain">(
    entryFilterPreset.matchFilter
  )
  const [entryGroupFilter, setEntryGroupFilter] = React.useState<string>("all")
  const [openAiGenerate, setOpenAiGenerate] = React.useState(false)
  const [aiPrompt, setAiPrompt] = React.useState("")
  const [aiDefaultGroup, setAiDefaultGroup] = React.useState("")
  const [aiEntryCount, setAiEntryCount] = React.useState(DEFAULT_AI_GENERATION_COUNT)
  const [aiGenerating, setAiGenerating] = React.useState(false)
  const [aiApplying, setAiApplying] = React.useState(false)
  const [aiError, setAiError] = React.useState<string | null>(null)
  const [aiSuggestions, setAiSuggestions] = React.useState<WorldBookGeneratedEntryDraft[]>([])
  const [aiRunInfo, setAiRunInfo] = React.useState<{ provider?: string; model: string } | null>(
    null
  )
  const [addMatchingOptionsOpen, setAddMatchingOptionsOpen] = React.useState<boolean>(() =>
    readSessionBoolean("worldbooks:add-matching-options-open", false)
  )
  const [editMatchingOptionsOpen, setEditMatchingOptionsOpen] = React.useState<boolean>(() =>
    readSessionBoolean("worldbooks:edit-matching-options-open", false)
  )
  const [keywordIndexOpen, setKeywordIndexOpen] = React.useState(false)
  const [bulkFormatsOpen, setBulkFormatsOpen] = React.useState(false)
  const keywordIndexContentId = React.useId()
  const editMatchingOptionsContentId = React.useId()
  const addMatchingOptionsContentId = React.useId()
  const bulkFormatsContentId = React.useId()
  const addRegexMatch = Form.useWatch('regex_match', form)
  const addCaseSensitive = Form.useWatch('case_sensitive', form)
  const addWholeWord = Form.useWatch('whole_word_match', form)
  const addKeywordsWatch = Form.useWatch('keywords', form)
  const addContentWatch = Form.useWatch('content', form)
  const editRegexMatch = Form.useWatch('regex_match', editForm)
  const editCaseSensitive = Form.useWatch('case_sensitive', editForm)
  const editWholeWord = Form.useWatch('whole_word_match', editForm)
  const editKeywordsWatch = Form.useWatch('keywords', editForm)
  const editContentWatch = Form.useWatch('content', editForm)

  React.useEffect(() => {
    setEntryEnabledFilter(entryFilterPreset.enabledFilter)
    setEntryMatchFilter(entryFilterPreset.matchFilter)
    setEntrySearch(entryFilterPreset.searchText)
    setEntryGroupFilter("all")
  }, [
    entryFilterPreset.enabledFilter,
    entryFilterPreset.matchFilter,
    entryFilterPreset.searchText,
    worldBookId
  ])

  const { data: entryQueryData, status } = useQuery({
    queryKey: ['tldw:listWorldBookEntries', worldBookId],
    queryFn: async () => {
      await tldwClient.initialize()
      const res = await tldwClient.listWorldBookEntries(worldBookId, false)
      return {
        entries: Array.isArray(res?.entries) ? res.entries : [],
        total: Number.isFinite(Number(res?.total))
          ? Number(res.total)
          : Array.isArray(res?.entries)
            ? res.entries.length
            : 0
      }
    }
  })
  const { entries, totalEntryCount } = React.useMemo(() => {
    if (Array.isArray(entryQueryData)) {
      return { entries: entryQueryData, totalEntryCount: entryQueryData.length }
    }
    const normalizedEntries = Array.isArray((entryQueryData as any)?.entries)
      ? (entryQueryData as any).entries
      : []
    const parsedTotal = Number((entryQueryData as any)?.total)
    const normalizedTotal =
      Number.isFinite(parsedTotal) && parsedTotal >= normalizedEntries.length
        ? parsedTotal
        : normalizedEntries.length
    return { entries: normalizedEntries, totalEntryCount: normalizedTotal }
  }, [entryQueryData])
  const configuredTokenBudget = React.useMemo(() => {
    const value = Number(tokenBudget)
    if (!Number.isFinite(value) || value <= 0) return null
    return value
  }, [tokenBudget])
  const estimatedEntryTokens = React.useMemo(
    () =>
      (entries || []).reduce(
        (total: number, entry: any) => total + estimateEntryTokens(entry?.content),
        0
      ),
    [entries]
  )
  const addFormProjectedTokens = React.useMemo(
    () => estimateEntryTokens(addContentWatch),
    [addContentWatch]
  )
  const addFormProjectedTotal = estimatedEntryTokens + addFormProjectedTokens
  const editFormProjectedTokens = React.useMemo(
    () => estimateEntryTokens(editContentWatch),
    [editContentWatch]
  )
  const editingEntryTokens = React.useMemo(
    () => estimateEntryTokens(editingEntry?.content),
    [editingEntry?.content]
  )
  const editFormProjectedTotal = Math.max(
    0,
    estimatedEntryTokens - editingEntryTokens + editFormProjectedTokens
  )
  const { mutate: addEntry, mutateAsync: addEntryAsync, isPending: adding } = useMutation({
    mutationFn: (v: any) =>
      tldwClient.addWorldBookEntry(worldBookId, {
        ...v,
        keywords: normalizeKeywords(v.keywords),
        group: normalizeEntryGroup(v?.group)
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tldw:listWorldBookEntries', worldBookId] }); form.resetFields() },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Failed to add entry' })
  })
  const { mutate: updateEntry, isPending: updating } = useMutation({
    mutationFn: (v: any) =>
      editingEntry
        ? tldwClient.updateWorldBookEntry(editingEntry.entry_id, {
            ...v,
            keywords: normalizeKeywords(v.keywords),
            group: normalizeEntryGroup(v?.group)
          })
        : Promise.resolve(null),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tldw:listWorldBookEntries', worldBookId] }); setEditingEntry(null); editForm.resetFields() },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Failed to update entry' })
  })
  const { mutate: deleteEntry } = useMutation({
    mutationFn: (id: number) => tldwClient.deleteWorldBookEntry(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tldw:listWorldBookEntries', worldBookId] })
  })
  const { mutateAsync: bulkOperate, isPending: bulkPending } = useMutation({
    mutationFn: (payload: { entry_ids: number[]; operation: string; priority?: number }) =>
      tldwClient.bulkWorldBookEntries(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tldw:listWorldBookEntries', worldBookId] })
      setSelectedRowKeys([])
    },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Bulk operation failed' })
  })

  const openEditModal = (entry: any) => {
    setEditingEntry(entry)
    const appendableValue =
      typeof entry?.appendable === "boolean"
        ? entry.appendable
        : Boolean(entry?.metadata?.appendable)
    editForm.setFieldsValue({
      keywords: entry.keywords || [],
      content: entry.content,
      group: normalizeEntryGroup(entry?.group ?? entry?.metadata?.group) || undefined,
      priority: entry.priority,
      enabled: entry.enabled,
      appendable: appendableValue,
      case_sensitive: entry.case_sensitive,
      regex_match: entry.regex_match,
      whole_word_match: entry.whole_word_match
    })
  }

  const keywordIndex = React.useMemo(() => {
    const map = new Map<string, { count: number; contentVariants: Set<string> }>()
    ;(entries || []).forEach((entry: any) => {
      ;(entry.keywords || []).forEach((kw: string) => {
        const key = String(kw).trim()
        if (!key) return
        const current = map.get(key) || { count: 0, contentVariants: new Set<string>() }
        current.count += 1
        current.contentVariants.add(entry.content || '')
        map.set(key, current)
      })
    })
    return Array.from(map.entries())
      .map(([keyword, info]) => ({
        keyword,
        count: info.count,
        conflict: info.contentVariants.size > 1,
        variantCount: info.contentVariants.size
      }))
      .sort((a, b) => b.count - a.count)
  }, [entries])
  const keywordConflictCount = React.useMemo(
    () => keywordIndex.filter((item) => item.conflict).length,
    [keywordIndex]
  )
  const entryById = React.useMemo(() => {
    const map = new Map<number, any>()
    ;(entries || []).forEach((entry: any) => {
      const id = Number(entry?.entry_id)
      if (Number.isFinite(id) && id > 0) map.set(id, entry)
    })
    return map
  }, [entries])
  const referencedBySignalMap = React.useMemo(
    () => buildReferencedBySignalMap(entries as any[]),
    [entries]
  )

  const entryGroupOptions = React.useMemo(() => {
    const groups = new Set<string>()
    ;(entries || []).forEach((entry: any) => {
      const group = normalizeEntryGroup(entry?.group ?? entry?.metadata?.group)
      if (group) groups.add(group)
    })
    return Array.from(groups).sort((a, b) => a.localeCompare(b))
  }, [entries])

  const filteredEntryData = React.useMemo(() => {
    const source = Array.isArray(entries) ? entries : []
    const query = entrySearch.trim().toLowerCase()
    let next = source

    if (query) {
      next = next.filter((entry: any) => {
        const content = String(entry?.content || "").toLowerCase()
        const keywords = normalizeKeywords(entry?.keywords).join(" ").toLowerCase()
        return content.includes(query) || keywords.includes(query)
      })
    }

    if (entryEnabledFilter !== "all") {
      const enabled = entryEnabledFilter === "enabled"
      next = next.filter((entry: any) => Boolean(entry?.enabled) === enabled)
    }

    if (entryMatchFilter !== "all") {
      const requiresRegex = entryMatchFilter === "regex"
      next = next.filter((entry: any) => Boolean(entry?.regex_match) === requiresRegex)
    }

    if (entryGroupFilter !== "all") {
      next = next.filter((entry: any) => {
        const group = normalizeEntryGroup(entry?.group ?? entry?.metadata?.group)
        return group === entryGroupFilter
      })
    }

    return next
  }, [entries, entryEnabledFilter, entryGroupFilter, entryMatchFilter, entrySearch])

  const filteredEntryIds = React.useMemo(
    () => normalizeBulkEntryIds(filteredEntryData.map((entry: any) => entry?.entry_id)),
    [filteredEntryData]
  )
  const visibleEntryCount = filteredEntryData.length
  const selectedEntryIds = React.useMemo(
    () => normalizeBulkEntryIds(selectedRowKeys),
    [selectedRowKeys]
  )
  const hasSelectedEntries = selectedEntryIds.length > 0
  const canEscalateSelectAll = hasSelectedEntries && selectedEntryIds.length < filteredEntryIds.length
  const moveDestinationOptions = React.useMemo(
    () =>
      (worldBooks || [])
        .filter((book) => Number(book?.id) !== worldBookId)
        .map((book) => ({
          label: String(book?.name || `World Book ${book?.id}`),
          value: Number(book?.id)
        }))
        .filter((book) => Number.isFinite(book.value) && book.value > 0),
    [worldBookId, worldBooks]
  )

  const bulkParse = React.useMemo(() => parseBulkEntries(bulkText), [bulkText])

  const getSelectedEntriesForMove = React.useCallback(() => {
    const sourceEntries = Array.isArray(entries) ? entries : []
    const selectedSet = new Set(selectedEntryIds)
    return sourceEntries.filter((entry: any) => selectedSet.has(Number(entry?.entry_id)))
  }, [entries, selectedEntryIds])

  const handleBulkAction = async (operation: "enable" | "disable" | "delete") => {
    if (!hasSelectedEntries) return
    if (operation === "delete") {
      const ok = await confirmDanger({
        title: "Delete selected entries?",
        content: `This will permanently remove ${selectedEntryIds.length} entries.`,
        okText: "Delete",
        cancelText: "Cancel"
      })
      if (!ok) return
    }
    try {
      const response = await bulkOperate({ entry_ids: selectedEntryIds, operation })
      const failedCount = Array.isArray(response?.failed_ids) ? response.failed_ids.length : 0
      const affectedCount = Number(response?.affected_count ?? selectedEntryIds.length - failedCount)

      if (failedCount > 0) {
        notification.warning({
          message: "Bulk action completed with errors",
          description: `${affectedCount} entries updated, ${failedCount} failed.`
        })
        return
      }

      notification.success({
        message: "Bulk action complete",
        description: `${affectedCount} entries updated.`
      })
    } catch {
      // Error notification is handled in the mutation onError callback.
    }
  }

  const handleSelectAllFilteredEntries = () => {
    if (filteredEntryIds.length === 0) return
    setSelectedRowKeys(filteredEntryIds)
  }

  const handleBulkSetPriority = async () => {
    const payload = buildBulkSetPriorityPayload(selectedEntryIds, bulkPriorityValue)
    if (payload.entry_ids.length === 0) return

    try {
      const response = await bulkOperate(payload)
      const failedCount = Array.isArray(response?.failed_ids) ? response.failed_ids.length : 0
      const affectedCount = Number(response?.affected_count ?? payload.entry_ids.length - failedCount)

      if (failedCount > 0) {
        notification.warning({
          message: "Bulk priority completed with errors",
          description: `${affectedCount} entries updated, ${failedCount} failed.`
        })
      } else {
        notification.success({
          message: "Bulk priority updated",
          description: `Set priority ${payload.priority} for ${affectedCount} entries.`
        })
      }

      setBulkPriorityPopoverOpen(false)
    } catch {
      // Error notification is handled in the mutation onError callback.
    }
  }

  const handleBulkMoveEntries = async () => {
    const destinationId = Number(bulkMoveTargetId)
    if (!Number.isFinite(destinationId) || destinationId <= 0) return

    const selectedEntries = getSelectedEntriesForMove()
    if (selectedEntries.length === 0) return

    const destinationBookName =
      moveDestinationOptions.find((book) => book.value === destinationId)?.label ||
      `World Book ${destinationId}`
    const ok = await confirmDanger({
      title: `Move ${selectedEntries.length} entries?`,
      content: `Entries will be moved from "${worldBookName || `World Book ${worldBookId}`}" to "${destinationBookName}".`,
      okText: "Move",
      cancelText: "Cancel"
    })
    if (!ok) return

    setBulkMovePending(true)
    try {
      const destinationEntriesResponse = await tldwClient.listWorldBookEntries(destinationId, false)
      const destinationEntries = Array.isArray(destinationEntriesResponse?.entries)
        ? destinationEntriesResponse.entries
        : []
      const destinationKeys = new Set(
        destinationEntries.map((entry: any) =>
          `${normalizeKeywordList(entry?.keywords).join("|")}::${String(entry?.content || "")}`
        )
      )

      let copiedCount = 0
      let skippedCount = 0
      const failedIds: number[] = []
      const copiedSourceIds: number[] = []

      for (const entry of selectedEntries) {
        const sourceEntryId = Number(entry?.entry_id)
        const normalizedKeywords = normalizeKeywordList(entry?.keywords)
        const content = String(entry?.content || "")
        const dedupeKey = `${normalizedKeywords.join("|")}::${content}`
        const destinationAlreadyHasEntry = destinationKeys.has(dedupeKey)
        if (bulkMoveStrategy === "skip_existing" && destinationAlreadyHasEntry) {
          skippedCount += 1
          continue
        }

        try {
          await tldwClient.addWorldBookEntry(destinationId, {
            keywords: normalizedKeywords,
            content,
            group: normalizeEntryGroup(entry?.group ?? entry?.metadata?.group),
            priority: clampBulkPriority(entry?.priority, 50),
            enabled: Boolean(entry?.enabled),
            case_sensitive: Boolean(entry?.case_sensitive),
            regex_match: Boolean(entry?.regex_match),
            whole_word_match:
              typeof entry?.whole_word_match === "boolean" ? entry.whole_word_match : true,
            appendable:
              typeof entry?.appendable === "boolean"
                ? entry.appendable
                : Boolean(entry?.metadata?.appendable)
          })
          copiedCount += 1
          if (Number.isFinite(sourceEntryId) && sourceEntryId > 0) copiedSourceIds.push(sourceEntryId)
          destinationKeys.add(dedupeKey)
        } catch {
          if (Number.isFinite(sourceEntryId) && sourceEntryId > 0) failedIds.push(sourceEntryId)
        }
      }

      let deletedCount = 0
      if (copiedSourceIds.length > 0) {
        const deleteResponse = await bulkOperate({ entry_ids: copiedSourceIds, operation: "delete" })
        const failedDeletes = Array.isArray(deleteResponse?.failed_ids) ? deleteResponse.failed_ids.length : 0
        const affectedDeletes = Number(deleteResponse?.affected_count ?? copiedSourceIds.length - failedDeletes)
        deletedCount = Math.max(0, affectedDeletes)
      }

      const failedCount = failedIds.length + Math.max(0, copiedCount - deletedCount)
      if (failedCount > 0 || skippedCount > 0) {
        notification.warning({
          message: "Move completed with warnings",
          description: `${deletedCount} moved, ${skippedCount} skipped, ${failedCount} failed.`
        })
      } else {
        notification.success({
          message: "Entries moved",
          description: `Moved ${deletedCount} entries to "${destinationBookName}".`
        })
      }

      setBulkMovePopoverOpen(false)
      setBulkMoveTargetId(null)
    } catch (error: any) {
      notification.error({
        message: "Move failed",
        description: error?.message || "Failed to move selected entries."
      })
    } finally {
      setBulkMovePending(false)
    }
  }

  const handleBulkAddEntries = async () => {
    if (bulkParse.entries.length === 0 || bulkParse.errors.length > 0) return
    setBulkAdding(true)
    setBulkFailures([])
    try {
      const result = await runBulkAddEntries({
        entries: bulkParse.entries,
        concurrency: DEFAULT_BULK_ADD_CONCURRENCY,
        addEntry: async (entry) => tldwClient.addWorldBookEntry(worldBookId, entry),
        onProgress: (progress) => setBulkProgress(progress)
      })

      setBulkFailures(result.failures)
      if (result.succeeded > 0) {
        qc.invalidateQueries({ queryKey: ['tldw:listWorldBookEntries', worldBookId] })
      }

      if (result.failed === 0) {
        notification.success({ message: `Added ${result.succeeded} entries` })
        setBulkText('')
      } else if (result.succeeded === 0) {
        notification.error({
          message: "Bulk add failed",
          description: "No entries were added. Review the failure summary below."
        })
      } else {
        notification.warning({
          message: "Bulk add completed with errors",
          description: `${result.succeeded} entries added, ${result.failed} failed.`
        })
      }
    } catch (error: any) {
      notification.error({ message: "Bulk add failed", description: error?.message || "Unexpected error" })
    } finally {
      setBulkAdding(false)
    }
  }

  const resolveAiGenerationConfig = React.useCallback(async () => {
    try {
      await tldwClient.initialize()
      const providersResponse = await tldwClient.getProviders()
      const payload = (providersResponse as any)?.data ?? providersResponse
      return resolveWorldBookGenerationProviderConfig(payload)
    } catch {
      return { model: FALLBACK_AI_GENERATION_MODEL }
    }
  }, [])

  const handleGenerateSuggestions = async () => {
    const topic = aiPrompt.trim()
    if (!topic) {
      setAiError("Add a topic or description before generating suggestions.")
      return
    }

    setAiGenerating(true)
    setAiError(null)

    try {
      const generationConfig = await resolveAiGenerationConfig()
      const preferredCount = clampBulkPriority(aiEntryCount, DEFAULT_AI_GENERATION_COUNT)
      const count = Math.min(MAX_AI_GENERATION_COUNT, Math.max(MIN_AI_GENERATION_COUNT, preferredCount))

      const defaultGroup = normalizeEntryGroup(aiDefaultGroup)
      const completionResponse = await tldwClient.createChatCompletion({
        model: generationConfig.model,
        api_provider: generationConfig.provider,
        temperature: 0.3,
        max_tokens: 900,
        messages: [
          {
            role: "system",
            content:
              "You generate world-book entry suggestions. Return plain text only with one suggestion per line. " +
              'Each line must be: \"keyword1, keyword2 -> content\". ' +
              "Do not include numbering, markdown, comments, or extra sections."
          },
          {
            role: "user",
            content:
              `Topic: ${topic}\n` +
              `Number of suggestions: ${count}\n` +
              `World book: ${worldBookName || `World Book ${worldBookId}`}\n` +
              (defaultGroup ? `Default group: ${defaultGroup}\n` : "") +
              "Keep each content sentence concise, specific, and immediately useful in chat context."
          }
        ]
      })
      const completionPayload = await completionResponse.json()
      const outputCandidate =
        completionPayload?.choices?.[0]?.message?.content ??
        completionPayload?.output ??
        completionPayload?.content
      const output = normalizeWorldBookCompletionText(outputCandidate)

      if (!output) {
        setAiSuggestions([])
        throw new Error("The model returned an empty result.")
      }

      const parsed = parseBulkEntries(output)
      const mappedSuggestions = parsed.entries.slice(0, count).map((entry, index) => ({
        id: `${Date.now()}-${entry.sourceLine}-${index}`,
        keywordsText: entry.keywords.join(", "),
        content: entry.content,
        priority: 50,
        group: defaultGroup || ""
      }))

      if (mappedSuggestions.length === 0) {
        const parseMessage = parsed.errors[0] || "Unable to parse generated suggestions."
        throw new Error(parseMessage)
      }

      if (parsed.errors.length > 0) {
        notification.warning({
          message: "Some suggestions could not be parsed",
          description: `${parsed.errors.length} lines were skipped.`
        })
      }

      setAiRunInfo(generationConfig)
      setAiSuggestions(mappedSuggestions)
    } catch (error: any) {
      setAiSuggestions([])
      setAiError(error?.message || "Failed to generate suggestions.")
    } finally {
      setAiGenerating(false)
    }
  }

  const updateAiSuggestion = (
    id: string,
    patch: Partial<WorldBookGeneratedEntryDraft>
  ) => {
    setAiSuggestions((previous) =>
      previous.map((item) => (item.id === id ? { ...item, ...patch } : item))
    )
  }

  const removeAiSuggestion = (id: string) => {
    setAiSuggestions((previous) => previous.filter((item) => item.id !== id))
  }

  const buildAiSuggestionPayload = (
    suggestion: WorldBookGeneratedEntryDraft
  ): Record<string, unknown> => {
    const keywords = normalizeKeywords(suggestion.keywordsText)
    const content = String(suggestion.content || "").trim()
    if (keywords.length === 0) {
      throw new Error("Generated suggestion needs at least one keyword.")
    }
    if (!content) {
      throw new Error("Generated suggestion needs content.")
    }
    return {
      keywords,
      content,
      group: normalizeEntryGroup(suggestion.group),
      priority: clampBulkPriority(suggestion.priority, 50),
      enabled: true,
      case_sensitive: false,
      regex_match: false,
      whole_word_match: true,
      appendable: false,
      metadata: {
        generated_with_ai: true,
        generated_provider: aiRunInfo?.provider || null,
        generated_model: aiRunInfo?.model || FALLBACK_AI_GENERATION_MODEL,
        generated_topic: aiPrompt.trim(),
        generated_at: new Date().toISOString()
      }
    }
  }

  const applyAiSuggestion = async (suggestion: WorldBookGeneratedEntryDraft) => {
    const payload = buildAiSuggestionPayload(suggestion)
    await addEntryAsync(payload)
    setAiSuggestions((previous) => previous.filter((item) => item.id !== suggestion.id))
  }

  const handleApplyAllAiSuggestions = async () => {
    if (aiSuggestions.length === 0) return
    setAiApplying(true)
    setAiError(null)
    try {
      for (const suggestion of aiSuggestions) {
        await applyAiSuggestion(suggestion)
      }
      notification.success({
        message: "Generated suggestions saved",
        description: "All generated entries were added to this world book."
      })
    } catch (error: any) {
      setAiError(error?.message || "Failed to apply all generated suggestions.")
    } finally {
      setAiApplying(false)
    }
  }

  const handleCloseAiGenerate = () => {
    if (aiGenerating || aiApplying) return
    setOpenAiGenerate(false)
    setAiError(null)
  }

  return (
    <div className="space-y-3">
      {status === 'pending' && <Skeleton active paragraph={{ rows: 4 }} />}
      {status === 'success' && entries.length === 0 && (
        <Empty
          description={
            <div className="text-center space-y-2">
              <p className="text-text-muted">No entries yet</p>
              <p className="text-sm text-text-muted">
                Entries define keyword→content mappings. When a keyword appears
                in chat, the content is injected into the AI's context.
              </p>
              <div className="text-left bg-surface-secondary p-3 rounded text-sm mt-3">
                <p className="font-medium mb-1">Example:</p>
                <p><strong>Keywords:</strong> Hermione, Granger</p>
                <p><strong>Content:</strong> Hermione Granger is a brilliant witch and one of Harry's closest friends. She values knowledge and justice.</p>
              </div>
            </div>
          }
        />
      )}
      {status === 'success' && entries.length > 0 && (
        <div className="space-y-3">
          {typeof configuredTokenBudget === "number" && configuredTokenBudget > 0 ? (
            <WorldBookBudgetBar
              estimatedTokens={estimatedEntryTokens}
              tokenBudget={configuredTokenBudget}
              className="mb-3"
            />
          ) : (
            <div
              className="rounded border border-border px-3 py-2 space-y-1"
              aria-label="Entry budget utilization"
            >
              <p className="text-xs text-text-muted">
                Configure a token budget in world book settings to track utilization.
              </p>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <Input
              allowClear
              value={entrySearch}
              onChange={(event) => setEntrySearch(event.target.value)}
              placeholder="Search entries…"
              aria-label="Search entries"
              className="w-full min-w-[220px] md:w-72"
            />
            <Select
              value={entryEnabledFilter}
              onChange={(value) => setEntryEnabledFilter(value)}
              aria-label="Filter entries by enabled status"
              className="w-40"
              options={[
                { label: "All entries", value: "all" },
                { label: "Enabled", value: "enabled" },
                { label: "Disabled", value: "disabled" }
              ]}
            />
            <Select
              value={entryMatchFilter}
              onChange={(value) => setEntryMatchFilter(value)}
              aria-label="Filter entries by match type"
              className="w-44"
              options={[
                { label: "All match types", value: "all" },
                { label: "Regex only", value: "regex" },
                { label: "Plain keywords", value: "plain" }
              ]}
            />
            <Select
              value={entryGroupFilter}
              onChange={(value) => setEntryGroupFilter(value)}
              aria-label="Filter entries by group"
              className="w-44"
              options={[
                { label: "All groups", value: "all" },
                ...entryGroupOptions.map((group) => ({
                  label: group,
                  value: group
                }))
              ]}
            />
          </div>
          {hasSelectedEntries && (
            <div className="rounded border border-border p-2 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-semibold">{selectedEntryIds.length} selected</span>
                <div className="flex flex-wrap items-center gap-2">
                  {canEscalateSelectAll && (
                    <Button
                      type="link"
                      size="small"
                      className="px-0"
                      onClick={handleSelectAllFilteredEntries}
                      aria-label={`Select all ${filteredEntryIds.length} entries`}
                    >
                      Select all {filteredEntryIds.length} entries
                    </Button>
                  )}
                  <Button
                    type="link"
                    size="small"
                    className="px-0"
                    onClick={() => setSelectedRowKeys([])}
                    aria-label="Clear selected entries"
                  >
                    Clear selection
                  </Button>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="small" loading={bulkPending} onClick={() => void handleBulkAction("enable")}>
                  Enable
                </Button>
                <Button size="small" loading={bulkPending} onClick={() => void handleBulkAction("disable")}>
                  Disable
                </Button>
                <Popover
                  trigger="click"
                  open={bulkMovePopoverOpen}
                  onOpenChange={setBulkMovePopoverOpen}
                  content={
                    <div className="space-y-2 w-64">
                      <p className="text-xs text-text-muted">Move selected entries to another world book.</p>
                      {moveDestinationOptions.length === 0 && (
                        <p className="text-xs text-text-muted">
                          Create another world book to enable move actions.
                        </p>
                      )}
                      {moveDestinationOptions.length > 0 && (
                        <>
                          <Select
                            placeholder="Destination world book"
                            value={bulkMoveTargetId ?? undefined}
                            onChange={(value) => setBulkMoveTargetId(Number(value))}
                            options={moveDestinationOptions}
                            aria-label="Bulk move destination"
                          />
                          <Select
                            value={bulkMoveStrategy}
                            onChange={(value) =>
                              setBulkMoveStrategy(value as "skip_existing" | "duplicate")
                            }
                            options={[
                              { label: "Skip existing entries", value: "skip_existing" },
                              { label: "Allow duplicates", value: "duplicate" }
                            ]}
                            aria-label="Bulk move conflict strategy"
                          />
                          <Button
                            size="small"
                            type="primary"
                            className="w-full"
                            loading={bulkMovePending}
                            disabled={!bulkMoveTargetId}
                            onClick={() => void handleBulkMoveEntries()}
                          >
                            Move Entries
                          </Button>
                        </>
                      )}
                    </div>
                  }
                >
                  <Button size="small" loading={bulkMovePending} disabled={moveDestinationOptions.length === 0}>
                    Move To
                  </Button>
                </Popover>
                <Popover
                  trigger="click"
                  open={bulkPriorityPopoverOpen}
                  onOpenChange={setBulkPriorityPopoverOpen}
                  content={
                    <div className="space-y-2 w-52">
                      <p className="text-xs text-text-muted">Set a single priority for all selected entries.</p>
                      <InputNumber
                        min={0}
                        max={100}
                        value={bulkPriorityValue}
                        onChange={(value) => setBulkPriorityValue(clampBulkPriority(value))}
                        style={{ width: "100%" }}
                        aria-label="Bulk priority value"
                      />
                      <Button
                        size="small"
                        type="primary"
                        className="w-full"
                        loading={bulkPending}
                        onClick={() => void handleBulkSetPriority()}
                      >
                        Apply Priority
                      </Button>
                    </div>
                  }
                >
                  <Button size="small" loading={bulkPending}>
                    Set Priority
                  </Button>
                </Popover>
                <Button size="small" danger loading={bulkPending} onClick={() => void handleBulkAction("delete")}>
                  Delete
                </Button>
              </div>
            </div>
          )}
          <p className="text-xs text-text-muted">
            Showing {visibleEntryCount} of {totalEntryCount} entries.
          </p>
          <p className="text-xs text-text-muted">
            Entry order is controlled by priority (0-100). Higher-priority entries are evaluated first.
          </p>
          <p className="text-xs text-text-muted">
            Referenced By is a heuristic based on keyword text found inside other entry content.
          </p>
          <Table
            size="small"
            virtual
            pagination={false}
            scroll={{ y: 420, x: isEntryManagerMobile ? 760 : 1060 }}
            rowKey={(r: any) => r.entry_id}
            rowSelection={{
              selectedRowKeys,
              onChange: (keys) => setSelectedRowKeys(keys)
            }}
            dataSource={filteredEntryData}
            locale={{
              emptyText: "No entries match the current filters."
            }}
            columns={[
              { title: 'Keywords', dataIndex: 'keywords', key: 'keywords', width: 200, render: (arr: string[]) => <div className="flex flex-wrap gap-1">{(arr||[]).map((k) => <Tag key={k}>{k}</Tag>)}</div> },
              {
                title: 'Group',
                dataIndex: 'group',
                key: 'group',
                width: 140,
                render: (_value: unknown, row: any) => {
                  const group = normalizeEntryGroup(row?.group ?? row?.metadata?.group)
                  return group ? (
                    <Tag color="geekblue">{group}</Tag>
                  ) : (
                    <span className="text-xs text-text-muted">Ungrouped</span>
                  )
                }
              },
              { title: 'Content', dataIndex: 'content', key: 'content', render: (v: string) => (
                <div>
                  <span className="line-clamp-2">{v}</span>
                  <span className="text-xs text-text-muted">~{estimateEntryTokens(v)} tokens</span>
                </div>
              ) },
              {
                title: (
                  <Tooltip title="Heuristic: counts entries whose content contains this entry's keywords.">
                    <span>Referenced By</span>
                  </Tooltip>
                ),
                key: "referenced_by",
                width: 150,
                render: (_: unknown, row: any) => {
                  const entryId = Number(row?.entry_id)
                  if (!Number.isFinite(entryId) || entryId <= 0) {
                    return <span className="text-xs text-text-muted">None</span>
                  }
                  const references = referencedBySignalMap[entryId] || []
                  if (references.length === 0) {
                    return <span className="text-xs text-text-muted">None</span>
                  }

                  return (
                    <Popover
                      trigger="click"
                      content={
                        <div className="max-w-xs space-y-1">
                          <p className="text-xs text-text-muted">
                            Heuristic keyword overlaps for this entry:
                          </p>
                          {references.map((reference) => {
                            const sourceEntry = entryById.get(reference.sourceEntryId)
                            const sourceKeywords = sourceEntry
                              ? normalizeKeywordList(sourceEntry?.keywords).slice(0, 2).join(", ")
                              : ""
                            return (
                              <div
                                key={`${entryId}-${reference.sourceEntryId}-${reference.matchedKeyword}`}
                                className="rounded border border-border px-2 py-1 text-xs"
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span>Entry #{reference.sourceEntryId}</span>
                                  <Button
                                    type="link"
                                    size="small"
                                    className="px-0"
                                    aria-label={`Open referencing entry ${reference.sourceEntryId}`}
                                    onClick={() => {
                                      if (sourceEntry) openEditModal(sourceEntry)
                                    }}
                                  >
                                    Open
                                  </Button>
                                </div>
                                <p className="text-text-muted">
                                  matches "{reference.matchedKeyword}"
                                  {sourceKeywords ? ` (${sourceKeywords})` : ""}
                                </p>
                              </div>
                            )
                          })}
                        </div>
                      }
                    >
                      <Button
                        type="link"
                        size="small"
                        className="px-0"
                        aria-label={`View references for entry ${entryId}`}
                      >
                        {references.length} entries
                      </Button>
                    </Popover>
                  )
                }
              },
              ...(screens.md
                ? [
                    {
                      title: 'Priority',
                      dataIndex: 'priority',
                      key: 'priority',
                      width: 110,
                      render: (value: number) => {
                        const band = getPriorityBand(value)
                        return <Tag color={getPriorityTagColor(band)}>{Number(value || 0)}/100</Tag>
                      }
                    },
                    {
                      title: 'Enabled',
                      dataIndex: 'enabled',
                      key: 'enabled',
                      width: 70,
                      render: (v: boolean) => (v ? <Tag color="green">Yes</Tag> : <Tag>No</Tag>)
                    }
                  ]
                : []),
              { title: 'Actions', key: 'actions', width: 80, render: (_: any, r: any) => (
                <div className="flex gap-2">
                  <Tooltip title="Edit">
                    <Button
                      type="text"
                      size={entryActionButtonSize}
                      className={entryActionButtonClassName}
                      aria-label="Edit entry"
                      icon={<Pen className="w-4 h-4" />}
                      onClick={() => openEditModal(r)}
                    />
                  </Tooltip>
                  <Tooltip title="Delete">
                    <Button
                      type="text"
                      size={entryActionButtonSize}
                      className={entryActionButtonClassName}
                      danger
                      aria-label="Delete entry"
                      icon={<Trash2 className="w-4 h-4" />}
                      onClick={async () => {
                        const ok = await confirmDanger({
                          title: 'Delete entry?',
                          content: `This will remove the entry with keywords: ${(r.keywords || []).join(', ') || '(none)'}`,
                          okText: 'Delete',
                          cancelText: 'Cancel'
                        })
                        if (ok) deleteEntry(r.entry_id)
                      }}
                    />
                  </Tooltip>
                </div>
              ) }
            ] as any}
          />
          <details
            className="mt-2"
            open={keywordIndexOpen}
            onToggle={(event) => {
              const nextOpen = (event.currentTarget as HTMLDetailsElement).open
              setKeywordIndexOpen(nextOpen)
            }}
          >
            <summary
              className="cursor-pointer text-sm text-text-muted hover:text-text"
              aria-label={`Keyword Index${keywordConflictCount > 0 ? `, ${keywordConflictCount} conflicts` : ""}`}
              aria-expanded={keywordIndexOpen}
              aria-controls={keywordIndexContentId}
            >
              Keyword Index{keywordConflictCount > 0 ? ` (${keywordConflictCount} conflicts)` : ""}
            </summary>
            <div id={keywordIndexContentId} className="mt-2 flex flex-wrap gap-1">
              <span className="sr-only" role="status" aria-live="polite">
                {keywordConflictCount > 0
                  ? `${keywordConflictCount} keyword conflicts detected.`
                  : "No keyword conflicts detected."}
              </span>
              {keywordIndex.length === 0 && <span className="text-sm text-text-muted">No keywords yet</span>}
              {keywordIndex.map((k) => (
                <Tooltip key={k.keyword} title={k.conflict ? `Conflict: ${k.variantCount} content variations` : `${k.count} entries`}>
                  <Tag
                    color={k.conflict ? "red" : undefined}
                    aria-label={
                      k.conflict
                        ? `${k.keyword}: conflict - ${k.variantCount} content variations`
                        : `${k.keyword}: ${k.count} entries`
                    }
                  >
                    {k.keyword} ({k.count})
                  </Tag>
                </Tooltip>
              ))}
            </div>
          </details>
        </div>
      )}

      {/* Edit Entry Modal */}
      <Modal
        title="Edit Entry"
        open={!!editingEntry}
        onCancel={async () => {
          if (editForm.isFieldsTouched()) {
            const ok = await confirmDanger({
              title: "Discard changes?",
              content: "You have unsaved changes. Are you sure you want to close?",
              okText: "Discard",
              cancelText: "Keep editing"
            })
            if (!ok) return
          }
          setEditingEntry(null)
          editForm.resetFields()
        }}
        footer={null}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        <Form layout="vertical" form={editForm} onFinish={(v) => updateEntry(v)}>
          <Form.Item
            name="keywords"
            label="Keywords"
            rules={[
              {
                validator: (_: any, value: unknown) => {
                  if (!editForm.getFieldValue("regex_match")) return Promise.resolve()
                  const message = validateRegexKeywords(value)
                  return message ? Promise.reject(new Error(message)) : Promise.resolve()
                }
              }
            ]}
          >
            <Select
              mode="tags"
              tokenSeparators={[","]}
              placeholder="Add keywords and press Enter"
            />
          </Form.Item>
          <KeywordPreview value={editKeywordsWatch} />
          <Form.Item name="content" label="Content" rules={[{ required: true }]}>
            <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
          </Form.Item>
          <Form.Item name="group" label="Group (optional)">
            <Input placeholder="e.g., Geography, Characters, History" />
          </Form.Item>
          <p className="text-xs text-text-muted -mt-4 mb-3">{formatEntryContentStats(editContentWatch)}</p>
          {typeof configuredTokenBudget === "number" && configuredTokenBudget > 0 && editFormProjectedTokens > 0 && (
            <WorldBookBudgetBar
              estimatedTokens={estimatedEntryTokens}
              tokenBudget={configuredTokenBudget}
              projectedTokens={editFormProjectedTotal}
              className="mt-2"
            />
          )}
          {typeof configuredTokenBudget === "number" && configuredTokenBudget > 0 && editFormProjectedTotal > configuredTokenBudget && editFormProjectedTokens > 0 && (
            <div className="mt-1 rounded border border-amber-300 bg-amber-50 px-2 py-1 text-xs text-amber-800">
              This entry would push usage over budget. The AI may not receive all entries. You can still save — or increase the budget.
            </div>
          )}
          <Form.Item
            name="priority"
            label={<LabelWithHelp label="Priority" help="Higher values = more important (0-100). Higher priority entries are selected first when token budget is limited." />}
          >
            <InputNumber style={{ width: '100%' }} min={0} max={100} />
          </Form.Item>
          <Form.Item name="enabled" label="Enabled" valuePropName="checked"><Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} /></Form.Item>
          <Form.Item
            name="appendable"
            label={
              <LabelWithHelp
                label="Appendable"
                help="When enabled, this entry's content is appended to other triggered content rather than replacing it."
              />
            }
            valuePropName="checked"
          >
            <Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} />
          </Form.Item>
          <details
            className="mb-4"
            open={editMatchingOptionsOpen}
            onToggle={(event) => {
              const nextOpen = (event.currentTarget as HTMLDetailsElement).open
              setEditMatchingOptionsOpen(nextOpen)
              writeSessionBoolean("worldbooks:edit-matching-options-open", nextOpen)
            }}
          >
            <summary
              className="cursor-pointer text-sm text-text-muted hover:text-text"
              aria-expanded={editMatchingOptionsOpen}
              aria-controls={editMatchingOptionsContentId}
            >
              Matching Options
            </summary>
            <div id={editMatchingOptionsContentId}>
            <p className="text-xs text-text-muted mt-1 mb-2">These options control how keywords are matched in chat messages.</p>
            <div className="mt-2 pl-2 border-l-2 border-border space-y-0">
              <Form.Item name="case_sensitive" label="Case Sensitive" valuePropName="checked"><Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} /></Form.Item>
              <Form.Item name="regex_match" label="Regex Match" valuePropName="checked">
                <Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} onChange={(checked) => { if (checked) editForm.setFieldsValue({ whole_word_match: false }) }} />
              </Form.Item>
              {!editRegexMatch && (
                <Form.Item name="whole_word_match" label="Whole Word Match" valuePropName="checked"><Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} /></Form.Item>
              )}
            </div>
            <p className="text-xs text-text-muted mt-2">
              {buildMatchPreview(editKeywordsWatch, {
                caseSensitive: editCaseSensitive,
                regexMatch: editRegexMatch,
                wholeWord: editWholeWord
              })}
            </p>
            </div>
          </details>
          <Button type="primary" htmlType="submit" loading={updating} className="w-full">Save Changes</Button>
        </Form>
      </Modal>

      {/* Add Entry Form */}
      <div className="border-t border-border pt-4 mt-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium">Add New Entry</h4>
          <div className="flex items-center gap-2">
            <Button
              size="small"
              onClick={() => setOpenAiGenerate(true)}
              aria-label="Generate entries with AI"
            >
              Generate with AI
            </Button>
            <Switch
              checked={bulkMode}
              onChange={setBulkMode}
              aria-label="Toggle bulk add mode"
              {...ACCESSIBLE_SWITCH_TEXT_PROPS}
            />
            <span className="text-xs text-text-muted">Bulk add mode</span>
          </div>
        </div>
        {!bulkMode && (
          <Form layout="vertical" form={form} onFinish={(v) => addEntry(v)}>
            <Form.Item
              name="keywords"
              label="Keywords"
              rules={[
                {
                  validator: (_: any, value: unknown) => {
                    if (!form.getFieldValue("regex_match")) return Promise.resolve()
                    const message = validateRegexKeywords(value)
                    return message ? Promise.reject(new Error(message)) : Promise.resolve()
                  }
                }
              ]}
            >
              <Select
                mode="tags"
                tokenSeparators={[","]}
                placeholder="Add keywords and press Enter"
              />
            </Form.Item>
            <KeywordPreview value={addKeywordsWatch} />
            <Form.Item name="content" label="Content" rules={[{ required: true }]}>
              <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
            </Form.Item>
            <Form.Item name="group" label="Group (optional)">
              <Input placeholder="e.g., Geography, Characters, History" />
            </Form.Item>
            <p className="text-xs text-text-muted -mt-4 mb-3">{formatEntryContentStats(addContentWatch)}</p>
            {typeof configuredTokenBudget === "number" && configuredTokenBudget > 0 && addFormProjectedTokens > 0 && (
              <WorldBookBudgetBar
                estimatedTokens={estimatedEntryTokens}
                tokenBudget={configuredTokenBudget}
                projectedTokens={addFormProjectedTotal}
                className="mt-2"
              />
            )}
            {typeof configuredTokenBudget === "number" && configuredTokenBudget > 0 && addFormProjectedTotal > configuredTokenBudget && addFormProjectedTokens > 0 && (
              <div className="mt-1 rounded border border-amber-300 bg-amber-50 px-2 py-1 text-xs text-amber-800">
                This entry would push usage over budget. The AI may not receive all entries. You can still save — or increase the budget.
              </div>
            )}
            <Form.Item
              name="priority"
              label={<LabelWithHelp label="Priority" help="Higher values = more important (0-100). Higher priority entries are selected first when token budget is limited." />}
            >
              <InputNumber style={{ width: '100%' }} min={0} max={100} placeholder="Default: 50" />
            </Form.Item>
            <Form.Item name="enabled" label="Enabled" valuePropName="checked"><Switch defaultChecked {...ACCESSIBLE_SWITCH_TEXT_PROPS} /></Form.Item>
            <Form.Item
              name="appendable"
              label={
                <LabelWithHelp
                  label="Appendable"
                  help="When enabled, this entry's content is appended to other triggered content rather than replacing it."
                />
              }
              valuePropName="checked"
            >
              <Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} />
            </Form.Item>
          <details
            className="mb-4"
            open={addMatchingOptionsOpen}
            onToggle={(event) => {
              const nextOpen = (event.currentTarget as HTMLDetailsElement).open
              setAddMatchingOptionsOpen(nextOpen)
              writeSessionBoolean("worldbooks:add-matching-options-open", nextOpen)
            }}
          >
            <summary
              className="cursor-pointer text-sm text-text-muted hover:text-text"
              aria-expanded={addMatchingOptionsOpen}
              aria-controls={addMatchingOptionsContentId}
            >
              Matching Options
            </summary>
            <div id={addMatchingOptionsContentId}>
            <p className="text-xs text-text-muted mt-1 mb-2">These options control how keywords are matched in chat messages.</p>
            <div className="mt-2 pl-2 border-l-2 border-border space-y-0">
              <Form.Item name="case_sensitive" label="Case Sensitive" valuePropName="checked"><Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} /></Form.Item>
              <Form.Item name="regex_match" label="Regex Match" valuePropName="checked">
                <Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} onChange={(checked) => { if (checked) form.setFieldsValue({ whole_word_match: false }) }} />
              </Form.Item>
              {!addRegexMatch && (
                <Form.Item name="whole_word_match" label="Whole Word Match" valuePropName="checked"><Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} /></Form.Item>
              )}
            </div>
            <p className="text-xs text-text-muted mt-2">
              {buildMatchPreview(addKeywordsWatch, {
                caseSensitive: addCaseSensitive,
                regexMatch: addRegexMatch,
                wholeWord: addWholeWord
              })}
            </p>
            </div>
          </details>
          <Button type="primary" htmlType="submit" loading={adding} className="w-full">Add Entry</Button>
        </Form>
      )}
        {bulkMode && (
          <div className="space-y-2">
            <details
              className="rounded border border-border p-2"
              open={bulkFormatsOpen}
              onToggle={(event) => {
                const nextOpen = (event.currentTarget as HTMLDetailsElement).open
                setBulkFormatsOpen(nextOpen)
              }}
            >
              <summary
                className="cursor-pointer text-sm text-text-muted hover:text-text"
                aria-expanded={bulkFormatsOpen}
                aria-controls={bulkFormatsContentId}
              >
                Supported formats
              </summary>
              <div id={bulkFormatsContentId} className="mt-2 space-y-1 text-xs text-text-muted">
                {BULK_ENTRY_FORMAT_EXAMPLES.filter((format) =>
                  SUPPORTED_BULK_SEPARATORS.includes(format.separator)
                ).map((format) => (
                  <p key={format.separator}>
                    <code>{format.example}</code>
                  </p>
                ))}
              </div>
            </details>
            <Input.TextArea
              value={bulkText}
              onChange={(e) => {
                setBulkText(e.target.value)
                if (bulkFailures.length > 0) setBulkFailures([])
              }}
              autoSize={{ minRows: 4, maxRows: 10 }}
              placeholder="One per line: keyword1, keyword2 -> content"
              aria-label="Bulk entry input"
            />
            {bulkParse.errors.length > 0 && (
              <div className="text-sm text-danger space-y-1">
                {bulkParse.errors.map((err, i) => <p key={i}>{err}</p>)}
              </div>
            )}
            <div className="text-xs text-text-muted">Parsed entries: {bulkParse.entries.length}</div>
            {bulkProgress && (
              <div className="space-y-1" aria-live="polite">
                <Progress
                  size="small"
                  percent={
                    bulkProgress.total > 0
                      ? Math.round((bulkProgress.completed / bulkProgress.total) * 100)
                      : 0
                  }
                  status={bulkAdding ? "active" : bulkProgress.failed > 0 ? "exception" : "success"}
                />
                <p className="text-xs text-text-muted">
                  Bulk progress: {bulkProgress.completed} / {bulkProgress.total} ({bulkProgress.succeeded} succeeded, {bulkProgress.failed} failed)
                </p>
              </div>
            )}
            {bulkFailures.length > 0 && (
              <div className="rounded border border-border p-2 space-y-1">
                <p className="text-sm font-medium text-danger">Failed entries ({bulkFailures.length})</p>
                <div className="max-h-36 overflow-auto space-y-1 text-xs">
                  {bulkFailures.map((failure, index) => (
                    <p key={`${failure.line}-${index}`} className="text-danger">
                      Line {failure.line} ({failure.keywords.join(", ") || "no keywords"}): {failure.message}
                    </p>
                  ))}
                </div>
              </div>
            )}
            <Button
              type="primary"
              className="w-full"
              loading={bulkAdding}
              disabled={bulkParse.entries.length === 0 || bulkParse.errors.length > 0}
              onClick={handleBulkAddEntries}
            >
              Add Entries
            </Button>
          </div>
        )}
      </div>

      <Modal
        title="Generate Entries with AI"
        open={openAiGenerate}
        onCancel={handleCloseAiGenerate}
        footer={null}
        width={900}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        <div className="space-y-3">
          <p className="text-xs text-text-muted">
            Generate draft keyword/content pairs, review them, and save only the ones you want.
          </p>
          <Form layout="vertical">
            <Form.Item label="Topic / Notes">
              <Input.TextArea
                value={aiPrompt}
                onChange={(event) => setAiPrompt(event.target.value)}
                autoSize={{ minRows: 3, maxRows: 7 }}
                placeholder="Describe the setting, concept, or facts you want entries for."
                aria-label="AI generation topic"
              />
            </Form.Item>
            <div className="grid gap-2 md:grid-cols-2">
              <Form.Item label="Suggestion count">
                <InputNumber
                  min={MIN_AI_GENERATION_COUNT}
                  max={MAX_AI_GENERATION_COUNT}
                  value={aiEntryCount}
                  onChange={(value) =>
                    setAiEntryCount(
                      Math.min(
                        MAX_AI_GENERATION_COUNT,
                        Math.max(
                          MIN_AI_GENERATION_COUNT,
                          Number(value || DEFAULT_AI_GENERATION_COUNT)
                        )
                      )
                    )
                  }
                  style={{ width: "100%" }}
                  aria-label="AI suggestion count"
                />
              </Form.Item>
              <Form.Item label="Default group (optional)">
                <Input
                  value={aiDefaultGroup}
                  onChange={(event) => setAiDefaultGroup(event.target.value)}
                  placeholder="e.g., Geography"
                  aria-label="AI default group"
                />
              </Form.Item>
            </div>
            <Button
              type="primary"
              loading={aiGenerating}
              onClick={() => void handleGenerateSuggestions()}
              aria-label="Run AI generation"
            >
              Generate suggestions
            </Button>
          </Form>

          {aiRunInfo && (
            <p className="text-xs text-text-muted">
              Generated with {aiRunInfo.provider || "default"} / {aiRunInfo.model}.
            </p>
          )}
          {aiError && (
            <div className="rounded border border-danger/40 bg-danger/5 px-3 py-2 text-xs text-danger">
              {aiError}
            </div>
          )}

          {aiSuggestions.length > 0 && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-medium">
                  Review generated suggestions ({aiSuggestions.length})
                </p>
                <Button
                  size="small"
                  type="primary"
                  loading={aiApplying}
                  onClick={() => void handleApplyAllAiSuggestions()}
                  aria-label="Add all AI suggestions"
                >
                  Add All Suggestions
                </Button>
              </div>
              <div className="space-y-2 max-h-[45vh] overflow-auto pr-1">
                {aiSuggestions.map((suggestion, index) => (
                  <div
                    key={suggestion.id}
                    className="rounded border border-border px-3 py-2 space-y-2"
                  >
                    <p className="text-xs text-text-muted">Suggestion {index + 1}</p>
                    <Form layout="vertical">
                      <Form.Item label="Keywords">
                        <Input
                          value={suggestion.keywordsText}
                          onChange={(event) =>
                            updateAiSuggestion(suggestion.id, {
                              keywordsText: event.target.value
                            })
                          }
                          aria-label={`Generated keywords ${index + 1}`}
                        />
                      </Form.Item>
                      <Form.Item label="Content">
                        <Input.TextArea
                          value={suggestion.content}
                          onChange={(event) =>
                            updateAiSuggestion(suggestion.id, {
                              content: event.target.value
                            })
                          }
                          autoSize={{ minRows: 2, maxRows: 6 }}
                          aria-label={`Generated content ${index + 1}`}
                        />
                      </Form.Item>
                      <div className="grid gap-2 md:grid-cols-2">
                        <Form.Item label="Group (optional)">
                          <Input
                            value={suggestion.group}
                            onChange={(event) =>
                              updateAiSuggestion(suggestion.id, {
                                group: event.target.value
                              })
                            }
                            aria-label={`Generated group ${index + 1}`}
                          />
                        </Form.Item>
                        <Form.Item label="Priority">
                          <InputNumber
                            min={0}
                            max={100}
                            value={suggestion.priority}
                            onChange={(value) =>
                              updateAiSuggestion(suggestion.id, {
                                priority: clampBulkPriority(value, 50)
                              })
                            }
                            style={{ width: "100%" }}
                            aria-label={`Generated priority ${index + 1}`}
                          />
                        </Form.Item>
                      </div>
                    </Form>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        size="small"
                        type="primary"
                        loading={aiApplying}
                        onClick={async () => {
                          setAiApplying(true)
                          setAiError(null)
                          try {
                            await applyAiSuggestion(suggestion)
                            notification.success({ message: "Generated entry added" })
                          } catch (error: any) {
                            setAiError(error?.message || "Failed to add generated entry.")
                          } finally {
                            setAiApplying(false)
                          }
                        }}
                        aria-label={`Add generated suggestion ${index + 1}`}
                      >
                        Add Suggestion
                      </Button>
                      <Button
                        size="small"
                        onClick={() => removeAiSuggestion(suggestion.id)}
                        disabled={aiApplying}
                        aria-label={`Discard generated suggestion ${index + 1}`}
                      >
                        Discard
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Modal>
    </div>
  )
}
