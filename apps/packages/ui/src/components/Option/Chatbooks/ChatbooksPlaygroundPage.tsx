import React from "react"
import {
  Button,
  Card,
  Divider,
  Empty,
  Input,
  Progress,
  Select,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { InboxOutlined } from "@ant-design/icons"
import { Download, RotateCcw, Trash2, XCircle } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { useDebounce } from "@/hooks/useDebounce"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { formatFileSize } from "@/utils/format"
import {
  buildServerLogHint,
  sanitizeServerErrorMessage
} from "@/utils/server-error-message"

const { Text, Title, Paragraph } = Typography

const POLL_INTERVALS_MS = [3000, 5000, 8000, 13000, 21000, 30000]
const SEARCH_FETCH_LIMIT = 1000

const CONTENT_TYPE_KEYS = [
  "conversation",
  "note",
  "character",
  "media",
  "embedding",
  "prompt",
  "evaluation",
  "world_book",
  "dictionary",
  "generated_document"
] as const

type ContentTypeKey = (typeof CONTENT_TYPE_KEYS)[number]

const MAX_RESULTS_BY_TYPE: Partial<Record<ContentTypeKey, number>> = {
  evaluation: 100,
  generated_document: 200
}

const getFetchLimitForType = (type: ContentTypeKey) => {
  const max = MAX_RESULTS_BY_TYPE[type]
  return max ? Math.min(SEARCH_FETCH_LIMIT, max) : SEARCH_FETCH_LIMIT
}

type ChatbookEntity = {
  id: string
  title: string
  description?: string | null
  updated_at?: string | null
  tags?: string[]
}

type EntityListResult = {
  items: ChatbookEntity[]
  total?: number
  truncated?: boolean
}

type FetchParams = {
  search?: string
  tags?: string[]
  page: number
  pageSize: number
  updatedAfter?: Date | null
}

type JobKind = "export" | "import"
type ImportSourceFormat = "chatbook" | "openwebui_json" | "openwebui_db"

type ChatbookJob = {
  job_id: string
  status?: string
  chatbook_name?: string
  chatbook_path?: string
  created_at?: string
  started_at?: string
  completed_at?: string
  error_message?: string
  progress_percentage?: number
  total_items?: number
  processed_items?: number
  file_size_bytes?: number
  download_url?: string
  expires_at?: string
  successful_items?: number
  failed_items?: number
  skipped_items?: number
  warnings?: string[]
  conflicts?: any[]
}

type OpenWebUIDatabasePreviewUser = {
  source_user_id: string
  display_label: string
  email?: string | null
  chat_count?: number
  folder_count?: number
  message_count?: number
  branched_chat_count?: number
  duplicate_chat_count?: number
  archived_chat_count?: number
  pinned_chat_count?: number
  attachment_reference_count?: number
  warning_count?: number
  warnings?: string[]
}

type OpenWebUIHydrationSummary = {
  referenced_files?: number
  resolved_files?: number
  image_files?: number
  media_files?: number
  missing_files?: number
  unsupported_files?: number
  failed_files?: number
  hydrated_images?: number
  registered_media_files?: number
  already_hydrated?: number
  processed_files?: number
  warning_count?: number
}

type OpenWebUIHydrationPreviewState = {
  summary?: OpenWebUIHydrationSummary
  warnings?: string[]
}

type OpenWebUIHydrationJobState = {
  job_id?: string
  status?: string
  result?: any
  error?: string | null
}

const parseIdList = (raw: string) =>
  raw
    .split(/[\n,]+/)
    .map((id) => id.trim())
    .filter(Boolean)

const normalizeTags = (value: any): string[] => {
  if (!value) return []
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return item
        if (item?.name) return String(item.name)
        if (item?.keyword) return String(item.keyword)
        return null
      })
      .filter(Boolean) as string[]
  }
  return []
}

const pickFirstString = (source: any, keys: string[], fallback = "") => {
  for (const key of keys) {
    const value = source?.[key]
    if (typeof value === "string" && value.trim()) return value
  }
  return fallback
}

const normalizeUpdatedAt = (source: any): string | null => {
  const value =
    source?.updated_at ??
    source?.updatedAt ??
    source?.last_modified ??
    source?.lastModified ??
    source?.modified_at ??
    source?.modifiedAt ??
    source?.created_at ??
    source?.createdAt ??
    null
  if (!value) return null
  if (typeof value === "string") return value
  if (typeof value === "number") {
    return value > 1e12
      ? new Date(value).toISOString()
      : new Date(value * 1000).toISOString()
  }
  return null
}

const normalizeTotal = (value: unknown): number | undefined => {
  if (value == null) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

const buildSearchText = (item: ChatbookEntity) =>
  `${item.title || ""} ${item.description || ""} ${item.tags?.join(" ") || ""}`
    .toLowerCase()
    .trim()

const applyFilters = (
  items: ChatbookEntity[],
  search: string,
  tags: string[],
  updatedAfter?: Date | null
) => {
  const searchValue = search.trim().toLowerCase()
  const tagFilters = tags.map((tag) => tag.toLowerCase()).filter(Boolean)

  return items.filter((item) => {
    if (searchValue) {
      if (!buildSearchText(item).includes(searchValue)) return false
    }
    if (tagFilters.length) {
      const itemTags = (item.tags || []).map((tag) => tag.toLowerCase())
      if (!tagFilters.every((tag) => itemTags.includes(tag))) return false
    }
    if (updatedAfter && item.updated_at) {
      const updated = Date.parse(item.updated_at)
      if (!Number.isNaN(updated) && updated < updatedAfter.getTime()) return false
    }
    return true
  })
}

const paginateItems = (items: ChatbookEntity[], page: number, pageSize: number) => {
  const start = (page - 1) * pageSize
  return items.slice(start, start + pageSize)
}

const statusColor = (status?: string) => {
  switch ((status || "").toLowerCase()) {
    case "completed":
      return "green"
    case "failed":
      return "red"
    case "cancelled":
      return "volcano"
    case "pending":
      return "gold"
    case "in_progress":
      return "blue"
    case "validating":
      return "cyan"
    default:
      return "default"
  }
}

const isActiveJobStatus = (status?: string) => {
  const value = (status || "").toLowerCase()
  return value === "pending" || value === "in_progress" || value === "validating"
}

const isRemovableJobStatus = (status?: string) =>
  ["completed", "cancelled", "expired", "failed"].includes((status || "").toLowerCase())

const buildJobSignature = (jobs: ChatbookJob[]) =>
  jobs
    .map(
      (job) =>
        `${job.job_id}:${job.status || ""}:${job.progress_percentage ?? ""}:${job.processed_items ?? ""}`
    )
    .join("|")

const mapConversation = (item: any): ChatbookEntity => {
  const title = pickFirstString(item, ["title", "name"], "Untitled conversation")
  return {
    id: String(item?.id ?? ""),
    title,
    description: pickFirstString(item, ["topic_label", "state", "source"], ""),
    updated_at: normalizeUpdatedAt(item),
    tags: normalizeTags(item?.tags)
  }
}

const mapNote = (item: any): ChatbookEntity => {
  const title =
    pickFirstString(item, ["title", "name"], "") ||
    pickFirstString(item, ["content"], "Untitled note").slice(0, 40)
  return {
    id: String(item?.id ?? ""),
    title,
    description: pickFirstString(item, ["content"], ""),
    updated_at: normalizeUpdatedAt(item),
    tags: normalizeTags(item?.tags || item?.keywords)
  }
}

const mapCharacter = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? ""),
  title: pickFirstString(item, ["name", "character_name", "title"], "Untitled character"),
  description: pickFirstString(item, ["description", "persona"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.tags)
})

const mapMedia = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? ""),
  title: pickFirstString(item, ["title", "name"], "Untitled media"),
  description: pickFirstString(item, ["url", "source_url", "summary", "type"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.tags || item?.keywords)
})

const mapPrompt = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? item?.name ?? ""),
  title: pickFirstString(item, ["name", "title"], "Untitled prompt"),
  description: pickFirstString(item, ["details", "description"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.keywords || item?.tags)
})

const mapEvaluation = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? ""),
  title: pickFirstString(item, ["name", "title"], "Untitled evaluation"),
  description: pickFirstString(item, ["description", "eval_type"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.tags)
})

const mapWorldBook = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? ""),
  title: pickFirstString(item, ["name", "title"], "Untitled world book"),
  description: pickFirstString(item, ["description"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.tags)
})

const mapDictionary = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? ""),
  title: pickFirstString(item, ["name", "title"], "Untitled dictionary"),
  description: pickFirstString(item, ["description"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.tags)
})

const mapGeneratedDocument = (item: any): ChatbookEntity => ({
  id: String(item?.id ?? item?.document_id ?? ""),
  title: pickFirstString(item, ["title", "document_type"], "Generated document"),
  description: pickFirstString(item, ["document_type", "source"], ""),
  updated_at: normalizeUpdatedAt(item),
  tags: normalizeTags(item?.tags)
})

const buildContentItems = async (
  type: ContentTypeKey,
  params: FetchParams
): Promise<EntityListResult> => {
  const { search = "", tags = [], page, pageSize, updatedAfter } = params
  const filterClientSide = Boolean(search || tags.length || updatedAfter)
  const limit = filterClientSide ? getFetchLimitForType(type) : pageSize
  const offset = filterClientSide ? 0 : (page - 1) * pageSize

  switch (type) {
    case "conversation": {
      const data = await bgRequest<any>({
        path: `/api/v1/chats?limit=${limit}&offset=${offset}`,
        method: "GET"
      })
      const list = data?.chats || data?.items || data?.results || data?.data || data || []
      const items = (Array.isArray(list) ? list : []).map(mapConversation)
      const totalAvailable = normalizeTotal(data?.total)
      const filtered = filterClientSide ? applyFilters(items, search, tags, updatedAfter) : items
      const paged = filterClientSide ? paginateItems(filtered, page, pageSize) : filtered
      const truncated =
        filterClientSide &&
        (totalAvailable != null ? totalAvailable > items.length : items.length >= limit)
      const total = filterClientSide ? filtered.length : totalAvailable
      return { items: paged, total, truncated }
    }
    case "note": {
      const data = await bgRequest<any>({
        path: `/api/v1/notes?limit=${limit}&offset=${offset}&include_keywords=true`,
        method: "GET"
      })
      const list = data?.notes || data?.items || data?.results || data || []
      const items = (Array.isArray(list) ? list : []).map(mapNote)
      const totalAvailable = normalizeTotal(data?.total)
      const filtered = filterClientSide ? applyFilters(items, search, tags, updatedAfter) : items
      const paged = filterClientSide ? paginateItems(filtered, page, pageSize) : filtered
      const truncated =
        filterClientSide &&
        (totalAvailable != null ? totalAvailable > items.length : items.length >= limit)
      const total = filterClientSide ? filtered.length : totalAvailable
      return { items: paged, total, truncated }
    }
    case "character": {
      const data = await bgRequest<any>({
        path: `/api/v1/characters?limit=${limit}&offset=${offset}`,
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      const items = (Array.isArray(list) ? list : []).map(mapCharacter)
      const totalAvailable = normalizeTotal(data?.total)
      const filtered = filterClientSide ? applyFilters(items, search, tags, updatedAfter) : items
      const paged = filterClientSide ? paginateItems(filtered, page, pageSize) : filtered
      const truncated =
        filterClientSide &&
        (totalAvailable != null ? totalAvailable > items.length : items.length >= limit)
      const total = filterClientSide ? filtered.length : totalAvailable
      return { items: paged, total, truncated }
    }
    case "media": {
      const pageParam = filterClientSide ? 1 : page
      const perPage = filterClientSide ? getFetchLimitForType(type) : pageSize
      const data = await bgRequest<any>({
        path: `/api/v1/media?page=${pageParam}&results_per_page=${perPage}`,
        method: "GET"
      })
      const list = data?.items || data?.results || data?.data || data || []
      const items = (Array.isArray(list) ? list : []).map(mapMedia)
      const totalAvailable = normalizeTotal(data?.pagination?.total_items)
      const filtered = filterClientSide ? applyFilters(items, search, tags, updatedAfter) : items
      const paged = filterClientSide ? paginateItems(filtered, page, pageSize) : filtered
      const truncated =
        filterClientSide &&
        (totalAvailable != null ? totalAvailable > items.length : items.length >= perPage)
      const total = filterClientSide ? filtered.length : totalAvailable
      return { items: paged, total, truncated }
    }
    case "prompt": {
      const data = await bgRequest<any>({
        path: "/api/v1/prompts",
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      const items = (Array.isArray(list) ? list : []).map(mapPrompt)
      const filtered = applyFilters(items, search, tags, updatedAfter)
      const paged = paginateItems(filtered, page, pageSize)
      return { items: paged, total: filtered.length }
    }
    case "evaluation": {
      const evalLimit = getFetchLimitForType(type)
      const data = await bgRequest<any>({
        path: `/api/v1/evaluations?limit=${evalLimit}`,
        method: "GET"
      })
      const list = data?.data || data?.items || data?.results || data || []
      const items = (Array.isArray(list) ? list : []).map(mapEvaluation)
      const filtered = applyFilters(items, search, tags, updatedAfter)
      const paged = paginateItems(filtered, page, pageSize)
      const truncated = items.length >= evalLimit
      return { items: paged, total: filtered.length, truncated }
    }
    case "world_book": {
      const data = await bgRequest<any>({
        path: "/api/v1/characters/world-books",
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      const items = (Array.isArray(list) ? list : []).map(mapWorldBook)
      const filtered = applyFilters(items, search, tags, updatedAfter)
      const paged = paginateItems(filtered, page, pageSize)
      return { items: paged, total: filtered.length }
    }
    case "dictionary": {
      const data = await bgRequest<any>({
        path: "/api/v1/chat/dictionaries",
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      const items = (Array.isArray(list) ? list : []).map(mapDictionary)
      const filtered = applyFilters(items, search, tags, updatedAfter)
      const paged = paginateItems(filtered, page, pageSize)
      return { items: paged, total: filtered.length }
    }
    case "generated_document": {
      const docLimit = getFetchLimitForType(type)
      const data = await bgRequest<any>({
        path: `/api/v1/chat/documents?limit=${docLimit}`,
        method: "GET"
      })
      const list = data?.documents || data?.items || data?.results || data?.data || data || []
      const items = (Array.isArray(list) ? list : []).map(mapGeneratedDocument)
      const filtered = applyFilters(items, search, tags, updatedAfter)
      const paged = paginateItems(filtered, page, pageSize)
      const truncated = items.length >= docLimit
      return { items: paged, total: filtered.length, truncated }
    }
    default:
      return { items: [], total: 0 }
  }
}

const FETCH_ALL_ONCE_TYPES = new Set<ContentTypeKey>([
  "prompt",
  "evaluation",
  "world_book",
  "dictionary",
  "generated_document"
])

const fetchAllItemsCache = new Map<ContentTypeKey, ChatbookEntity[]>()

const clearFetchAllItemsCache = () => {
  fetchAllItemsCache.clear()
}

const fetchAllItemsForType = async (type: ContentTypeKey): Promise<ChatbookEntity[]> => {
  if (fetchAllItemsCache.has(type)) {
    return fetchAllItemsCache.get(type) ?? []
  }
  let items: ChatbookEntity[] = []

  switch (type) {
    case "prompt": {
      const data = await bgRequest<any>({
        path: "/api/v1/prompts",
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      items = (Array.isArray(list) ? list : []).map(mapPrompt)
      break
    }
    case "evaluation": {
      const evalLimit = getFetchLimitForType(type)
      const data = await bgRequest<any>({
        path: `/api/v1/evaluations?limit=${evalLimit}`,
        method: "GET"
      })
      const list = data?.data || data?.items || data?.results || data || []
      items = (Array.isArray(list) ? list : []).map(mapEvaluation)
      break
    }
    case "world_book": {
      const data = await bgRequest<any>({
        path: "/api/v1/characters/world-books",
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      items = (Array.isArray(list) ? list : []).map(mapWorldBook)
      break
    }
    case "dictionary": {
      const data = await bgRequest<any>({
        path: "/api/v1/chat/dictionaries",
        method: "GET"
      })
      const list = Array.isArray(data) ? data : data?.items || data?.results || data || []
      items = (Array.isArray(list) ? list : []).map(mapDictionary)
      break
    }
    case "generated_document": {
      const docLimit = getFetchLimitForType(type)
      const data = await bgRequest<any>({
        path: `/api/v1/chat/documents?limit=${docLimit}`,
        method: "GET"
      })
      const list = data?.documents || data?.items || data?.results || data?.data || data || []
      items = (Array.isArray(list) ? list : []).map(mapGeneratedDocument)
      break
    }
    default:
      items = []
  }

  fetchAllItemsCache.set(type, items)
  return items
}

const INCLUDE_ALL_PAGE_SIZES: Record<ContentTypeKey, number> = {
  conversation: 200,
  note: 1000,
  character: 200,
  media: 200,
  embedding: 0,
  prompt: 200,
  evaluation: 200,
  world_book: 200,
  dictionary: 200,
  generated_document: 200
}

const fetchAllIds = async (type: ContentTypeKey) => {
  if (FETCH_ALL_ONCE_TYPES.has(type)) {
    const items = await fetchAllItemsForType(type)
    return Array.from(new Set(items.map((item) => item.id).filter(Boolean)))
  }
  const pageSize = INCLUDE_ALL_PAGE_SIZES[type] || 200
  if (!pageSize) return []
  const ids: string[] = []
  let page = 1
  let total: number | undefined = undefined
  while (page <= 50) {
    const result = await buildContentItems(type, {
      search: "",
      tags: [],
      page,
      pageSize
    })
    const batch = result.items.map((item) => item.id).filter(Boolean)
    ids.push(...batch)
    total = result.total ?? total
    if (batch.length < pageSize) break
    if (total != null && ids.length >= total) break
    page += 1
  }
  return Array.from(new Set(ids))
}

type ContentTypePickerProps = {
  typeKey: ContentTypeKey
  label: string
  helper?: string
  includeAll: boolean
  onIncludeAllChange: (next: boolean) => void
  selectedIds: string[]
  onSelectionChange: (ids: string[]) => void
  fetcher?: (params: FetchParams) => Promise<EntityListResult>
  itemsOverride?: ChatbookEntity[]
  disabled?: boolean
  allowIncludeAll?: boolean
  allowManualEntry?: boolean
  manualPlaceholder?: string
  emptyHint?: string
  truncationLimit?: number
  /** When true, suppress per-section error alerts for auth/connection errors (a consolidated banner is shown above). */
  suppressAuthErrors?: boolean
}

export const ContentTypePicker: React.FC<ContentTypePickerProps> = ({
  typeKey,
  label,
  helper,
  includeAll,
  onIncludeAllChange,
  selectedIds,
  onSelectionChange,
  fetcher,
  itemsOverride,
  disabled,
  allowIncludeAll = true,
  allowManualEntry = false,
  manualPlaceholder,
  emptyHint,
  truncationLimit,
  suppressAuthErrors = false
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [search, setSearch] = React.useState("")
  const [tagFilters, setTagFilters] = React.useState<string[]>([])
  const [updatedWindow, setUpdatedWindow] = React.useState<string>("any")
  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(10)
  const [manualInput, setManualInput] = React.useState("")
  const debouncedSearch = useDebounce(search, 300)

  const updatedAfter = React.useMemo(() => {
    if (updatedWindow === "any") return null
    const days = Number(updatedWindow)
    if (!Number.isFinite(days)) return null
    const date = new Date()
    date.setDate(date.getDate() - days)
    return date
  }, [updatedWindow])

  React.useEffect(() => {
    setPage(1)
  }, [debouncedSearch, tagFilters, updatedWindow])

  const query = useQuery({
    queryKey: ["chatbooks", "picker", typeKey, debouncedSearch, tagFilters, updatedWindow, page, pageSize],
    queryFn: () =>
      fetcher
        ? fetcher({
            search: debouncedSearch,
            tags: tagFilters,
            page,
            pageSize,
            updatedAfter
          })
        : Promise.resolve({ items: [], total: 0 }),
    enabled: Boolean(fetcher) && !includeAll && !disabled,
    staleTime: 60_000
  })

  const filteredOverride = React.useMemo(() => {
    if (!itemsOverride) return null
    return applyFilters(itemsOverride, debouncedSearch, tagFilters, updatedAfter)
  }, [itemsOverride, debouncedSearch, tagFilters, updatedAfter])

  const items = React.useMemo(() => {
    if (itemsOverride && filteredOverride) {
      return paginateItems(filteredOverride, page, pageSize)
    }
    return query.data?.items || []
  }, [itemsOverride, filteredOverride, page, pageSize, query.data?.items])

  const total = itemsOverride
    ? filteredOverride?.length
    : query.data?.total ?? (query.data?.items?.length || 0)
  const showTruncatedWarning = Boolean(query.data?.truncated)
  const effectiveTruncationLimit = truncationLimit ?? SEARCH_FETCH_LIMIT
  const loadErrorDescription = React.useMemo(
    () =>
      sanitizeServerErrorMessage(
        query.error,
        t("settings:chatbooksPlayground.loadErrorDescription", {
          defaultValue: "Check your server connection and try again."
        })
      ),
    [query.error, t]
  )
  const loadErrorHint = React.useMemo(
    () =>
      buildServerLogHint(
        query.error,
        t("settings:chatbooksPlayground.loadErrorHint", {
          defaultValue: "If this keeps failing, check server logs for details."
        })
      ),
    [query.error, t]
  )

  const columns = React.useMemo<ColumnsType<ChatbookEntity>>(
    () => [
      {
        title: t("settings:chatbooksPlayground.name", "Name"),
        dataIndex: "title",
        key: "title",
        render: (value) => <Text strong>{value}</Text>
      },
      {
        title: t("settings:chatbooksPlayground.updated", "Updated"),
        dataIndex: "updated_at",
        key: "updated_at",
        width: 140,
        render: (value) =>
          value ? formatRelativeTime(String(value), t) : t("common:unknown", "Unknown")
      },
      {
        title: t("settings:chatbooksPlayground.tags", "Tags"),
        dataIndex: "tags",
        key: "tags",
        render: (value: string[]) => {
          const tags = value || []
          if (!tags.length) {
            return <Text type="secondary">—</Text>
          }
          return (
            <Space size={[4, 4]} wrap>
              {tags.map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </Space>
          )
        }
      },
      {
        title: t("settings:chatbooksPlayground.description", "Description"),
        dataIndex: "description",
        key: "description",
        render: (value) => (
          <Paragraph ellipsis={{ rows: 2 }} className="mb-0 text-sm">
            {value || "—"}
          </Paragraph>
        )
      }
    ],
    [t]
  )

  const showTable = !includeAll && (fetcher || itemsOverride)
  const showManual = !includeAll && allowManualEntry

  return (
    <Card size="small" className="border-border" title={label}>
      <div className="flex flex-col gap-3">
        {helper && <Text type="secondary">{helper}</Text>}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Space wrap>
            <Switch
              checked={includeAll}
              disabled={!allowIncludeAll || disabled}
              onChange={(checked) => onIncludeAllChange(checked)}
            />
            <Text>
              {t("settings:chatbooksPlayground.includeAll", "Include all")}
            </Text>
          </Space>
          <Text type="secondary" className="text-xs">
            {t("settings:chatbooksPlayground.selectedCount", {
              defaultValue: "Selected: {{count}}",
              count: selectedIds.length
            })}
          </Text>
        </div>

        {includeAll && (
          <DesignSystemAlert
            variant="info"
            title={t("settings:chatbooksPlayground.includeAllHint", {
              defaultValue: "All {{label}} will be included.",
              label: label.toLowerCase()
            })}
          />
        )}

        {showTable && (
          <div className="flex flex-col gap-3">
            <Space wrap className="w-full">
              <Input
                placeholder={t(
                  "settings:chatbooksPlayground.searchPlaceholder",
                  "Search"
                )}
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                className="min-w-[180px]"
                disabled={disabled}
              />
              <Select
                mode="tags"
                allowClear
                placeholder={t(
                  "settings:chatbooksPlayground.tagsPlaceholder",
                  "Filter tags"
                )}
                value={tagFilters}
                onChange={(values) => setTagFilters(values)}
                className="min-w-[200px]"
                disabled={disabled}
              />
              <Select
                value={updatedWindow}
                onChange={(value) => setUpdatedWindow(value)}
                className="min-w-[160px]"
                disabled={disabled}
                options={[
                  { value: "any", label: t("settings:chatbooksPlayground.updatedAny", "Any time") },
                  { value: "7", label: t("settings:chatbooksPlayground.updated7", "Last 7 days") },
                  { value: "30", label: t("settings:chatbooksPlayground.updated30", "Last 30 days") },
                  { value: "90", label: t("settings:chatbooksPlayground.updated90", "Last 90 days") }
                ]}
              />
            </Space>

            {showTruncatedWarning && (
              <DesignSystemAlert
                variant="warning"
                title={t("settings:chatbooksPlayground.resultsTruncated", {
                  defaultValue:
                    "Results limited to the first {{limit}} items. Refine filters to narrow the list.",
                  limit: effectiveTruncationLimit
                })}
              />
            )}

            <Table
              rowKey={(record) => record.id}
              size="small"
              dataSource={items}
              loading={query.isLoading}
              columns={columns}
              pagination={
                total
                  ? {
                      current: page,
                      pageSize,
                      total,
                      showSizeChanger: true,
                      onChange: (nextPage, nextPageSize) => {
                        setPage(nextPage)
                        if (nextPageSize) setPageSize(nextPageSize)
                      }
                    }
                  : false
              }
              rowSelection={{
                selectedRowKeys: selectedIds,
                onChange: (keys) => onSelectionChange(keys.map(String)),
                getCheckboxProps: () => ({ disabled })
              }}
              locale={{
                emptyText:
                  emptyHint ||
                  t("settings:chatbooksPlayground.emptyList", "No items yet")
              }}
            />
          </div>
        )}

        {showManual && (
          <div className="flex flex-col gap-2">
            <Text type="secondary">
              {t(
                "settings:chatbooksPlayground.manualHint",
                "Paste IDs separated by commas or new lines."
              )}
            </Text>
            <Input.TextArea
              rows={3}
              value={manualInput}
              placeholder={manualPlaceholder}
              onChange={(event) => {
                const nextValue = event.target.value
                setManualInput(nextValue)
                onSelectionChange(parseIdList(nextValue))
              }}
              disabled={disabled}
            />
          </div>
        )}

        {query.isError && !suppressAuthErrors && (
          <DesignSystemAlert
            variant="warning"
            title={t(
              "settings:chatbooksPlayground.loadError",
              "Unable to load items"
            )}
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0 flex-1">
                <p className="text-sm">{loadErrorDescription}</p>
                <p className="mt-1 text-xs text-text-muted">{loadErrorHint}</p>
              </div>
              <Button
                size="small"
                onClick={() => {
                  void query.refetch()
                }}>
                {t("common:retry", { defaultValue: "Retry" })}
              </Button>
            </div>
          </DesignSystemAlert>
        )}
      </div>
    </Card>
  )
}

const buildSelectionState = <T,>(initializer: (key: ContentTypeKey) => T) =>
  CONTENT_TYPE_KEYS.reduce((acc, key) => {
    acc[key] = initializer(key)
    return acc
  }, {} as Record<ContentTypeKey, T>)

const getContentTypeLabel = (t: (key: string, fallback: string) => string) => ({
  conversation: t("settings:chatbooksPlayground.typeConversation", "Conversations"),
  note: t("settings:chatbooksPlayground.typeNote", "Notes"),
  character: t("settings:chatbooksPlayground.typeCharacter", "Characters"),
  media: t("settings:chatbooksPlayground.typeMedia", "Media"),
  embedding: t("settings:chatbooksPlayground.typeEmbedding", "Embeddings"),
  prompt: t("settings:chatbooksPlayground.typePrompt", "Prompts"),
  evaluation: t("settings:chatbooksPlayground.typeEvaluation", "Evaluations"),
  world_book: t("settings:chatbooksPlayground.typeWorldBook", "World books"),
  dictionary: t("settings:chatbooksPlayground.typeDictionary", "Dictionaries"),
  generated_document: t("settings:chatbooksPlayground.typeGenerated", "Generated documents")
})

const getJobLabels = (t: (key: string, fallback: string) => string): Record<JobKind, string> => ({
  export: t("settings:chatbooksPlayground.tabExport", "Export"),
  import: t("settings:chatbooksPlayground.tabImport", "Import")
})

const extractJobList = (payload: any): ChatbookJob[] => {
  if (Array.isArray(payload)) return payload
  return payload?.jobs || payload?.items || payload?.results || payload?.data || []
}

const computeProgress = (job: ChatbookJob) => {
  if (typeof job.progress_percentage === "number") return job.progress_percentage
  if (job.total_items && job.processed_items != null) {
    return Math.round((job.processed_items / job.total_items) * 100)
  }
  return undefined
}

const formatTimestamp = (value?: string) => {
  if (!value) return "—"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString()
}

const getPreviewCount = (manifest: any, key: ContentTypeKey) => {
  if (!manifest) return 0
  switch (key) {
    case "conversation":
      return manifest.total_conversations || 0
    case "note":
      return manifest.total_notes || 0
    case "character":
      return manifest.total_characters || 0
    case "media":
      return manifest.total_media_items || 0
    case "prompt":
      return manifest.total_prompts || 0
    case "evaluation":
      return manifest.total_evaluations || 0
    case "embedding":
      return manifest.total_embeddings || 0
    case "world_book":
      return manifest.total_world_books || 0
    case "dictionary":
      return manifest.total_dictionaries || 0
    case "generated_document":
      return manifest.total_documents || 0
    default:
      return 0
  }
}

const groupPreviewItems = (items: any[]): Record<ContentTypeKey, ChatbookEntity[]> => {
  const grouped = buildSelectionState(() => [])
  items.forEach((item) => {
    const type = item?.type as ContentTypeKey
    if (!type || !CONTENT_TYPE_KEYS.includes(type)) return
    grouped[type].push({
      id: String(item.id ?? ""),
      title: item.title || String(item.id ?? ""),
      description: item.description || "",
      updated_at: item.updated_at || null,
      tags: normalizeTags(item.tags)
    })
  })
  return grouped
}

export const ChatbooksPlaygroundPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common", "option"])
  const isOnline = useServerOnline()
  const notification = useAntdNotification()
  const { capabilities } = useServerCapabilities()

  const [activeTab, setActiveTab] = React.useState("export")
  const contentLabels = React.useMemo(
    () => getContentTypeLabel((key, fallback) => t(key, fallback) as string),
    [t]
  )
  const jobLabels = React.useMemo(
    () => getJobLabels((key, fallback) => t(key, fallback) as string),
    [t]
  )

  const [exportName, setExportName] = React.useState("")
  const [exportDescription, setExportDescription] = React.useState("")
  const [exportAuthor, setExportAuthor] = React.useState("")
  const [exportTags, setExportTags] = React.useState<string[]>([])
  const [exportCategories, setExportCategories] = React.useState<string[]>([])
  const [includeMedia, setIncludeMedia] = React.useState(true)
  const [mediaQuality, setMediaQuality] = React.useState("compressed")
  const [includeEmbeddings, setIncludeEmbeddings] = React.useState(false)
  const [includeGenerated, setIncludeGenerated] = React.useState(true)
  const [exportAsync, setExportAsync] = React.useState(true)
  const [exportSelections, setExportSelections] = React.useState(
    buildSelectionState(() => [] as string[])
  )
  const [exportIncludeAll, setExportIncludeAll] = React.useState(
    buildSelectionState(() => false)
  )
  const [exportSubmitting, setExportSubmitting] = React.useState(false)

  const [importFile, setImportFile] = React.useState<File | null>(null)
  const [importSourceFormat, setImportSourceFormat] = React.useState<ImportSourceFormat>("chatbook")
  const [importPreviewVersion, setImportPreviewVersion] = React.useState(0)
  const [previewManifest, setPreviewManifest] = React.useState<any | null>(null)
  const [openwebuiPreview, setOpenwebuiPreview] = React.useState<any | null>(null)
  const [openwebuiDbPreview, setOpenwebuiDbPreview] = React.useState<any | null>(null)
  const [selectedOpenWebUIUserId, setSelectedOpenWebUIUserId] = React.useState("")
  const [previewError, setPreviewError] = React.useState<string | null>(null)
  const [previewLoading, setPreviewLoading] = React.useState(false)
  const [conflictResolution, setConflictResolution] = React.useState("skip")
  const [prefixImported, setPrefixImported] = React.useState(false)
  const [importMedia, setImportMedia] = React.useState(true)
  const [importEmbeddings, setImportEmbeddings] = React.useState(false)
  const [importAsync, setImportAsync] = React.useState(true)
  const [importSelections, setImportSelections] = React.useState(
    buildSelectionState(() => [] as string[])
  )
  const [importIncludeAll, setImportIncludeAll] = React.useState(
    buildSelectionState(() => true)
  )
  const [importSubmitting, setImportSubmitting] = React.useState(false)
  const [hydrationDataRoot, setHydrationDataRoot] = React.useState("")
  const [hydrationConversationIdsRaw, setHydrationConversationIdsRaw] = React.useState("")
  const [hydrationSourceUserId, setHydrationSourceUserId] = React.useState("")
  const [hydrationProcessSupportedFiles, setHydrationProcessSupportedFiles] = React.useState(false)
  const [hydrationPreview, setHydrationPreview] =
    React.useState<OpenWebUIHydrationPreviewState | null>(null)
  const [hydrationPreviewSignature, setHydrationPreviewSignature] = React.useState("")
  const [hydrationPreviewError, setHydrationPreviewError] = React.useState<string | null>(null)
  const [hydrationPreviewLoading, setHydrationPreviewLoading] = React.useState(false)
  const [hydrationJob, setHydrationJob] = React.useState<OpenWebUIHydrationJobState | null>(null)
  const [hydrationJobError, setHydrationJobError] = React.useState<string | null>(null)
  const [hydrationJobLoading, setHydrationJobLoading] = React.useState(false)

  const [exportJobs, setExportJobs] = React.useState<ChatbookJob[]>([])
  const [importJobs, setImportJobs] = React.useState<ChatbookJob[]>([])
  const [jobsLoading, setJobsLoading] = React.useState(false)
  const [jobsError, setJobsError] = React.useState<string | null>(null)
  const [pollIndex, setPollIndex] = React.useState(0)
  const lastSignatureRef = React.useRef<string>("")
  const previewRequestIdRef = React.useRef(0)

  const canUseChatbooks = capabilities?.hasChatbooks !== false
  const isOpenWebUIJsonImport = importSourceFormat === "openwebui_json"
  const isOpenWebUIDatabaseImport = importSourceFormat === "openwebui_db"
  const isOpenWebUIImport = isOpenWebUIJsonImport || isOpenWebUIDatabaseImport

  React.useEffect(() => {
    previewRequestIdRef.current += 1
    setImportFile(null)
    setPreviewManifest(null)
    setOpenwebuiPreview(null)
    setOpenwebuiDbPreview(null)
    setImportPreviewVersion(0)
    setSelectedOpenWebUIUserId("")
    setPreviewError(null)
    setPreviewLoading(false)
    setImportSelections(buildSelectionState(() => [] as string[]))
    setImportIncludeAll(buildSelectionState(() => true))
    setHydrationPreview(null)
    setHydrationPreviewSignature("")
    setHydrationPreviewError(null)
    setHydrationPreviewLoading(false)
    setHydrationJob(null)
    setHydrationJobError(null)
    if (importSourceFormat === "openwebui_json" || importSourceFormat === "openwebui_db") {
      setImportMedia(false)
      setImportEmbeddings(false)
      setConflictResolution((current) =>
        current === "rename" || current === "skip" ? current : "skip"
      )
    }
  }, [importSourceFormat])

  React.useEffect(() => {
    clearFetchAllItemsCache()
    return () => clearFetchAllItemsCache()
  }, [])

  const previewItemsByType = React.useMemo(() => {
    if (!previewManifest?.content_items?.length) return null
    return groupPreviewItems(previewManifest.content_items)
  }, [previewManifest])

  const previewTypes = React.useMemo(() => {
    if (!previewManifest) return []
    const fromCounts = CONTENT_TYPE_KEYS.filter(
      (key) => getPreviewCount(previewManifest, key) > 0
    )
    const fromItems = previewItemsByType
      ? CONTENT_TYPE_KEYS.filter((key) => previewItemsByType[key]?.length)
      : []
    return Array.from(new Set([...fromCounts, ...fromItems]))
  }, [previewManifest, previewItemsByType])

  const openwebuiDbUsers = React.useMemo<OpenWebUIDatabasePreviewUser[]>(() => {
    const users = openwebuiDbPreview?.users
    return Array.isArray(users) ? users : []
  }, [openwebuiDbPreview])

  const selectedOpenWebUIUser = React.useMemo(
    () =>
      openwebuiDbUsers.find(
        (user) => user.source_user_id === selectedOpenWebUIUserId
      ) || null,
    [openwebuiDbUsers, selectedOpenWebUIUserId]
  )

  const effectiveHydrationSourceUserId =
    isOpenWebUIDatabaseImport && selectedOpenWebUIUserId
      ? selectedOpenWebUIUserId
      : hydrationSourceUserId.trim()

  const hydrationPayload = React.useMemo(() => {
    const scope: {
      conversation_ids: string[]
      source_user_id?: string
    } = {
      conversation_ids: parseIdList(hydrationConversationIdsRaw)
    }
    if (effectiveHydrationSourceUserId) {
      scope.source_user_id = effectiveHydrationSourceUserId
    }
    return {
      openwebui_data_root: hydrationDataRoot.trim(),
      scope,
      process_supported_files: hydrationProcessSupportedFiles
    }
  }, [
    effectiveHydrationSourceUserId,
    hydrationConversationIdsRaw,
    hydrationDataRoot,
    hydrationProcessSupportedFiles
  ])

  const hydrationPayloadSignature = React.useMemo(
    () =>
      JSON.stringify({
        hydrationPayload,
        import_preview: {
          source_format: importSourceFormat,
          version: importPreviewVersion,
          file_name: importFile?.name || "",
          file_size: importFile?.size || 0,
          file_last_modified: importFile?.lastModified || 0
        }
      }),
    [hydrationPayload, importFile, importPreviewVersion, importSourceFormat]
  )

  const hydrationPreviewIsCurrent = Boolean(
    hydrationPreview && hydrationPreviewSignature === hydrationPayloadSignature
  )

  const activeJobs = React.useMemo(() => {
    const exportActive = exportJobs.filter((job) => isActiveJobStatus(job.status))
    const importActive = importJobs.filter((job) => isActiveJobStatus(job.status))
    return {
      exportActive,
      importActive,
      all: [...exportActive, ...importActive]
    }
  }, [exportJobs, importJobs])

  const resetPolling = React.useCallback(() => {
    lastSignatureRef.current = ""
    setPollIndex(0)
  }, [])

  const loadJobs = React.useCallback(async () => {
    if (!canUseChatbooks) return
    setJobsLoading(true)
    setJobsError(null)
    try {
      await tldwClient.initialize().catch(() => null)
      const [exports, imports] = await Promise.all([
        tldwClient.listChatbookExportJobs({ limit: 50, offset: 0 }),
        tldwClient.listChatbookImportJobs({ limit: 50, offset: 0 })
      ])
      setExportJobs(extractJobList(exports))
      setImportJobs(extractJobList(imports))
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error)
      setJobsError(msg)
    } finally {
      setJobsLoading(false)
    }
  }, [canUseChatbooks])

  const pollJobStatus = React.useCallback(async () => {
    if (!canUseChatbooks) return false
    const exportActive = exportJobs.filter((job) => isActiveJobStatus(job.status))
    const importActive = importJobs.filter((job) => isActiveJobStatus(job.status))
    if (!exportActive.length && !importActive.length) return false

    await tldwClient.initialize().catch(() => null)

    const exportUpdates = await Promise.all(
      exportActive.map((job) => tldwClient.getChatbookExportJob(job.job_id).catch(() => job))
    )
    const importUpdates = await Promise.all(
      importActive.map((job) => tldwClient.getChatbookImportJob(job.job_id).catch(() => job))
    )

    const mergeJobs = (prev: ChatbookJob[], updates: ChatbookJob[]) => {
      const map = new Map(prev.map((job) => [job.job_id, job]))
      updates.forEach((job) => map.set(job.job_id, { ...map.get(job.job_id), ...job }))
      return Array.from(map.values())
    }

    const nextExport = mergeJobs(exportJobs, exportUpdates)
    const nextImport = mergeJobs(importJobs, importUpdates)
    setExportJobs(nextExport)
    setImportJobs(nextImport)

    return buildJobSignature([...nextExport, ...nextImport])
  }, [canUseChatbooks, exportJobs, importJobs])

  React.useEffect(() => {
    if (!canUseChatbooks) return
    void loadJobs()
  }, [canUseChatbooks, loadJobs])

  React.useEffect(() => {
    if (!activeJobs.all.length) return

    const timeout = window.setTimeout(async () => {
      try {
        const signature = await pollJobStatus()
        const changed = Boolean(signature && signature !== lastSignatureRef.current)
        if (signature) lastSignatureRef.current = signature
        setPollIndex((prev) => (changed ? 0 : Math.min(prev + 1, POLL_INTERVALS_MS.length - 1)))
      } catch {
        setPollIndex((prev) => Math.min(prev + 1, POLL_INTERVALS_MS.length - 1))
      }
    }, POLL_INTERVALS_MS[pollIndex])

    return () => window.clearTimeout(timeout)
  }, [activeJobs.all.length, exportJobs, importJobs, pollIndex, pollJobStatus])

  const handleDownload = async (jobId: string) => {
    try {
      await tldwClient.initialize().catch(() => null)
      const { blob, filename } = await tldwClient.downloadChatbookExport(jobId)
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      window.setTimeout(() => {
        link.remove()
        URL.revokeObjectURL(url)
      }, 0)
      resetPolling()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.downloadError", "Download failed"),
        description: error instanceof Error ? error.message : String(error)
      })
    }
  }

  const handleCancelJob = async (job: ChatbookJob, kind: JobKind) => {
    try {
      await tldwClient.initialize().catch(() => null)
      if (kind === "export") {
        await tldwClient.cancelChatbookExportJob(job.job_id)
      } else {
        await tldwClient.cancelChatbookImportJob(job.job_id)
      }
      notification.success({
        message: t("settings:chatbooksPlayground.jobCancelled", "Job cancelled")
      })
      resetPolling()
      await loadJobs()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.jobCancelError", "Unable to cancel job"),
        description: error instanceof Error ? error.message : String(error)
      })
    }
  }

  const handleCleanup = async () => {
    try {
      await tldwClient.initialize().catch(() => null)
      const res = await tldwClient.cleanupChatbooks()
      notification.success({
        message: t("settings:chatbooksPlayground.cleanupSuccess", "Cleanup complete"),
        description:
          res?.deleted_count != null
            ? t("settings:chatbooksPlayground.cleanupCount", {
                defaultValue: "Deleted {{count}} files",
                count: res.deleted_count
              })
            : undefined
      })
      resetPolling()
      await loadJobs()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.cleanupError", "Cleanup failed"),
        description: error instanceof Error ? error.message : String(error)
      })
    }
  }

  const handleRemoveJob = async (job: ChatbookJob, kind: JobKind) => {
    try {
      await tldwClient.initialize().catch(() => null)
      if (kind === "export") {
        await tldwClient.removeChatbookExportJob(job.job_id)
      } else {
        await tldwClient.removeChatbookImportJob(job.job_id)
      }
      notification.success({
        message: t("settings:chatbooksPlayground.jobRemoved", "Job removed")
      })
      resetPolling()
      await loadJobs()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.jobRemoveError", "Unable to remove job"),
        description: error instanceof Error ? error.message : String(error)
      })
    }
  }

  const handleRemoveAllFinished = async () => {
    const removableExports = exportJobs.filter(j => isRemovableJobStatus(j.status))
    const removableImports = importJobs.filter(j => isRemovableJobStatus(j.status))

    if (removableExports.length === 0 && removableImports.length === 0) {
      notification.info({
        message: t("settings:chatbooksPlayground.noJobsToRemove", "No finished jobs to remove")
      })
      return
    }

    try {
      await tldwClient.initialize().catch(() => null)
      await Promise.all([
        ...removableExports.map(j => tldwClient.removeChatbookExportJob(j.job_id)),
        ...removableImports.map(j => tldwClient.removeChatbookImportJob(j.job_id))
      ])
      notification.success({
        message: t("settings:chatbooksPlayground.allFinishedRemoved", "All finished jobs removed")
      })
      resetPolling()
      await loadJobs()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.removeAllError", "Failed to remove some jobs"),
        description: error instanceof Error ? error.message : String(error)
      })
      await loadJobs()
    }
  }

  const resolveExportSelections = async () => {
    const entries: Array<readonly [ContentTypeKey, string[]] | null> = []
    for (const key of CONTENT_TYPE_KEYS) {
      if (key === "generated_document" && !includeGenerated) {
        entries.push(null)
        continue
      }
      if (exportIncludeAll[key]) {
        const ids = await fetchAllIds(key)
        entries.push([key, ids] as const)
        continue
      }
      if (exportSelections[key].length) {
        entries.push([key, exportSelections[key]] as const)
        continue
      }
      entries.push(null)
    }

    return entries
      .filter((entry): entry is [ContentTypeKey, string[]] => Boolean(entry))
      .reduce((acc, [key, ids]) => {
        acc[key] = ids
        return acc
      }, {} as Record<ContentTypeKey, string[]>)
  }

  const handleExport = async () => {
    if (!exportName.trim() || !exportDescription.trim()) {
      notification.error({
        message: t("settings:chatbooksPlayground.exportValidation", "Name and description are required.")
      })
      return
    }

    setExportSubmitting(true)
    try {
      await tldwClient.initialize().catch(() => null)
      const contentSelections = await resolveExportSelections()
      const totalSelected = Object.values(contentSelections).reduce(
        (sum, ids) => sum + ids.length,
        0
      )
      if (!Object.keys(contentSelections).length || totalSelected === 0) {
        notification.error({
          message: t("settings:chatbooksPlayground.exportMissing", "Select at least one item to export.")
        })
        return
      }

      const payload = {
        name: exportName.trim(),
        description: exportDescription.trim(),
        author: exportAuthor.trim() || undefined,
        tags: exportTags,
        categories: exportCategories,
        content_selections: contentSelections,
        include_media: includeMedia,
        media_quality: mediaQuality,
        include_embeddings: includeEmbeddings,
        include_generated_content: includeGenerated,
        async_mode: exportAsync
      }

      const res = await tldwClient.exportChatbook(payload)
      notification.success({
        message: res?.job_id
          ? t("settings:chatbooksPlayground.exportQueued", "Export job created")
          : t("settings:chatbooksPlayground.exportComplete", "Export complete")
      })
      clearFetchAllItemsCache()
      resetPolling()
      await loadJobs()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.exportError", "Export failed"),
        description: error instanceof Error ? error.message : String(error)
      })
    } finally {
      setExportSubmitting(false)
    }
  }

  const handlePreview = async (file: File) => {
    const requestId = previewRequestIdRef.current + 1
    previewRequestIdRef.current = requestId
    const sourceFormat = importSourceFormat
    const isCurrentRequest = () => previewRequestIdRef.current === requestId

    setPreviewLoading(true)
    setPreviewError(null)
    setPreviewManifest(null)
    setOpenwebuiPreview(null)
    setOpenwebuiDbPreview(null)
    setImportPreviewVersion(requestId)
    setSelectedOpenWebUIUserId("")
    setImportFile(file)
    setImportSelections(buildSelectionState(() => [] as string[]))
    setImportIncludeAll(buildSelectionState(() => true))
    setHydrationPreview(null)
    setHydrationPreviewSignature("")
    setHydrationPreviewError(null)
    setHydrationJob(null)
    setHydrationJobError(null)
    try {
      await tldwClient.initialize().catch(() => null)
      const res = await tldwClient.previewChatbook(file, {
        source_format: sourceFormat
      })
      if (!isCurrentRequest()) return
      if (res?.error) {
        setPreviewError(res.error)
      } else if (sourceFormat === "openwebui_json") {
        setOpenwebuiPreview(res?.openwebui_preview || null)
      } else if (sourceFormat === "openwebui_db") {
        const dbPreview = res?.openwebui_db_preview || null
        const users = Array.isArray(dbPreview?.users) ? dbPreview.users : []
        setOpenwebuiDbPreview(dbPreview)
        setSelectedOpenWebUIUserId(users.length === 1 ? users[0]?.source_user_id || "" : "")
      } else {
        setPreviewManifest(res?.manifest || null)
      }
    } catch (error) {
      if (isCurrentRequest()) {
        setPreviewError(error instanceof Error ? error.message : String(error))
      }
    } finally {
      if (isCurrentRequest()) {
        setPreviewLoading(false)
      }
    }
  }

  const resolveImportSelections = () => {
    if (isOpenWebUIImport) return undefined
    const anySelective = previewTypes.some((key) => !importIncludeAll[key])
    if (!anySelective) return undefined
    const selections: Record<string, string[]> = {}
    const missingPreviewMessage = t(
      "settings:chatbooksPlayground.previewMissingItems",
      "This preview does not include item details. Import all or re-run preview on a newer server."
    )

    if (!previewItemsByType) {
      previewTypes.forEach((key) => {
        if (importIncludeAll[key]) return
        const selectedIds = importSelections[key]
        if (!selectedIds || selectedIds.length === 0) {
          throw new Error(missingPreviewMessage)
        }
        selections[key] = selectedIds
      })
      return Object.keys(selections).length ? selections : undefined
    }

    previewTypes.forEach((key) => {
      if (importIncludeAll[key]) {
        selections[key] = previewItemsByType[key].map((item) => item.id)
      } else if (importSelections[key].length) {
        selections[key] = importSelections[key]
      }
    })

    return selections
  }

  const handleImport = async () => {
    if (!importFile) {
      notification.error({
        message: t("settings:chatbooksPlayground.importMissing", "Select a chatbook file first.")
      })
      return
    }
    if (isOpenWebUIDatabaseImport && !selectedOpenWebUIUserId) {
      notification.error({
        message: t(
          "settings:chatbooksPlayground.openwebuiDbUserRequired",
          "Select an OpenWebUI user to import."
        )
      })
      return
    }

    setImportSubmitting(true)
    try {
      await tldwClient.initialize().catch(() => null)
      const contentSelections = resolveImportSelections()
      const normalizedSelections =
        !isOpenWebUIImport && contentSelections && Object.keys(contentSelections).length
          ? contentSelections
          : undefined
      if (normalizedSelections) {
        const totalSelected = Object.values(normalizedSelections).reduce(
          (sum, ids) => sum + ids.length,
          0
        )
        if (totalSelected === 0) {
          notification.error({
            message: t(
              "settings:chatbooksPlayground.importMissingSelection",
              "Select at least one item to import."
            )
          })
          return
        }
      }
      const res = await tldwClient.importChatbook(importFile, {
        source_format: importSourceFormat,
        conflict_resolution: conflictResolution,
        prefix_imported: prefixImported,
        import_media: isOpenWebUIImport ? false : importMedia,
        import_embeddings: isOpenWebUIImport ? false : importEmbeddings,
        async_mode: importAsync,
        content_selections: normalizedSelections,
        selected_openwebui_user_id: isOpenWebUIDatabaseImport
          ? selectedOpenWebUIUserId
          : undefined
      })

      notification.success({
        message: res?.job_id
          ? t("settings:chatbooksPlayground.importQueued", "Import job created")
          : t("settings:chatbooksPlayground.importComplete", "Import complete")
      })
      clearFetchAllItemsCache()
      resetPolling()
      await loadJobs()
    } catch (error) {
      notification.error({
        message: t("settings:chatbooksPlayground.importError", "Import failed"),
        description: error instanceof Error ? error.message : String(error)
      })
    } finally {
      setImportSubmitting(false)
    }
  }

  const handleHydrationPreview = async () => {
    if (!hydrationPayload.openwebui_data_root) {
      notification.error({
        message: t(
          "settings:chatbooksPlayground.openwebuiHydrationRootRequired",
          "OpenWebUI data root is required."
        )
      })
      return
    }

    setHydrationPreviewLoading(true)
    setHydrationPreviewError(null)
    setHydrationJob(null)
    setHydrationJobError(null)
    try {
      await tldwClient.initialize().catch(() => null)
      const res = await tldwClient.previewOpenWebUIHydration(hydrationPayload)
      setHydrationPreview(res || null)
      setHydrationPreviewSignature(hydrationPayloadSignature)
    } catch (error) {
      setHydrationPreview(null)
      setHydrationPreviewSignature("")
      setHydrationPreviewError(error instanceof Error ? error.message : String(error))
    } finally {
      setHydrationPreviewLoading(false)
    }
  }

  const handleCreateHydrationJob = async () => {
    if (!hydrationPreviewIsCurrent) {
      notification.error({
        message: t(
          "settings:chatbooksPlayground.openwebuiHydrationPreviewRequired",
          "Preview attachments before creating a hydration job."
        )
      })
      return
    }

    setHydrationJobLoading(true)
    setHydrationJobError(null)
    try {
      await tldwClient.initialize().catch(() => null)
      const res = await tldwClient.createOpenWebUIHydrationJob(hydrationPayload)
      setHydrationJob(res || null)
      notification.success({
        message: t(
          "settings:chatbooksPlayground.openwebuiHydrationJobQueued",
          "Hydration job created"
        )
      })
    } catch (error) {
      setHydrationJobError(error instanceof Error ? error.message : String(error))
    } finally {
      setHydrationJobLoading(false)
    }
  }

  const handleRefreshHydrationJob = async () => {
    if (!hydrationJob?.job_id) return

    setHydrationJobLoading(true)
    setHydrationJobError(null)
    try {
      await tldwClient.initialize().catch(() => null)
      const res = await tldwClient.getOpenWebUIHydrationJob(hydrationJob.job_id)
      setHydrationJob(res || null)
    } catch (error) {
      setHydrationJobError(error instanceof Error ? error.message : String(error))
    } finally {
      setHydrationJobLoading(false)
    }
  }

  const jobTrackerList = React.useMemo(() => {
    const exportCards = exportJobs.map((job) => ({ ...job, kind: "export" as const }))
    const importCards = importJobs.map((job) => ({ ...job, kind: "import" as const }))
    return [...exportCards, ...importCards]
      .sort((a, b) => {
        const aTime = Date.parse(a.started_at || a.created_at || "") || 0
        const bTime = Date.parse(b.started_at || b.created_at || "") || 0
        return bTime - aTime
      })
      .slice(0, 8)
  }, [exportJobs, importJobs])

  const exportPickerConfig = React.useMemo(
    () =>
      CONTENT_TYPE_KEYS.map((key) => ({
        key,
        label: contentLabels[key],
        supportsList: key !== "embedding",
        allowManualEntry: key === "embedding",
        allowIncludeAll: key !== "embedding"
      })),
    [contentLabels]
  )

  const exportTab = (
    <>
      <Card
        className="border-border"
        title={t("settings:chatbooksPlayground.exportTitle", "Export chatbook")}
        extra={
          <Text type="secondary">
            {t("settings:chatbooksPlayground.asyncDefault", "Async by default")}
          </Text>
        }
      >
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Input
              value={exportName}
              onChange={(event) => setExportName(event.target.value)}
              placeholder={t("settings:chatbooksPlayground.exportName", "Name")}
            />
            <Input
              value={exportAuthor}
              onChange={(event) => setExportAuthor(event.target.value)}
              placeholder={t("settings:chatbooksPlayground.exportAuthor", "Author (optional)")}
            />
          </div>
          <Input.TextArea
            rows={3}
            value={exportDescription}
            onChange={(event) => setExportDescription(event.target.value)}
            placeholder={t("settings:chatbooksPlayground.exportDescription", "Description")}
          />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Select
              mode="tags"
              placeholder={t("settings:chatbooksPlayground.tagsPlaceholder", "Tags")}
              value={exportTags}
              onChange={(values) => setExportTags(values)}
            />
            <Select
              mode="tags"
              placeholder={t("settings:chatbooksPlayground.categoriesPlaceholder", "Categories")}
              value={exportCategories}
              onChange={(values) => setExportCategories(values)}
            />
          </div>

          <Divider />

          <Space wrap>
            <Switch checked={includeMedia} onChange={setIncludeMedia} />
            <Text>{t("settings:chatbooksPlayground.includeMedia", "Include media")}</Text>
            <Select
              value={mediaQuality}
              onChange={setMediaQuality}
              options={[
                { value: "thumbnail", label: t("settings:chatbooksPlayground.mediaThumbnail", "Thumbnail") },
                { value: "compressed", label: t("settings:chatbooksPlayground.mediaCompressed", "Compressed") },
                { value: "original", label: t("settings:chatbooksPlayground.mediaOriginal", "Original") }
              ]}
              disabled={!includeMedia}
              className="min-w-[160px]"
            />
          </Space>

          <Space wrap>
            <Switch checked={includeEmbeddings} onChange={setIncludeEmbeddings} />
            <Text>{t("settings:chatbooksPlayground.includeEmbeddings", "Include embeddings")}</Text>
          </Space>

          <Space wrap>
            <Switch checked={includeGenerated} onChange={setIncludeGenerated} />
            <Text>
              {t(
                "settings:chatbooksPlayground.includeGenerated",
                "Include generated content"
              )}
            </Text>
          </Space>

          <Space wrap>
            <Switch checked={exportAsync} onChange={setExportAsync} />
            <Text>{t("settings:chatbooksPlayground.runAsync", "Run as background job")}</Text>
          </Space>
        </div>
      </Card>

      <div className="flex flex-col gap-4">
        {!isOnline && (
          <DesignSystemAlert
            variant="info"
            title={t(
              "settings:chatbooksPlayground.serverOffline",
              "Server connection required"
            )}
          >
            {t(
              "settings:chatbooksPlayground.serverOfflineHint",
              "Connect to your tldw server in Settings to browse and select content for export."
            )}
          </DesignSystemAlert>
        )}
        {exportPickerConfig.map((config) => {
          const generatedDisabled =
            config.key === "generated_document" && !includeGenerated
          return (
            <ContentTypePicker
              key={config.key}
              typeKey={config.key}
              label={config.label}
              helper={
                generatedDisabled
                  ? t(
                      "settings:chatbooksPlayground.generatedDisabled",
                      "Enable generated content to include generated documents."
                    )
                  : undefined
              }
              includeAll={exportIncludeAll[config.key]}
              onIncludeAllChange={(next) =>
                setExportIncludeAll((prev) => ({ ...prev, [config.key]: next }))
              }
              selectedIds={exportSelections[config.key]}
              onSelectionChange={(ids) =>
                setExportSelections((prev) => ({ ...prev, [config.key]: ids }))
              }
              fetcher={
                config.supportsList
                  ? (params) => buildContentItems(config.key, params)
                  : undefined
              }
              disabled={!canUseChatbooks || generatedDisabled}
              allowIncludeAll={config.allowIncludeAll}
              allowManualEntry={config.allowManualEntry}
              manualPlaceholder={t(
                "settings:chatbooksPlayground.manualEmbeddings",
                "Embedding IDs (one per line)"
              )}
              emptyHint={t(
                "settings:chatbooksPlayground.noItems",
                "No items found"
              )}
              truncationLimit={getFetchLimitForType(config.key)}
              suppressAuthErrors={!isOnline}
            />
          )
        })}
      </div>

      <div className="flex justify-end">
        <Button
          type="primary"
          onClick={handleExport}
          loading={exportSubmitting}
          disabled={!canUseChatbooks}
        >
          {t("settings:chatbooksPlayground.exportCta", "Export chatbook")}
        </Button>
      </div>
    </>
  )

  const importAccept = isOpenWebUIDatabaseImport
    ? ".db,.sqlite"
    : isOpenWebUIJsonImport
      ? ".json"
      : ".zip,.chatbook"
  const importDropText = isOpenWebUIDatabaseImport
    ? t(
        "settings:chatbooksPlayground.importDropOpenWebUIDb",
        "Drop an OpenWebUI webui.db or .sqlite database or click to browse"
      )
    : isOpenWebUIJsonImport
      ? t(
          "settings:chatbooksPlayground.importDropOpenWebUI",
          "Drop an OpenWebUI .json export or click to browse"
        )
      : t(
          "settings:chatbooksPlayground.importDrop",
          "Drop a .zip or .chatbook archive or click to browse"
        )

  const hydrationSummary = hydrationPreview?.summary || {}
  const hydrationWarnings = Array.isArray(hydrationPreview?.warnings)
    ? hydrationPreview.warnings
    : []
  const hydrationJobResult = hydrationJob?.result
  const hydrationJobSummary =
    hydrationJobResult && typeof hydrationJobResult === "object"
      ? (hydrationJobResult as { summary?: OpenWebUIHydrationSummary }).summary
      : undefined

  const importTab = (
    <>
      <Card
        className="border-border"
        title={t("settings:chatbooksPlayground.importTitle", "Import chatbook")}
        extra={
          <Text type="secondary">
            {t("settings:chatbooksPlayground.asyncDefault", "Async by default")}
          </Text>
        }
      >
        <div className="flex flex-col gap-4">
          <Space wrap>
            <Text>{t("settings:chatbooksPlayground.importSource", "Source")}</Text>
            <Select
              value={importSourceFormat}
              onChange={(value) => setImportSourceFormat(value as ImportSourceFormat)}
              className="min-w-[220px]"
              options={[
                {
                  value: "chatbook",
                  label: t("settings:chatbooksPlayground.sourceChatbook", "Chatbook archive")
                },
                {
                  value: "openwebui_json",
                  label: t("settings:chatbooksPlayground.sourceOpenWebUI", "OpenWebUI JSON")
                },
                {
                  value: "openwebui_db",
                  label: t("settings:chatbooksPlayground.sourceOpenWebUIDatabase", "OpenWebUI database")
                }
              ]}
              disabled={!canUseChatbooks || previewLoading || importSubmitting}
            />
          </Space>

          <Upload.Dragger
            accept={importAccept}
            multiple={false}
            showUploadList={false}
            beforeUpload={(file) => {
              void handlePreview(file as File)
              return false
            }}
            disabled={!canUseChatbooks || previewLoading || importSubmitting}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              {importDropText}
            </p>
            <p className="ant-upload-hint">
              {importFile?.name || t("settings:chatbooksPlayground.importHint", "Preview before import")}
            </p>
          </Upload.Dragger>

          {previewLoading && (
            <DesignSystemAlert
              variant="info"
              title={t("settings:chatbooksPlayground.previewLoading", "Previewing chatbook...")}
            />
          )}

          {previewError && (
            <DesignSystemAlert
              variant="error"
              title={t("settings:chatbooksPlayground.previewError", "Preview failed")}
            >
              {previewError}
            </DesignSystemAlert>
          )}

          {previewManifest && (
            <Card
              size="small"
              className="border-border"
              title={t("settings:chatbooksPlayground.previewSummary", "Manifest summary")}
            >
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.previewName", "Name")}</Text>
                  <div>{previewManifest.name || "—"}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.previewAuthor", "Author")}</Text>
                  <div>{previewManifest.author || "—"}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.previewDescription", "Description")}</Text>
                  <div>{previewManifest.description || "—"}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.previewSize", "Archive size")}</Text>
                  <div>
                    {previewManifest.total_size_bytes != null
                      ? formatFileSize(Number(previewManifest.total_size_bytes))
                      : "—"}
                  </div>
                </div>
              </div>
              {previewTypes.length > 0 && (
                <div className="mt-3">
                  <Text type="secondary">
                    {t("settings:chatbooksPlayground.previewContents", "Contents")}
                  </Text>
                  <Space wrap className="mt-2">
                    {previewTypes.map((key) => (
                      <Tag key={key}>
                        {contentLabels[key]} · {getPreviewCount(previewManifest, key)}
                      </Tag>
                    ))}
                  </Space>
                </div>
              )}
            </Card>
          )}

          {openwebuiPreview && (
            <Card
              size="small"
              className="border-border"
              title={t("settings:chatbooksPlayground.openwebuiPreviewSummary", "OpenWebUI preview")}
            >
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.openwebuiChats", "Chats")}</Text>
                  <div>{openwebuiPreview.chat_count ?? 0}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.openwebuiMessages", "Messages")}</Text>
                  <div>{openwebuiPreview.message_count ?? 0}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.openwebuiBranches", "Branched")}</Text>
                  <div>{openwebuiPreview.branched_chat_count ?? 0}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.openwebuiDuplicates", "Duplicates")}</Text>
                  <div>{openwebuiPreview.duplicate_chat_count ?? 0}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.openwebuiAttachments", "Attachment refs")}</Text>
                  <div>{openwebuiPreview.attachment_reference_count ?? 0}</div>
                </div>
                <div>
                  <Text type="secondary">{t("settings:chatbooksPlayground.openwebuiMalformed", "Malformed")}</Text>
                  <div>{openwebuiPreview.malformed_chat_count ?? 0}</div>
                </div>
              </div>
              {Boolean(openwebuiPreview.warnings?.length) && (
                <DesignSystemAlert
                  variant="warning"
                  className="mt-3"
                  title={t("settings:chatbooksPlayground.openwebuiWarnings", "Import warnings")}
                >
                  {openwebuiPreview.warnings.slice(0, 3).join("\n")}
                </DesignSystemAlert>
              )}
            </Card>
          )}

          {openwebuiDbPreview && (
            <Card
              size="small"
              className="border-border"
              title={t(
                "settings:chatbooksPlayground.openwebuiDbPreviewSummary",
                "OpenWebUI database users"
              )}
            >
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {openwebuiDbUsers.map((user) => (
                  <div
                    key={user.source_user_id}
                    className="rounded border border-border p-3"
                  >
                    <Text strong>{user.display_label || user.source_user_id}</Text>
                    <div className="text-sm text-muted-foreground">
                      {user.email || user.source_user_id}
                    </div>
                    <Space wrap size="small" className="mt-2">
                      <Tag>
                        {t("settings:chatbooksPlayground.openwebuiChats", "Chats")} ·{" "}
                        {user.chat_count ?? 0}
                      </Tag>
                      <Tag>
                        {t("settings:chatbooksPlayground.openwebuiMessages", "Messages")} ·{" "}
                        {user.message_count ?? 0}
                      </Tag>
                      <Tag>
                        {t("settings:chatbooksPlayground.openwebuiFolders", "Folders")} ·{" "}
                        {user.folder_count ?? 0}
                      </Tag>
                      <Tag>
                        {t("settings:chatbooksPlayground.openwebuiAttachments", "Attachment refs")} ·{" "}
                        {user.attachment_reference_count ?? 0}
                      </Tag>
                    </Space>
                  </div>
                ))}
              </div>

              <div className="mt-3">
                <Select
                  placeholder={t(
                    "settings:chatbooksPlayground.openwebuiDbUserSelect",
                    "Select source user"
                  )}
                  value={selectedOpenWebUIUserId || undefined}
                  onChange={(value) => setSelectedOpenWebUIUserId(String(value || ""))}
                  className="min-w-[260px]"
                  options={openwebuiDbUsers.map((user) => ({
                    value: user.source_user_id,
                    label: `${user.display_label || user.source_user_id} (${user.source_user_id})`
                  }))}
                  disabled={previewLoading || importSubmitting}
                />
              </div>

              {selectedOpenWebUIUser && (
                <DesignSystemAlert
                  variant="info"
                  className="mt-3"
                  title={t("settings:chatbooksPlayground.openwebuiDbDestination", "Destination")}
                >
                  {`Destination: OpenWebUI / ${
                    selectedOpenWebUIUser.display_label || selectedOpenWebUIUser.source_user_id
                  } (${selectedOpenWebUIUser.source_user_id}) / source folders`}
                </DesignSystemAlert>
              )}

              <DesignSystemAlert
                variant="warning"
                className="mt-3"
                title={t(
                  "settings:chatbooksPlayground.openwebuiDbAttachmentRefs",
                  "Files and images"
                )}
              >
                {t(
                  "settings:chatbooksPlayground.openwebuiDbAttachmentRefsDescription",
                  "Files, images, and artifacts import as metadata references first. Use attachment hydration below to copy images and register supported files from your OpenWebUI data root."
                )}
              </DesignSystemAlert>

              {Boolean(openwebuiDbPreview.warnings?.length) && (
                <DesignSystemAlert
                  variant="warning"
                  className="mt-3"
                  title={t("settings:chatbooksPlayground.openwebuiWarnings", "Import warnings")}
                >
                  {openwebuiDbPreview.warnings.slice(0, 3).join("\n")}
                </DesignSystemAlert>
              )}
            </Card>
          )}

          {isOpenWebUIImport && (
            <Card
              size="small"
              className="border-border"
              title={t(
                "settings:chatbooksPlayground.openwebuiHydrationTitle",
                "OpenWebUI attachment hydration"
              )}
            >
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div>
                  <Text type="secondary">
                    {t(
                      "settings:chatbooksPlayground.openwebuiHydrationRoot",
                      "Data root"
                    )}
                  </Text>
                  <Input
                    aria-label={t(
                      "settings:chatbooksPlayground.openwebuiHydrationRootLabel",
                      "OpenWebUI data root"
                    ) as string}
                    value={hydrationDataRoot}
                    onChange={(event) => setHydrationDataRoot(event.target.value)}
                    placeholder="/app/backend/data"
                    disabled={!canUseChatbooks || hydrationPreviewLoading || hydrationJobLoading}
                  />
                </div>
                <div>
                  <Text type="secondary">
                    {t(
                      "settings:chatbooksPlayground.openwebuiHydrationSourceUser",
                      "Source user"
                    )}
                  </Text>
                  <Input
                    aria-label={t(
                      "settings:chatbooksPlayground.openwebuiHydrationSourceUserLabel",
                      "OpenWebUI source user id"
                    ) as string}
                    value={
                      isOpenWebUIDatabaseImport && selectedOpenWebUIUserId
                        ? selectedOpenWebUIUserId
                        : hydrationSourceUserId
                    }
                    onChange={(event) => setHydrationSourceUserId(event.target.value)}
                    placeholder={t(
                      "settings:chatbooksPlayground.openwebuiHydrationSourceUserPlaceholder",
                      "Optional source user id"
                    ) as string}
                    disabled={
                      !canUseChatbooks ||
                      hydrationPreviewLoading ||
                      hydrationJobLoading ||
                      (isOpenWebUIDatabaseImport && Boolean(selectedOpenWebUIUserId))
                    }
                  />
                </div>
                <div className="md:col-span-2">
                  <Text type="secondary">
                    {t(
                      "settings:chatbooksPlayground.openwebuiHydrationConversationIds",
                      "Imported conversation IDs"
                    )}
                  </Text>
                  <Input.TextArea
                    aria-label={t(
                      "settings:chatbooksPlayground.openwebuiHydrationConversationIdsLabel",
                      "Imported conversation IDs"
                    ) as string}
                    value={hydrationConversationIdsRaw}
                    onChange={(event) => setHydrationConversationIdsRaw(event.target.value)}
                    autoSize={{ minRows: 2, maxRows: 4 }}
                    placeholder={t(
                      "settings:chatbooksPlayground.openwebuiHydrationConversationIdsPlaceholder",
                      "Paste tldw conversation ids, one per line or comma-separated"
                    ) as string}
                    disabled={!canUseChatbooks || hydrationPreviewLoading || hydrationJobLoading}
                  />
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Space>
                  <Switch
                    aria-label={t(
                      "settings:chatbooksPlayground.openwebuiHydrationProcessFiles",
                      "Process supported files"
                    ) as string}
                    checked={hydrationProcessSupportedFiles}
                    onChange={setHydrationProcessSupportedFiles}
                    disabled={!canUseChatbooks || hydrationPreviewLoading || hydrationJobLoading}
                  />
                  <Text>
                    {t(
                      "settings:chatbooksPlayground.openwebuiHydrationProcessFiles",
                      "Process supported files"
                    )}
                  </Text>
                </Space>
                <Button
                  aria-label={t(
                    "settings:chatbooksPlayground.openwebuiHydrationPreview",
                    "Preview attachments"
                  ) as string}
                  onClick={() => void handleHydrationPreview()}
                  loading={hydrationPreviewLoading}
                  disabled={
                    !canUseChatbooks ||
                    !hydrationPayload.openwebui_data_root ||
                    hydrationJobLoading
                  }
                >
                  {t(
                    "settings:chatbooksPlayground.openwebuiHydrationPreview",
                    "Preview attachments"
                  )}
                </Button>
                <Button
                  aria-label={t(
                    "settings:chatbooksPlayground.openwebuiHydrationRun",
                    "Run hydration job"
                  ) as string}
                  type="primary"
                  onClick={() => void handleCreateHydrationJob()}
                  loading={hydrationJobLoading}
                  disabled={
                    !canUseChatbooks ||
                    !hydrationPreviewIsCurrent ||
                    hydrationPreviewLoading
                  }
                >
                  {t(
                    "settings:chatbooksPlayground.openwebuiHydrationRun",
                    "Run hydration job"
                  )}
                </Button>
              </div>

              {hydrationPreview && !hydrationPreviewIsCurrent && (
                <DesignSystemAlert
                  variant="info"
                  className="mt-3"
                  title={t(
                    "settings:chatbooksPlayground.openwebuiHydrationPreviewStale",
                    "Preview needs refresh"
                  )}
                />
              )}

              {hydrationPreviewError && (
                <DesignSystemAlert
                  variant="error"
                  className="mt-3"
                  title={t(
                    "settings:chatbooksPlayground.openwebuiHydrationPreviewError",
                    "Attachment preview failed"
                  )}
                >
                  {hydrationPreviewError}
                </DesignSystemAlert>
              )}

              {hydrationPreview && (
                <div className="mt-3">
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationReferenced",
                          "Referenced files"
                        )}
                      </Text>
                      <div>{hydrationSummary.referenced_files ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationResolved",
                          "Resolved files"
                        )}
                      </Text>
                      <div>{hydrationSummary.resolved_files ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationImages",
                          "Image files"
                        )}
                      </Text>
                      <div>{hydrationSummary.image_files ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationMedia",
                          "Media files"
                        )}
                      </Text>
                      <div>{hydrationSummary.media_files ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationMissing",
                          "Missing files"
                        )}
                      </Text>
                      <div>{hydrationSummary.missing_files ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationUnsupported",
                          "Unsupported files"
                        )}
                      </Text>
                      <div>{hydrationSummary.unsupported_files ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationHydratedImages",
                          "Hydrated images"
                        )}
                      </Text>
                      <div>{hydrationSummary.hydrated_images ?? 0}</div>
                    </div>
                    <div>
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationRegisteredMedia",
                          "Registered media"
                        )}
                      </Text>
                      <div>{hydrationSummary.registered_media_files ?? 0}</div>
                    </div>
                  </div>

                  {Boolean(hydrationWarnings.length) && (
                    <DesignSystemAlert
                      variant="warning"
                      className="mt-3"
                      title={t(
                        "settings:chatbooksPlayground.openwebuiHydrationWarnings",
                        "Hydration warnings"
                      )}
                    >
                      {hydrationWarnings.slice(0, 3).join("\n")}
                    </DesignSystemAlert>
                  )}
                </div>
              )}

              {hydrationJobError && (
                <DesignSystemAlert
                  variant="error"
                  className="mt-3"
                  title={t(
                    "settings:chatbooksPlayground.openwebuiHydrationJobError",
                    "Hydration job failed"
                  )}
                >
                  {hydrationJobError}
                </DesignSystemAlert>
              )}

              {hydrationJob && (
                <DesignSystemAlert
                  variant={hydrationJob.status === "failed" ? "error" : "info"}
                  className="mt-3"
                  title={t(
                    "settings:chatbooksPlayground.openwebuiHydrationJobStatus",
                    "Hydration job"
                  )}
                >
                  <div className="flex flex-col gap-1">
                    <Space wrap>
                      {hydrationJob.job_id && <Tag>{hydrationJob.job_id}</Tag>}
                      {hydrationJob.status && <Tag>{hydrationJob.status}</Tag>}
                      <Button
                        size="small"
                        onClick={() => void handleRefreshHydrationJob()}
                        loading={hydrationJobLoading}
                      >
                        {t(
                          "settings:chatbooksPlayground.refresh",
                          "Refresh"
                        )}
                      </Button>
                    </Space>
                    {hydrationJobSummary && (
                      <Text type="secondary">
                        {t(
                          "settings:chatbooksPlayground.openwebuiHydrationJobSummary",
                          {
                            defaultValue:
                              "Hydrated {{images}} images and registered {{media}} files.",
                            images: hydrationJobSummary.hydrated_images ?? 0,
                            media: hydrationJobSummary.registered_media_files ?? 0
                          }
                        )}
                      </Text>
                    )}
                  </div>
                </DesignSystemAlert>
              )}
            </Card>
          )}

          <Divider />

          <Space wrap>
            <Select
              value={conflictResolution}
              onChange={setConflictResolution}
              options={
                isOpenWebUIImport
                  ? [
                      { value: "skip", label: t("settings:chatbooksPlayground.conflictSkip", "Skip") },
                      { value: "rename", label: t("settings:chatbooksPlayground.conflictRename", "Rename") }
                    ]
                  : [
                      { value: "skip", label: t("settings:chatbooksPlayground.conflictSkip", "Skip") },
                      { value: "overwrite", label: t("settings:chatbooksPlayground.conflictOverwrite", "Overwrite") },
                      { value: "rename", label: t("settings:chatbooksPlayground.conflictRename", "Rename") },
                      { value: "merge", label: t("settings:chatbooksPlayground.conflictMerge", "Merge") }
                    ]
              }
              className="min-w-[160px]"
            />
            <Switch checked={prefixImported} onChange={setPrefixImported} />
            <Text>{t("settings:chatbooksPlayground.prefixImported", "Prefix imported")}</Text>
          </Space>

          {!isOpenWebUIImport && (
            <>
              <Space wrap>
                <Switch checked={importMedia} onChange={setImportMedia} />
                <Text>{t("settings:chatbooksPlayground.importMedia", "Import media")}</Text>
              </Space>

              <Space wrap>
                <Switch checked={importEmbeddings} onChange={setImportEmbeddings} />
                <Text>{t("settings:chatbooksPlayground.importEmbeddings", "Import embeddings")}</Text>
              </Space>
            </>
          )}

          <Space wrap>
            <Switch checked={importAsync} onChange={setImportAsync} />
            <Text>{t("settings:chatbooksPlayground.runAsync", "Run as background job")}</Text>
          </Space>
        </div>
      </Card>

      {!isOpenWebUIImport && previewManifest && previewTypes.length > 0 && (
        <div className="flex flex-col gap-4">
          {previewTypes.map((key) => (
            <ContentTypePicker
              key={key}
              typeKey={key}
              label={contentLabels[key]}
              includeAll={importIncludeAll[key]}
              onIncludeAllChange={(next) =>
                setImportIncludeAll((prev) => ({ ...prev, [key]: next }))
              }
              selectedIds={importSelections[key]}
              onSelectionChange={(ids) =>
                setImportSelections((prev) => ({ ...prev, [key]: ids }))
              }
              itemsOverride={previewItemsByType?.[key]}
              allowIncludeAll={true}
              allowManualEntry={!previewItemsByType}
              manualPlaceholder={t(
                "settings:chatbooksPlayground.manualImport",
                "Paste IDs to import"
              )}
              emptyHint={t(
                "settings:chatbooksPlayground.previewEmpty",
                "No items found in preview"
              )}
              truncationLimit={getFetchLimitForType(key)}
            />
          ))}
        </div>
      )}

      {!isOpenWebUIImport && previewManifest && previewTypes.length === 0 && (
        <DesignSystemAlert
          variant="info"
          title={t(
            "settings:chatbooksPlayground.previewNoTypes",
            "Preview did not return item details."
          )}
        />
      )}

      <div className="flex justify-end">
        <Button
          type="primary"
          onClick={handleImport}
          loading={importSubmitting}
          disabled={
            !canUseChatbooks ||
            !importFile ||
            previewLoading ||
            (isOpenWebUIDatabaseImport && !selectedOpenWebUIUserId)
          }
        >
          {t("settings:chatbooksPlayground.importCta", "Import chatbook")}
        </Button>
      </div>
    </>
  )

  const jobsTab = (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Title level={4} className="m-0">
          {t("settings:chatbooksPlayground.jobsTitle", "Job status")}
        </Title>
        <Space>
          <Button
            icon={<RotateCcw className="h-4 w-4" />}
            onClick={() => {
              resetPolling()
              void loadJobs()
            }}
          >
            {t("settings:chatbooksPlayground.refresh", "Refresh")}
          </Button>
          <Button
            icon={<Trash2 className="h-4 w-4" />}
            onClick={handleCleanup}
          >
            {t("settings:chatbooksPlayground.cleanup", "Cleanup exports")}
          </Button>
          <Button
            icon={<Trash2 className="h-4 w-4" />}
            onClick={handleRemoveAllFinished}
          >
            {t("settings:chatbooksPlayground.removeAllFinished", "Remove finished")}
          </Button>
        </Space>
      </div>

      {jobsError && (
        <DesignSystemAlert
          variant="warning"
          title={t("settings:chatbooksPlayground.jobsError", "Unable to load jobs")}
        >
          {jobsError}
        </DesignSystemAlert>
      )}

      <Card title={t("settings:chatbooksPlayground.exportJobs", "Export jobs")}>
        <Table
          rowKey={(record) => record.job_id}
          dataSource={exportJobs}
          loading={jobsLoading}
          pagination={false}
          columns={[
            {
              title: t("settings:chatbooksPlayground.jobName", "Name"),
              dataIndex: "chatbook_name"
            },
            {
              title: t("settings:chatbooksPlayground.jobStatus", "Status"),
              dataIndex: "status",
              render: (value: string) => <Tag color={statusColor(value)}>{value}</Tag>
            },
            {
              title: t("settings:chatbooksPlayground.jobProgress", "Progress"),
              render: (job: ChatbookJob) => {
                const progress = computeProgress(job)
                return progress != null ? <Progress percent={progress} size="small" /> : "—"
              }
            },
            {
              title: t("settings:chatbooksPlayground.jobCreated", "Created"),
              dataIndex: "created_at",
              render: (value: string) => formatTimestamp(value)
            },
            {
              title: t("settings:chatbooksPlayground.jobError", "Error"),
              dataIndex: "error_message",
              render: (value: string) => value || "—"
            },
            {
              title: t("settings:chatbooksPlayground.jobWarnings", "Warnings"),
              render: (job: ChatbookJob) =>
                job.warnings?.length ? job.warnings.length : "—"
            },
            {
              title: t("settings:chatbooksPlayground.jobConflicts", "Conflicts"),
              render: (job: ChatbookJob) =>
                job.conflicts?.length ? job.conflicts.length : "—"
            },
            {
              title: t("settings:chatbooksPlayground.jobActions", "Actions"),
              render: (job: ChatbookJob) => (
                <Space>
                  {job.status === "completed" && (
                    <Button
                      size="small"
                      icon={<Download className="h-4 w-4" />}
                      onClick={() => handleDownload(job.job_id)}
                    >
                      {t("settings:chatbooksPlayground.download", "Download")}
                    </Button>
                  )}
                  {isActiveJobStatus(job.status) && (
                    <Button
                      size="small"
                      danger
                      icon={<XCircle className="h-4 w-4" />}
                      onClick={() => handleCancelJob(job, "export")}
                    >
                      {t("settings:chatbooksPlayground.cancel", "Cancel")}
                    </Button>
                  )}
                  {isRemovableJobStatus(job.status) && (
                    <Button
                      size="small"
                      icon={<Trash2 className="h-4 w-4" />}
                      onClick={() => handleRemoveJob(job, "export")}
                    >
                      {t("settings:chatbooksPlayground.remove", "Remove")}
                    </Button>
                  )}
                </Space>
              )
            }
          ]}
        />
      </Card>

      <Card title={t("settings:chatbooksPlayground.importJobs", "Import jobs")}>
        <Table
          rowKey={(record) => record.job_id}
          dataSource={importJobs}
          loading={jobsLoading}
          pagination={false}
          columns={[
            {
              title: t("settings:chatbooksPlayground.jobId", "Job ID"),
              dataIndex: "job_id"
            },
            {
              title: t("settings:chatbooksPlayground.jobStatus", "Status"),
              dataIndex: "status",
              render: (value: string) => <Tag color={statusColor(value)}>{value}</Tag>
            },
            {
              title: t("settings:chatbooksPlayground.jobProgress", "Progress"),
              render: (job: ChatbookJob) => {
                const progress = computeProgress(job)
                return progress != null ? <Progress percent={progress} size="small" /> : "—"
              }
            },
            {
              title: t("settings:chatbooksPlayground.jobCreated", "Created"),
              dataIndex: "created_at",
              render: (value: string) => formatTimestamp(value)
            },
            {
              title: t("settings:chatbooksPlayground.jobError", "Error"),
              dataIndex: "error_message",
              render: (value: string) => value || "—"
            },
            {
              title: t("settings:chatbooksPlayground.jobActions", "Actions"),
              render: (job: ChatbookJob) => (
                <Space>
                  {isActiveJobStatus(job.status) && (
                    <Button
                      size="small"
                      danger
                      icon={<XCircle className="h-4 w-4" />}
                      onClick={() => handleCancelJob(job, "import")}
                    >
                      {t("settings:chatbooksPlayground.cancel", "Cancel")}
                    </Button>
                  )}
                  {isRemovableJobStatus(job.status) && (
                    <Button
                      size="small"
                      icon={<Trash2 className="h-4 w-4" />}
                      onClick={() => handleRemoveJob(job, "import")}
                    >
                      {t("settings:chatbooksPlayground.remove", "Remove")}
                    </Button>
                  )}
                </Space>
              )
            }
          ]}
        />
      </Card>
    </div>
  )

  const tabItems = [
    {
      key: "export",
      label: t("settings:chatbooksPlayground.tabExport", "Export"),
      children: exportTab
    },
    {
      key: "import",
      label: t("settings:chatbooksPlayground.tabImport", "Import"),
      children: importTab
    },
    {
      key: "jobs",
      label: t("settings:chatbooksPlayground.tabJobs", "Jobs"),
      children: jobsTab
    }
  ]

  return (
    <WorkspaceConnectionGate
      featureName={t("settings:chatbooksPlayground.title", "Chatbooks")}
      setupDescription={t(
        "settings:chatbooksPlayground.offline",
        "Connect to your tldw server to use Chatbooks."
      )}
      maxWidthClassName="max-w-6xl"
    >
      <PageShell maxWidthClassName="max-w-6xl">
      <div className="flex flex-col gap-6">
        <div>
          <Title level={3}>{t("settings:chatbooksPlayground.title", "Chatbooks Playground")}</Title>
          <Paragraph type="secondary">
            {t(
              "settings:chatbooksPlayground.subtitle",
              "Export and import chatbooks with full control over content selection and job tracking."
            )}
          </Paragraph>
        </div>

        {!canUseChatbooks && (
          <DesignSystemAlert
            variant="warning"
            title={t(
              "settings:chatbooksPlayground.unavailable",
              "Chatbooks is not available on this server."
            )}
          />
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="flex flex-col gap-6">
            <Tabs activeKey={activeTab} onChange={setActiveTab} items={tabItems} />
          </div>

          <div className="flex flex-col gap-4 lg:sticky lg:top-24 self-start">
            <Card
              title={t("settings:chatbooksPlayground.jobTracker", "Job tracker")}
              extra={
                <Button
                  size="small"
                  icon={<RotateCcw className="h-4 w-4" />}
                  onClick={() => {
                    resetPolling()
                    void loadJobs()
                  }}
                >
                  {t("settings:chatbooksPlayground.refresh", "Refresh")}
                </Button>
              }
            >
              {jobTrackerList.length === 0 ? (
                <Empty
                  description={t(
                    "settings:chatbooksPlayground.noJobs",
                    "No chatbook jobs yet."
                  )}
                />
              ) : (
                <div className="divide-y divide-border">
                  {jobTrackerList.map((job: ChatbookJob & { kind: JobKind }) => {
                    const progress = computeProgress(job)
                    return (
                      <div
                        key={`${job.kind}-${job.job_id}`}
                        className="flex w-full flex-col gap-2 py-3 first:pt-0 last:pb-0"
                      >
                        <div className="flex items-center justify-between">
                          <Text strong>
                            {jobLabels[job.kind]} · {job.chatbook_name || job.job_id}
                          </Text>
                          <Tag color={statusColor(job.status)}>{job.status}</Tag>
                        </div>
                        {progress != null && <Progress percent={progress} size="small" />}
                        {job.error_message && (
                          <Text type="danger" className="text-xs">
                            {job.error_message}
                          </Text>
                        )}
                        <Space wrap>
                          {job.kind === "export" && job.status === "completed" && (
                            <Button
                              size="small"
                              icon={<Download className="h-4 w-4" />}
                              onClick={() => handleDownload(job.job_id)}
                            >
                              {t("settings:chatbooksPlayground.download", "Download")}
                            </Button>
                          )}
                          {isActiveJobStatus(job.status) && (
                            <Button
                              size="small"
                              danger
                              icon={<XCircle className="h-4 w-4" />}
                              onClick={() => handleCancelJob(job, job.kind)}
                            >
                              {t("settings:chatbooksPlayground.cancel", "Cancel")}
                            </Button>
                          )}
                          {isRemovableJobStatus(job.status) && (
                            <Button
                              size="small"
                              icon={<Trash2 className="h-4 w-4" />}
                              onClick={() => handleRemoveJob(job, job.kind)}
                            >
                              {t("settings:chatbooksPlayground.remove", "Remove")}
                            </Button>
                          )}
                        </Space>
                        <Text type="secondary" className="text-xs">
                          {formatTimestamp(job.started_at || job.created_at)}
                        </Text>
                      </div>
                    )
                  })}
                </div>
              )}
            </Card>

            <Card className="border-border" title={t("settings:chatbooksPlayground.polling", "Polling")}
                  size="small">
              <Text type="secondary">
                {t(
                  "settings:chatbooksPlayground.pollingHint",
                  "Polling backs off from 3s to 30s while jobs are in progress."
                )}
              </Text>
              <div className="mt-2 text-xs text-text-muted">
                {t("settings:chatbooksPlayground.pollingNext", {
                  defaultValue: "Next poll in ~{{seconds}}s",
                  seconds: Math.round(POLL_INTERVALS_MS[pollIndex] / 1000)
                })}
              </div>
            </Card>
          </div>
        </div>
      </div>
      </PageShell>
    </WorkspaceConnectionGate>
  )
}

export default ChatbooksPlaygroundPage
