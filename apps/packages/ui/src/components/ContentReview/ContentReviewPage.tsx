import React from "react"
import {
  Button,
  Checkbox,
  Collapse,
  Empty,
  Input,
  List,
  Select,
  Space,
  Tag,
  Tooltip,
  Typography,
  message
} from "antd"
import { useLocation, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { DiffViewModal } from "@/components/Media/DiffViewModal"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { buildContentFromSections, detectSections } from "@/utils/content-review"
import {
  AI_CORRECTION_PROMPT,
  CONTENT_REVIEW_TEMPLATES,
  extractChatContent,
  wrapDraftForPrompt
} from "@/utils/content-review-ai"
import { getTextStats } from "@/utils/text-stats"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open"
import { bgRequest, bgUpload } from "@/services/background-proxy"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"
import { resolvePerformChunking } from "@/services/tldw/ingest-defaults"
import { normalizeMediaTypeForUpload } from "@/services/tldw/media-routing"
import { fetchChatModels } from "@/services/tldw-server"
import {
  DRAFT_STORAGE_CAP_BYTES,
  getDraftAsset,
  getDraftBatches,
  getDraftById,
  getDraftsByBatch,
  storeDraftAsset,
  upsertContentDraft
} from "@/db/dexie/drafts"
import type { ContentDraft, DraftBatch, DraftSection } from "@/db/dexie/types"
import { db } from "@/db/dexie/schema"

const statusColor = (status?: ContentDraft["status"]) => {
  switch (status) {
    case "reviewed":
      return "gold"
    case "committed":
      return "green"
    case "discarded":
      return "red"
    case "in_progress":
      return "blue"
    default:
      return "default"
  }
}

const extractMediaId = (
  data: any,
  visited?: WeakSet<object>,
  depth = 0
): string | null => {
  if (!data || typeof data !== "object") return null
  if (depth > 10) return null
  if (!visited) visited = new WeakSet<object>()
  if (visited.has(data as object)) return null
  visited.add(data as object)

  const direct =
    (data as any).media_id ??
    (data as Record<string, unknown>).db_id ??
    (data as any).id ??
    (data as any).pk ??
    (data as any).uuid
  if (direct !== undefined && direct !== null) {
    return String(direct)
  }
  if ((data as any).media && typeof (data as any).media === "object") {
    return extractMediaId((data as any).media, visited, depth + 1)
  }
  if ((data as any).result) {
    return extractMediaId((data as any).result, visited, depth + 1)
  }
  if (Array.isArray((data as any).results) && (data as any).results.length > 0) {
    return extractMediaId((data as any).results[0], visited, depth + 1)
  }
  return null
}

const buildFields = (draft: ContentDraft) => {
  const fields: Record<string, any> = {
    media_type: normalizeMediaTypeForUpload(draft.mediaType),
    perform_analysis: Boolean(draft.processingOptions?.perform_analysis),
    perform_chunking: resolvePerformChunking(draft.processingOptions?.perform_chunking),
    overwrite_existing: Boolean(draft.processingOptions?.overwrite_existing)
  }
  const nested: Record<string, any> = {}
  const assignPath = (obj: any, path: string[], val: any) => {
    let cur = obj
    for (let i = 0; i < path.length; i++) {
      const seg = path[i]
      if (i === path.length - 1) cur[seg] = val
      else cur = (cur[seg] = cur[seg] || {})
    }
  }
  for (const [key, value] of Object.entries(
    draft.processingOptions?.advancedValues || {}
  )) {
    if (key.includes(".")) assignPath(nested, key.split("."), value)
    else fields[key] = value
  }
  for (const [key, value] of Object.entries(nested)) fields[key] = value
  return fields
}

const buildSyntheticFile = (draft: ContentDraft) => {
  const format = draft.contentFormat || "plain"
  const ext = format === "markdown" ? "md" : "txt"
  const nameBase = draft.title?.trim() || "review-draft"
  const safeName = nameBase.replace(/[^\w.-]+/g, "_").slice(0, 64)
  const fileName = `${safeName}.${ext}`
  const mimeType = format === "markdown" ? "text/markdown" : "text/plain"
  const encoder = new TextEncoder()
  const data = encoder.encode(draft.content || "")
  return { fileName, mimeType, data }
}

const buildSafeMetadata = (
  draft: ContentDraft,
  includeOriginalType: boolean
) => {
  const meta =
    draft.metadata && typeof draft.metadata === "object" ? draft.metadata : {}
  const safeMetadata = { ...meta }
  if (includeOriginalType) {
    safeMetadata.original_media_type = draft.mediaType
  }
  return Object.keys(safeMetadata).length > 0 ? safeMetadata : null
}

const commitDraftToServer = async (
  draft: ContentDraft,
  onWarning?: (msg: string) => void
): Promise<ContentDraft> => {
  const fields = buildFields(draft)
  let includeOriginalType = false
  let uploadResp: any

  if (draft.source.kind === "url" && draft.source.url) {
    fields.urls = [draft.source.url]
    uploadResp = await bgUpload({
      path: "/api/v1/media/add",
      method: "POST",
      fields
    })
  } else if (draft.sourceAssetId) {
    const asset = await getDraftAsset(draft.sourceAssetId)
    if (!asset) {
      throw new Error("Source file missing. Please reattach to commit.")
    }
    const data = await asset.blob.arrayBuffer()
    uploadResp = await bgUpload({
      path: "/api/v1/media/add",
      method: "POST",
      fields,
      file: {
        name: asset.fileName,
        type: asset.mimeType,
        data
      }
    })
  } else if (draft.mediaType === "audio" || draft.mediaType === "video") {
    throw new Error(
      "Audio/video drafts require the original source file to commit."
    )
  } else {
    includeOriginalType = true
    const synthetic = buildSyntheticFile(draft)
    uploadResp = await bgUpload({
      path: "/api/v1/media/add",
      method: "POST",
      fields: {
        ...fields,
        media_type: "document"
      },
      file: {
        name: synthetic.fileName,
        type: synthetic.mimeType,
        data: synthetic.data.buffer
      }
    })
  }

  const mediaId = extractMediaId(uploadResp)
  if (!mediaId) {
    throw new Error("Media ID not returned from server.")
  }

  const updatePayload: Record<string, any> = {}
  if (draft.title) updatePayload.title = draft.title
  if (draft.content) updatePayload.content = draft.content
  if (draft.keywords) updatePayload.keywords = draft.keywords
  if (draft.analysis) updatePayload.analysis = draft.analysis
  if (draft.prompt) updatePayload.prompt = draft.prompt

  if (Object.keys(updatePayload).length > 0) {
    await bgRequest({
      path: `/api/v1/media/${mediaId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: updatePayload
    })
  }

  const safeMetadata = buildSafeMetadata(draft, includeOriginalType)
  if (safeMetadata) {
    await bgRequest({
      path: `/api/v1/media/${mediaId}/metadata`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: {
        safe_metadata: safeMetadata,
        merge: true
      }
    })
  }

  // Trigger reprocessing when content was edited so chunks and embeddings
  // reflect the updated text.  Failures here are non-fatal – the content
  // is already persisted, so we only warn.
  const contentEdited =
    draft.content != null &&
    draft.originalContent != null &&
    draft.content !== draft.originalContent
  if (contentEdited) {
    try {
      await bgRequest({
        path: `/api/v1/media/${mediaId}/reprocess`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          perform_chunking: true,
          generate_embeddings: true,
          force_regenerate_embeddings: true
        }
      })
    } catch (reprocessErr) {
      console.warn(
        "[ContentReview] Reprocess after content edit failed — chunks/embeddings may be stale:",
        reprocessErr
      )
      onWarning?.(
        "Content saved but re-indexing failed. Search results may be stale until manually reprocessed."
      )
    }
  }

  const now = Date.now()
  return {
    ...draft,
    status: "committed",
    reviewedAt: draft.reviewedAt || now,
    committedAt: now,
    updatedAt: now
  }
}

const formatTimestamp = (ts?: number) => {
  if (!ts) return null
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return null
  }
}

const cloneDraft = (draft: ContentDraft): ContentDraft => {
  try {
    return structuredClone(draft)
  } catch {
    try {
      return JSON.parse(JSON.stringify(draft))
    } catch {
      return { ...draft }
    }
  }
}


export const ContentReviewPage: React.FC = () => {
  const { t } = useTranslation(["option"])
  const [messageApi, contextHolder] = message.useMessage()
  const location = useLocation()
  const navigate = useNavigate()
  const confirmDanger = useConfirmDanger()
  const [batches, setBatches] = React.useState<DraftBatch[]>([])
  const [activeBatchId, setActiveBatchId] = React.useState<string | null>(null)
  const [drafts, setDrafts] = React.useState<ContentDraft[]>([])
  const [activeDraftId, setActiveDraftId] = React.useState<string | null>(null)
  const [draftContent, setDraftContent] = React.useState<ContentDraft | null>(
    null
  )
  const [isLoading, setIsLoading] = React.useState<boolean>(false)
  const [isSaving, setIsSaving] = React.useState<boolean>(false)
  const [isCommitting, setIsCommitting] = React.useState<boolean>(false)
  const [isDirty, setIsDirty] = React.useState<boolean>(false)
  const [lastSavedAt, setLastSavedAt] = React.useState<number | null>(null)
  const [diffOpen, setDiffOpen] = React.useState<boolean>(false)
  const [aiWarningSeen, setAiWarningSeen] = useStorage<boolean>(
    "contentReviewAiWarningSeen",
    false
  )
  const [selectedModel, setSelectedModel] =
    useStorage<string | null>("selectedModel")
  const [hasChat, setHasChat] = React.useState<boolean>(true)
  const [aiBusy, setAiBusy] = React.useState<boolean>(false)
  const [templateBusy, setTemplateBusy] = React.useState<boolean>(false)
  const [selectedTemplateId, setSelectedTemplateId] = React.useState<
    string | null
  >(null)
  const [modelOptions, setModelOptions] = React.useState<
    Array<{ value: string; label: string }>
  >([])
  const [modelsLoading, setModelsLoading] = React.useState<boolean>(false)
  const saveTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement | null>(null)

  const query = React.useMemo(() => {
    return new URLSearchParams(location.search)
  }, [location.search])
  const batchFromQuery = query.get("batch")
  const draftFromQuery = query.get("draft")
  const draftFromQueryRef = React.useRef<string | null>(draftFromQuery)
  draftFromQueryRef.current = draftFromQuery

  const refreshBatches = React.useCallback(async () => {
    const data = await getDraftBatches()
    setBatches(data)
    return data
  }, [])

  const refreshDrafts = React.useCallback(async (batchId: string) => {
    const data = await getDraftsByBatch(batchId)
    const sorted = [...data].sort((a, b) => a.createdAt - b.createdAt)
    setDrafts(sorted)
    return sorted
  }, [])

  React.useEffect(() => {
    let mounted = true
    getServerCapabilities()
      .then((caps) => {
        if (mounted) setHasChat(caps.hasChat)
      })
      .catch((err) => {
        console.debug(
          "[ContentReview] Failed to fetch server capabilities:",
          err
        )
      })
    return () => {
      mounted = false
    }
  }, [])

  React.useEffect(() => {
    let mounted = true
    setModelsLoading(true)
    fetchChatModels({ returnEmpty: true })
      .then((models) => {
        if (!mounted) return
        const options = (models || []).map((model) => {
          const provider = getProviderDisplayName(model.provider)
          const modelLabel = model.nickname || model.model
          return {
            value: model.model,
            label: provider ? `${provider} - ${modelLabel}` : modelLabel
          }
        })
        setModelOptions(options)
      })
      .catch((err) => {
        console.debug("[ContentReview] Failed to fetch chat models:", err)
      })
      .finally(() => {
        if (mounted) setModelsLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [])

  const syncRoute = React.useCallback(
    (batchId: string | null, draftId?: string | null) => {
      const params = new URLSearchParams()
      if (batchId) params.set("batch", batchId)
      if (draftId) params.set("draft", draftId)
      const search = params.toString()
      const nextSearch = search ? `?${search}` : ""
      if (
        location.pathname === "/content-review" &&
        location.search === nextSearch
      ) {
        return
      }
      navigate(`/content-review${nextSearch}`, {
        replace: true
      })
    },
    [location.pathname, location.search, navigate]
  )
  const syncRouteRef = React.useRef(syncRoute)
  syncRouteRef.current = syncRoute

  React.useEffect(() => {
    let mounted = true
    setIsLoading(true)
    refreshBatches()
      .then((data) => {
        if (!mounted) return
        const initialBatch =
          (batchFromQuery &&
            data.find((batch) => batch.id === batchFromQuery)?.id) ||
          data[0]?.id ||
          null
        setActiveBatchId(initialBatch)
        if (initialBatch) {
          syncRouteRef.current(initialBatch, draftFromQueryRef.current)
        }
      })
      .finally(() => {
        if (mounted) setIsLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [batchFromQuery, refreshBatches])

  React.useEffect(() => {
    if (!activeBatchId) {
      setDrafts([])
      setActiveDraftId(null)
      setDraftContent(null)
      return
    }
    let mounted = true
    setIsLoading(true)
    refreshDrafts(activeBatchId)
      .then((items) => {
        if (!mounted) return
        const preferred =
          (draftFromQuery &&
            items.find((draft) => draft.id === draftFromQuery)?.id) ||
          items[0]?.id ||
          null
        setActiveDraftId(preferred)
        if (activeBatchId) {
          syncRouteRef.current(activeBatchId, preferred)
        }
      })
      .finally(() => {
        if (mounted) setIsLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [activeBatchId, draftFromQuery, refreshDrafts])

  React.useEffect(() => {
    if (!activeDraftId) {
      setDraftContent(null)
      return
    }
    let mounted = true
    const loadDraft = async () => {
      const draft = await getDraftById(activeDraftId)
      if (!mounted || !draft) return
      let resolved = draft
      if (draft.status === "pending") {
        const now = Date.now()
        const updated: ContentDraft = {
          ...draft,
          status: "in_progress",
          updatedAt: now
        }
        await upsertContentDraft(updated)
        if (!mounted) return
        setDrafts((prev) =>
          prev.map((d) => (d.id === draft.id ? updated : d))
        )
        resolved = updated
      }
      if (!mounted) return
      setDraftContent(cloneDraft(resolved))
      setIsDirty(false)
      setLastSavedAt(resolved.updatedAt)
    }
    void loadDraft()
    return () => {
      mounted = false
    }
  }, [activeDraftId])

  const totalCount = drafts.length
  const reviewedCount = drafts.filter((d) => d.status === "reviewed").length
  const committedCount = drafts.filter((d) => d.status === "committed").length

  const updateDraftField = <K extends keyof ContentDraft>(
    field: K,
    value: ContentDraft[K]
  ) => {
    setDraftContent((prev) => {
      if (!prev) return prev
      return { ...prev, [field]: value }
    })
    setIsDirty(true)
  }

  // Use a ref for draftContent inside callbacks to break the circular dependency:
  // saveDraftLocally depends on draftContent → effect depends on saveDraftLocally → loop
  const draftContentRef = React.useRef(draftContent)
  draftContentRef.current = draftContent

  const saveDraftLocally = React.useCallback(
    async (label?: string) => {
      const current = draftContentRef.current
      if (!current) return
      setIsSaving(true)
      const now = Date.now()
      const revisions =
        label && current.content !== current.originalContent
          ? [
              {
                id: crypto.randomUUID(),
                content: current.content,
                metadata: {
                  title: current.title,
                  keywords: current.keywords
                },
                timestamp: now,
                changeDescription: label
              },
              ...(current.revisions || [])
            ].slice(0, 10)
          : current.revisions || []
      const updated: ContentDraft = {
        ...current,
        revisions,
        updatedAt: now
      }
      await upsertContentDraft(updated)
      setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
      setDraftContent(updated)
      setIsDirty(false)
      setLastSavedAt(now)
      setIsSaving(false)
    },
    []
  )

  const applyDraftUpdate = React.useCallback(
    async (
      nextContent: string,
      label: string,
      extras?: Partial<ContentDraft>
    ) => {
      const current = draftContentRef.current
      if (!current) return
      const now = Date.now()
      const revisions =
        label && nextContent !== current.originalContent
          ? [
              {
                id: crypto.randomUUID(),
                content: nextContent,
                metadata: {
                  title: current.title,
                  keywords: current.keywords
                },
                timestamp: now,
                changeDescription: label
              },
              ...(current.revisions || [])
            ].slice(0, 10)
          : current.revisions || []
      const updated: ContentDraft = {
        ...current,
        ...extras,
        content: nextContent,
        revisions,
        updatedAt: now
      }
      await upsertContentDraft(updated)
      setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
      setDraftContent(updated)
      setIsDirty(false)
      setLastSavedAt(now)
    },
    []
  )

  React.useEffect(() => {
    if (!draftContentRef.current || !isDirty) return
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
    }
    saveTimerRef.current = setTimeout(() => {
      void saveDraftLocally()
    }, 2000)
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    }
  }, [isDirty, saveDraftLocally])

  const handleSectionToggle = (section: DraftSection, include: boolean) => {
    if (!draftContent || !draftContent.sections) return
    const excluded = new Set(draftContent.excludedSectionIds || [])
    if (include) excluded.delete(section.id)
    else excluded.add(section.id)
    const excludedIds = Array.from(excluded)
    const content = buildContentFromSections(
      draftContent.sections,
      excludedIds
    )
    updateDraftField("excludedSectionIds", excludedIds)
    updateDraftField("content", content)
  }

  const handleRedetectSections = async () => {
    if (!draftContent) return
    const hasSections = (draftContent.sections || []).length > 0
    if (hasSections) {
      const ok = await confirmDanger({
        title: t(
          "contentReview.detectSectionsTitle",
          "Re-detect sections?"
        ),
        content: t(
          "contentReview.detectSectionsBody",
          "This will replace existing sections and reset your selections."
        ),
        okText: t("contentReview.detectSectionsConfirm", "Re-detect"),
        cancelText: t("contentReview.detectSectionsCancel", "Cancel"),
        danger: false,
        autoFocusButton: "cancel"
      })
      if (!ok) return
    }
    const { sections, strategy } = detectSections(
      draftContent.content || ""
    )
    if (!sections.length) {
      messageApi.info(
        t("contentReview.detectSectionsEmpty", "No sections detected.")
      )
      return
    }
    const now = Date.now()
    const updated: ContentDraft = {
      ...draftContent,
      sections,
      excludedSectionIds: [],
      sectionStrategy: strategy || undefined,
      updatedAt: now
    }
    await upsertContentDraft(updated)
    setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
    setDraftContent(updated)
    setIsDirty(false)
    setLastSavedAt(now)
  }

  const ensureAiConsent = React.useCallback(async () => {
    if (aiWarningSeen) return true
    const ok = await confirmDanger({
      title: t("contentReview.aiConsentTitle", "Send draft to server?"),
      content: t(
        "contentReview.aiConsentBody",
        "This will send the draft content to your tldw server for AI processing. Drafts stay local until you commit."
      ),
      okText: t("contentReview.aiConsentConfirm", "Continue"),
      cancelText: t("contentReview.aiConsentCancel", "Cancel"),
      danger: false,
      autoFocusButton: "cancel"
    })
    if (!ok) return false
    setAiWarningSeen(true)
    return true
  }, [aiWarningSeen, confirmDanger, setAiWarningSeen, t])

  const runChatRewrite = React.useCallback(
    async (
      systemPrompt: string,
      instruction: string,
      content: string,
      modelOverride?: string | null
    ) => {
      const body = {
        model: modelOverride || selectedModel || "default",
        stream: false,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: wrapDraftForPrompt(content, instruction) }
        ]
      }
      const resp = await bgRequest<any>({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body
      })
      return extractChatContent(resp)
    },
    [selectedModel]
  )

  const handleAiCorrections = async () => {
    if (!draftContent || isFinalized) return
    if (!hasChat) {
      messageApi.error(
        t("contentReview.aiUnavailable", "Chat completions unavailable.")
      )
      return
    }
    if (!draftContent.content?.trim()) return
    const consented = await ensureAiConsent()
    if (!consented) return
    setAiBusy(true)
    try {
      const nextContent = await runChatRewrite(
        AI_CORRECTION_PROMPT.system,
        AI_CORRECTION_PROMPT.instruction,
        draftContent.content
      )
      if (!nextContent) {
        messageApi.error(
          t("contentReview.aiEmpty", "AI returned no content.")
        )
        return
      }
      if (nextContent.trim() === draftContent.content.trim()) {
        messageApi.info(
          t("contentReview.aiNoChange", "No changes returned.")
        )
        return
      }
      await applyDraftUpdate(nextContent, "AI corrections")
      messageApi.success(
        t("contentReview.aiApplied", "AI corrections applied.")
      )
    } catch (err: any) {
      const msg = err?.message || "AI corrections failed."
      messageApi.error(msg)
    } finally {
      setAiBusy(false)
    }
  }

  const handleApplyTemplate = async () => {
    if (!draftContent || isFinalized) return
    if (!hasChat) {
      messageApi.error(
        t("contentReview.aiUnavailable", "Chat completions unavailable.")
      )
      return
    }
    if (!draftContent.content?.trim()) return
    if (!selectedTemplateId) {
      messageApi.info(
        t("contentReview.templateMissing", "Select a template first.")
      )
      return
    }
    const consented = await ensureAiConsent()
    if (!consented) return
    const template = CONTENT_REVIEW_TEMPLATES.find(
      (item) => item.id === selectedTemplateId
    )
    if (!template) return
    setTemplateBusy(true)
    try {
      const nextContent = await runChatRewrite(
        template.systemPrompt,
        template.instruction,
        draftContent.content
      )
      if (!nextContent) {
        messageApi.error(
          t("contentReview.aiEmpty", "AI returned no content.")
        )
        return
      }
      if (nextContent.trim() === draftContent.content.trim()) {
        messageApi.info(
          t("contentReview.aiNoChange", "No changes returned.")
        )
        return
      }
      await applyDraftUpdate(nextContent, "Template formatting", {
        contentFormat: template.outputFormat
      })
      messageApi.success(
        t("contentReview.templateApplied", "Template applied.")
      )
    } catch (err: any) {
      const msg = err?.message || "Template application failed."
      messageApi.error(msg)
    } finally {
      setTemplateBusy(false)
    }
  }

  const handleResetContent = async () => {
    if (!draftContent) return
    const ok = await confirmDanger({
      title: t("contentReview.resetTitle", "Reset draft?"),
      content: t(
        "contentReview.resetBody",
        "This will restore the original content for this draft."
      ),
      okText: t("contentReview.resetConfirm", "Reset"),
      cancelText: t("contentReview.resetCancel", "Cancel"),
      danger: false,
      autoFocusButton: "cancel"
    })
    if (!ok) return
    updateDraftField("content", draftContent.originalContent)
    updateDraftField("excludedSectionIds", [])
  }

  const handleMarkReviewed = async () => {
    if (!draftContent) return
    const now = Date.now()
    const updated = {
      ...draftContent,
      status: "reviewed" as const,
      reviewedAt: now,
      updatedAt: now
    }
    await upsertContentDraft(updated)
    setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
    setDraftContent(updated)
    setIsDirty(false)
    setLastSavedAt(now)
    navigateToAdjacent(1)
  }

  const handleDiscard = async () => {
    if (!draftContent) return
    const ok = await confirmDanger({
      title: t("contentReview.discardTitle", "Discard draft?"),
      content: t(
        "contentReview.discardBody",
        "This draft will be marked as discarded and removed from review."
      ),
      okText: t("contentReview.discardConfirm", "Discard"),
      cancelText: t("contentReview.discardCancel", "Cancel"),
      danger: true,
      autoFocusButton: "cancel"
    })
    if (!ok) return
    const now = Date.now()
    const updated = {
      ...draftContent,
      status: "discarded" as const,
      updatedAt: now
    }
    await upsertContentDraft(updated)
    setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
    setDraftContent(updated)
    setIsDirty(false)
  }

  const navigateToAdjacent = (direction: 1 | -1) => {
    if (!activeDraftId) return
    const index = drafts.findIndex((d) => d.id === activeDraftId)
    if (index === -1) return
    const next = drafts[index + direction]
    if (next) {
      setActiveDraftId(next.id)
      syncRoute(activeBatchId, next.id)
    }
  }

  const handleCommit = async () => {
    if (!draftContent) return
    if (isDirty) {
      await saveDraftLocally("Pre-commit save")
    }
    setIsCommitting(true)
    try {
      const updated = await commitDraftToServer(draftContent, (warn) =>
        messageApi.warning(warn)
      )
      await upsertContentDraft(updated)
      setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
      setDraftContent(updated)
      setIsDirty(false)
      setLastSavedAt(updated.updatedAt)
      messageApi.success(
        t("contentReview.commitSuccess", "Draft committed.")
      )
    } catch (err: any) {
      const msg = err?.message || "Commit failed."
      messageApi.error(msg)
    } finally {
      setIsCommitting(false)
    }
  }

  const handleCommitSingle = async (draft: ContentDraft) => {
    const updated = await commitDraftToServer(draft, (warn) =>
      messageApi.warning(warn)
    )
    await upsertContentDraft(updated)
    setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
    if (draftContent?.id === updated.id) {
      setDraftContent(updated)
      setIsDirty(false)
      setLastSavedAt(updated.updatedAt)
    }
  }

  const handleCommitAll = async () => {
    if (reviewedCount === 0) return
    const ok = await confirmDanger({
      title: t("contentReview.commitAllTitle", "Commit reviewed drafts?"),
      content: t(
        "contentReview.commitAllBody",
        "Only drafts marked as reviewed will be committed."
      ),
      okText: t("contentReview.commitAllConfirm", "Commit all"),
      cancelText: t("contentReview.commitAllCancel", "Cancel"),
      danger: false,
      autoFocusButton: "ok"
    })
    if (!ok) return

    if (draftContent && isDirty) {
      await saveDraftLocally("Pre-commit save")
    }

    const reviewed = drafts.filter((d) => d.status === "reviewed")
    setIsCommitting(true)
    let successCount = 0
    let failCount = 0
    const errors: Array<{ title: string; error: string }> = []
    for (const draft of reviewed) {
      try {
        await handleCommitSingle(draft)
        successCount += 1
      } catch (err: any) {
        failCount += 1
        errors.push({
          title: draft.title || draft.id,
          error: err?.message || "Unknown error"
        })
      }
    }
    setIsCommitting(false)
    if (errors.length > 0) {
      console.warn("[ContentReview] Bulk commit errors:", errors)
    }
    messageApi.info(
      t(
        "contentReview.commitAllResult",
        "{{success}} committed · {{failed}} failed",
        { success: successCount, failed: failCount }
      )
    )
  }

  const handleReattachFile = async (file?: File) => {
    if (!draftContent || !file) return
    const stored = await storeDraftAsset(draftContent.id, file)
    if (!stored.asset) {
      messageApi.warning(
        t(
          "contentReview.storageCapWarning",
          "This file exceeds the local draft cap ({{cap}}).",
          { cap: `${Math.round(DRAFT_STORAGE_CAP_BYTES / 1024 / 1024)}MB` }
        )
      )
      return
    }
    const updated: ContentDraft = {
      ...draftContent,
      source: {
        kind: "file",
        fileName: file.name,
        mimeType: file.type,
        sizeBytes: file.size,
        lastModified: file.lastModified
      },
      sourceAssetId: stored.asset.id
    }
    await upsertContentDraft(updated)
    setDrafts((prev) => prev.map((d) => (d.id === updated.id ? updated : d)))
    setDraftContent(updated)
    setIsDirty(false)
  }

  const handleClearDrafts = async () => {
    const ok = await confirmDanger({
      title: t("contentReview.clearTitle", "Clear all drafts?"),
      content: t(
        "contentReview.clearBody",
        "This removes all local content review drafts and assets."
      ),
      okText: t("contentReview.clearConfirm", "Clear drafts"),
      cancelText: t("contentReview.clearCancel", "Cancel"),
      danger: true,
      autoFocusButton: "cancel"
    })
    if (!ok) return
    await db.transaction("rw", [db.contentDrafts, db.draftAssets, db.draftBatches], async () => {
      await db.contentDrafts.clear()
      await db.draftAssets.clear()
      await db.draftBatches.clear()
    })
    setBatches([])
    setDrafts([])
    setActiveBatchId(null)
    setActiveDraftId(null)
    setDraftContent(null)
    messageApi.success(t("contentReview.clearSuccess", "Drafts cleared."))
  }

  const openQuickIngest = () => {
    requestQuickIngestOpen()
  }

  const { wordCount, charCount } = getTextStats(draftContent?.content || "")
  const missingRequiredAsset =
    !!draftContent &&
    draftContent.source.kind === "file" &&
    !draftContent.sourceAssetId &&
    (draftContent.mediaType === "audio" || draftContent.mediaType === "video")
  const isFinalized =
    !!draftContent &&
    (draftContent.status === "committed" || draftContent.status === "discarded")
  const commitDisabled =
    !!draftContent &&
    (draftContent.status === "committed" ||
      draftContent.status === "discarded" ||
      missingRequiredAsset)

  return (
    <div className="space-y-4">
      {contextHolder}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <Typography.Title level={3} className="!mb-1">
            {t("contentReview.title", "Content Review")}
          </Typography.Title>
          <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
            <Tag color="blue">
              {t("contentReview.readyToSubmit", {
                defaultValue: "{{count}} ready",
                count: reviewedCount
              })}
            </Tag>
            <Tag color="green">
              {t("contentReview.committedCount", {
                defaultValue: "{{count}} committed",
                count: committedCount
              })}
            </Tag>
            <Tag>
              {t("contentReview.totalCount", {
                defaultValue: "{{count}} total",
                count: totalCount
              })}
            </Tag>
          </div>
        </div>
        <Space>
          <Button
            onClick={handleCommitAll}
            disabled={reviewedCount === 0 || isCommitting}
            type="primary"
          >
            {t("contentReview.commitAll", "Commit All")}
          </Button>
          <Button onClick={handleClearDrafts} danger>
            {t("contentReview.clearDrafts", "Clear drafts")}
          </Button>
        </Space>
      </div>

      {batches.length === 0 ? (
        <Empty
          description={t(
            "contentReview.emptyDescription",
            "No drafts yet. Use Quick Ingest with Review before saving to create drafts."
          )}
        >
          <Button type="primary" onClick={openQuickIngest}>
            {t("contentReview.openQuickIngest", "Open Quick Ingest")}
          </Button>
        </Empty>
      ) : (
        <div className="flex flex-col gap-4 lg:flex-row">
          <div className="lg:w-80 space-y-3">
            <div className="rounded-lg border border-border bg-surface p-3 shadow-sm ">
              <Typography.Text strong>
                {t("contentReview.batchLabel", "Batch")}
              </Typography.Text>
              <Select
                className="mt-2 w-full"
                value={activeBatchId || undefined}
                onChange={(value) => {
                  setActiveBatchId(value)
                  syncRoute(value, null)
                }}
                options={batches.map((batch, index) => ({
                  value: batch.id,
                  label:
                    batch.name ||
                    t("contentReview.batchName", "Batch {{index}}", {
                      index: batches.length - index
                    })
                }))}
              />
              {activeBatchId ? (
                <div className="mt-2 text-xs text-text-muted">
                  {formatTimestamp(
                    batches.find((b) => b.id === activeBatchId)?.createdAt
                  )}
                </div>
              ) : null}
            </div>
            <div className="rounded-lg border border-border bg-surface shadow-sm ">
              <div className="px-3 py-2">
                <Typography.Text strong>
                  {t("contentReview.draftsLabel", "Drafts")}
                </Typography.Text>
              </div>
              <List
                dataSource={drafts}
                loading={isLoading}
                renderItem={(draft) => (
                  <List.Item
                    onClick={() => {
                      setActiveDraftId(draft.id)
                      syncRoute(activeBatchId, draft.id)
                    }}
                    className={`cursor-pointer px-3 py-2 ${
                      draft.id === activeDraftId
                        ? "bg-primary/10 "
                        : ""
                    }`}
                  >
                    <div className="w-full">
                      <div className="flex items-start justify-between gap-2">
                        <div className="text-sm font-medium text-text text-text">
                          {draft.title || draft.source.url || draft.source.fileName}
                        </div>
                        <Tag color={statusColor(draft.status)}>
                          {draft.status.replace("_", " ")}
                        </Tag>
                      </div>
                      <div className="mt-1 text-xs text-text-muted">
                        {draft.mediaType.toUpperCase()}
                      </div>
                    </div>
                  </List.Item>
                )}
              />
            </div>
          </div>

          <div className="flex-1 space-y-4">
            {!draftContent ? (
              <Empty
                description={t(
                  "contentReview.selectDraft",
                  "Select a draft to begin reviewing."
                )}
              />
            ) : (
              <>
                <div className="rounded-lg border border-border bg-surface p-4 shadow-sm ">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div className="flex-1 space-y-2">
                      <Input
                        value={draftContent.title}
                        onChange={(e) =>
                          updateDraftField("title", e.target.value)
                        }
                        disabled={isFinalized}
                        placeholder={t(
                          "contentReview.titlePlaceholder",
                          "Draft title"
                        )}
                      />
                      <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
                        <Tag color={statusColor(draftContent.status)}>
                          {draftContent.status.replace("_", " ")}
                        </Tag>
                        {draftContent.source.url ? (
                          <Tooltip title={draftContent.source.url}>
                            <span className="truncate max-w-[320px]">
                              {draftContent.source.url}
                            </span>
                          </Tooltip>
                        ) : draftContent.source.fileName ? (
                          <span>{draftContent.source.fileName}</span>
                        ) : null}
                        {lastSavedAt ? (
                          <span>
                            {t("contentReview.lastSaved", "Saved {{time}}", {
                              time: formatTimestamp(lastSavedAt)
                            })}
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <Space>
                      <Button onClick={handleResetContent} disabled={isFinalized}>
                        {t("contentReview.resetContent", "Reset")}
                      </Button>
                      <Button onClick={() => setDiffOpen(true)}>
                        {t("contentReview.diffView", "Diff view")}
                      </Button>
                      <Button
                        onClick={() => saveDraftLocally("Manual save")}
                        loading={isSaving}
                        disabled={isFinalized}
                      >
                        {t("contentReview.saveDraft", "Save draft")}
                      </Button>
                    </Space>
                  </div>
                </div>

                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
                  <div className="space-y-4">
                    <div className="rounded-lg border border-border bg-surface p-4 shadow-sm ">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <Typography.Text strong>
                            {t("contentReview.editorLabel", "Content")}
                          </Typography.Text>
                          <div className="text-xs text-text-muted">
                            {t("contentReview.wordCount", "{{words}} words", {
                              words: wordCount
                            })}{" "}
                            ·{" "}
                            {t("contentReview.charCount", "{{chars}} chars", {
                              chars: charCount
                            })}
                          </div>
                        </div>
                        <Space size="small" wrap>
                          <Select
                            size="small"
                            allowClear
                            className="min-w-[180px]"
                            value={selectedModel || undefined}
                            onChange={(value) =>
                              setSelectedModel(value || null)
                            }
                            placeholder={t(
                              "contentReview.modelPlaceholder",
                              "Server default"
                            )}
                            loading={modelsLoading}
                            disabled={!hasChat || modelsLoading}
                            options={modelOptions}
                            showSearch
                            filterOption={(input, option) =>
                              String(option?.label || "")
                                .toLowerCase()
                                .includes(input.toLowerCase())
                            }
                          />
                          <Tooltip
                            title={t(
                              "contentReview.aiHint",
                              "Sends draft content to your server for AI processing."
                            )}
                          >
                            <Button
                              size="small"
                              onClick={handleAiCorrections}
                              disabled={
                                isFinalized || aiBusy || templateBusy || !hasChat
                              }
                              loading={aiBusy}
                            >
                              {t("contentReview.aiFix", "AI fix")}
                            </Button>
                          </Tooltip>
                          <Select
                            size="small"
                            allowClear
                            className="min-w-[180px]"
                            value={selectedTemplateId || undefined}
                            onChange={(value) =>
                              setSelectedTemplateId(value || null)
                            }
                            placeholder={t(
                              "contentReview.templatePlaceholder",
                              "Choose template"
                            )}
                            options={CONTENT_REVIEW_TEMPLATES.map((template) => ({
                              value: template.id,
                              label: t(
                                `contentReview.templates.${template.id}.label`,
                                template.label
                              )
                            }))}
                          />
                          <Button
                            size="small"
                            onClick={handleApplyTemplate}
                            disabled={
                              isFinalized ||
                              aiBusy ||
                              templateBusy ||
                              !selectedTemplateId ||
                              !hasChat
                            }
                            loading={templateBusy}
                          >
                            {t("contentReview.applyTemplate", "Apply template")}
                          </Button>
                          <Button
                            size="small"
                            onClick={handleRedetectSections}
                            disabled={isFinalized || aiBusy || templateBusy}
                          >
                            {t("contentReview.detectSections", "Detect sections")}
                          </Button>
                        </Space>
                      </div>
                      <Input.TextArea
                        className="mt-2"
                        autoSize={{ minRows: 14 }}
                        value={draftContent.content}
                        onChange={(e) =>
                          updateDraftField("content", e.target.value)
                        }
                        readOnly={isFinalized}
                      />
                    </div>

                    {draftContent.sections && draftContent.sections.length > 0 ? (
                      <Collapse
                        items={[
                          {
                            key: "sections",
                            label: t(
                              "contentReview.sectionsLabel",
                              "Section filter"
                            ),
                            children: (
                              <div className="space-y-2">
                                {draftContent.sections.map((section) => {
                                  const excluded = new Set(
                                    draftContent.excludedSectionIds || []
                                  )
                                  const included = !excluded.has(section.id)
                                  return (
                                    <label
                                      key={section.id}
                                      className="flex items-start gap-2 text-sm"
                                    >
                                      <Checkbox
                                        checked={included}
                                        disabled={isFinalized}
                                        onChange={(e) =>
                                          handleSectionToggle(
                                            section,
                                            e.target.checked
                                          )
                                        }
                                      />
                                      <div>
                                        <div className="font-medium">
                                          {section.label}
                                        </div>
                                        <div className="text-xs text-text-muted">
                                          {section.content.slice(0, 140)}
                                          {section.content.length > 140 ? "…" : ""}
                                        </div>
                                      </div>
                                    </label>
                                  )
                                })}
                              </div>
                            )
                          }
                        ]}
                      />
                    ) : null}
                  </div>

                  <div className="space-y-4">
                    <div className="rounded-lg border border-border bg-surface p-4 shadow-sm ">
                      <Typography.Text strong>
                        {t("contentReview.metadataLabel", "Metadata")}
                      </Typography.Text>
                      <div className="mt-2 space-y-3">
                        <div>
                          <Typography.Text className="text-xs text-text-muted">
                            {t("contentReview.mediaType", "Media type")}
                          </Typography.Text>
                          <div className="text-sm font-medium">
                            {draftContent.mediaType.toUpperCase()}
                          </div>
                        </div>
                        <div>
                          <Typography.Text className="text-xs text-text-muted">
                            {t("contentReview.keywords", "Keywords")}
                          </Typography.Text>
                          <Select
                            mode="tags"
                            className="mt-1 w-full"
                            value={draftContent.keywords || []}
                            onChange={(value) =>
                              updateDraftField("keywords", value)
                            }
                            disabled={isFinalized}
                            placeholder={t(
                              "contentReview.keywordsPlaceholder",
                              "Add keywords"
                            )}
                          />
                        </div>
                        <div>
                          <Typography.Text className="text-xs text-text-muted">
                            {t("contentReview.reviewNotes", "Review notes")}
                          </Typography.Text>
                          <Input.TextArea
                            className="mt-1"
                            autoSize={{ minRows: 3 }}
                            value={draftContent.reviewNotes || ""}
                            onChange={(e) =>
                              updateDraftField("reviewNotes", e.target.value)
                            }
                            disabled={isFinalized}
                          />
                        </div>
                        {draftContent.source.kind === "file" &&
                        !draftContent.sourceAssetId ? (
                          <div className="rounded-md border border-warn/40 bg-warn/10 p-3 text-xs text-warn">
                            <div className="font-medium">
                              {t(
                                "contentReview.missingSource",
                                "Source file missing"
                              )}
                            </div>
                            <div className="mt-1">
                              {t(
                                "contentReview.missingSourceBody",
                                "Attach the original file to commit this draft."
                              )}
                            </div>
                            <div className="mt-2">
                              <Button
                                size="small"
                                onClick={() => fileInputRef.current?.click()}
                              >
                                {t(
                                  "contentReview.attachSource",
                                  "Attach file"
                                )}
                              </Button>
                              <input
                                ref={fileInputRef}
                                type="file"
                                className="hidden"
                                onChange={(e) => {
                                  const file = e.target.files?.[0]
                                  void handleReattachFile(file || undefined)
                                  e.target.value = ""
                                }}
                              />
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </div>

                    <div className="rounded-lg border border-border bg-surface p-4 shadow-sm ">
                      <Typography.Text strong>
                        {t("contentReview.actionsLabel", "Actions")}
                      </Typography.Text>
                      <Space orientation="vertical" className="mt-3 w-full">
                        <Button
                          type="primary"
                          onClick={handleCommit}
                          loading={isCommitting}
                          disabled={commitDisabled}
                        >
                          {t("contentReview.commit", "Commit")}
                        </Button>
                        {missingRequiredAsset ? (
                          <div className="text-xs text-warn">
                            {t(
                              "contentReview.commitDisabled",
                              "Attach the original audio/video file before committing."
                            )}
                          </div>
                        ) : null}
                        <Button onClick={handleMarkReviewed} disabled={isFinalized}>
                          {t("contentReview.markReviewed", "Mark reviewed")}
                        </Button>
                        <Button onClick={() => navigateToAdjacent(1)}>
                          {t("contentReview.skipNext", "Skip")}
                        </Button>
                        <Button danger onClick={handleDiscard} disabled={isFinalized}>
                          {t("contentReview.discard", "Discard")}
                        </Button>
                      </Space>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {draftContent ? (
        <DiffViewModal
          open={diffOpen}
          onClose={() => setDiffOpen(false)}
          leftText={draftContent.originalContent || ""}
          rightText={draftContent.content || ""}
          leftLabel={t("contentReview.originalLabel", "Original")}
          rightLabel={t("contentReview.editedLabel", "Edited")}
        />
      ) : null}
    </div>
  )
}

export default ContentReviewPage
