import React from "react"
import { useTranslation } from "react-i18next"
import {
  AlertCircle,
  ChevronDown,
  FileText,
  Send,
  MessageSquarePlus,
  Square,
  Trash2,
  RotateCcw,
  SlidersHorizontal,
  Loader2,
  Share2,
  Cpu,
  Plus,
  BookOpen,
  Users,
  Briefcase,
  WifiOff
} from "lucide-react"
import { Modal, Tag, Tooltip, Input, Slider, Switch, Button, message } from "antd"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildWorkspaceChatSessionKey } from "@/store/workspace-chat-session-key"
import { useWorkspaceStore } from "@/store/workspace"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import type { Message } from "@/store/option"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useSmartScroll } from "@/hooks/useSmartScroll"
import { useMobile } from "@/hooks/useMediaQuery"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase } from "@/types/connection"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { formatCost } from "@/utils/model-pricing"
import { trackWorkspacePlaygroundTelemetry } from "@/utils/workspace-playground-telemetry"
import type { WorkspaceSource, WorkspaceSourceType } from "@/types/workspace"
import type { ChatScope } from "@/types/chat-scope"
import {
  applyVariantToMessage,
  normalizeMessageVariants
} from "@/utils/message-variants"
import { buildConversationShareUrl } from "@/components/Layouts/chat-share-links"
import { PlaygroundMessage } from "@/components/Common/Playground/Message"
import { Link } from "react-router-dom"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { buildChatLorebookDebugPath } from "@/routes/route-paths"
import {
  WORKSPACE_SOURCE_DRAG_TYPE,
  parseWorkspaceSourceDragPayload
} from "../drag-source"
import {
  WORKSPACE_TEMPLATE_PRESETS,
  type WorkspaceTemplatePreset
} from "../workspace-header.utils"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "../undo-manager"
import { getWorkspaceChatNoSourcesHint } from "../source-location-copy"
import { getWorkspaceChatSearchMessageId } from "../workspace-global-search"

const { TextArea } = Input
const VISIBLE_SOURCE_TAG_COUNT = 5

type RetrievalDiagnostics = {
  chunkCount: number | null
  sourceCount: number | null
  sourceLabels: string[]
  averageRelevanceScore: number | null
  faithfulnessScore: number | null
  promptTokens: number | null
  completionTokens: number | null
  totalTokens: number | null
  costUsd: number | null
  confidenceLevel: "high" | "medium" | "low" | null
}

type ChatModePreference = "normal" | "rag"
type ChatModelOption = {
  id: string
  label: string
  provider: string
}
type ChatPaneContentWidthMode = "comfortable" | "expanded" | "full"
type LorebookActivityTurn = {
  turnNumber: number
  assistantPreview: string
  entryCount: number
}

const LOREBOOK_ACTIVITY_PAGE_SIZE = 8
const LOREBOOK_ACTIVITY_EXPORT_PAGE_SIZE = 200
const RETRY_BURST_WINDOW_MS = 30_000
const RETRY_BURST_THRESHOLD = 3
const DUPLICATE_SUBMISSION_WINDOW_MS = 12_000
const LOREBOOK_DEBUG_ENTRYPOINT_HREF = buildChatLorebookDebugPath({
  from: "workspace-playground"
})
const EMPTY_STRING_ARRAY: string[] = []

const useEffectiveSelectedSources = (): WorkspaceSource[] => {
  const selectedSourceIds = useWorkspaceStore((s) => s.selectedSourceIds)
  const selectedSourceFolderIds =
    useWorkspaceStore((s) => s.selectedSourceFolderIds) ?? EMPTY_STRING_ARRAY
  const sources = useWorkspaceStore((s) => s.sources)
  const getSelectedSources = useWorkspaceStore((s) => s.getSelectedSources)
  const getEffectiveSelectedSources = useWorkspaceStore(
    (s) => s.getEffectiveSelectedSources
  )

  return React.useMemo(
    () =>
      typeof getEffectiveSelectedSources === "function"
        ? getEffectiveSelectedSources()
        : getSelectedSources(),
    [
      getEffectiveSelectedSources,
      getSelectedSources,
      selectedSourceFolderIds,
      selectedSourceIds,
      sources
    ]
  )
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

const toPositiveInt = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10)
    if (Number.isInteger(parsed) && parsed > 0) {
      return parsed
    }
  }
  return null
}

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

const normalizeText = (value: unknown): string =>
  typeof value === "string" ? value.trim().toLowerCase() : ""

const extractStringCandidate = (value: unknown): string => {
  if (typeof value !== "string") return ""
  return value.trim()
}

const extractSourceFullText = (detail: unknown): string => {
  if (!isRecord(detail)) return ""
  const content = isRecord(detail.content) ? detail.content : null
  const processing = isRecord(detail.processing) ? detail.processing : null

  const candidates: unknown[] = [
    content?.text,
    content?.content,
    content?.full_text,
    content?.transcript,
    content?.raw_text,
    processing?.analysis,
    processing?.transcript,
    processing?.text
  ]

  for (const candidate of candidates) {
    const normalized = extractStringCandidate(candidate)
    if (normalized.length > 0) {
      return normalized
    }
  }

  return ""
}

const extractCitationMediaId = (citation: unknown): number | null => {
  if (!isRecord(citation)) return null
  const metadata = isRecord(citation.metadata) ? citation.metadata : null
  const includeMediaIds = Array.isArray(metadata?.include_media_ids)
    ? metadata?.include_media_ids
    : []
  const mediaCandidates: unknown[] = [
    citation.mediaId,
    citation.media_id,
    citation.mediaID,
    metadata?.mediaId,
    metadata?.media_id,
    metadata?.mediaID,
    ...includeMediaIds
  ]

  for (const candidate of mediaCandidates) {
    const mediaId = toPositiveInt(candidate)
    if (mediaId !== null) {
      return mediaId
    }
  }
  return null
}

const findSourceIdFromCitation = (
  citation: unknown,
  sources: WorkspaceSource[]
): string | null => {
  if (!isRecord(citation)) return null

  const citationMediaId = extractCitationMediaId(citation)
  if (citationMediaId !== null) {
    const byMediaId = sources.find((source) => source.mediaId === citationMediaId)
    if (byMediaId) return byMediaId.id
  }

  const metadata = isRecord(citation.metadata) ? citation.metadata : null
  const citationUrl = normalizeText(citation.url || metadata?.url)
  if (citationUrl) {
    const byUrl = sources.find(
      (source) => normalizeText(source.url) === citationUrl
    )
    if (byUrl) return byUrl.id
  }

  const candidateLabels = new Set(
    [
      citation.name,
      citation.title,
      citation.source,
      metadata?.source,
      metadata?.title
    ]
      .map(normalizeText)
      .filter(Boolean)
  )

  if (candidateLabels.size === 0) {
    return null
  }

  for (const source of sources) {
    const normalizedTitle = normalizeText(source.title)
    if (!normalizedTitle) continue
    if (candidateLabels.has(normalizedTitle)) {
      return source.id
    }
  }

  for (const source of sources) {
    const normalizedTitle = normalizeText(source.title)
    if (!normalizedTitle) continue
    for (const label of candidateLabels) {
      if (label.includes(normalizedTitle) || normalizedTitle.includes(label)) {
        return source.id
      }
    }
  }

  return null
}

const extractSourceScore = (source: unknown): number | null => {
  if (!isRecord(source)) return null
  const metadata = isRecord(source.metadata) ? source.metadata : null
  const scoreCandidates: unknown[] = [
    source.score,
    source.relevance,
    source.relevance_score,
    source.similarity,
    source.rerank_score,
    metadata?.score,
    metadata?.relevance,
    metadata?.relevance_score,
    metadata?.similarity,
    metadata?.rerank_score
  ]

  for (const candidate of scoreCandidates) {
    const parsed = toFiniteNumber(candidate)
    if (parsed !== null) {
      return parsed
    }
  }
  return null
}

const extractSourceLabel = (source: unknown): string | null => {
  if (!isRecord(source)) return null
  const metadata = isRecord(source.metadata) ? source.metadata : null
  const candidates: unknown[] = [
    source.name,
    source.title,
    source.source,
    source.url,
    metadata?.title,
    metadata?.source,
    metadata?.url
  ]

  for (const candidate of candidates) {
    const normalized = extractStringCandidate(candidate)
    if (normalized.length > 0) {
      return normalized
    }
  }

  return null
}

const normalizeConfidenceScore = (score: number | null): number | null => {
  if (score === null || !Number.isFinite(score)) {
    return null
  }
  if (score > 1 && score <= 100) {
    return score / 100
  }
  if (score < 0) {
    return 0
  }
  if (score > 1) {
    return 1
  }
  return score
}

const getConfidenceLevel = (
  score: number | null
): RetrievalDiagnostics["confidenceLevel"] => {
  const normalized = normalizeConfidenceScore(score)
  if (normalized === null) {
    return null
  }
  if (normalized >= 0.8) {
    return "high"
  }
  if (normalized >= 0.55) {
    return "medium"
  }
  return "low"
}

const getNumericField = (
  generationInfo: unknown,
  paths: Array<string[]>
): number | null => {
  if (!isRecord(generationInfo)) return null

  for (const path of paths) {
    let current: unknown = generationInfo
    for (const key of path) {
      if (!isRecord(current)) {
        current = undefined
        break
      }
      current = current[key]
    }
    const parsed = toFiniteNumber(current)
    if (parsed !== null) {
      return parsed
    }
  }

  return null
}

const buildRetrievalDiagnostics = (
  messageSources: unknown[] | undefined,
  generationInfo: unknown
): RetrievalDiagnostics | null => {
  const sources = Array.isArray(messageSources) ? messageSources : []

  const chunkCountFromGeneration = getNumericField(generationInfo, [
    ["retrieval", "chunks_retrieved"],
    ["retrieval", "chunk_count"],
    ["retrieval", "chunks"],
    ["chunks_retrieved"],
    ["chunk_count"],
    ["retrieved_chunks"],
    ["context_chunks"]
  ])

  const sourceCountFromGeneration = getNumericField(generationInfo, [
    ["retrieval", "source_count"],
    ["retrieval", "sources_used"],
    ["source_count"],
    ["sources_used"]
  ])

  const avgScoreFromGeneration = getNumericField(generationInfo, [
    ["retrieval", "avg_relevance_score"],
    ["retrieval", "average_relevance_score"],
    ["avg_relevance_score"],
    ["average_relevance_score"],
    ["relevance_score"]
  ])
  const faithfulnessScore = getNumericField(generationInfo, [
    ["faithfulness", "score"],
    ["faithfulness", "faithfulness_score"],
    ["faithfulness", "overall_score"],
    ["faithfulness_score"],
    ["claim_verification", "faithfulness_score"],
    ["retrieval", "faithfulness_score"],
    ["confidence"],
    ["confidence_score"]
  ])
  const promptTokensFromGeneration = getNumericField(generationInfo, [
    ["usage", "prompt_tokens"],
    ["usage", "input_tokens"],
    ["prompt_tokens"],
    ["input_tokens"],
    ["prompt_eval_count"]
  ])
  const completionTokensFromGeneration = getNumericField(generationInfo, [
    ["usage", "completion_tokens"],
    ["usage", "output_tokens"],
    ["completion_tokens"],
    ["output_tokens"],
    ["eval_count"]
  ])
  const totalTokensFromGeneration = getNumericField(generationInfo, [
    ["usage", "total_tokens"],
    ["total_tokens"],
    ["total_token_count"]
  ])
  const costUsdFromGeneration = getNumericField(generationInfo, [
    ["usage", "total_cost_usd"],
    ["usage", "estimated_cost_usd"],
    ["usage", "cost_usd"],
    ["pricing", "total_cost_usd"],
    ["pricing", "estimated_cost_usd"],
    ["pricing", "cost_usd"],
    ["total_cost_usd"],
    ["estimated_cost_usd"],
    ["cost_usd"],
    ["total_cost"],
    ["estimated_cost"]
  ])

  const uniqueSourceKeys = new Set<string>()
  const uniqueSourceLabels = new Set<string>()
  for (const source of sources) {
    if (!isRecord(source)) continue
    const sourceLabel = extractSourceLabel(source)
    if (sourceLabel) {
      uniqueSourceLabels.add(sourceLabel)
    }

    const mediaId = extractCitationMediaId(source)
    if (mediaId !== null) {
      uniqueSourceKeys.add(`media:${mediaId}`)
    } else {
      const metadata = isRecord(source.metadata) ? source.metadata : null
      const label = normalizeText(
        source.name || source.url || metadata?.source || metadata?.title
      )
      if (label) {
        uniqueSourceKeys.add(`label:${label}`)
      }
    }
  }

  const sourceScores = sources
    .map((source) => extractSourceScore(source))
    .filter((score): score is number => score !== null)

  const averageScoreFromSources =
    sourceScores.length > 0
      ? sourceScores.reduce((sum, score) => sum + score, 0) /
        sourceScores.length
      : null

  const diagnostics: RetrievalDiagnostics = {
    chunkCount:
      chunkCountFromGeneration !== null
        ? Math.max(0, Math.round(chunkCountFromGeneration))
        : sources.length > 0
          ? sources.length
          : null,
    sourceCount:
      sourceCountFromGeneration !== null
        ? Math.max(0, Math.round(sourceCountFromGeneration))
        : uniqueSourceKeys.size > 0
          ? uniqueSourceKeys.size
          : null,
    sourceLabels: Array.from(uniqueSourceLabels),
    averageRelevanceScore:
      avgScoreFromGeneration !== null
        ? avgScoreFromGeneration
        : averageScoreFromSources,
    faithfulnessScore,
    promptTokens:
      promptTokensFromGeneration !== null
        ? Math.max(0, Math.round(promptTokensFromGeneration))
        : null,
    completionTokens:
      completionTokensFromGeneration !== null
        ? Math.max(0, Math.round(completionTokensFromGeneration))
        : null,
    totalTokens:
      totalTokensFromGeneration !== null
        ? Math.max(0, Math.round(totalTokensFromGeneration))
        : promptTokensFromGeneration !== null ||
            completionTokensFromGeneration !== null
          ? Math.max(
              0,
              Math.round(
                (promptTokensFromGeneration || 0) +
                  (completionTokensFromGeneration || 0)
              )
            )
          : null,
    costUsd:
      costUsdFromGeneration !== null && costUsdFromGeneration >= 0
        ? costUsdFromGeneration
        : null,
    confidenceLevel: getConfidenceLevel(
      faithfulnessScore !== null
        ? faithfulnessScore
        : avgScoreFromGeneration !== null
          ? avgScoreFromGeneration
          : averageScoreFromSources
    )
  }

  if (
    diagnostics.chunkCount === null &&
    diagnostics.sourceCount === null &&
    diagnostics.averageRelevanceScore === null &&
    diagnostics.faithfulnessScore === null &&
    diagnostics.sourceLabels.length === 0 &&
    diagnostics.totalTokens === null &&
    diagnostics.costUsd === null
  ) {
    return null
  }

  return diagnostics
}

/**
 * ChatContextIndicator - Shows sources as horizontally scrollable tags
 */
const ChatContextIndicator: React.FC = () => {
  const { t } = useTranslation(["playground"])
  const selectedSourceIds = useWorkspaceStore((s) => s.selectedSourceIds)
  const selectedSourceFolderIds =
    useWorkspaceStore((s) => s.selectedSourceFolderIds) ?? EMPTY_STRING_ARRAY
  const selectedSources = useEffectiveSelectedSources()
  const [showAllSources, setShowAllSources] = React.useState(false)

  React.useEffect(() => {
    setShowAllSources(false)
  }, [selectedSourceFolderIds, selectedSourceIds, selectedSources])

  if (selectedSources.length === 0) return null

  const hiddenSourceCount = Math.max(
    0,
    selectedSources.length - VISIBLE_SOURCE_TAG_COUNT
  )
  const visibleSources = showAllSources
    ? selectedSources
    : selectedSources.slice(0, VISIBLE_SOURCE_TAG_COUNT)

  return (
    <div className="shrink-0 border-b border-border bg-surface px-4 py-2">
      <div className="flex items-center gap-2">
        <span className="shrink-0 text-xs font-medium text-text-muted">
          <FileText className="mr-1 inline h-3 w-3" />
          {t("playground:chat.usingSourcesLabel", "Sources:")}
        </span>
        {/* Horizontally scrollable source tags */}
        <div className="custom-scrollbar flex min-w-0 flex-1 gap-1.5 overflow-x-auto pb-0.5">
          {visibleSources.map((source) => (
            <Tooltip key={source.id} title={source.title}>
              <Tag
                color="blue"
                className="shrink-0 cursor-default !m-0 max-w-[150px] truncate"
              >
                {source.title}
              </Tag>
            </Tooltip>
          ))}
          {hiddenSourceCount > 0 && !showAllSources && (
            <button
              type="button"
              onClick={() => setShowAllSources(true)}
              className="shrink-0 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary transition hover:bg-primary/15"
              aria-label={t("playground:chat.showMoreSources", "Show more sources")}
            >
              +{hiddenSourceCount}{" "}
              {t("playground:chat.moreSources", "more")}
            </button>
          )}
          {hiddenSourceCount > 0 && showAllSources && (
            <button
              type="button"
              onClick={() => setShowAllSources(false)}
              className="shrink-0 rounded-full border border-border bg-surface2 px-2 py-0.5 text-xs font-medium text-text-muted transition hover:bg-surface"
              aria-label={t("playground:chat.showFewerSources", "Show fewer sources")}
            >
              {t("playground:chat.showLess", "Show less")}
            </button>
          )}
        </div>
      </div>
      <p className="mt-1 text-xs text-text-muted">
        {t(
          "playground:chat.ragModeHint",
          "Answers will be grounded in your selected sources"
        )}
      </p>
    </div>
  )
}

const RetrievalDiagnosticsPanel: React.FC<{
  diagnostics: RetrievalDiagnostics
}> = ({ diagnostics }) => {
  const { t } = useTranslation(["playground"])
  const tokenCostTelemetryTrackedRef = React.useRef(false)
  const confidenceStyles =
    diagnostics.confidenceLevel === "high"
      ? "text-success"
      : diagnostics.confidenceLevel === "medium"
        ? "text-warn"
        : diagnostics.confidenceLevel === "low"
          ? "text-error"
          : "text-text-muted"
  const confidenceText =
    diagnostics.confidenceLevel === "high"
      ? t("playground:chat.confidenceHigh", "High")
      : diagnostics.confidenceLevel === "medium"
        ? t("playground:chat.confidenceMedium", "Medium")
        : diagnostics.confidenceLevel === "low"
          ? t("playground:chat.confidenceLow", "Low")
          : t("playground:chat.confidenceUnknown", "Unknown")
  const displayedSources = diagnostics.sourceLabels.slice(0, 3)
  const hiddenSourceCount = Math.max(0, diagnostics.sourceLabels.length - 3)
  const normalizedFaithfulnessScore = normalizeConfidenceScore(
    diagnostics.faithfulnessScore
  )

  React.useEffect(() => {
    if (tokenCostTelemetryTrackedRef.current) return
    if (diagnostics.totalTokens === null && diagnostics.costUsd === null) return
    tokenCostTelemetryTrackedRef.current = true
    void trackWorkspacePlaygroundTelemetry({
      type: "token_cost_rendered",
      has_tokens: diagnostics.totalTokens !== null,
      has_cost: diagnostics.costUsd !== null
    })
  }, [diagnostics.costUsd, diagnostics.totalTokens])

  return (
    <details
      className="rounded-md border border-border bg-surface2/40 px-3 py-2 text-xs text-text-muted"
      onToggle={(event) => {
        void trackWorkspacePlaygroundTelemetry({
          type: "diagnostics_toggled",
          expanded: event.currentTarget.open
        })
      }}
    >
      <summary className="cursor-pointer font-medium text-text-subtle">
        {t("playground:chat.retrievalInfo", "Retrieval info")}
      </summary>
      <div className="mt-2 space-y-1">
        {diagnostics.chunkCount !== null && (
          <p>
            {t("playground:chat.retrievalChunks", "Chunks retrieved")}:{" "}
            <span className="font-medium text-text">{diagnostics.chunkCount}</span>
          </p>
        )}
        {diagnostics.sourceCount !== null && (
          <p>
            {t("playground:chat.retrievalSources", "Sources used")}:{" "}
            <span className="font-medium text-text">{diagnostics.sourceCount}</span>
          </p>
        )}
        {displayedSources.length > 0 && (
          <p>
            {t("playground:chat.retrievalSourceList", "Source list")}:{" "}
            <span className="font-medium text-text">
              {displayedSources.join(", ")}
              {hiddenSourceCount > 0 ? ` +${hiddenSourceCount} more` : ""}
            </span>
          </p>
        )}
        {diagnostics.averageRelevanceScore !== null && (
          <p>
            {t("playground:chat.retrievalAverageScore", "Avg relevance score")}:{" "}
            <span className="font-medium text-text">
              {diagnostics.averageRelevanceScore.toFixed(3)}
            </span>
          </p>
        )}
        {diagnostics.totalTokens !== null && (
          <p>
            {t("playground:chat.retrievalTokens", "Tokens")}:{" "}
            <span className="font-medium text-text">
              {diagnostics.promptTokens ?? 0}{" "}
              {t("playground:tokens.prompt", "prompt")} +{" "}
              {diagnostics.completionTokens ?? 0}{" "}
              {t("playground:tokens.completion", "completion")} ={" "}
              {diagnostics.totalTokens}{" "}
              {t("playground:tokens.total", "tokens")}
            </span>
          </p>
        )}
        {diagnostics.costUsd !== null && (
          <p>
            {t("playground:chat.retrievalCost", "Cost")}:{" "}
            <span className="font-medium text-text">
              {formatCost(diagnostics.costUsd)}
            </span>
          </p>
        )}
        {diagnostics.faithfulnessScore !== null && (
          <p>
            {t("playground:chat.retrievalFaithfulnessScore", "Faithfulness score")}:{" "}
            <span className="font-medium text-text">
              {(normalizedFaithfulnessScore ?? diagnostics.faithfulnessScore).toFixed(
                3
              )}
            </span>
          </p>
        )}
        <p>
          {t("playground:chat.retrievalConfidence", "Confidence")}:{" "}
          <span className={`font-medium ${confidenceStyles}`}>{confidenceText}</span>
        </p>
      </div>
    </details>
  )
}

/**
 * WorkspaceChatEmpty - Empty state for the workspace chat
 */
const TEMPLATE_ICONS: Record<string, React.ElementType> = {
  literature_review: BookOpen,
  interview_analysis: Users,
  product_brief: Briefcase
}

const WorkspaceChatEmpty: React.FC<{
  hasSelectedSources: boolean
  sourceCount: number
  totalSourceCount: number
  selectedSourceTypes: WorkspaceSourceType[]
  isMobile: boolean
  layoutMode?: ChatPaneContentWidthMode
  onExamplePromptSelect?: (prompt: string) => void
  onAddSource?: () => void
  onCreateFromTemplate?: (template: WorkspaceTemplatePreset) => void
}> = ({
  hasSelectedSources,
  sourceCount,
  totalSourceCount,
  selectedSourceTypes,
  isMobile,
  layoutMode = "comfortable",
  onExamplePromptSelect,
  onAddSource,
  onCreateFromTemplate
}) => {
  const { t } = useTranslation(["playground"])
  const sourceTypeSet = React.useMemo(
    () => new Set(selectedSourceTypes),
    [selectedSourceTypes]
  )
  const hasPromptActions = typeof onExamplePromptSelect === "function"
  const emptyStateMaxWidthClass = "max-w-none"

  const examples = React.useMemo(() => {
    if (!hasSelectedSources) {
      return [
        t(
          "playground:chat.exampleGeneral1",
          "Help me frame a research question on this topic"
        ),
        t(
          "playground:chat.exampleGeneral2",
          "Give me a quick overview before I add sources"
        ),
        t(
          "playground:chat.exampleGeneral3",
          "What should I look for in high-quality sources?"
        )
      ]
    }

    const adaptiveExamples: string[] = []
    if (sourceTypeSet.has("video")) {
      adaptiveExamples.push(
        t(
          "playground:chat.exampleVideo",
          "What was discussed around minute 12?"
        )
      )
    }
    if (sourceTypeSet.has("audio")) {
      adaptiveExamples.push(
        t(
          "playground:chat.exampleAudio",
          "List key takeaways and speaker viewpoints from this audio"
        )
      )
    }
    if (
      sourceTypeSet.has("pdf") ||
      sourceTypeSet.has("document") ||
      sourceTypeSet.has("text")
    ) {
      adaptiveExamples.push(
        t(
          "playground:chat.exampleDocument",
          "Summarize chapter 3 and cite the strongest supporting passages"
        )
      )
    }
    if (sourceTypeSet.has("website")) {
      adaptiveExamples.push(
        t(
          "playground:chat.exampleWebsite",
          "Extract the main claims and supporting evidence from these pages"
        )
      )
    }

    const fallbackExamples = [
      t("playground:chat.example1", "Summarize the key points from these sources"),
      t("playground:chat.example2", "What are the main arguments presented?"),
      t("playground:chat.example3", "Compare and contrast the different perspectives")
    ]

    return [...new Set([...adaptiveExamples, ...fallbackExamples])].slice(0, 3)
  }, [hasSelectedSources, sourceTypeSet, t])

  return (
    <div className={`mx-auto w-full ${emptyStateMaxWidthClass} px-4 pb-2 pt-3`}>
      <FeatureEmptyState
        className="max-w-none"
        icon={MessageSquarePlus}
        title={t("playground:chat.emptyTitle", "Start your research")}
        description={
          <div className="space-y-1">
            <p className="text-xs font-medium text-text-muted">
              {t("playground:chat.emptyTagline", "Your research assistant — grounded in your sources")}
            </p>
            <p className="text-xs text-text-subtle">
              {hasSelectedSources
                ? t(
                    "playground:chat.emptyWithSources",
                    "Ask questions about your {{count}} selected source(s)",
                    { count: sourceCount }
                  )
                : t(
                    "playground:chat.emptyNoSources",
                    getWorkspaceChatNoSourcesHint(isMobile)
                  )}
            </p>
          </div>
        }
        examples={examples.map((example, index) => (
          <button
            key={example}
            type="button"
            data-testid={`workspace-chat-empty-prompt-chip-${index}`}
            onClick={() => {
              if (typeof onExamplePromptSelect !== "function") return
              onExamplePromptSelect(example)
            }}
            disabled={!hasPromptActions}
            className="w-full rounded-md border border-border/70 bg-surface2/70 px-2 py-1 text-left text-xs text-text-muted transition hover:border-primary/40 hover:bg-primary/5 hover:text-text disabled:cursor-default disabled:opacity-80"
          >
            {example}
          </button>
        ))}
      />

      <p className="mt-3 text-center text-[11px] text-text-subtle">
        {t("playground:chat.knowledgeHint", "Quick document search?")}{" "}
        <Link to="/knowledge" className="text-primary/70 hover:text-primary transition-colors">
          {t("playground:chat.knowledgeHintLink", "Try Knowledge QA \u2192")}
        </Link>
      </p>

      {totalSourceCount === 0 && (
        <div className="mt-4 space-y-3" data-testid="workspace-chat-empty-guidance">
          <p className="text-center text-xs text-text-muted" data-testid="workspace-chat-sources-explainer">
            {t(
              "playground:chat.sourcesExplainer",
              "Sources are documents, PDFs, web pages, or other content you upload. Add sources in the left panel to start asking questions about them."
            )}
          </p>
          {onAddSource && (
            <button
              type="button"
              data-testid="workspace-chat-add-source-button"
              onClick={onAddSource}
              className="mx-auto flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition hover:bg-primary/20"
            >
              <Plus className="h-3.5 w-3.5" />
              {t("playground:chat.addSourceCta", "Add sources to get started")}
            </button>
          )}

          {onCreateFromTemplate && (
            <div className="space-y-1.5">
              <p className="text-center text-[11px] text-text-subtle">
                {t("playground:chat.templateHint", "Or start from a template")}
              </p>
              <div className={`grid gap-2 ${isMobile ? "grid-cols-1" : "grid-cols-3"}`}>
                {WORKSPACE_TEMPLATE_PRESETS.map((template) => {
                  const Icon = TEMPLATE_ICONS[template.id] || FileText
                  return (
                    <button
                      key={template.id}
                      type="button"
                      data-testid={`workspace-template-card-${template.id}`}
                      onClick={() => onCreateFromTemplate(template)}
                      className="flex items-center gap-2 rounded-lg border border-border/70 bg-surface2/50 px-3 py-2 text-left text-xs transition hover:border-primary/40 hover:bg-primary/5"
                    >
                      <Icon className="h-4 w-4 shrink-0 text-primary/70" />
                      <span className="text-text">{template.label}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * SimpleChatInput - A simple chat input component with slash command autocomplete (UX-006)
 */
const SimpleChatInput: React.FC<{
  onSubmit: (message: string) => void
  onStop: () => void
  isLoading: boolean
  isPreparingContext?: boolean
  isChatUnavailable?: boolean
  placeholder?: string
  seededValue?: string | null
  onSeedConsumed?: () => void
  slashCommands?: Array<{ name: string; description: string }>
}> = ({
  onSubmit,
  onStop,
  isLoading,
  isPreparingContext = false,
  isChatUnavailable = false,
  placeholder,
  seededValue,
  onSeedConsumed,
  slashCommands = []
}) => {
  const { t } = useTranslation(["playground", "common"])
  const [value, setValue] = React.useState("")
  const [showSlashMenu, setShowSlashMenu] = React.useState(false)
  const [slashMenuIndex, setSlashMenuIndex] = React.useState(0)

  React.useEffect(() => {
    if (typeof seededValue !== "string") return
    setValue(seededValue)
    onSeedConsumed?.()
  }, [onSeedConsumed, seededValue])

  // Slash command filtering
  const filteredSlashCommands = React.useMemo(() => {
    if (!value.startsWith("/") || value.includes(" ")) return []
    const query = value.slice(1).toLowerCase()
    return slashCommands.filter((cmd) =>
      cmd.name.toLowerCase().startsWith(query)
    )
  }, [value, slashCommands])

  React.useEffect(() => {
    setShowSlashMenu(filteredSlashCommands.length > 0 && value.startsWith("/") && !value.includes(" "))
    setSlashMenuIndex(0)
  }, [filteredSlashCommands.length, value])

  const selectSlashCommand = (cmd: { name: string }) => {
    setValue(`/${cmd.name} `)
    setShowSlashMenu(false)
  }

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || isLoading || isPreparingContext || isChatUnavailable) return
    onSubmit(trimmed)
    setValue("")
    setShowSlashMenu(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Slash command menu navigation
    if (showSlashMenu && filteredSlashCommands.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault()
        setSlashMenuIndex((prev) => Math.min(prev + 1, filteredSlashCommands.length - 1))
        return
      }
      if (e.key === "ArrowUp") {
        e.preventDefault()
        setSlashMenuIndex((prev) => Math.max(prev - 1, 0))
        return
      }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
        e.preventDefault()
        selectSlashCommand(filteredSlashCommands[slashMenuIndex])
        return
      }
      if (e.key === "Escape") {
        e.preventDefault()
        setShowSlashMenu(false)
        return
      }
    }

    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSubmit()
      return
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="rounded-lg border border-border/70 bg-surface/90 p-1.5 shadow-sm">
      {isChatUnavailable && (
        <div className="px-3 py-2 text-xs text-warning bg-warning/10 border-b border-warning/20 flex items-center gap-2 rounded-t-md">
          <WifiOff className="h-3.5 w-3.5 shrink-0" />
          <span>
            {t(
              "playground:chat.disconnectedWarning",
              "Can't reach the server. Check your connection or server status."
            )}
          </span>
        </div>
      )}
      <form onSubmit={handleSubmit} className="flex items-end gap-1.5">
        <div className="relative flex-1">
          {/* Slash command autocomplete dropdown (UX-006) */}
          {showSlashMenu && filteredSlashCommands.length > 0 && (
            <div
              role="listbox"
              className="absolute bottom-full left-0 z-20 mb-1 max-h-48 w-full overflow-y-auto rounded-lg border border-border bg-surface shadow-lg"
            >
              {filteredSlashCommands.map((cmd, idx) => (
                <button
                  key={cmd.name}
                  type="button"
                  role="option"
                  aria-selected={idx === slashMenuIndex}
                  onClick={() => selectSlashCommand(cmd)}
                  className={`flex w-full items-start gap-2 px-3 py-2 text-left text-sm transition ${
                    idx === slashMenuIndex
                      ? "bg-primary/10 text-text"
                      : "text-text-muted hover:bg-surface2"
                  }`}
                >
                  <span className="shrink-0 font-mono text-xs text-primary">/{cmd.name}</span>
                  {cmd.description && (
                    <span className="truncate text-xs text-text-muted">
                      {cmd.description}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
          <TextArea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isChatUnavailable
                ? t("playground:chat.inputPlaceholderDisconnected", "Server disconnected...")
                : placeholder || t("playground:chat.inputPlaceholder", "Type / for commands or a message...")
            }
            autoSize={{ minRows: 1, maxRows: 6 }}
            disabled={isLoading || isPreparingContext}
            className="pr-10 text-sm"
          />
        </div>
        {isLoading ? (
          <button
            type="button"
            onClick={onStop}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-error text-white transition hover:bg-error/90"
            aria-label={t("common:stop", "Stop")}
            title={t("common:stop", "Stop") as string}
          >
            <Square className="h-4 w-4 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim() || isPreparingContext || isChatUnavailable}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-white transition hover:bg-primaryStrong disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={
              isChatUnavailable
                ? t("playground:chat.serverDisconnected", "Server disconnected")
                : isPreparingContext
                ? t(
                    "playground:chat.preparingSourceContext",
                    "Preparing source context"
                  )
                : t("common:send", "Send")
            }
          >
            {isPreparingContext ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        )}
      </form>
      <p className="mt-1 px-0.5 text-[11px] text-text-muted">
        {t(
          "playground:chat.inputKeyboardHint",
          "Enter or Cmd/Ctrl+Enter to send, Shift+Enter for new line"
        )}
      </p>
    </div>
  )
}

// Generate a stable conversation instance ID for the workspace
const WORKSPACE_CONVERSATION_ID = "workspace-playground-conversation"
const WORKSPACE_DISCUSS_EVENT = "workspace-playground:discuss-artifact"
const WORKSPACE_SOURCE_CONTEXT_WARNING_KEY =
  "workspace-playground:source-context-warning"

type WorkspaceDiscussArtifactDetail = {
  artifactId: string
  artifactType: string
  title: string
  content: string
}

type TemporarySourceScope = {
  sourceId: string
  sourceTitle: string
  previousSelectedSourceIds: string[]
  previousPreferredChatMode: ChatModePreference | null
}

const buildCapturedMessageTitle = (
  isBot: boolean,
  text: string,
  assistantPrefix: string,
  userPrefix: string
): string => {
  const prefix = isBot ? assistantPrefix : userPrefix
  const collapsed = text.replace(/\s+/g, " ").trim()
  if (!collapsed) return prefix
  const excerpt = collapsed.length > 56 ? `${collapsed.slice(0, 56)}...` : collapsed
  return `${prefix}: ${excerpt}`
}

/**
 * ChatPane - Middle pane for RAG-powered conversation
 */
interface ChatPaneProps {
  provenanceEnabled?: boolean
  statusGuardrailsEnabled?: boolean
  contentWidthMode?: ChatPaneContentWidthMode
}

export const ChatPane: React.FC<ChatPaneProps> = ({
  provenanceEnabled = true,
  statusGuardrailsEnabled = true,
  contentWidthMode = "comfortable"
}) => {
  const { t } = useTranslation(["playground", "common"])
  const isMobile = useMobile()
  const [messageApi, messageContextHolder] = message.useMessage()

  // Workspace store
  const selectedSourceIds = useWorkspaceStore((s) => s.selectedSourceIds)
  const selectedSourceFolderIds =
    useWorkspaceStore((s) => s.selectedSourceFolderIds) ?? EMPTY_STRING_ARRAY
  const sources = useWorkspaceStore((s) => s.sources)
  const getSelectedMediaIds = useWorkspaceStore((s) => s.getSelectedMediaIds)
  const getEffectiveSelectedMediaIds = useWorkspaceStore(
    (s) => s.getEffectiveSelectedMediaIds
  )
  const setSelectedSourceIds = useWorkspaceStore((s) => s.setSelectedSourceIds)
  const focusSourceById = useWorkspaceStore((s) => s.focusSourceById)
  const focusSourceByMediaId = useWorkspaceStore((s) => s.focusSourceByMediaId)
  const captureToCurrentNote = useWorkspaceStore((s) => s.captureToCurrentNote)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const workspaceChatReferenceId = useWorkspaceStore(
    (s) => s.workspaceChatReferenceId
  )
  const chatFocusTarget = useWorkspaceStore((s) => s.chatFocusTarget)
  const clearChatFocusTarget = useWorkspaceStore((s) => s.clearChatFocusTarget)
  const openAddSourceModal = useWorkspaceStore((s) => s.openAddSourceModal)
  const createNewWorkspace = useWorkspaceStore((s) => s.createNewWorkspace)
  const setCurrentNote = useWorkspaceStore((s) => s.setCurrentNote)
  const switchWorkspace = useWorkspaceStore((s) => s.switchWorkspace)

  // Model settings for model badge (UX-009)
  const selectedModel = useStoreMessageOption((s) => s.selectedModel)
  const setSelectedModel = useStoreMessageOption((s) => s.setSelectedModel)
  const chatApiProvider = useStoreChatModelSettings((s) => s.apiProvider)
  const chatScope = React.useMemo<ChatScope | undefined>(
    () =>
      workspaceId
        ? {
            type: "workspace",
            workspaceId
          }
        : undefined,
    [workspaceId]
  )

  // Message option hook
  const {
    messages,
    setMessages,
    history,
    setHistory,
    streaming,
    setStreaming,
    isProcessing,
    setIsProcessing,
    onSubmit,
    stopStreamingRequest,
    regenerateLastMessage,
    createChatBranch,
    deleteMessage,
    editMessage,
    historyId,
    setHistoryId,
    serverChatId,
    setServerChatId
  } = useMessageOption({ scope: chatScope })

  // RAG state from store
  const setRagMediaIds = useStoreMessageOption((s) => s.setRagMediaIds)
  const setChatMode = useStoreMessageOption((s) => s.setChatMode)
  const setFileRetrievalEnabled = useStoreMessageOption(
    (s) => s.setFileRetrievalEnabled
  )
  const ragTopK = useStoreMessageOption((s) => s.ragTopK)
  const setRagTopK = useStoreMessageOption((s) => s.setRagTopK)
  const ragAdvancedOptions = useStoreMessageOption((s) => s.ragAdvancedOptions)
  const setRagAdvancedOptions = useStoreMessageOption(
    (s) => s.setRagAdvancedOptions
  )
  const saveWorkspaceChatSession = useWorkspaceStore(
    (s) => s.saveWorkspaceChatSession
  )
  const getWorkspaceChatSession = useWorkspaceStore(
    (s) => s.getWorkspaceChatSession
  )
  const checkConnectionOnce = useConnectionStore((s) => s.checkOnce)
  const connectionState = useConnectionStore((s) => s.state)
  const [preferredChatMode, setPreferredChatMode] = React.useState<
    ChatModePreference | null
  >(null)
  const [dropZoneActive, setDropZoneActive] = React.useState(false)
  const [seededPrompt, setSeededPrompt] = React.useState<string | null>(null)
  const [highlightedChatMessageId, setHighlightedChatMessageId] = React.useState<
    string | null
  >(null)
  const [showAdvancedRagSettings, setShowAdvancedRagSettings] = React.useState(
    false
  )
  const [temporarySourceScope, setTemporarySourceScope] =
    React.useState<TemporarySourceScope | null>(null)
  const [submitError, setSubmitError] = React.useState<string | null>(null)
  const [includeFullSourceContents, setIncludeFullSourceContents] =
    React.useState(false)
  const [preparingSourceContext, setPreparingSourceContext] =
    React.useState(false)
  const [lorebookActivityTurns, setLorebookActivityTurns] = React.useState<
    LorebookActivityTurn[]
  >([])
  const [lorebookActivityTotal, setLorebookActivityTotal] = React.useState(0)
  const [lorebookActivityLoading, setLorebookActivityLoading] =
    React.useState(false)
  const [lorebookActivityError, setLorebookActivityError] = React.useState<
    string | null
  >(null)
  const [lorebookActivityForbidden, setLorebookActivityForbidden] =
    React.useState(false)
  const [exportingLorebookActivity, setExportingLorebookActivity] =
    React.useState(false)
  const [sharingConversation, setSharingConversation] = React.useState(false)
  const [slashCommands, setSlashCommands] = React.useState<
    Array<{ name: string; description: string }>
  >([])
  const [availableModels, setAvailableModels] = React.useState<ChatModelOption[]>(
    []
  )
  const [loadingModels, setLoadingModels] = React.useState(false)
  const slashCommandsFetchedRef = React.useRef(false)
  const modelsFetchedRef = React.useRef(false)
  const workspaceSessionRef = React.useRef<string | null>(null)
  const chatMessageItemRefs = React.useRef<Record<string, HTMLDivElement | null>>(
    {}
  )
  const previousSelectedSourcesRef = React.useRef<string[]>(selectedSourceIds)
  const selectedSourceIdsRef = React.useRef<string[]>(selectedSourceIds)
  const effectiveSelectedSourceIdsRef = React.useRef<string[]>([])
  const temporarySourceScopeRef = React.useRef<TemporarySourceScope | null>(null)
  const suppressSourceContextWarningRef = React.useRef(false)
  const selectedSourcesInitializedRef = React.useRef(false)
  const retryAttemptTimestampsRef = React.useRef<number[]>([])
  const lastRetryBurstSignalRef = React.useRef(0)
  const duplicateSubmissionTrackerRef = React.useRef<{
    signature: string
    count: number
    lastAttemptAt: number
  }>({
    signature: "",
    count: 0,
    lastAttemptAt: 0
  })
  const workspaceSessionId = React.useMemo(
    () =>
      buildWorkspaceChatSessionKey(
        workspaceId || WORKSPACE_CONVERSATION_ID,
        workspaceChatReferenceId || workspaceId || WORKSPACE_CONVERSATION_ID
      ),
    [workspaceChatReferenceId, workspaceId]
  )

  // Smart scroll for chat messages
  const { containerRef, isAutoScrollToBottom, autoScrollToBottom } =
    useSmartScroll(messages, streaming, 120)

  const selectedSources = useEffectiveSelectedSources()
  const effectiveSelectedSourceIds = React.useMemo(
    () => selectedSources.map((source) => source.id),
    [selectedSources]
  )
  const hasMessages = messages.length > 0
  const hasSelectedSources = selectedSources.length > 0

  const handleCreateFromTemplate = React.useCallback(
    (template: WorkspaceTemplatePreset) => {
      createNewWorkspace(template.workspaceName)
      setCurrentNote({
        id: undefined,
        title: template.noteTitle,
        content: template.noteContent,
        keywords: [...template.keywords],
        version: undefined,
        isDirty: true
      })
    },
    [createNewWorkspace, setCurrentNote]
  )

  const normalizedRagAdvancedOptions = React.useMemo(() => {
    return isRecord(ragAdvancedOptions) ? ragAdvancedOptions : {}
  }, [ragAdvancedOptions])
  const resolvedTopK = React.useMemo(() => {
    const value =
      typeof ragTopK === "number" && Number.isFinite(ragTopK)
        ? ragTopK
        : DEFAULT_RAG_SETTINGS.top_k
    return Math.max(1, Math.min(50, Math.round(value)))
  }, [ragTopK])
  const similarityThreshold = React.useMemo(() => {
    const raw = normalizedRagAdvancedOptions.min_score
    const value =
      typeof raw === "number" && Number.isFinite(raw)
        ? raw
        : DEFAULT_RAG_SETTINGS.min_score
    return Math.max(0, Math.min(1, value))
  }, [normalizedRagAdvancedOptions.min_score])
  const rerankingEnabled = React.useMemo(() => {
    const raw = normalizedRagAdvancedOptions.enable_reranking
    return typeof raw === "boolean"
      ? raw
      : DEFAULT_RAG_SETTINGS.enable_reranking
  }, [normalizedRagAdvancedOptions.enable_reranking])
  const requestedChatMode: ChatModePreference =
    preferredChatMode ?? (hasSelectedSources ? "rag" : "normal")
  const effectiveChatMode: ChatModePreference =
    hasSelectedSources && requestedChatMode === "rag" ? "rag" : "normal"

  React.useEffect(() => {
    if (hasSelectedSources || !includeFullSourceContents) return
    setIncludeFullSourceContents(false)
  }, [hasSelectedSources, includeFullSourceContents])

  const updateRagAdvancedOptions = React.useCallback(
    (patch: Record<string, unknown>) => {
      setRagAdvancedOptions({
        ...normalizedRagAdvancedOptions,
        ...patch
      })
    },
    [normalizedRagAdvancedOptions, setRagAdvancedOptions]
  )

  const handleTopKChange = (value: number | number[]) => {
    const raw = Array.isArray(value) ? value[0] : value
    if (typeof raw !== "number" || !Number.isFinite(raw)) return
    const nextTopK = Math.max(1, Math.min(50, Math.round(raw)))
    setRagTopK(nextTopK)
    updateRagAdvancedOptions({ top_k: nextTopK })
  }

  const handleSimilarityThresholdChange = (value: number | number[]) => {
    const raw = Array.isArray(value) ? value[0] : value
    if (typeof raw !== "number" || !Number.isFinite(raw)) return
    const nextThreshold = Math.max(0, Math.min(1, raw))
    updateRagAdvancedOptions({ min_score: Number(nextThreshold.toFixed(2)) })
  }

  const handleRerankingToggle = (checked: boolean) => {
    updateRagAdvancedOptions({ enable_reranking: checked })
  }

  const handleSwitchMessageVariant = React.useCallback(
    (messageIndex: number, direction: "prev" | "next") => {
      setMessages((previousMessages) => {
        if (!Array.isArray(previousMessages)) return previousMessages
        const targetMessage = previousMessages[messageIndex] as Message | undefined
        if (!targetMessage || !targetMessage.isBot) return previousMessages

        const variants = normalizeMessageVariants(targetMessage)
        if (variants.length <= 1) return previousMessages

        const resolvedIndex =
          typeof targetMessage.activeVariantIndex === "number"
            ? Math.max(
                0,
                Math.min(targetMessage.activeVariantIndex, variants.length - 1)
              )
            : variants.length - 1
        const nextIndex =
          direction === "prev"
            ? Math.max(0, resolvedIndex - 1)
            : Math.min(variants.length - 1, resolvedIndex + 1)
        if (nextIndex === resolvedIndex) {
          return previousMessages
        }

        const nextVariant = variants[nextIndex]
        if (!nextVariant) return previousMessages

        const nextMessages = [...previousMessages]
        nextMessages[messageIndex] = applyVariantToMessage(
          {
            ...targetMessage,
            variants
          },
          nextVariant,
          nextIndex
        )
        return nextMessages
      })
    },
    [setMessages]
  )

  const handleCreateChatBranch = React.useCallback(
    (messageIndex: number) => {
      if (!Number.isInteger(messageIndex) || messageIndex < 0) return
      void createChatBranch(messageIndex)
    },
    [createChatBranch]
  )

  React.useEffect(() => {
    if (!hasSelectedSources && showAdvancedRagSettings) {
      setShowAdvancedRagSettings(false)
    }
  }, [hasSelectedSources, showAdvancedRagSettings])

  React.useEffect(() => {
    if (modelsFetchedRef.current) return
    modelsFetchedRef.current = true
    if (typeof tldwClient.getModels !== "function") return

    let isMounted = true
    setLoadingModels(true)
    void tldwClient
      .getModels()
      .then((models) => {
        if (!isMounted) return
        if (!Array.isArray(models) || models.length === 0) {
          setAvailableModels([])
          return
        }
        const uniqueById = new Map<string, ChatModelOption>()
        for (const model of models) {
          if (!model || typeof model !== "object") continue
          const modelId = extractStringCandidate((model as { id?: unknown }).id)
          if (!modelId) continue
          const provider = extractStringCandidate(
            (model as { provider?: unknown }).provider
          )
          const modelName = extractStringCandidate(
            (model as { name?: unknown }).name
          )
          const label =
            modelName && modelName !== modelId
              ? `${modelName} (${modelId})`
              : modelName || modelId
          uniqueById.set(modelId, {
            id: modelId,
            label: provider ? `${provider} · ${label}` : label,
            provider
          })
        }
        setAvailableModels(
          Array.from(uniqueById.values()).sort((a, b) =>
            a.label.localeCompare(b.label, undefined, { sensitivity: "base" })
          )
        )
      })
      .catch(() => {
        if (!isMounted) return
        setAvailableModels([])
      })
      .finally(() => {
        if (isMounted) {
          setLoadingModels(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [])

  React.useEffect(() => {
    selectedSourceIdsRef.current = selectedSourceIds
  }, [selectedSourceIds])

  React.useEffect(() => {
    effectiveSelectedSourceIdsRef.current = effectiveSelectedSourceIds
  }, [effectiveSelectedSourceIds])

  React.useEffect(() => {
    temporarySourceScopeRef.current = temporarySourceScope
  }, [temporarySourceScope])

  React.useEffect(() => {
    if (!temporarySourceScope) return
    if (selectedSourceIds.includes(temporarySourceScope.sourceId)) return
    setTemporarySourceScope(null)
  }, [selectedSourceIds, temporarySourceScope])

  const applyScopedSelection = React.useCallback(
    (nextSelectedSourceIds: string[]) => {
      suppressSourceContextWarningRef.current = true
      setSelectedSourceIds(nextSelectedSourceIds)
    },
    [setSelectedSourceIds]
  )

  const restoreTemporaryScope = React.useCallback(
    (reason: "manual" | "auto", explicitScope?: TemporarySourceScope) => {
      const activeScope = explicitScope ?? temporarySourceScopeRef.current
      if (!activeScope) return

      applyScopedSelection(activeScope.previousSelectedSourceIds)
      setPreferredChatMode(activeScope.previousPreferredChatMode)
      setTemporarySourceScope(null)

      if (reason === "manual") {
        messageApi.info({
          duration: 3,
          content: t(
            "playground:chat.dragScopedRestored",
            "Restored your previous source selection."
          )
        })
      }
    },
    [applyScopedSelection, messageApi, t]
  )

  React.useEffect(() => {
    if (!selectedSourcesInitializedRef.current) {
      previousSelectedSourcesRef.current = effectiveSelectedSourceIds
      selectedSourcesInitializedRef.current = true
      return
    }

    if (suppressSourceContextWarningRef.current) {
      suppressSourceContextWarningRef.current = false
      previousSelectedSourcesRef.current = effectiveSelectedSourceIds
      return
    }

    const previous = previousSelectedSourcesRef.current
    const removedSourceCount = previous.filter(
      (sourceId) => !effectiveSelectedSourceIds.includes(sourceId)
    ).length

    if (removedSourceCount > 0 && hasMessages) {
      messageApi.info({
        key: WORKSPACE_SOURCE_CONTEXT_WARNING_KEY,
        duration: 4,
        content: t(
          "playground:chat.sourceContextChangedWarning",
          "Source context changed. Previous answers may reference sources no longer selected."
        )
      })
    }

    previousSelectedSourcesRef.current = effectiveSelectedSourceIds
  }, [effectiveSelectedSourceIds, hasMessages, messageApi, t])

  // Sync selected sources + user mode preference with RAG context
  React.useEffect(() => {
    const mediaIds =
      typeof getEffectiveSelectedMediaIds === "function"
        ? getEffectiveSelectedMediaIds()
        : getSelectedMediaIds()
    const hasScopedMediaIds = mediaIds.length > 0
    const autoMode: ChatModePreference = hasScopedMediaIds ? "rag" : "normal"
    const resolvedMode = preferredChatMode ?? autoMode
    const nextMode: ChatModePreference =
      hasScopedMediaIds && resolvedMode === "rag" ? "rag" : "normal"

    if (nextMode === "rag") {
      setRagMediaIds(mediaIds)
      setChatMode("rag")
      setFileRetrievalEnabled(true)
    } else {
      setRagMediaIds(null)
      setChatMode("normal")
      setFileRetrievalEnabled(false)
    }
  }, [
    effectiveSelectedSourceIds,
    preferredChatMode,
    getEffectiveSelectedMediaIds,
    getSelectedMediaIds,
    setChatMode,
    setFileRetrievalEnabled,
    setRagMediaIds
  ])

  React.useEffect(() => {
    if (!workspaceSessionId) return

    saveWorkspaceChatSession(workspaceSessionId, {
      messages,
      history,
      historyId,
      serverChatId
    })
  }, [
    workspaceSessionId,
    messages,
    history,
    historyId,
    serverChatId,
    saveWorkspaceChatSession
  ])

  React.useEffect(() => {
    if (!workspaceSessionId) return

    const previousWorkspaceSessionId = workspaceSessionRef.current
    if (previousWorkspaceSessionId === workspaceSessionId) return

    if (previousWorkspaceSessionId) {
      saveWorkspaceChatSession(previousWorkspaceSessionId, {
        messages,
        history,
        historyId,
        serverChatId
      })
    }

    const nextSession = getWorkspaceChatSession(workspaceSessionId)
    if (nextSession) {
      setMessages(nextSession.messages)
      setHistory(nextSession.history)
      setHistoryId(nextSession.historyId, { preserveServerChatId: true })
      setServerChatId(nextSession.serverChatId)
    } else {
      setMessages([])
      setHistory([])
      setHistoryId(null, { preserveServerChatId: true })
      setServerChatId(null)
    }

    setStreaming(false)
    setIsProcessing(false)
    setSubmitError(null)
    workspaceSessionRef.current = workspaceSessionId
  }, [
    workspaceSessionId,
    getWorkspaceChatSession,
    history,
    historyId,
    messages,
    saveWorkspaceChatSession,
    serverChatId,
    setHistory,
    setHistoryId,
    setIsProcessing,
    setMessages,
    setServerChatId,
    setStreaming
  ])

  React.useEffect(() => {
    const targetMessageId = chatFocusTarget?.messageId
    if (!targetMessageId) return

    const targetElement = chatMessageItemRefs.current[targetMessageId]
    if (!targetElement) {
      clearChatFocusTarget()
      return
    }

    const revealTimer = window.setTimeout(() => {
      targetElement.scrollIntoView({ behavior: "smooth", block: "nearest" })
      setHighlightedChatMessageId(targetMessageId)
      clearChatFocusTarget()
    }, 0)
    const highlightTimer = window.setTimeout(() => {
      setHighlightedChatMessageId((current) =>
        current === targetMessageId ? null : current
      )
    }, 1800)

    return () => {
      window.clearTimeout(revealTimer)
      window.clearTimeout(highlightTimer)
    }
  }, [chatFocusTarget, clearChatFocusTarget])

  const buildFullSourceContextPrompt = React.useCallback(
    async (message: string): Promise<string> => {
      if (!includeFullSourceContents || selectedSources.length === 0) {
        return message
      }

      setPreparingSourceContext(true)

      try {
        const detailResults = await Promise.allSettled(
          selectedSources.map(async (source) => {
            const detail = await tldwClient.getMediaDetails(source.mediaId, {
              include_content: true,
              include_versions: false,
              include_version_content: false
            })
            const fullText = extractSourceFullText(detail)
            return {
              source,
              fullText
            }
          })
        )

        const resolvedSourceContexts: Array<{
          source: WorkspaceSource
          fullText: string
        }> = []
        let skippedSourceCount = 0

        for (const detailResult of detailResults) {
          if (detailResult.status !== "fulfilled") {
            skippedSourceCount += 1
            continue
          }

          if (!detailResult.value.fullText) {
            skippedSourceCount += 1
            continue
          }

          resolvedSourceContexts.push(detailResult.value)
        }

        if (resolvedSourceContexts.length === 0) {
          messageApi.warning(
            t(
              "playground:chat.fullSourceContextUnavailable",
              "Couldn't load full source contents; sending your question without inline source text."
            )
          )
          return message
        }

        if (skippedSourceCount > 0) {
          messageApi.info(
            t(
              "playground:chat.partialSourceContextLoaded",
              "Loaded full content for {{loaded}} source(s); {{skipped}} source(s) had no extracted text yet.",
              {
                loaded: resolvedSourceContexts.length,
                skipped: skippedSourceCount
              }
            )
          )
        }

        const fullSourceBlock = resolvedSourceContexts
          .map(({ source, fullText }) => {
            const sourceHeader = `Source ${source.mediaId}: ${source.title}`
            return `<<SOURCE START: ${sourceHeader}>>\n${fullText}\n<<SOURCE END: ${sourceHeader}>>`
          })
          .join("\n\n")

        return [
          t(
            "playground:chat.fullSourceContextInstruction",
            "Use the complete source contents below when answering the user question."
          ),
          t(
            "playground:chat.fullSourceContextCitationInstruction",
            "When relevant, cite source titles directly."
          ),
          "",
          fullSourceBlock,
          "",
          `${t("playground:chat.fullSourceContextQuestionLabel", "User question")}: ${message}`
        ].join("\n")
      } catch {
        messageApi.warning(
          t(
            "playground:chat.fullSourceContextFailed",
            "Failed to fetch full source contents; sending your question without inline source text."
          )
        )
        return message
      } finally {
        setPreparingSourceContext(false)
      }
    },
    [includeFullSourceContents, messageApi, selectedSources, t]
  )

  const handleSubmit = async (message: string) => {
    if (preparingSourceContext) return
    const normalizedMessage = message.trim().replace(/\s+/g, " ").toLowerCase()
    const sourceScopeSignature = [...effectiveSelectedSourceIdsRef.current]
      .sort((a, b) => a.localeCompare(b))
      .join(",")
    const submissionSignature = `${effectiveChatMode}|${sourceScopeSignature}|${normalizedMessage}`
    const now = Date.now()
    const duplicateState = duplicateSubmissionTrackerRef.current
    const isDuplicateAttempt =
      duplicateState.signature === submissionSignature &&
      now - duplicateState.lastAttemptAt <= DUPLICATE_SUBMISSION_WINDOW_MS

    if (isDuplicateAttempt) {
      const nextCount = duplicateState.count + 1
      duplicateSubmissionTrackerRef.current = {
        signature: submissionSignature,
        count: nextCount,
        lastAttemptAt: now
      }
      if (statusGuardrailsEnabled && nextCount === 2) {
        void trackWorkspacePlaygroundTelemetry({
          type: "confusion_duplicate_submission",
          workspace_id: workspaceSessionId || null,
          duplicate_count: nextCount,
          window_ms: DUPLICATE_SUBMISSION_WINDOW_MS,
          source_scope_count: effectiveSelectedSourceIdsRef.current.length,
          message_length: normalizedMessage.length
        })
      }
    } else {
      duplicateSubmissionTrackerRef.current = {
        signature: submissionSignature,
        count: 1,
        lastAttemptAt: now
      }
    }

    setSubmitError(null)
    try {
      const preparedMessage = await buildFullSourceContextPrompt(message)
      await onSubmit({ message: preparedMessage, image: "" })
      const activeScope = temporarySourceScopeRef.current
      if (activeScope) {
        const selectedIds = selectedSourceIdsRef.current
        const isStillScopedToDroppedSource =
          selectedIds.length === 1 && selectedIds[0] === activeScope.sourceId
        if (isStillScopedToDroppedSource) {
          restoreTemporaryScope("auto", activeScope)
        }
      }
    } catch {
      setSubmitError(
        t(
          "playground:chat.connectionError",
          "Unable to reach server. Please check your connection and retry."
        )
      )
    }
  }

  React.useEffect(() => {
    if (typeof window === "undefined") return

    const onDiscussArtifact = (event: Event) => {
      const customEvent = event as CustomEvent<WorkspaceDiscussArtifactDetail>
      const detail = customEvent.detail
      if (!detail || typeof detail.content !== "string") return

      const trimmedContent = detail.content.trim()
      if (!trimmedContent) return
      const excerpt =
        trimmedContent.length > 6000
          ? `${trimmedContent.slice(0, 6000)}\n\n[truncated]`
          : trimmedContent

      const prompt = [
        `I generated this ${detail.artifactType.replaceAll("_", " ")} in Studio:`,
        `Title: ${detail.title}`,
        "",
        excerpt,
        "",
        "Please review it, identify gaps, and suggest concrete improvements."
      ].join("\n")

      void handleSubmit(prompt)
    }

    window.addEventListener(WORKSPACE_DISCUSS_EVENT, onDiscussArtifact)
    return () => {
      window.removeEventListener(WORKSPACE_DISCUSS_EVENT, onDiscussArtifact)
    }
  }, [handleSubmit])

  const handleDropZoneDragOver = React.useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      const hasWorkspaceSource =
        event.dataTransfer.types.includes(WORKSPACE_SOURCE_DRAG_TYPE) ||
        Boolean(event.dataTransfer.getData(WORKSPACE_SOURCE_DRAG_TYPE))
      if (!hasWorkspaceSource) return
      event.preventDefault()
      event.dataTransfer.dropEffect = "copy"
      if (!dropZoneActive) {
        setDropZoneActive(true)
      }
    },
    [dropZoneActive]
  )

  const handleDropZoneDragLeave = React.useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      const nextTarget = event.relatedTarget as Node | null
      if (nextTarget && event.currentTarget.contains(nextTarget)) {
        return
      }
      setDropZoneActive(false)
    },
    []
  )

  const handleDropZoneDrop = React.useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      setDropZoneActive(false)

      const payload = parseWorkspaceSourceDragPayload(
        event.dataTransfer.getData(WORKSPACE_SOURCE_DRAG_TYPE)
      )
      if (!payload) return

      setTemporarySourceScope({
        sourceId: payload.sourceId,
        sourceTitle: payload.title,
        previousSelectedSourceIds: [...selectedSourceIdsRef.current],
        previousPreferredChatMode: preferredChatMode
      })
      applyScopedSelection([payload.sourceId])
      setPreferredChatMode("rag")

      const promptTemplate = t(
        "playground:chat.dragPromptTemplate",
        'Focus on "{{title}}" only. Summarize key findings and cite supporting evidence.',
        { title: payload.title }
      ).replace("{{title}}", payload.title)
      setSeededPrompt(promptTemplate)
      messageApi.info({
        duration: 3,
        content: t(
          "playground:chat.dragScopedInfo",
          'Scoped chat context to "{{title}}".',
          { title: payload.title }
        ).replace("{{title}}", payload.title)
      })
    },
    [applyScopedSelection, messageApi, preferredChatMode, t]
  )

  const handleClearChat = () => {
    if (!hasMessages) return

    Modal.confirm({
      title: t("playground:chat.clearTitle", "Clear chat?"),
      content: t(
        "playground:chat.clearMessage",
        "This will remove all messages in this workspace chat."
      ),
      okText: t("common:clear", "Clear"),
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => {
        const previousMessages = [...messages]
        const previousHistory = [...history]
        const previousHistoryId = historyId
        const previousServerChatId = serverChatId

        const undoHandle = scheduleWorkspaceUndoAction({
          apply: () => {
            setMessages([])
            setHistory([])
            setHistoryId(null, { preserveServerChatId: true })
            setServerChatId(null)
            setStreaming(false)
            setIsProcessing(false)
            setSubmitError(null)
            saveWorkspaceChatSession(workspaceSessionId, {
              messages: [],
              history: [],
              historyId: null,
              serverChatId: null
            })
          },
          undo: () => {
            setMessages(previousMessages)
            setHistory(previousHistory)
            setHistoryId(previousHistoryId, { preserveServerChatId: true })
            setServerChatId(previousServerChatId)
            setStreaming(false)
            setIsProcessing(false)
            setSubmitError(null)
            saveWorkspaceChatSession(workspaceSessionId, {
              messages: previousMessages,
              history: previousHistory,
              historyId: previousHistoryId,
              serverChatId: previousServerChatId
            })
          }
        })

        const undoMessageKey = `workspace-chat-clear-undo-${undoHandle.id}`
        const maybeOpen = (
          messageApi as { open?: (config: unknown) => void }
        ).open
        const messageConfig = {
          key: undoMessageKey,
          type: "warning",
          duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
          content: t("playground:chat.cleared", "Chat cleared."),
          btn: (
            <Button
              size="small"
              type="link"
              onClick={() => {
                if (undoWorkspaceAction(undoHandle.id)) {
                  messageApi.success(
                    t("playground:chat.restored", "Chat restored")
                  )
                }
                messageApi.destroy(undoMessageKey)
              }}
            >
              {t("common:undo", "Undo")}
            </Button>
          )
        }
        if (typeof maybeOpen === "function") {
          maybeOpen(messageConfig)
        } else {
          const maybeWarning = (
            messageApi as { warning?: (content: string) => void }
          ).warning
          if (typeof maybeWarning === "function") {
            maybeWarning(t("playground:chat.cleared", "Chat cleared."))
          }
        }
      }
    })
  }

  const handleDeleteMessageWithUndo = React.useCallback(
    async (messageIndex: number) => {
      if (!Number.isInteger(messageIndex) || messageIndex < 0) return
      const previousMessages = [...messages]
      const previousHistory = [...history]
      const previousHistoryId = historyId
      const previousServerChatId = serverChatId
      if (!previousMessages[messageIndex]) return

      const nextMessages = previousMessages.filter(
        (_message, index) => index !== messageIndex
      )
      const nextHistory = previousHistory.filter(
        (_entry, index) => index !== messageIndex
      )

      await deleteMessage(messageIndex)

      const undoHandle = scheduleWorkspaceUndoAction({
        apply: () => {
          saveWorkspaceChatSession(workspaceSessionId, {
            messages: nextMessages,
            history: nextHistory,
            historyId: previousHistoryId,
            serverChatId: previousServerChatId
          })
        },
        undo: () => {
          setMessages(previousMessages)
          setHistory(previousHistory)
          setHistoryId(previousHistoryId, { preserveServerChatId: true })
          setServerChatId(previousServerChatId)
          setStreaming(false)
          setIsProcessing(false)
          setSubmitError(null)
          saveWorkspaceChatSession(workspaceSessionId, {
            messages: previousMessages,
            history: previousHistory,
            historyId: previousHistoryId,
            serverChatId: previousServerChatId
          })
        }
      })

      const undoMessageKey = `workspace-message-delete-undo-${undoHandle.id}`
      const maybeOpen = (
        messageApi as { open?: (config: unknown) => void }
      ).open
      const messageConfig = {
        key: undoMessageKey,
        type: "warning",
        duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
        content: t("playground:chat.messageDeleted", "Message deleted."),
        btn: (
          <Button
            size="small"
            type="link"
            onClick={() => {
              if (undoWorkspaceAction(undoHandle.id)) {
                messageApi.success(
                  t("playground:chat.messageRestored", "Message restored")
                )
              }
              messageApi.destroy(undoMessageKey)
            }}
          >
            {t("common:undo", "Undo")}
          </Button>
        )
      }
      if (typeof maybeOpen === "function") {
        maybeOpen(messageConfig)
      } else {
        const maybeWarning = (
          messageApi as { warning?: (content: string) => void }
        ).warning
        if (typeof maybeWarning === "function") {
          maybeWarning(t("playground:chat.messageDeleted", "Message deleted."))
        }
      }
    },
    [
      deleteMessage,
      history,
      historyId,
      messageApi,
      messages,
      saveWorkspaceChatSession,
      serverChatId,
      setHistory,
      setHistoryId,
      setIsProcessing,
      setMessages,
      setServerChatId,
      setStreaming,
      t,
      workspaceSessionId
    ]
  )

  const handleStopStreaming = React.useCallback(() => {
    void trackWorkspacePlaygroundTelemetry({
      type: "operation_cancelled",
      workspace_id: workspaceSessionId || null,
      operation: "chat_stream"
    })
    stopStreamingRequest()
  }, [stopStreamingRequest, workspaceSessionId])

  const handleRetryConnection = async () => {
    const now = Date.now()
    const recentRetryAttempts = retryAttemptTimestampsRef.current.filter(
      (timestamp) => now - timestamp <= RETRY_BURST_WINDOW_MS
    )
    recentRetryAttempts.push(now)
    retryAttemptTimestampsRef.current = recentRetryAttempts

    if (
      statusGuardrailsEnabled &&
      recentRetryAttempts.length >= RETRY_BURST_THRESHOLD &&
      now - lastRetryBurstSignalRef.current > RETRY_BURST_WINDOW_MS
    ) {
      lastRetryBurstSignalRef.current = now
      void trackWorkspacePlaygroundTelemetry({
        type: "confusion_retry_burst",
        workspace_id: workspaceSessionId || null,
        retry_count: recentRetryAttempts.length,
        window_ms: RETRY_BURST_WINDOW_MS
      })
    }

    setSubmitError(null)
    await checkConnectionOnce()
  }

  const handleCitationSourceClick = React.useCallback(
    (citation: unknown) => {
      if (!provenanceEnabled) return
      const citationMediaId = extractCitationMediaId(citation)
      void trackWorkspacePlaygroundTelemetry({
        type: "citation_provenance_opened",
        workspace_id: workspaceSessionId || null,
        has_media_id: citationMediaId !== null
      })
      if (citationMediaId !== null && focusSourceByMediaId(citationMediaId)) {
        return
      }

      const matchedSourceId = findSourceIdFromCitation(citation, sources)
      if (matchedSourceId) {
        focusSourceById(matchedSourceId)
      }
    },
    [
      focusSourceById,
      focusSourceByMediaId,
      provenanceEnabled,
      sources,
      workspaceSessionId
    ]
  )

  const loadLorebookActivity = React.useCallback(async () => {
    if (!serverChatId) {
      setLorebookActivityTurns([])
      setLorebookActivityTotal(0)
      setLorebookActivityLoading(false)
      setLorebookActivityError(null)
      setLorebookActivityForbidden(false)
      return
    }

    setLorebookActivityLoading(true)
    setLorebookActivityError(null)
    setLorebookActivityForbidden(false)
    try {
      const response = await tldwClient.getChatLorebookDiagnostics(
        serverChatId,
        {
          page: 1,
          size: LOREBOOK_ACTIVITY_PAGE_SIZE,
          order: "desc"
        },
        chatScope ? { scope: chatScope } : undefined
      )
      const turns = Array.isArray(response?.turns) ? response.turns : []
      const normalizedTurns: LorebookActivityTurn[] = turns
        .slice(0, LOREBOOK_ACTIVITY_PAGE_SIZE)
        .map((turn: any) => ({
        turnNumber:
          typeof turn?.turn_number === "number"
            ? turn.turn_number
            : Number(turn?.turn_number || 0),
        assistantPreview: String(turn?.message_preview || ""),
        entryCount: Array.isArray(turn?.diagnostics) ? turn.diagnostics.length : 0
      }))
      setLorebookActivityTurns(normalizedTurns)
      setLorebookActivityTotal(
        typeof response?.total_turns_with_diagnostics === "number"
          ? response.total_turns_with_diagnostics
          : normalizedTurns.length
      )
    } catch (error: any) {
      const messageText = String(error?.message || "Failed to load lorebook activity.")
      const forbidden = /(403|forbidden|not authorized|permission)/i.test(messageText)
      setLorebookActivityForbidden(forbidden)
      setLorebookActivityError(messageText)
      setLorebookActivityTurns([])
      setLorebookActivityTotal(0)
    } finally {
      setLorebookActivityLoading(false)
    }
  }, [chatScope, serverChatId])

  const handleExportLorebookActivity = React.useCallback(async () => {
    if (!serverChatId || exportingLorebookActivity) return
    setExportingLorebookActivity(true)
    try {
      const response = await tldwClient.getChatLorebookDiagnostics(
        serverChatId,
        {
          page: 1,
          size: LOREBOOK_ACTIVITY_EXPORT_PAGE_SIZE,
          order: "asc"
        },
        chatScope ? { scope: chatScope } : undefined
      )
      const payload = {
        exported_at: new Date().toISOString(),
        chat_id: String(serverChatId),
        total_turns_with_diagnostics: response?.total_turns_with_diagnostics || 0,
        turns: Array.isArray(response?.turns) ? response.turns : []
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json;charset=utf-8"
      })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = `lorebook-activity-${String(serverChatId)}.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error: any) {
      messageApi.error(
        error?.message || "Failed to export lorebook activity diagnostics."
      )
    } finally {
      setExportingLorebookActivity(false)
    }
  }, [chatScope, exportingLorebookActivity, messageApi, serverChatId])

  React.useEffect(() => {
    if (!hasMessages || !serverChatId) {
      setLorebookActivityTurns([])
      setLorebookActivityTotal(0)
      setLorebookActivityLoading(false)
      setLorebookActivityError(null)
      setLorebookActivityForbidden(false)
      return
    }
    void loadLorebookActivity()
  }, [hasMessages, loadLorebookActivity, serverChatId])

  const handleSaveMessageToNotes = React.useCallback(
    (msg: {
      isBot: boolean
      message: string
      name?: string
    }) => {
      const snippet = String(msg.message || "").trim()
      if (!snippet) return
      captureToCurrentNote({
        title: buildCapturedMessageTitle(
          msg.isBot,
          snippet,
          t("playground:chat.savedAssistantPrefix", "Assistant"),
          t("playground:chat.savedUserPrefix", "User")
        ),
        content: snippet,
        mode: "append"
      })
    },
    [captureToCurrentNote, t]
  )

  // Share conversation handler (UX-044)
  const handleShareConversation = React.useCallback(async () => {
    if (!serverChatId || sharingConversation) return
    setSharingConversation(true)
    try {
      const result = await tldwClient.createConversationShareLink(
        serverChatId,
        {
          label: "Workspace share"
        },
        chatScope ? { scope: chatScope } : undefined
      )
      const shareUrl = buildConversationShareUrl(window.location.origin, {
        share_path: result?.share_path ?? null,
        token: result?.token ?? null
      })
      if (shareUrl) {
        await navigator.clipboard.writeText(String(shareUrl))
        messageApi.success(
          t("playground:chat.shareUrlCopied", "Share link copied to clipboard")
        )
      } else {
        messageApi.info(
          t("playground:chat.shareLinkCreated", "Share link created")
        )
      }
    } catch {
      messageApi.error(
        t("playground:chat.shareError", "Failed to create share link")
      )
    } finally {
      setSharingConversation(false)
    }
  }, [chatScope, serverChatId, sharingConversation, messageApi, t])

  // Fetch slash commands once (UX-006)
  React.useEffect(() => {
    if (slashCommandsFetchedRef.current) return
    slashCommandsFetchedRef.current = true
    if (typeof tldwClient.listChatCommands !== "function") return
    try {
      tldwClient.listChatCommands().then((data: unknown) => {
        const record = isRecord(data) ? data : {}
        const commands = Array.isArray(data)
          ? data
          : Array.isArray(record.commands)
            ? record.commands
            : []
        setSlashCommands(
          commands
            .filter(
              (cmd: unknown): cmd is { name: string; description: string } =>
                isRecord(cmd) && typeof cmd.name === "string"
            )
            .map((cmd: { name: string; description?: string }) => ({
              name: String(cmd.name),
              description: String(cmd.description || "")
            }))
        )
      }).catch(() => {
        // Slash commands unavailable - silently degrade
      })
    } catch {
      // Slash commands unavailable - silently degrade
    }
  }, [])

  // Model display label (UX-009)
  const modelDisplayLabel = React.useMemo(() => {
    if (selectedModel) return selectedModel
    if (chatApiProvider) return chatApiProvider
    return null
  }, [selectedModel, chatApiProvider])

  // Conversation instance ID (use workspace ID or fallback)
  const conversationInstanceId = workspaceSessionId
  const hasConnectionFailure =
    connectionState.phase === ConnectionPhase.ERROR &&
    !connectionState.isChecking
  const isChatUnavailable =
    statusGuardrailsEnabled &&
    (submitError !== null || hasConnectionFailure)
  const showConnectionBanner = isChatUnavailable
  const connectionDescription =
    submitError ||
    connectionState.lastError ||
    t(
      "playground:chat.connectionErrorGeneric",
      "Unable to reach server. Please retry."
    )
  const contentMaxWidthClass = "max-w-none"

  return (
    <div className="flex h-full w-full min-h-0 flex-1 flex-col">
      {messageContextHolder}

      {/* Context indicator */}
      <ChatContextIndicator />

      {/* Connection banner */}
      {showConnectionBanner && (
        <div className="border-b border-error/30 bg-error/5 px-4 py-2">
          <div className={`mx-auto flex w-full ${contentMaxWidthClass} items-center justify-between gap-3`}>
            <div className="flex min-w-0 items-center gap-2 text-sm text-error">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span className="truncate">
                {t("playground:chat.connectionBanner", "Unable to reach server")}
                : {connectionDescription}
              </span>
            </div>
            <button
              type="button"
              onClick={() => {
                void handleRetryConnection()
              }}
              className="shrink-0 rounded border border-error/40 bg-surface px-2 py-1 text-xs font-medium text-error transition hover:bg-error/10"
            >
              <RotateCcw className="mr-1 inline h-3.5 w-3.5" />
              {t("common:retry", "Retry")}
            </button>
          </div>
        </div>
      )}

      {/* Chat controls */}
      <div className="border-b border-border bg-surface px-4 py-2">
        <div className={`mx-auto flex w-full ${contentMaxWidthClass} items-center justify-end`}>
          <button
            type="button"
            onClick={handleClearChat}
            disabled={!hasMessages}
            className="rounded p-1.5 text-text-muted transition hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={t("playground:chat.clearChat", "Clear chat")}
            title={t("playground:chat.clearChat", "Clear chat") as string}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {hasMessages && (
        <div className="border-b border-border bg-surface px-4 py-2">
          <div className={`mx-auto w-full ${contentMaxWidthClass} space-y-2`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-medium text-text">Lorebook Activity</p>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void loadLorebookActivity()}
                  className="rounded border border-border px-2 py-1 text-xs text-text-muted transition hover:border-primary/40 hover:text-text"
                >
                  Refresh
                </button>
                <button
                  type="button"
                  onClick={() => void handleExportLorebookActivity()}
                  disabled={!serverChatId || exportingLorebookActivity}
                  className="rounded border border-border px-2 py-1 text-xs text-text-muted transition hover:border-primary/40 hover:text-text disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {exportingLorebookActivity ? "Exporting…" : "Export"}
                </button>
                <a
                  href={LOREBOOK_DEBUG_ENTRYPOINT_HREF}
                  className="text-xs text-primary hover:underline"
                  aria-label="Open full lorebook diagnostics"
                >
                  View Full Diagnostics
                </a>
              </div>
            </div>

            {!serverChatId && (
              <p className="text-xs text-text-muted">
                Lorebook activity appears once this chat is saved to server.
              </p>
            )}

            {serverChatId && lorebookActivityLoading && (
              <p className="text-xs text-text-muted">Loading lorebook activity…</p>
            )}

            {serverChatId && lorebookActivityError && (
              <p className="text-xs text-danger">
                {lorebookActivityForbidden
                  ? "Lorebook activity is unavailable for this account."
                  : lorebookActivityError}
              </p>
            )}

            {serverChatId &&
              !lorebookActivityLoading &&
              !lorebookActivityError &&
              lorebookActivityTurns.length === 0 && (
                <p className="text-xs text-text-muted">
                  No lorebook entries have fired yet for this chat.
                </p>
              )}

            {serverChatId &&
              !lorebookActivityLoading &&
              !lorebookActivityError &&
              lorebookActivityTurns.length > 0 && (
                <div className="space-y-1">
                  {lorebookActivityTurns.map((turn) => (
                    <div
                      key={`lorebook-turn-${turn.turnNumber}`}
                      className="rounded border border-border px-2 py-1"
                    >
                      <p className="text-xs font-medium text-text">
                        Turn {turn.turnNumber}: {turn.entryCount} entries fired
                      </p>
                      {turn.assistantPreview && (
                        <p className="line-clamp-2 text-xs text-text-muted">
                          {turn.assistantPreview}
                        </p>
                      )}
                    </div>
                  ))}
                  {lorebookActivityTotal > lorebookActivityTurns.length && (
                    <p className="text-[11px] text-text-muted">
                      Showing {lorebookActivityTurns.length} of {lorebookActivityTotal} turns with diagnostics.
                    </p>
                  )}
                </div>
              )}
          </div>
        </div>
      )}

      {/* Chat messages area */}
      <div className="relative flex min-h-0 flex-1 flex-col">
        <div
          ref={containerRef}
          role="log"
          aria-live="polite"
          aria-relevant="additions"
          aria-label={t("playground:aria.chatTranscript", "Chat messages")}
          className="custom-scrollbar flex-1 min-h-0 overflow-x-hidden overflow-y-auto px-4"
        >
          <div className={`mx-auto w-full ${contentMaxWidthClass} pb-6`}>
            {hasMessages ? (
              <div className="space-y-4 py-4">
                {messages.map((msg, idx) => {
                  const diagnostics = msg.isBot
                    ? buildRetrievalDiagnostics(msg.sources, msg.generationInfo)
                    : null
                  const chatSearchMessageId = getWorkspaceChatSearchMessageId(
                    msg,
                    idx
                  )
                  const isHighlighted =
                    highlightedChatMessageId === chatSearchMessageId

                  return (
                    <div
                      key={msg.id || `msg-${idx}`}
                      data-chat-message-id={chatSearchMessageId}
                      ref={(element) => {
                        chatMessageItemRefs.current[chatSearchMessageId] = element
                      }}
                      className={`space-y-2 rounded-md transition ${
                        isHighlighted ? "ring-2 ring-primary/40 bg-primary/5 p-1.5" : ""
                      }`}
                    >
                      <PlaygroundMessage
                        isBot={msg.isBot}
                        message={msg.message}
                        name={msg.name}
                        images={msg.images}
                        generationInfo={msg.generationInfo}
                        sources={msg.sources}
                        toolCalls={msg.toolCalls}
                        toolResults={msg.toolResults}
                        reasoningTimeTaken={msg.reasoning_time_taken}
                        currentMessageIndex={idx}
                        totalMessages={messages.length}
                        isProcessing={isProcessing}
                        isStreaming={streaming && idx === messages.length - 1}
                        conversationInstanceId={conversationInstanceId}
                        historyId={historyId || undefined}
                        serverChatId={serverChatId}
                        serverMessageId={msg.serverMessageId}
                        scope={chatScope}
                        messageId={msg.id}
                        discoSkillComment={msg.discoSkillComment}
                        createdAt={msg.createdAt}
                        variants={msg.variants}
                        activeVariantIndex={msg.activeVariantIndex}
                        onSwipePrev={
                          msg.isBot
                            ? (id: string) => {
                                const foundIdx = messages.findIndex((m) => m.id === id)
                                if (foundIdx >= 0) handleSwitchMessageVariant(foundIdx, "prev")
                              }
                            : undefined
                        }
                        onSwipeNext={
                          msg.isBot
                            ? (id: string) => {
                                const foundIdx = messages.findIndex((m) => m.id === id)
                                if (foundIdx >= 0) handleSwitchMessageVariant(foundIdx, "next")
                              }
                            : undefined
                        }
                        onNewBranch={
                          msg.isBot ? (idx: number) => handleCreateChatBranch(idx) : undefined
                        }
                        modelName={msg.modelName}
                        modelImage={msg.modelImage}
                        onSourceClick={
                          provenanceEnabled ? handleCitationSourceClick : undefined
                        }
                        onSaveToWorkspaceNotes={() =>
                          handleSaveMessageToNotes({
                            isBot: msg.isBot,
                            message: msg.message,
                            name: msg.name
                          })
                        }
                        onRegenerate={
                          msg.isBot && idx === messages.length - 1
                            ? () => regenerateLastMessage()
                            : () => {}
                        }
                        onDeleteMessage={(idx: number) => handleDeleteMessageWithUndo(idx)}
                        suppressDeleteSuccessToast
                        onEditFormSubmit={(idx: number, value: string, isUser: boolean, isSend?: boolean) => {
                          editMessage(idx, value, isUser, isSend)
                        }}
                        hideEditAndRegenerate={!msg.isBot && idx !== messages.length - 1}
                        hideContinue={true}
                        temporaryChat={false}
                      />
                      {msg.isBot && diagnostics && provenanceEnabled && (
                        <div className="px-2">
                          <RetrievalDiagnosticsPanel diagnostics={diagnostics} />
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="flex min-h-full flex-col justify-end">
                <WorkspaceChatEmpty
                  hasSelectedSources={hasSelectedSources}
                  sourceCount={selectedSources.length}
                  totalSourceCount={sources.length}
                  selectedSourceTypes={selectedSources.map((source) => source.type)}
                  isMobile={isMobile}
                  layoutMode={contentWidthMode}
                  onExamplePromptSelect={(prompt) => setSeededPrompt(prompt)}
                  onAddSource={openAddSourceModal}
                  onCreateFromTemplate={handleCreateFromTemplate}
                />
              </div>
            )}
          </div>
        </div>

        {/* Scroll to bottom button */}
        {!isAutoScrollToBottom && hasMessages && (
          <div className="pointer-events-none absolute bottom-24 left-0 right-0 flex justify-center">
            <button
              onClick={() => autoScrollToBottom()}
              aria-label={t(
                "playground:composer.scrollToLatest",
                "Scroll to latest messages"
              )}
              title={
                t(
                  "playground:composer.scrollToLatest",
                  "Scroll to latest messages"
                ) as string
              }
              className="pointer-events-auto rounded-full border border-border bg-surface p-2 text-text-subtle shadow-card transition-colors hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              <ChevronDown className="size-4 text-text-subtle" aria-hidden="true" />
            </button>
          </div>
        )}
      </div>

      {/* Chat input */}
      <div
        data-testid="chat-drop-zone"
        data-drop-active={dropZoneActive ? "true" : "false"}
        onDragOver={handleDropZoneDragOver}
        onDrop={handleDropZoneDrop}
        onDragLeave={handleDropZoneDragLeave}
        className={`sticky bottom-0 z-20 border-t border-border bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/85 transition-colors ${
          dropZoneActive ? "bg-primary/5" : ""
        }`}
      >
        <div className={`mx-auto w-full ${contentMaxWidthClass} px-4 pb-[calc(env(safe-area-inset-bottom)+0.5rem)] pt-2.5`}>
          {dropZoneActive && (
            <div className="mb-2 rounded-md border border-primary/40 bg-primary/10 px-3 py-2 text-xs text-primary">
              {t(
                "playground:chat.dropZoneHint",
                "Drop source to scope chat and start a source-specific question."
              )}
            </div>
          )}
          {temporarySourceScope && (
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-xs">
              <p className="text-text">
                {t(
                  "playground:chat.dragScopedTemporary",
                  'Temporarily scoped to "{{title}}".',
                  { title: temporarySourceScope.sourceTitle }
                ).replace("{{title}}", temporarySourceScope.sourceTitle)}
              </p>
              <button
                type="button"
                onClick={() => restoreTemporaryScope("manual")}
                className="rounded border border-border bg-surface px-2 py-1 text-text-muted transition hover:text-text"
              >
                {t(
                  "playground:chat.restorePreviousSelection",
                  "Restore previous selection"
                )}
              </button>
            </div>
          )}
          <div
            data-testid="workspace-chat-controls-toolbar"
            role="toolbar"
            aria-label={t("playground:chat.controlsToolbarAria", "Chat controls")}
            className="mb-2 flex flex-wrap items-center gap-1.5 text-xs"
          >
            <div className="inline-flex items-center rounded-full border border-border/70 bg-surface/80 p-0.5 shadow-sm">
              <button
                type="button"
                onClick={() => setPreferredChatMode("normal")}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition ${
                  effectiveChatMode === "normal"
                    ? "bg-primary text-white"
                    : "text-text-muted hover:text-text"
                }`}
                aria-pressed={effectiveChatMode === "normal"}
              >
                {t("playground:chat.generalMode", "General chat")}
              </button>
              <Tooltip
                title={
                  hasSelectedSources
                    ? undefined
                    : t(
                        "playground:chat.ragModeRequiresSources",
                        "Select sources to enable RAG mode"
                      )
                }
              >
                <button
                  type="button"
                  disabled={!hasSelectedSources}
                  onClick={() => setPreferredChatMode("rag")}
                  className={`rounded-full px-2.5 py-1 text-xs font-medium transition ${
                    effectiveChatMode === "rag"
                      ? "bg-primary text-white"
                      : "text-text-muted hover:text-text"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                  aria-pressed={effectiveChatMode === "rag"}
                >
                  {t("playground:chat.ragMode", "RAG mode")}
                </button>
              </Tooltip>
            </div>
            {hasSelectedSources && (
              <label
                className="inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-[11px] text-text-muted"
                title={
                  t(
                    "playground:chat.includeFullSourcesHint",
                    "Inject complete extracted source text with your question (helpful for synopsis-style prompts)."
                  ) as string
                }
              >
                <span className="whitespace-nowrap">
                  {t(
                    "playground:chat.includeFullSourcesCompactLabel",
                    "Full source text"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={includeFullSourceContents}
                  onChange={setIncludeFullSourceContents}
                  aria-label={t(
                    "playground:chat.includeFullSourcesAria",
                    "Include full source contents"
                  )}
                />
              </label>
            )}
            <div className="ml-auto flex flex-wrap items-center gap-1">
              {provenanceEnabled && (loadingModels || availableModels.length > 0) && (
                <label
                  className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-[11px] text-text-muted"
                  title={
                    modelDisplayLabel
                      ? t(
                          "playground:chat.currentModelTooltip",
                          "Current model: {{model}}",
                          { model: modelDisplayLabel }
                        ).replace("{{model}}", modelDisplayLabel)
                      : undefined
                  }
                >
                  <Cpu className="h-3 w-3" />
                  <span>{t("playground:chat.modelPickerLabel", "Model")}</span>
                  <select
                    aria-label={t(
                      "playground:chat.modelPickerAria",
                      "Select model"
                    )}
                    value={selectedModel ?? ""}
                    onChange={(event) => {
                      if (typeof setSelectedModel !== "function") return
                      const value = event.target.value.trim()
                      setSelectedModel(value.length > 0 ? value : null)
                    }}
                    className="max-w-[180px] truncate bg-transparent text-[11px] text-text focus:outline-none"
                    disabled={loadingModels}
                  >
                    <option value="">
                      {t("playground:chat.modelPickerAuto", "Auto")}
                    </option>
                    {availableModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {/* Share conversation button (UX-044) */}
              {serverChatId && hasMessages && (
                <Tooltip
                  title={t("playground:chat.shareConversation", "Share conversation")}
                >
                  <button
                    type="button"
                    onClick={handleShareConversation}
                    disabled={sharingConversation}
                    className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-xs text-text-muted transition hover:bg-surface2 hover:text-text disabled:opacity-50"
                    aria-label={t("playground:chat.shareConversation", "Share conversation")}
                  >
                    {sharingConversation ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Share2 className="h-3.5 w-3.5" />
                    )}
                  </button>
                </Tooltip>
              )}
              {preferredChatMode !== null && (
                <button
                  type="button"
                  onClick={() => setPreferredChatMode(null)}
                  className="text-xs text-text-muted transition hover:text-text hover:underline"
                >
                  {t("playground:chat.modeAuto", "Auto")}
                </button>
              )}
              {hasSelectedSources && (
                <button
                  type="button"
                  onClick={() =>
                    setShowAdvancedRagSettings((current) => !current)
                  }
                  className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-xs text-text-muted transition hover:bg-surface2 hover:text-text"
                  aria-expanded={showAdvancedRagSettings}
                >
                  <SlidersHorizontal className="h-3.5 w-3.5" />
                  {t(
                    "playground:chat.advancedRagSettings",
                    "Advanced RAG settings"
                  )}
                </button>
              )}
            </div>
          </div>
          {hasSelectedSources && effectiveChatMode === "normal" && (
            <p className="mb-2 text-xs text-warn">
              {t(
                "playground:chat.generalModeWithSourcesHint",
                "General chat mode is active. Selected sources will not be used unless RAG mode is enabled."
              )}
            </p>
          )}
          {preparingSourceContext && (
            <p className="mb-2 flex items-center gap-1.5 text-xs text-text-muted">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              {t(
                "playground:chat.preparingSourceContext",
                "Preparing source context"
              )}
            </p>
          )}
          {hasSelectedSources && showAdvancedRagSettings && (
            <div className="mb-2 rounded-lg border border-border/70 bg-surface2/40 p-3">
              <div className="space-y-3">
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                    <span>{t("playground:chat.ragTopK", "Top K")}</span>
                    <span className="font-medium text-text">{resolvedTopK}</span>
                  </div>
                  <Slider
                    min={1}
                    max={50}
                    step={1}
                    value={resolvedTopK}
                    onChange={handleTopKChange}
                  />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                    <span>
                      {t(
                        "playground:chat.ragSimilarityThreshold",
                        "Similarity threshold"
                      )}
                    </span>
                    <span className="font-medium text-text">
                      {similarityThreshold.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={similarityThreshold}
                    onChange={handleSimilarityThresholdChange}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-muted">
                    {t("playground:chat.ragReranking", "Enable reranking")}
                  </span>
                  <Switch
                    size="small"
                    checked={rerankingEnabled}
                    onChange={handleRerankingToggle}
                    aria-label={t("playground:chat.ragReranking", "Enable reranking")}
                  />
                </div>
              </div>
            </div>
          )}
          <SimpleChatInput
            onSubmit={handleSubmit}
            onStop={handleStopStreaming}
            isLoading={streaming}
            isPreparingContext={preparingSourceContext}
            isChatUnavailable={isChatUnavailable}
            seededValue={seededPrompt}
            onSeedConsumed={() => setSeededPrompt(null)}
            slashCommands={slashCommands}
            placeholder={
              hasSelectedSources
                ? t(
                    "playground:chat.inputPlaceholderWithSources",
                    "Ask about your sources..."
                  )
                : t(
                    "playground:chat.inputPlaceholder",
                    "Type a message..."
                  )
            }
          />
        </div>
      </div>
    </div>
  )
}

export default ChatPane
