import React from "react"
import {
  Button,
  Checkbox,
  Collapse,
  Dropdown,
  Input,
  List,
  Modal,
  Select,
  Spin,
  Switch
} from "antd"
import { X } from "lucide-react"
import { browser } from "wxt/browser"
import { useTranslation } from "react-i18next"
import type { UploadedFile } from "@/db/dexie/types"
import type { TabInfo } from "@/hooks/useTabMentions"
import { formatRagResult } from "@/utils/rag-format"
import { withFullMediaTextIfAvailable } from "@/components/Knowledge/hooks"
import { formatFileSize } from "@/utils/format"

import { useRagSearchState } from "./hooks/useRagSearchState"
import {
  useRagResultsDisplay,
  getResultText,
  getResultTitle,
  getResultScore,
  getResultType,
  getResultDate,
  toPinnedResult,
  formatScore,
  formatDate,
  highlightText,
} from "./hooks/useRagResultsDisplay"
import {
  useRagFilterPanel,
  STRATEGY_OPTIONS,
  SEARCH_MODE_OPTIONS,
  FTS_LEVEL_OPTIONS,
  EXPANSION_OPTIONS,
  SENSITIVITY_OPTIONS,
  TABLE_METHOD_OPTIONS,
  CHUNK_TYPE_OPTIONS,
  CLAIM_EXTRACTOR_OPTIONS,
  CLAIM_VERIFIER_OPTIONS,
  RERANK_STRATEGY_OPTIONS,
  CITATION_STYLE_OPTIONS,
  ABSTENTION_OPTIONS,
  CONTENT_POLICY_TYPES,
  CONTENT_POLICY_MODES,
  NUMERIC_FIDELITY_OPTIONS,
  LOW_CONFIDENCE_OPTIONS,
  parseNumericIdList,
  parseStringIdList,
  stringifyIdList,
} from "./hooks/useRagFilterPanel"
import { useRagSearchHistory } from "./hooks/useRagSearchHistory"
import { getRagSourceOptions } from "@/services/rag/sourceMetadata"

type Props = {
  onInsert: (text: string) => void
  onAsk: (text: string, options?: { ignorePinnedResults?: boolean }) => void
  isConnected?: boolean
  open?: boolean
  onOpenChange?: (nextOpen: boolean) => void
  autoFocus?: boolean
  showToggle?: boolean
  variant?: "card" | "embedded"
  currentMessage?: string
  showAttachedContext?: boolean
  attachedTabs?: TabInfo[]
  availableTabs?: TabInfo[]
  attachedFiles?: UploadedFile[]
  onRemoveTab?: (tabId: number) => void
  onAddTab?: (tab: TabInfo) => void
  onClearTabs?: () => void
  onRefreshTabs?: () => void
  onAddFile?: () => void
  onRemoveFile?: (fileId: string) => void
  onClearFiles?: () => void
}

export const RagSearchBar: React.FC<Props> = ({
  onInsert,
  onAsk,
  isConnected = true,
  open,
  onOpenChange,
  autoFocus = true,
  showToggle = true,
  variant = "card",
  currentMessage,
  showAttachedContext = false,
  attachedTabs = [],
  availableTabs = [],
  attachedFiles = [],
  onRemoveTab,
  onAddTab,
  onClearTabs,
  onRefreshTabs,
  onAddFile,
  onRemoveFile,
  onClearFiles
}) => {
  const { t } = useTranslation(["sidepanel", "playground", "common"])
  const sourceOptions = React.useMemo(
    () => getRagSourceOptions((key, fallback) => t(key, fallback)),
    [t]
  )
  const [internalOpen, setInternalOpen] = React.useState(false)
  const isControlled = typeof open === "boolean"
  const isOpen = isControlled ? open : internalOpen
  const setOpenState = React.useCallback(
    (next: boolean) => {
      if (isControlled) {
        onOpenChange?.(next)
        return
      }
      setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange]
  )

  const search = useRagSearchState({ currentMessage, t })
  const resultsDisplay = useRagResultsDisplay({
    results: search.results,
    batchResults: search.batchResults,
    ragPinnedResults: search.ragPinnedResults,
    setRagPinnedResults: search.setRagPinnedResults,
    onInsert,
    onAsk,
    t
  })
  const filter = useRagFilterPanel({
    draftSettings: search.draftSettings,
    updateSetting: search.updateSetting,
    t
  })
  const history = useRagSearchHistory({
    resolvedQuery: search.resolvedQuery,
    resultsLength: search.results.length,
    batchResultsLength: search.batchResults.length,
    draftSettings: search.draftSettings
  })

  React.useEffect(() => {
    const handler = () => setOpenState(!isOpen)
    window.addEventListener("tldw:toggle-rag", handler)
    return () => window.removeEventListener("tldw:toggle-rag", handler)
  }, [isOpen, setOpenState])

  React.useEffect(() => {
    if (!isOpen || !autoFocus) return
    const id = requestAnimationFrame(() => search.searchInputRef.current?.focus())
    return () => cancelAnimationFrame(id)
  }, [isOpen, autoFocus])

  const wrapperClassName = variant === "embedded" ? "w-full" : "w-full mb-2"
  const panelClassName =
    variant === "embedded"
      ? "panel-elevated p-2 relative"
      : "panel-card p-2 mb-2 relative"

  const { matchesAny } = filter

  const advancedItems = [
    matchesAny(
      t("sidepanel:rag.sourceScope", "Source scope") as string,
      t("sidepanel:rag.corpus", "Corpus") as string,
      t("sidepanel:rag.indexNamespace", "Index namespace") as string
    ) && {
      key: "source-scope",
      label: t("sidepanel:rag.sourceScope", "Source scope"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {matchesAny(
            t("sidepanel:rag.sourceScope", "Source scope") as string,
            t("sidepanel:rag.corpus", "Corpus") as string
          ) &&
            filter.renderTextInput(
              t("sidepanel:rag.corpus", "Corpus"),
              search.draftSettings.corpus,
              (next) => search.updateSetting("corpus", next)
            )}
          {matchesAny(
            t("sidepanel:rag.sourceScope", "Source scope") as string,
            t("sidepanel:rag.indexNamespace", "Index namespace") as string
          ) &&
            filter.renderTextInput(
              t("sidepanel:rag.indexNamespace", "Index namespace"),
              search.draftSettings.index_namespace,
              (next) => search.updateSetting("index_namespace", next)
            )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.queryExpansion", "Query expansion") as string,
      t("sidepanel:rag.expandQuery", "Expand query") as string,
      t("sidepanel:rag.expansionStrategies", "Expansion strategies") as string,
      t("sidepanel:rag.spellCheck", "Spell check") as string
    ) && {
      key: "query-expansion",
      label: t("sidepanel:rag.queryExpansion", "Query expansion"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {matchesAny(
            t("sidepanel:rag.queryExpansion", "Query expansion") as string,
            t("sidepanel:rag.expandQuery", "Expand query") as string,
            t("sidepanel:rag.expansionStrategies", "Expansion strategies") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.expand_query}
                onChange={(checked) => search.updateSetting("expand_query", checked)}
                aria-label={t("sidepanel:rag.expandQuery", "Expand query")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.expandQuery", "Expand query")}
              </span>
            </div>
          )}
          {search.draftSettings.expand_query &&
            matchesAny(
              t("sidepanel:rag.queryExpansion", "Query expansion") as string,
              t("sidepanel:rag.expansionStrategies", "Expansion strategies") as string
            ) &&
            filter.renderMultiSelect(
              t("sidepanel:rag.expansionStrategies", "Expansion strategies"),
              search.draftSettings.expansion_strategies,
              (next) => search.updateSetting("expansion_strategies", next as any),
              EXPANSION_OPTIONS
            )}
          {matchesAny(
            t("sidepanel:rag.queryExpansion", "Query expansion") as string,
            t("sidepanel:rag.spellCheck", "Spell check") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.spell_check}
                onChange={(checked) => search.updateSetting("spell_check", checked)}
                aria-label={t("sidepanel:rag.spellCheck", "Spell check")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.spellCheck", "Spell check")}
              </span>
            </div>
          )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.caching", "Caching") as string,
      t("sidepanel:rag.enableCache", "Enable cache") as string,
      t("sidepanel:rag.cacheThreshold", "Cache threshold") as string,
      t("sidepanel:rag.adaptiveCache", "Adaptive cache") as string
    ) && {
      key: "caching",
      label: t("sidepanel:rag.caching", "Caching"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {matchesAny(
            t("sidepanel:rag.caching", "Caching") as string,
            t("sidepanel:rag.enableCache", "Enable cache") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.enable_cache}
                onChange={(checked) => search.updateSetting("enable_cache", checked)}
                aria-label={t("sidepanel:rag.enableCache", "Enable cache")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.enableCache", "Enable cache")}
              </span>
            </div>
          )}
          {search.draftSettings.enable_cache && (
            <>
              {matchesAny(
                t("sidepanel:rag.caching", "Caching") as string,
                t("sidepanel:rag.cacheThreshold", "Cache threshold") as string
              ) &&
                filter.renderNumberInput(
                  t("sidepanel:rag.cacheThreshold", "Cache threshold"),
                  search.draftSettings.cache_threshold,
                  (next) => search.updateSetting("cache_threshold", next),
                  { min: 0, max: 1, step: 0.05 }
                )}
              {matchesAny(
                t("sidepanel:rag.caching", "Caching") as string,
                t("sidepanel:rag.adaptiveCache", "Adaptive cache") as string
              ) && (
                <div className="flex items-center gap-2">
                  <Switch
                    checked={search.draftSettings.adaptive_cache}
                    onChange={(checked) => search.updateSetting("adaptive_cache", checked)}
                    aria-label={t("sidepanel:rag.adaptiveCache", "Adaptive cache")}
                  />
                  <span className="text-xs text-text">
                    {t("sidepanel:rag.adaptiveCache", "Adaptive cache")}
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.documentProcessing", "Document processing") as string,
      t("sidepanel:rag.enableTableProcessing", "Enable table processing") as string,
      t("sidepanel:rag.tableMethod", "Table method") as string
    ) && {
      key: "document-processing",
      label: t("sidepanel:rag.documentProcessing", "Document processing"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {matchesAny(
            t("sidepanel:rag.documentProcessing", "Document processing") as string,
            t("sidepanel:rag.enableTableProcessing", "Enable table processing") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.enable_table_processing}
                onChange={(checked) => search.updateSetting("enable_table_processing", checked)}
                aria-label={t("sidepanel:rag.enableTableProcessing", "Enable table processing")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.enableTableProcessing", "Enable table processing")}
              </span>
            </div>
          )}
          {search.draftSettings.enable_table_processing &&
            matchesAny(
              t("sidepanel:rag.documentProcessing", "Document processing") as string,
              t("sidepanel:rag.tableMethod", "Table method") as string
            ) &&
            filter.renderSelect(
              t("sidepanel:rag.tableMethod", "Table method"),
              search.draftSettings.table_method,
              (next) => search.updateSetting("table_method", next as any),
              TABLE_METHOD_OPTIONS
            )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.vlm", "VLM late chunking") as string,
      t("sidepanel:rag.enableVlm", "Enable VLM late chunking") as string,
      t("sidepanel:rag.vlmBackend", "VLM backend") as string,
      t("sidepanel:rag.vlmDetectTables", "Detect tables only") as string,
      t("sidepanel:rag.vlmMaxPages", "Max pages") as string,
      t("sidepanel:rag.vlmTopKDocs", "Top K docs") as string
    ) && {
      key: "vlm",
      label: t("sidepanel:rag.vlm", "VLM late chunking"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {matchesAny(
            t("sidepanel:rag.vlm", "VLM late chunking") as string,
            t("sidepanel:rag.enableVlm", "Enable VLM late chunking") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.enable_vlm_late_chunking}
                onChange={(checked) => search.updateSetting("enable_vlm_late_chunking", checked)}
                aria-label={t("sidepanel:rag.enableVlm", "Enable VLM late chunking")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.enableVlm", "Enable VLM late chunking")}
              </span>
            </div>
          )}
          {search.draftSettings.enable_vlm_late_chunking && (
            <>
              {matchesAny(
                t("sidepanel:rag.vlm", "VLM late chunking") as string,
                t("sidepanel:rag.vlmBackend", "VLM backend") as string
              ) &&
                filter.renderTextInput(
                  t("sidepanel:rag.vlmBackend", "VLM backend"),
                  search.draftSettings.vlm_backend || "",
                  (next) => search.updateSetting("vlm_backend", next || null)
                )}
              {matchesAny(
                t("sidepanel:rag.vlm", "VLM late chunking") as string,
                t("sidepanel:rag.vlmDetectTables", "Detect tables only") as string
              ) && (
                <div className="flex items-center gap-2">
                  <Switch
                    checked={search.draftSettings.vlm_detect_tables_only}
                    onChange={(checked) => search.updateSetting("vlm_detect_tables_only", checked)}
                    aria-label={t("sidepanel:rag.vlmDetectTables", "Detect tables only")}
                  />
                  <span className="text-xs text-text">
                    {t("sidepanel:rag.vlmDetectTables", "Detect tables only")}
                  </span>
                </div>
              )}
              {matchesAny(
                t("sidepanel:rag.vlm", "VLM late chunking") as string,
                t("sidepanel:rag.vlmMaxPages", "Max pages") as string
              ) &&
                filter.renderNumberInput(
                  t("sidepanel:rag.vlmMaxPages", "Max pages"),
                  search.draftSettings.vlm_max_pages,
                  (next) => search.updateSetting("vlm_max_pages", next),
                  { min: 1 }
                )}
              {matchesAny(
                t("sidepanel:rag.vlm", "VLM late chunking") as string,
                t("sidepanel:rag.vlmTopKDocs", "Top K docs") as string
              ) &&
                filter.renderNumberInput(
                  t("sidepanel:rag.vlmTopKDocs", "Top K docs"),
                  search.draftSettings.vlm_late_chunk_top_k_docs,
                  (next) => search.updateSetting("vlm_late_chunk_top_k_docs", next),
                  { min: 1 }
                )}
            </>
          )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
      t("sidepanel:rag.multiVector", "Multi-vector passages") as string,
      t("sidepanel:rag.mvSpanChars", "Span chars") as string,
      t("sidepanel:rag.mvStride", "Stride") as string,
      t("sidepanel:rag.mvMaxSpans", "Max spans") as string,
      t("sidepanel:rag.mvFlatten", "Flatten to spans") as string,
      t("sidepanel:rag.numericTableBoost", "Numeric table boost") as string
    ) && {
      key: "advanced-retrieval",
      label: t("sidepanel:rag.advancedRetrieval", "Advanced retrieval"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {matchesAny(
            t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
            t("sidepanel:rag.multiVector", "Multi-vector passages") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.enable_multi_vector_passages}
                onChange={(checked) => search.updateSetting("enable_multi_vector_passages", checked)}
                aria-label={t("sidepanel:rag.multiVector", "Multi-vector passages")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.multiVector", "Multi-vector passages")}
              </span>
            </div>
          )}
          {search.draftSettings.enable_multi_vector_passages && (
            <>
              {matchesAny(
                t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
                t("sidepanel:rag.mvSpanChars", "Span chars") as string
              ) &&
                filter.renderNumberInput(
                  t("sidepanel:rag.mvSpanChars", "Span chars"),
                  search.draftSettings.mv_span_chars,
                  (next) => search.updateSetting("mv_span_chars", next),
                  { min: 1 }
                )}
              {matchesAny(
                t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
                t("sidepanel:rag.mvStride", "Stride") as string
              ) &&
                filter.renderNumberInput(
                  t("sidepanel:rag.mvStride", "Stride"),
                  search.draftSettings.mv_stride,
                  (next) => search.updateSetting("mv_stride", next),
                  { min: 1 }
                )}
              {matchesAny(
                t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
                t("sidepanel:rag.mvMaxSpans", "Max spans") as string
              ) &&
                filter.renderNumberInput(
                  t("sidepanel:rag.mvMaxSpans", "Max spans"),
                  search.draftSettings.mv_max_spans,
                  (next) => search.updateSetting("mv_max_spans", next),
                  { min: 1 }
                )}
              {matchesAny(
                t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
                t("sidepanel:rag.mvFlatten", "Flatten to spans") as string
              ) && (
                <div className="flex items-center gap-2">
                  <Switch
                    checked={search.draftSettings.mv_flatten_to_spans}
                    onChange={(checked) => search.updateSetting("mv_flatten_to_spans", checked)}
                    aria-label={t("sidepanel:rag.mvFlatten", "Flatten to spans")}
                  />
                  <span className="text-xs text-text">
                    {t("sidepanel:rag.mvFlatten", "Flatten to spans")}
                  </span>
                </div>
              )}
            </>
          )}
          {matchesAny(
            t("sidepanel:rag.advancedRetrieval", "Advanced retrieval") as string,
            t("sidepanel:rag.numericTableBoost", "Numeric table boost") as string
          ) && (
            <div className="flex items-center gap-2">
              <Switch
                checked={search.draftSettings.enable_numeric_table_boost}
                onChange={(checked) => search.updateSetting("enable_numeric_table_boost", checked)}
                aria-label={t("sidepanel:rag.numericTableBoost", "Numeric table boost")}
              />
              <span className="text-xs text-text">
                {t("sidepanel:rag.numericTableBoost", "Numeric table boost")}
              </span>
            </div>
          )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.claims", "Claims & factuality") as string,
      t("sidepanel:rag.enableClaims", "Enable claims") as string,
      t("sidepanel:rag.claimExtractor", "Claim extractor") as string,
      t("sidepanel:rag.claimVerifier", "Claim verifier") as string,
      t("sidepanel:rag.claimsTopK", "Claims top_k") as string,
      t("sidepanel:rag.claimsThreshold", "Confidence threshold") as string,
      t("sidepanel:rag.claimsMax", "Claims max") as string,
      t("sidepanel:rag.claimsConcurrency", "Concurrency") as string,
      t("sidepanel:rag.nliModel", "NLI model") as string
    ) && {
      key: "claims",
      label: t("sidepanel:rag.claims", "Claims & factuality"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch
              checked={search.draftSettings.enable_claims}
              onChange={(checked) => search.updateSetting("enable_claims", checked)}
              aria-label={t("sidepanel:rag.enableClaims", "Enable claims")}
            />
            <span className="text-xs text-text">
              {t("sidepanel:rag.enableClaims", "Enable claims")}
            </span>
          </div>
          {search.draftSettings.enable_claims && (
            <>
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.claimExtractor", "Claim extractor") as string) &&
                filter.renderSelect(t("sidepanel:rag.claimExtractor", "Claim extractor"), search.draftSettings.claim_extractor, (next) => search.updateSetting("claim_extractor", next as any), CLAIM_EXTRACTOR_OPTIONS)}
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.claimVerifier", "Claim verifier") as string) &&
                filter.renderSelect(t("sidepanel:rag.claimVerifier", "Claim verifier"), search.draftSettings.claim_verifier, (next) => search.updateSetting("claim_verifier", next as any), CLAIM_VERIFIER_OPTIONS)}
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.claimsTopK", "Claims top_k") as string) &&
                filter.renderNumberInput(t("sidepanel:rag.claimsTopK", "Claims top_k"), search.draftSettings.claims_top_k, (next) => search.updateSetting("claims_top_k", next), { min: 1 })}
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.claimsThreshold", "Confidence threshold") as string) &&
                filter.renderNumberInput(t("sidepanel:rag.claimsThreshold", "Confidence threshold"), search.draftSettings.claims_conf_threshold, (next) => search.updateSetting("claims_conf_threshold", next), { min: 0, max: 1, step: 0.05 })}
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.claimsMax", "Claims max") as string) &&
                filter.renderNumberInput(t("sidepanel:rag.claimsMax", "Claims max"), search.draftSettings.claims_max, (next) => search.updateSetting("claims_max", next), { min: 1 })}
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.claimsConcurrency", "Concurrency") as string) &&
                filter.renderNumberInput(t("sidepanel:rag.claimsConcurrency", "Concurrency"), search.draftSettings.claims_concurrency, (next) => search.updateSetting("claims_concurrency", next), { min: 1 })}
              {matchesAny(t("sidepanel:rag.claims", "Claims & factuality") as string, t("sidepanel:rag.nliModel", "NLI model") as string) &&
                filter.renderTextInput(t("sidepanel:rag.nliModel", "NLI model"), search.draftSettings.nli_model, (next) => search.updateSetting("nli_model", next))}
            </>
          )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.guardrails", "Generation guardrails") as string,
      t("sidepanel:rag.contentPolicy", "Content policy filter") as string,
      t("sidepanel:rag.htmlSanitizer", "HTML sanitizer") as string,
      t("sidepanel:rag.ocrThreshold", "OCR confidence threshold") as string
    ) && {
      key: "guardrails",
      label: t("sidepanel:rag.guardrails", "Generation guardrails"),
      children: !search.draftSettings.enable_generation ? (
        <div className="text-xs text-text-muted">
          {t("sidepanel:rag.guardrailsRequiresGeneration", "Enable generation to configure guardrails.")}
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_content_policy_filter} onChange={(checked) => search.updateSetting("enable_content_policy_filter", checked)} aria-label={t("sidepanel:rag.contentPolicy", "Content policy filter")} />
            <span className="text-xs text-text">{t("sidepanel:rag.contentPolicy", "Content policy filter")}</span>
          </div>
          {search.draftSettings.enable_content_policy_filter && (
            <>
              {filter.renderMultiSelect(t("sidepanel:rag.contentPolicyTypes", "Policy types"), search.draftSettings.content_policy_types, (next) => search.updateSetting("content_policy_types", next as any), CONTENT_POLICY_TYPES)}
              {filter.renderSelect(t("sidepanel:rag.contentPolicyMode", "Policy mode"), search.draftSettings.content_policy_mode, (next) => search.updateSetting("content_policy_mode", next as any), CONTENT_POLICY_MODES)}
            </>
          )}
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_html_sanitizer} onChange={(checked) => search.updateSetting("enable_html_sanitizer", checked)} aria-label={t("sidepanel:rag.htmlSanitizer", "HTML sanitizer")} />
            <span className="text-xs text-text">{t("sidepanel:rag.htmlSanitizer", "HTML sanitizer")}</span>
          </div>
          {search.draftSettings.enable_html_sanitizer && (
            <>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text">{t("sidepanel:rag.allowedTags", "Allowed tags")}</span>
                <Select mode="tags" value={search.draftSettings.html_allowed_tags} onChange={(next) => search.updateSetting("html_allowed_tags", next as string[])} aria-label={t("sidepanel:rag.allowedTags", "Allowed tags")} />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text">{t("sidepanel:rag.allowedAttrs", "Allowed attrs")}</span>
                <Select mode="tags" value={search.draftSettings.html_allowed_attrs} onChange={(next) => search.updateSetting("html_allowed_attrs", next as string[])} aria-label={t("sidepanel:rag.allowedAttrs", "Allowed attrs")} />
              </div>
            </>
          )}
          {filter.renderNumberInput(t("sidepanel:rag.ocrThreshold", "OCR confidence threshold"), search.draftSettings.ocr_confidence_threshold, (next) => search.updateSetting("ocr_confidence_threshold", next), { min: 0, max: 1, step: 0.05 })}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.postVerification", "Post-verification") as string,
      t("sidepanel:rag.enablePostVerification", "Enable post verification") as string,
      t("sidepanel:rag.lowConfidence", "Low confidence behavior") as string
    ) && {
      key: "post-verification",
      label: t("sidepanel:rag.postVerification", "Post-verification"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_post_verification} onChange={(checked) => search.updateSetting("enable_post_verification", checked)} aria-label={t("sidepanel:rag.enablePostVerification", "Enable post verification")} />
            <span className="text-xs text-text">{t("sidepanel:rag.enablePostVerification", "Enable post verification")}</span>
          </div>
          {search.draftSettings.enable_post_verification && (
            <>
              {filter.renderNumberInput(t("sidepanel:rag.adaptiveRetries", "Max retries"), search.draftSettings.adaptive_max_retries, (next) => search.updateSetting("adaptive_max_retries", next), { min: 0 })}
              {filter.renderNumberInput(t("sidepanel:rag.adaptiveUnsupported", "Unsupported threshold"), search.draftSettings.adaptive_unsupported_threshold, (next) => search.updateSetting("adaptive_unsupported_threshold", next), { min: 0, max: 1, step: 0.05 })}
              {filter.renderNumberInput(t("sidepanel:rag.adaptiveMaxClaims", "Max claims"), search.draftSettings.adaptive_max_claims, (next) => search.updateSetting("adaptive_max_claims", next), { min: 1 })}
              {filter.renderNumberInput(t("sidepanel:rag.adaptiveBudget", "Time budget"), search.draftSettings.adaptive_time_budget_sec, (next) => search.updateSetting("adaptive_time_budget_sec", next), { min: 1 })}
              {filter.renderSelect(t("sidepanel:rag.lowConfidence", "Low confidence behavior"), search.draftSettings.low_confidence_behavior, (next) => search.updateSetting("low_confidence_behavior", next as any), LOW_CONFIDENCE_OPTIONS)}
              <div className="flex items-center gap-2">
                <Switch checked={search.draftSettings.adaptive_advanced_rewrites} onChange={(checked) => search.updateSetting("adaptive_advanced_rewrites", checked)} aria-label={t("sidepanel:rag.advancedRewrites", "Advanced rewrites")} />
                <span className="text-xs text-text">{t("sidepanel:rag.advancedRewrites", "Advanced rewrites")}</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={search.draftSettings.adaptive_rerun_on_low_confidence} onChange={(checked) => search.updateSetting("adaptive_rerun_on_low_confidence", checked)} aria-label={t("sidepanel:rag.rerunLowConfidence", "Rerun on low confidence")} />
                <span className="text-xs text-text">{t("sidepanel:rag.rerunLowConfidence", "Rerun on low confidence")}</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={search.draftSettings.adaptive_rerun_include_generation} onChange={(checked) => search.updateSetting("adaptive_rerun_include_generation", checked)} aria-label={t("sidepanel:rag.rerunIncludeGeneration", "Rerun include generation")} />
                <span className="text-xs text-text">{t("sidepanel:rag.rerunIncludeGeneration", "Rerun include generation")}</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={search.draftSettings.adaptive_rerun_bypass_cache} onChange={(checked) => search.updateSetting("adaptive_rerun_bypass_cache", checked)} aria-label={t("sidepanel:rag.rerunBypassCache", "Rerun bypass cache")} />
                <span className="text-xs text-text">{t("sidepanel:rag.rerunBypassCache", "Rerun bypass cache")}</span>
              </div>
              {filter.renderNumberInput(t("sidepanel:rag.rerunTimeBudget", "Rerun time budget"), search.draftSettings.adaptive_rerun_time_budget_sec, (next) => search.updateSetting("adaptive_rerun_time_budget_sec", next), { min: 1 })}
              {filter.renderNumberInput(t("sidepanel:rag.rerunDocBudget", "Rerun doc budget"), search.draftSettings.adaptive_rerun_doc_budget, (next) => search.updateSetting("adaptive_rerun_doc_budget", next), { min: 1 })}
            </>
          )}
        </div>
      )
    },
    matchesAny(
      t("sidepanel:rag.agentic", "Agentic strategy") as string,
      t("sidepanel:rag.agenticTopK", "Top K docs") as string
    ) && {
      key: "agentic",
      label: t("sidepanel:rag.agentic", "Agentic strategy"),
      children:
        search.draftSettings.strategy !== "agentic" ? (
          <div className="text-xs text-text-muted">
            {t("sidepanel:rag.agenticOnly", "Switch strategy to Agentic to configure these settings.")}
          </div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {filter.renderNumberInput(t("sidepanel:rag.agenticTopK", "Top K docs"), search.draftSettings.agentic_top_k_docs, (next) => search.updateSetting("agentic_top_k_docs", next), { min: 1 })}
            {filter.renderNumberInput(t("sidepanel:rag.agenticWindow", "Window chars"), search.draftSettings.agentic_window_chars, (next) => search.updateSetting("agentic_window_chars", next), { min: 1 })}
            {filter.renderNumberInput(t("sidepanel:rag.agenticMaxTokens", "Max tokens read"), search.draftSettings.agentic_max_tokens_read, (next) => search.updateSetting("agentic_max_tokens_read", next), { min: 1 })}
            {filter.renderNumberInput(t("sidepanel:rag.agenticToolCalls", "Max tool calls"), search.draftSettings.agentic_max_tool_calls, (next) => search.updateSetting("agentic_max_tool_calls", next), { min: 0 })}
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_extractive_only} onChange={(checked) => search.updateSetting("agentic_extractive_only", checked)} aria-label={t("sidepanel:rag.agenticExtractive", "Extractive only")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticExtractive", "Extractive only")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_quote_spans} onChange={(checked) => search.updateSetting("agentic_quote_spans", checked)} aria-label={t("sidepanel:rag.agenticQuoteSpans", "Quote spans")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticQuoteSpans", "Quote spans")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_debug_trace} onChange={(checked) => search.updateSetting("agentic_debug_trace", checked)} aria-label={t("sidepanel:rag.agenticDebug", "Debug trace")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticDebug", "Debug trace")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_tools} onChange={(checked) => search.updateSetting("agentic_enable_tools", checked)} aria-label={t("sidepanel:rag.agenticTools", "Enable tools")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticTools", "Enable tools")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_use_llm_planner} onChange={(checked) => search.updateSetting("agentic_use_llm_planner", checked)} aria-label={t("sidepanel:rag.agenticPlanner", "Use LLM planner")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticPlanner", "Use LLM planner")}</span>
            </div>
            {filter.renderNumberInput(t("sidepanel:rag.agenticTimeBudget", "Time budget"), search.draftSettings.agentic_time_budget_sec, (next) => search.updateSetting("agentic_time_budget_sec", next), { min: 1 })}
            {filter.renderNumberInput(t("sidepanel:rag.agenticCacheTtl", "Cache TTL"), search.draftSettings.agentic_cache_ttl_sec, (next) => search.updateSetting("agentic_cache_ttl_sec", next), { min: 1 })}
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_query_decomposition} onChange={(checked) => search.updateSetting("agentic_enable_query_decomposition", checked)} aria-label={t("sidepanel:rag.agenticDecomposition", "Query decomposition")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticDecomposition", "Query decomposition")}</span>
            </div>
            {filter.renderNumberInput(t("sidepanel:rag.agenticSubgoalMax", "Subgoal max"), search.draftSettings.agentic_subgoal_max, (next) => search.updateSetting("agentic_subgoal_max", next), { min: 1 })}
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_semantic_within} onChange={(checked) => search.updateSetting("agentic_enable_semantic_within", checked)} aria-label={t("sidepanel:rag.agenticSemanticWithin", "Semantic within")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticSemanticWithin", "Semantic within")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_section_index} onChange={(checked) => search.updateSetting("agentic_enable_section_index", checked)} aria-label={t("sidepanel:rag.agenticSectionIndex", "Section index")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticSectionIndex", "Section index")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_prefer_structural_anchors} onChange={(checked) => search.updateSetting("agentic_prefer_structural_anchors", checked)} aria-label={t("sidepanel:rag.agenticAnchors", "Prefer anchors")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticAnchors", "Prefer anchors")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_table_support} onChange={(checked) => search.updateSetting("agentic_enable_table_support", checked)} aria-label={t("sidepanel:rag.agenticTableSupport", "Table support")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticTableSupport", "Table support")}</span>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_vlm_late_chunking} onChange={(checked) => search.updateSetting("agentic_enable_vlm_late_chunking", checked)} aria-label={t("sidepanel:rag.agenticVlm", "Agentic VLM late chunking")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticVlm", "Agentic VLM late chunking")}</span>
            </div>
            {search.draftSettings.agentic_enable_vlm_late_chunking && (
              <>
                {filter.renderTextInput(t("sidepanel:rag.agenticVlmBackend", "VLM backend"), search.draftSettings.agentic_vlm_backend || "", (next) => search.updateSetting("agentic_vlm_backend", next || null))}
                <div className="flex items-center gap-2">
                  <Switch checked={search.draftSettings.agentic_vlm_detect_tables_only} onChange={(checked) => search.updateSetting("agentic_vlm_detect_tables_only", checked)} aria-label={t("sidepanel:rag.agenticVlmDetect", "Detect tables only")} />
                  <span className="text-xs text-text">{t("sidepanel:rag.agenticVlmDetect", "Detect tables only")}</span>
                </div>
                {filter.renderNumberInput(t("sidepanel:rag.agenticVlmPages", "Max pages"), search.draftSettings.agentic_vlm_max_pages, (next) => search.updateSetting("agentic_vlm_max_pages", next), { min: 1 })}
                {filter.renderNumberInput(t("sidepanel:rag.agenticVlmTopK", "Top K docs"), search.draftSettings.agentic_vlm_late_chunk_top_k_docs, (next) => search.updateSetting("agentic_vlm_late_chunk_top_k_docs", next), { min: 1 })}
              </>
            )}
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_use_provider_embeddings_within} onChange={(checked) => search.updateSetting("agentic_use_provider_embeddings_within", checked)} aria-label={t("sidepanel:rag.agenticProviderEmbeddings", "Use provider embeddings")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticProviderEmbeddings", "Use provider embeddings")}</span>
            </div>
            {search.draftSettings.agentic_use_provider_embeddings_within &&
              filter.renderTextInput(t("sidepanel:rag.agenticProviderModel", "Provider model id"), search.draftSettings.agentic_provider_embedding_model_id, (next) => search.updateSetting("agentic_provider_embedding_model_id", next))}
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_adaptive_budgets} onChange={(checked) => search.updateSetting("agentic_adaptive_budgets", checked)} aria-label={t("sidepanel:rag.agenticAdaptiveBudgets", "Adaptive budgets")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticAdaptiveBudgets", "Adaptive budgets")}</span>
            </div>
            {filter.renderNumberInput(t("sidepanel:rag.agenticCoverage", "Coverage target"), search.draftSettings.agentic_coverage_target, (next) => search.updateSetting("agentic_coverage_target", next), { min: 0, max: 1, step: 0.05 })}
            {filter.renderNumberInput(t("sidepanel:rag.agenticMinCorroborating", "Min corroborating docs"), search.draftSettings.agentic_min_corroborating_docs, (next) => search.updateSetting("agentic_min_corroborating_docs", next), { min: 1 })}
            {filter.renderNumberInput(t("sidepanel:rag.agenticMaxRedundancy", "Max redundancy"), search.draftSettings.agentic_max_redundancy, (next) => search.updateSetting("agentic_max_redundancy", next), { min: 0 })}
            <div className="flex items-center gap-2">
              <Switch checked={search.draftSettings.agentic_enable_metrics} onChange={(checked) => search.updateSetting("agentic_enable_metrics", checked)} aria-label={t("sidepanel:rag.agenticMetrics", "Enable metrics")} />
              <span className="text-xs text-text">{t("sidepanel:rag.agenticMetrics", "Enable metrics")}</span>
            </div>
          </div>
        )
    },
    matchesAny(t("sidepanel:rag.monitoring", "Monitoring & analytics") as string) && {
      key: "monitoring",
      label: t("sidepanel:rag.monitoring", "Monitoring & analytics"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_monitoring} onChange={(checked) => search.updateSetting("enable_monitoring", checked)} aria-label={t("sidepanel:rag.enableMonitoring", "Enable monitoring")} />
            <span className="text-xs text-text">{t("sidepanel:rag.enableMonitoring", "Enable monitoring")}</span>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_analytics} onChange={(checked) => search.updateSetting("enable_analytics", checked)} aria-label={t("sidepanel:rag.enableAnalytics", "Enable analytics")} />
            <span className="text-xs text-text">{t("sidepanel:rag.enableAnalytics", "Enable analytics")}</span>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_observability} onChange={(checked) => search.updateSetting("enable_observability", checked)} aria-label={t("sidepanel:rag.enableObservability", "Enable observability")} />
            <span className="text-xs text-text">{t("sidepanel:rag.enableObservability", "Enable observability")}</span>
          </div>
          {filter.renderTextInput(t("sidepanel:rag.traceId", "Trace ID"), search.draftSettings.trace_id, (next) => search.updateSetting("trace_id", next))}
        </div>
      )
    },
    matchesAny(t("sidepanel:rag.performance", "Performance") as string) && {
      key: "performance",
      label: t("sidepanel:rag.performance", "Performance"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.use_connection_pool} onChange={(checked) => search.updateSetting("use_connection_pool", checked)} aria-label={t("sidepanel:rag.connectionPool", "Use connection pool")} />
            <span className="text-xs text-text">{t("sidepanel:rag.connectionPool", "Use connection pool")}</span>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.use_embedding_cache} onChange={(checked) => search.updateSetting("use_embedding_cache", checked)} aria-label={t("sidepanel:rag.embeddingCache", "Use embedding cache")} />
            <span className="text-xs text-text">{t("sidepanel:rag.embeddingCache", "Use embedding cache")}</span>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_performance_analysis} onChange={(checked) => search.updateSetting("enable_performance_analysis", checked)} aria-label={t("sidepanel:rag.performanceAnalysis", "Performance analysis")} />
            <span className="text-xs text-text">{t("sidepanel:rag.performanceAnalysis", "Performance analysis")}</span>
          </div>
          {filter.renderNumberInput(t("sidepanel:rag.timeout", "Timeout (s)"), search.draftSettings.timeout_seconds, (next) => search.updateSetting("timeout_seconds", next), { min: 1 })}
        </div>
      )
    },
    matchesAny(t("sidepanel:rag.resilience", "Resilience") as string) && {
      key: "resilience",
      label: t("sidepanel:rag.resilience", "Resilience"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_resilience} onChange={(checked) => search.updateSetting("enable_resilience", checked)} aria-label={t("sidepanel:rag.enableResilience", "Enable resilience")} />
            <span className="text-xs text-text">{t("sidepanel:rag.enableResilience", "Enable resilience")}</span>
          </div>
          {search.draftSettings.enable_resilience && (
            <>
              {filter.renderNumberInput(t("sidepanel:rag.retryAttempts", "Retry attempts"), search.draftSettings.retry_attempts, (next) => search.updateSetting("retry_attempts", next), { min: 0 })}
              <div className="flex items-center gap-2">
                <Switch checked={search.draftSettings.circuit_breaker} onChange={(checked) => search.updateSetting("circuit_breaker", checked)} aria-label={t("sidepanel:rag.circuitBreaker", "Circuit breaker")} />
                <span className="text-xs text-text">{t("sidepanel:rag.circuitBreaker", "Circuit breaker")}</span>
              </div>
            </>
          )}
        </div>
      )
    },
    matchesAny(t("sidepanel:rag.batch", "Batch") as string) && {
      key: "batch",
      label: t("sidepanel:rag.batch", "Batch"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.enable_batch} onChange={(checked) => search.updateSetting("enable_batch", checked)} aria-label={t("sidepanel:rag.enableBatch", "Enable batch")} />
            <span className="text-xs text-text">{t("sidepanel:rag.enableBatch", "Enable batch")}</span>
          </div>
          {search.draftSettings.enable_batch && (
            <>
              {filter.renderNumberInput(t("sidepanel:rag.batchConcurrent", "Batch concurrent"), search.draftSettings.batch_concurrent, (next) => search.updateSetting("batch_concurrent", next), { min: 1 })}
              <div className="flex flex-col gap-1 md:col-span-2">
                <span className="text-xs text-text">{t("sidepanel:rag.batchQueries", "Batch queries")}</span>
                <Input.TextArea
                  value={search.draftSettings.batch_queries.join("\n")}
                  onChange={(e) => search.updateSetting("batch_queries", search.parseBatchQueries(e.target.value), { transient: true })}
                  rows={4}
                  aria-label={t("sidepanel:rag.batchQueries", "Batch queries")}
                />
              </div>
            </>
          )}
        </div>
      )
    },
    matchesAny(t("sidepanel:rag.feedback", "Feedback") as string) && {
      key: "feedback",
      label: t("sidepanel:rag.feedback", "Feedback"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.collect_feedback} onChange={(checked) => search.updateSetting("collect_feedback", checked)} aria-label={t("sidepanel:rag.collectFeedback", "Collect feedback")} />
            <span className="text-xs text-text">{t("sidepanel:rag.collectFeedback", "Collect feedback")}</span>
          </div>
          {filter.renderTextInput(t("sidepanel:rag.feedbackUserId", "Feedback user id"), search.draftSettings.feedback_user_id, (next) => search.updateSetting("feedback_user_id", next))}
          <div className="flex items-center gap-2">
            <Switch checked={search.draftSettings.apply_feedback_boost} onChange={(checked) => search.updateSetting("apply_feedback_boost", checked)} aria-label={t("sidepanel:rag.feedbackBoost", "Apply feedback boost")} />
            <span className="text-xs text-text">{t("sidepanel:rag.feedbackBoost", "Apply feedback boost")}</span>
          </div>
        </div>
      )
    },
    matchesAny(t("sidepanel:rag.userContext", "User context") as string) && {
      key: "user-context",
      label: t("sidepanel:rag.userContext", "User context"),
      children: (
        <div className="grid gap-3 md:grid-cols-2">
          {filter.renderTextInput(t("sidepanel:rag.userId", "User ID"), search.draftSettings.user_id || "", (next) => search.updateSetting("user_id", next || null))}
          {filter.renderTextInput(t("sidepanel:rag.sessionId", "Session ID"), search.draftSettings.session_id || "", (next) => search.updateSetting("session_id", next || null))}
        </div>
      )
    }
  ].filter(Boolean) as any[]

  const renderResultItem = (item: any) => {
    const snippet = getResultText(item).slice(0, 240)
    const title = getResultTitle(item)
    const scoreLabel = formatScore(getResultScore(item))
    const typeLabel = getResultType(item)
    const dateLabel = formatDate(getResultDate(item))
    const metaText = [typeLabel, dateLabel].filter(Boolean).join(" - ")
    return (
      <List.Item
        className={search.draftSettings.highlight_results ? "bg-surface2/40" : undefined}
        actions={[
          <button key="insert" type="button" onClick={() => resultsDisplay.handleInsert(item)} className="text-primary hover:text-primaryStrong">{t("sidepanel:rag.actions.insert", "Insert")}</button>,
          <button key="ask" type="button" onClick={() => resultsDisplay.handleAsk(item)} className="text-primary hover:text-primaryStrong">{t("sidepanel:rag.actions.ask", "Ask")}</button>,
          <button key="preview" type="button" onClick={() => resultsDisplay.setPreviewItem(toPinnedResult(item))} className="text-primary hover:text-primaryStrong">{t("sidepanel:rag.actions.preview", "Preview")}</button>,
          <button key="open" type="button" onClick={() => resultsDisplay.handleOpen(item)} className="text-primary hover:text-primaryStrong">{t("sidepanel:rag.actions.open", "Open")}</button>,
          <Dropdown key="copy" menu={resultsDisplay.copyMenu(item)}><button type="button" className="text-primary hover:text-primaryStrong">{t("sidepanel:rag.actions.copy", "Copy")}</button></Dropdown>,
          <button key="pin" type="button" onClick={() => resultsDisplay.handlePin(item)} className="text-primary hover:text-primaryStrong">{t("sidepanel:rag.actions.pin", "Pin")}</button>
        ]}
      >
        <List.Item.Meta
          title={
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-text">{title || t("sidepanel:rag.untitled", "Untitled")}</span>
              {scoreLabel && <span className="text-[10px] text-text-muted">{scoreLabel}</span>}
            </div>
          }
          description={
            <div className="space-y-1">
              {metaText && <div className="text-[10px] text-text-muted">{metaText}</div>}
              <div className="text-xs text-text-muted line-clamp-3">
                {search.draftSettings.highlight_query_terms ? highlightText(snippet, search.resolvedQuery) : snippet}
              </div>
            </div>
          }
        />
      </List.Item>
    )
  }

  return (
    <div className={wrapperClassName}>
      {showToggle && (
        <div className="flex items-center justify-between mb-1">
          <button
            type="button"
            aria-expanded={isOpen}
            aria-controls="rag-search-panel"
            className="text-caption text-text-muted underline"
            onClick={() => setOpenState(!isOpen)}
            title={isOpen ? t("sidepanel:rag.hide", "Hide Search & Context") : t("sidepanel:rag.show", "Show Search & Context")}
          >
            {isOpen ? t("sidepanel:rag.hide", "Hide Search & Context") : t("sidepanel:rag.show", "Show Search & Context")}
          </button>
        </div>
      )}
      {isOpen && (
        <div id="rag-search-panel" data-testid="rag-search-panel" className={panelClassName}>
          {!isConnected && (
            <div className="absolute inset-0 z-10 flex items-center justify-center rounded bg-surface2">
              <span className="text-sm text-text-muted">{t("sidepanel:rag.disconnected", "Connect to server to search knowledge base")}</span>
            </div>
          )}
          {!search.ragHintSeen && !search.hasAttemptedSearch && (
            <div className="mb-2 flex items-start gap-2 rounded border-l-2 border-primary bg-surface2 p-2">
              <div className="flex-1">
                <p className="text-xs text-text">{t("sidepanel:rag.hint.message", "Search your knowledge base and insert results into your message.")}</p>
              </div>
              <button type="button" onClick={() => search.setRagHintSeen(true)} className="rounded p-1 text-text-subtle hover:bg-surface" aria-label={t("sidepanel:rag.hint.dismiss", "Dismiss")} title={t("sidepanel:rag.hint.dismiss", "Dismiss")}>
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="text-xs font-semibold text-text">{t("sidepanel:rag.title", "Search & Context")}</span>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-text-muted">{t("sidepanel:rag.preset", "Preset")}</span>
              <Select size="small" value={search.preset} onChange={(value) => search.applyPresetSelection(value as any)} options={[{ label: "Fast", value: "fast" }, { label: "Balanced", value: "balanced" }, { label: "Thorough", value: "thorough" }, { label: "Custom", value: "custom" }]} aria-label={t("sidepanel:rag.preset", "Preset")} className="min-w-28" />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-text-muted">{t("sidepanel:rag.strategy", "Strategy")}</span>
              <Select size="small" value={search.draftSettings.strategy} onChange={(value) => search.updateSetting("strategy", value as any)} options={STRATEGY_OPTIONS} aria-label={t("sidepanel:rag.strategy", "Strategy")} className="min-w-28" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-text-muted">{t("sidepanel:rag.explainOnly", "Explain only")}</span>
              <Switch size="small" checked={search.draftSettings.explain_only} onChange={(checked) => search.updateSetting("explain_only", checked)} aria-label={t("sidepanel:rag.explainOnly", "Explain only")} />
            </div>
            <Button size="small" type="link" onClick={search.resetToBalanced}>
              {t("sidepanel:rag.reset", "Reset to Balanced")}
            </Button>
          </div>

          <div className="mb-2 flex items-center gap-2">
            <Checkbox checked={search.useCurrentMessage} onChange={(e) => search.setUseCurrentMessage(e.target.checked)}>
              {t("sidepanel:rag.useCurrentMessage", "Use current message")}
            </Checkbox>
          </div>

          <div className="mb-2 flex items-center gap-2">
            <Input
              ref={search.searchInputRef}
              placeholder={t("sidepanel:rag.searchPlaceholder", "Search query")}
              value={search.draftSettings.query}
              aria-label={t("sidepanel:rag.searchPlaceholder", "Search query")}
              onChange={(e) => search.updateSetting("query", e.target.value, { transient: true })}
              onPressEnter={() => search.runSearch()}
            />
            <Button onClick={() => search.runSearch()} type="default">
              {t("sidepanel:rag.search", "Search")}
            </Button>
          </div>
          {search.queryError && <div className="text-xs text-danger mb-2">{search.queryError}</div>}

          <div className="space-y-3">
            {matchesAny(t("sidepanel:rag.sourcesFilters", "Sources & Filters") as string, t("sidepanel:rag.sources", "Sources") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.sourcesFilters", "Sources & Filters")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  {filter.renderMultiSelect(t("sidepanel:rag.sources", "Sources"), search.draftSettings.sources, (next) => search.updateSetting("sources", next as any), sourceOptions)}
                  {filter.renderTextInput(t("sidepanel:rag.keywordFilter", "Keyword filter"), search.draftSettings.keyword_filter, (next) => search.updateSetting("keyword_filter", next), { placeholder: t("sidepanel:rag.keywordFilterPlaceholder", "Comma-separated keywords") as string })}
                  {filter.renderTextInput(t("sidepanel:rag.includeMediaIds", "Include media IDs"), stringifyIdList(search.draftSettings.include_media_ids), (next) => search.updateSetting("include_media_ids", parseNumericIdList(next)), { placeholder: "1, 2, 3" })}
                  {filter.renderTextInput(t("sidepanel:rag.includeNoteIds", "Include note IDs"), stringifyIdList(search.draftSettings.include_note_ids), (next) => search.updateSetting("include_note_ids", parseStringIdList(next)), { placeholder: "10, 11, 12" })}
                </div>
              </div>
            )}

            {matchesAny(t("sidepanel:rag.retrieval", "Retrieval") as string, t("sidepanel:rag.searchMode", "Retrieval mode") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.retrieval", "Retrieval")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  {filter.renderSelect(t("sidepanel:rag.searchMode", "Retrieval mode"), search.draftSettings.search_mode, (next) => search.updateSetting("search_mode", next as any), SEARCH_MODE_OPTIONS)}
                  {search.draftSettings.search_mode !== "vector" && filter.renderSelect(t("sidepanel:rag.ftsLevel", "FTS level"), search.draftSettings.fts_level, (next) => search.updateSetting("fts_level", next as any), FTS_LEVEL_OPTIONS)}
                  {search.draftSettings.search_mode === "hybrid" && filter.renderNumberInput(t("sidepanel:rag.hybridAlpha", "Hybrid alpha"), search.draftSettings.hybrid_alpha, (next) => search.updateSetting("hybrid_alpha", next), { min: 0, max: 1, step: 0.05 })}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_intent_routing} onChange={(checked) => search.updateSetting("enable_intent_routing", checked)} aria-label={t("sidepanel:rag.intentRouting", "Intent routing")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.intentRouting", "Intent routing")}</span>
                  </div>
                  {filter.renderNumberInput(t("sidepanel:rag.topK", "Results (top_k)"), search.draftSettings.top_k, (next) => search.updateSetting("top_k", next), { min: 1 })}
                  {filter.renderNumberInput(t("sidepanel:rag.minScore", "Minimum relevance"), search.draftSettings.min_score, (next) => search.updateSetting("min_score", next), { min: 0, max: 1, step: 0.05 })}
                </div>
              </div>
            )}

            {matchesAny(t("sidepanel:rag.reranking", "Reranking") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.reranking", "Reranking")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_reranking} onChange={(checked) => search.updateSetting("enable_reranking", checked)} aria-label={t("sidepanel:rag.enableReranking", "Enable reranking")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.enableReranking", "Enable reranking")}</span>
                  </div>
                  {search.draftSettings.enable_reranking && (
                    <>
                      {filter.renderSelect(t("sidepanel:rag.rerankStrategy", "Strategy"), search.draftSettings.reranking_strategy, (next) => search.updateSetting("reranking_strategy", next as any), RERANK_STRATEGY_OPTIONS)}
                      {filter.renderNumberInput(t("sidepanel:rag.rerankTopK", "Rerank top_k"), search.draftSettings.rerank_top_k, (next) => search.updateSetting("rerank_top_k", next), { min: 1 })}
                      {filter.renderTextInput(t("sidepanel:rag.rerankingModel", "Reranking model"), search.draftSettings.reranking_model, (next) => search.updateSetting("reranking_model", next))}
                      {filter.renderNumberInput(t("sidepanel:rag.rerankMinProb", "Min relevance prob"), search.draftSettings.rerank_min_relevance_prob, (next) => search.updateSetting("rerank_min_relevance_prob", next), { min: 0, max: 1, step: 0.05 })}
                      {filter.renderNumberInput(t("sidepanel:rag.rerankSentinel", "Sentinel margin"), search.draftSettings.rerank_sentinel_margin, (next) => search.updateSetting("rerank_sentinel_margin", next), { min: 0, max: 1, step: 0.05 })}
                    </>
                  )}
                </div>
              </div>
            )}

            {matchesAny(t("sidepanel:rag.answerCitations", "Answer & Citations") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.answerCitations", "Answer & Citations")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_generation} onChange={(checked) => search.updateSetting("enable_generation", checked)} aria-label={t("sidepanel:rag.enableGeneration", "Enable generation")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.enableGeneration", "Enable generation")}</span>
                  </div>
                  {search.draftSettings.enable_generation && (
                    <>
                      <div className="flex items-center gap-2">
                        <Switch checked={search.draftSettings.strict_extractive} onChange={(checked) => search.updateSetting("strict_extractive", checked)} aria-label={t("sidepanel:rag.strictExtractive", "Strict extractive")} />
                        <span className="text-xs text-text">{t("sidepanel:rag.strictExtractive", "Strict extractive")}</span>
                      </div>
                      {filter.renderTextInput(t("sidepanel:rag.generationModel", "Generation model"), search.draftSettings.generation_model || "", (next) => search.updateSetting("generation_model", next || null))}
                      {filter.renderTextInput(t("sidepanel:rag.generationPrompt", "Generation prompt"), search.draftSettings.generation_prompt || "", (next) => search.updateSetting("generation_prompt", next || null))}
                      {filter.renderNumberInput(t("sidepanel:rag.maxTokens", "Max tokens"), search.draftSettings.max_generation_tokens, (next) => search.updateSetting("max_generation_tokens", next), { min: 1 })}
                      <div className="flex items-center gap-2">
                        <Switch checked={search.draftSettings.enable_abstention} onChange={(checked) => search.updateSetting("enable_abstention", checked)} aria-label={t("sidepanel:rag.enableAbstention", "Enable abstention")} />
                        <span className="text-xs text-text">{t("sidepanel:rag.enableAbstention", "Enable abstention")}</span>
                      </div>
                      {search.draftSettings.enable_abstention && filter.renderSelect(t("sidepanel:rag.abstentionBehavior", "Abstention behavior"), search.draftSettings.abstention_behavior, (next) => search.updateSetting("abstention_behavior", next as any), ABSTENTION_OPTIONS)}
                      <div className="flex items-center gap-2">
                        <Switch checked={search.draftSettings.enable_multi_turn_synthesis} onChange={(checked) => search.updateSetting("enable_multi_turn_synthesis", checked)} aria-label={t("sidepanel:rag.enableSynthesis", "Multi-turn synthesis")} />
                        <span className="text-xs text-text">{t("sidepanel:rag.enableSynthesis", "Multi-turn synthesis")}</span>
                      </div>
                      {search.draftSettings.enable_multi_turn_synthesis && (
                        <>
                          {filter.renderNumberInput(t("sidepanel:rag.synthesisBudget", "Synthesis time budget"), search.draftSettings.synthesis_time_budget_sec, (next) => search.updateSetting("synthesis_time_budget_sec", next), { min: 1 })}
                          {filter.renderNumberInput(t("sidepanel:rag.synthesisDraft", "Draft tokens"), search.draftSettings.synthesis_draft_tokens, (next) => search.updateSetting("synthesis_draft_tokens", next), { min: 1 })}
                          {filter.renderNumberInput(t("sidepanel:rag.synthesisRefine", "Refine tokens"), search.draftSettings.synthesis_refine_tokens, (next) => search.updateSetting("synthesis_refine_tokens", next), { min: 1 })}
                        </>
                      )}
                    </>
                  )}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_citations} onChange={(checked) => search.updateSetting("enable_citations", checked)} aria-label={t("sidepanel:rag.enableCitations", "Enable citations")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.enableCitations", "Enable citations")}</span>
                  </div>
                  {search.draftSettings.enable_citations && (
                    <>
                      {filter.renderSelect(t("sidepanel:rag.citationStyle", "Citation style"), search.draftSettings.citation_style, (next) => search.updateSetting("citation_style", next as any), CITATION_STYLE_OPTIONS)}
                      <div className="flex items-center gap-2">
                        <Switch checked={search.draftSettings.include_page_numbers} onChange={(checked) => search.updateSetting("include_page_numbers", checked)} aria-label={t("sidepanel:rag.includePageNumbers", "Include page numbers")} />
                        <span className="text-xs text-text">{t("sidepanel:rag.includePageNumbers", "Include page numbers")}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch checked={search.draftSettings.enable_chunk_citations} onChange={(checked) => search.updateSetting("enable_chunk_citations", checked)} aria-label={t("sidepanel:rag.chunkCitations", "Chunk citations")} />
                        <span className="text-xs text-text">{t("sidepanel:rag.chunkCitations", "Chunk citations")}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch checked={search.draftSettings.require_hard_citations} onChange={(checked) => search.updateSetting("require_hard_citations", checked)} aria-label={t("sidepanel:rag.requireHardCitations", "Require hard citations")} />
                        <span className="text-xs text-text">{t("sidepanel:rag.requireHardCitations", "Require hard citations")}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {matchesAny(t("sidepanel:rag.safetyIntegrity", "Safety & Integrity") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.safetyIntegrity", "Safety & Integrity")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_security_filter} onChange={(checked) => search.updateSetting("enable_security_filter", checked)} aria-label={t("sidepanel:rag.securityFilter", "Security filter")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.securityFilter", "Security filter")}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.content_filter} onChange={(checked) => search.updateSetting("content_filter", checked)} aria-label={t("sidepanel:rag.contentFilter", "Content filter")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.contentFilter", "Content filter")}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.detect_pii} onChange={(checked) => search.updateSetting("detect_pii", checked)} aria-label={t("sidepanel:rag.detectPii", "PII detect")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.detectPii", "PII detect")}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.redact_pii} onChange={(checked) => search.updateSetting("redact_pii", checked)} aria-label={t("sidepanel:rag.redactPii", "PII redact")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.redactPii", "PII redact")}</span>
                  </div>
                  {filter.renderSelect(t("sidepanel:rag.sensitivity", "Sensitivity"), search.draftSettings.sensitivity_level, (next) => search.updateSetting("sensitivity_level", next as any), SENSITIVITY_OPTIONS)}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_injection_filter} onChange={(checked) => search.updateSetting("enable_injection_filter", checked)} aria-label={t("sidepanel:rag.injectionFilter", "Injection filter")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.injectionFilter", "Injection filter")}</span>
                  </div>
                  {search.draftSettings.enable_injection_filter && filter.renderNumberInput(t("sidepanel:rag.injectionStrength", "Injection strength"), search.draftSettings.injection_filter_strength, (next) => search.updateSetting("injection_filter_strength", next), { min: 0, max: 1, step: 0.05 })}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_numeric_fidelity} onChange={(checked) => search.updateSetting("enable_numeric_fidelity", checked)} aria-label={t("sidepanel:rag.numericFidelity", "Numeric fidelity")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.numericFidelity", "Numeric fidelity")}</span>
                  </div>
                  {search.draftSettings.enable_numeric_fidelity && filter.renderSelect(t("sidepanel:rag.numericFidelityBehavior", "Numeric fidelity behavior"), search.draftSettings.numeric_fidelity_behavior, (next) => search.updateSetting("numeric_fidelity_behavior", next as any), NUMERIC_FIDELITY_OPTIONS)}
                </div>
              </div>
            )}

            {matchesAny(t("sidepanel:rag.contextConstruction", "Context Construction") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.contextConstruction", "Context Construction")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  {filter.renderMultiSelect(t("sidepanel:rag.chunkTypeFilter", "Chunk types"), search.draftSettings.chunk_type_filter, (next) => search.updateSetting("chunk_type_filter", next as any), CHUNK_TYPE_OPTIONS)}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.enable_parent_expansion} onChange={(checked) => search.updateSetting("enable_parent_expansion", checked)} aria-label={t("sidepanel:rag.parentExpansion", "Parent expansion")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.parentExpansion", "Parent expansion")}</span>
                  </div>
                  {search.draftSettings.enable_parent_expansion && (
                    <>
                      {filter.renderNumberInput(t("sidepanel:rag.parentContextSize", "Parent context size"), search.draftSettings.parent_context_size, (next) => search.updateSetting("parent_context_size", next), { min: 1 })}
                      {filter.renderNumberInput(t("sidepanel:rag.parentMaxTokens", "Parent max tokens"), search.draftSettings.parent_max_tokens, (next) => search.updateSetting("parent_max_tokens", next), { min: 1 })}
                    </>
                  )}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.include_sibling_chunks} onChange={(checked) => search.updateSetting("include_sibling_chunks", checked)} aria-label={t("sidepanel:rag.includeSiblings", "Include siblings")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.includeSiblings", "Include siblings")}</span>
                  </div>
                  {search.draftSettings.include_sibling_chunks && filter.renderNumberInput(t("sidepanel:rag.siblingWindow", "Sibling window"), search.draftSettings.sibling_window, (next) => search.updateSetting("sibling_window", next), { min: 0 })}
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.include_parent_document} onChange={(checked) => search.updateSetting("include_parent_document", checked)} aria-label={t("sidepanel:rag.includeParentDoc", "Include parent document")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.includeParentDoc", "Include parent document")}</span>
                  </div>
                </div>
              </div>
            )}

            {matchesAny(t("sidepanel:rag.quickWins", "Quick Wins") as string) && (
              <div className="rounded border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-text mb-2">{t("sidepanel:rag.quickWins", "Quick Wins")}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.highlight_results} onChange={(checked) => search.updateSetting("highlight_results", checked)} aria-label={t("sidepanel:rag.highlightResults", "Highlight results")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.highlightResults", "Highlight results")}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.highlight_query_terms} onChange={(checked) => search.updateSetting("highlight_query_terms", checked)} aria-label={t("sidepanel:rag.highlightQuery", "Highlight query terms")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.highlightQuery", "Highlight query terms")}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.track_cost} onChange={(checked) => search.updateSetting("track_cost", checked)} aria-label={t("sidepanel:rag.trackCost", "Track cost")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.trackCost", "Track cost")}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={search.draftSettings.debug_mode} onChange={(checked) => search.updateSetting("debug_mode", checked)} aria-label={t("sidepanel:rag.debugMode", "Debug mode")} />
                    <span className="text-xs text-text">{t("sidepanel:rag.debugMode", "Debug mode")}</span>
                  </div>
                </div>
              </div>
            )}

            <div className="rounded border border-border bg-surface p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-text">{t("sidepanel:rag.advanced", "Advanced")}</span>
                <Button size="small" type="link" onClick={() => filter.setAdvancedOpen((prev) => !prev)}>
                  {filter.advancedOpen ? t("sidepanel:rag.hideAdvanced", "Hide") : t("sidepanel:rag.showAdvanced", "Show")}
                </Button>
              </div>
              {filter.advancedOpen && (
                <div className="space-y-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-[11px] text-text-muted">{t("sidepanel:rag.searchAllSettings", "Search all settings")}</span>
                    <Input placeholder={t("sidepanel:rag.searchSettings", "Search settings")} value={filter.advancedSearch} aria-label={t("sidepanel:rag.searchAllSettings", "Search all settings")} onChange={(e) => filter.setAdvancedSearch(e.target.value)} />
                  </div>
                  {advancedItems.length === 0 ? (
                    <div className="text-xs text-text-muted">{t("sidepanel:rag.advancedNoMatches", "No matching advanced settings.")}</div>
                  ) : (
                    <Collapse size="small" items={advancedItems} />
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="mt-3">
            {search.loading ? (
              <div className="py-4 text-center"><Spin size="small" /></div>
            ) : search.timedOut ? (
              <div className="text-xs text-text-muted">
                {t("sidepanel:rag.timeout.message", "Request timed out.")}
                <div className="mt-1 flex items-center gap-2">
                  <Button size="small" type="primary" onClick={() => search.runSearch()}>{t("sidepanel:rag.timeout.retry", "Retry")}</Button>
                  <Button size="small" onClick={() => search.updateSetting("timeout_seconds", search.draftSettings.timeout_seconds + 5)}>{t("sidepanel:rag.timeout.increase", "Increase timeout")}</Button>
                  <Button size="small" type="link" onClick={() => { try { const url = browser.runtime.getURL("/options.html#/settings/health"); browser.tabs.create({ url }) } catch { window.open("#/settings/health", "_blank") } }}>{t("sidepanel:rag.timeout.checkHealth", "Check server health")}</Button>
                </div>
              </div>
            ) : search.results.length === 0 && search.batchResults.length === 0 ? (
              <div className="text-xs text-text-subtle">{t("sidepanel:rag.noResults", "No results yet. Enter a query to search.")}</div>
            ) : (
              <>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-text">{t("sidepanel:rag.results", "Results")}</span>
                  <Select size="small" value={resultsDisplay.sortMode} onChange={(value) => resultsDisplay.setSortMode(value as any)} options={[{ label: "Relevance", value: "relevance" }, { label: "Date", value: "date" }, { label: "Type", value: "type" }]} aria-label={t("sidepanel:rag.sort", "Sort results")} />
                </div>
                {search.batchResults.length > 0 ? (
                  <div className="space-y-4">
                    {search.batchResults.map((group) => (
                      <div key={group.query}>
                        <div className="mb-2 text-xs font-semibold text-text">{group.query}</div>
                        <List size="small" dataSource={group.results} renderItem={renderResultItem} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <List size="small" dataSource={search.results} renderItem={renderResultItem} />
                )}
              </>
            )}
          </div>

          {showAttachedContext ? (
            <div className="mt-3">
              <div className="mb-2 text-xs font-semibold text-text">{t("sidepanel:rag.attachedContext", "Attached context")}</div>
              <div className="space-y-3">
                <div className="rounded border border-border bg-surface p-2">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-text">{t("playground:composer.contextTabsTitle", "Tabs in context")}</span>
                    <div className="flex items-center gap-2">
                      {onRefreshTabs && <Button size="small" type="link" onClick={onRefreshTabs}>{t("common:refresh", "Refresh")}</Button>}
                      {onClearTabs && attachedTabs.length > 0 && <Button size="small" type="link" onClick={onClearTabs}>{t("playground:composer.clearTabs", "Remove all")}</Button>}
                    </div>
                  </div>
                  {attachedTabs.length === 0 ? (
                    <div className="text-xs text-text-muted">{t("playground:composer.contextTabsEmpty", "No tabs selected yet.")}</div>
                  ) : (
                    <div className="space-y-2">
                      {attachedTabs.map((tab) => (
                        <div key={tab.id} className="flex items-center justify-between gap-2 rounded border border-border bg-surface2 px-2 py-1">
                          <div className="min-w-0">
                            <div className="truncate text-xs text-text">{tab.title || tab.url}</div>
                            <div className="truncate text-[10px] text-text-muted">{tab.url}</div>
                          </div>
                          {onRemoveTab && <Button size="small" type="link" onClick={() => onRemoveTab(tab.id)}>{t("common:remove", "Remove")}</Button>}
                        </div>
                      ))}
                    </div>
                  )}
                  {onAddTab && (
                    <div className="mt-3">
                      <div className="text-xs font-semibold text-text">{t("playground:composer.contextTabsAvailable", "Open tabs")}</div>
                      <div className="mt-2 max-h-40 space-y-2 overflow-y-auto">
                        {availableTabs.length > 0 ? availableTabs.map((tab) => (
                          <div key={tab.id} className="flex items-center justify-between gap-2 rounded border border-border bg-surface2 px-2 py-1">
                            <div className="min-w-0">
                              <div className="truncate text-xs text-text">{tab.title || tab.url}</div>
                              <div className="truncate text-[10px] text-text-muted">{tab.url}</div>
                            </div>
                            <Button size="small" type="link" onClick={() => onAddTab(tab)}>{t("common:add", "Add")}</Button>
                          </div>
                        )) : (
                          <div className="text-xs text-text-muted">{t("playground:composer.noTabsFound", "No eligible open tabs found.")}</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="rounded border border-border bg-surface p-2">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-text">{t("playground:composer.contextFilesTitle", "Files in context")}</span>
                    <div className="flex items-center gap-2">
                      {onAddFile && <Button size="small" type="link" onClick={onAddFile}>{t("playground:composer.addFile", "Add file")}</Button>}
                      {onClearFiles && attachedFiles.length > 0 && <Button size="small" type="link" onClick={onClearFiles}>{t("playground:composer.clearFiles", "Remove all")}</Button>}
                    </div>
                  </div>
                  {attachedFiles.length === 0 ? (
                    <div className="text-xs text-text-muted">{t("playground:composer.contextFilesEmpty", "No files attached yet.")}</div>
                  ) : (
                    <div className="space-y-2">
                      {attachedFiles.map((file) => (
                        <div key={file.id} className="flex items-center justify-between gap-2 rounded border border-border bg-surface2 px-2 py-1">
                          <div className="min-w-0">
                            <div className="truncate text-xs text-text">{file.filename}</div>
                            <div className="text-[10px] text-text-muted">{formatFileSize(file.size)}</div>
                          </div>
                          {onRemoveFile && <Button size="small" type="link" onClick={() => onRemoveFile(file.id)}>{t("common:remove", "Remove")}</Button>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="rounded border border-border bg-surface p-2">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-text">{t("sidepanel:rag.pinned", "Pinned results")}</span>
                    {(search.ragPinnedResults || []).length > 0 && <Button size="small" type="link" onClick={resultsDisplay.handleClearPins}>{t("sidepanel:rag.clearPins", "Clear all")}</Button>}
                  </div>
                  {(search.ragPinnedResults || []).length === 0 ? (
                    <div className="text-xs text-text-muted">{t("sidepanel:rag.pinsEmpty", "No pinned results yet.")}</div>
                  ) : (
                    <div className="space-y-2">
                      {(search.ragPinnedResults || []).map((item) => (
                        <div key={item.id} className="flex items-center justify-between gap-2 rounded border border-border bg-surface2 px-2 py-1">
                          <div className="min-w-0 truncate text-xs text-text">{item.title || item.source || item.url || "Untitled"}</div>
                          <Button size="small" type="link" onClick={() => resultsDisplay.handleUnpin(item.id)}>{t("sidepanel:rag.remove", "Remove")}</Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="mt-3">
              <div className="mb-2 text-xs font-semibold text-text">{t("sidepanel:rag.pinned", "Pinned results")}</div>
              {(search.ragPinnedResults || []).length === 0 ? (
                <div className="text-xs text-text-muted">{t("sidepanel:rag.pinsEmpty", "No pinned results yet.")}</div>
              ) : (
                <div className="space-y-2">
                  {(search.ragPinnedResults || []).map((item) => (
                    <div key={item.id} className="flex items-center justify-between rounded border border-border bg-surface p-2">
                      <div className="text-xs text-text">{item.title || item.source || item.url || "Untitled"}</div>
                      <Button size="small" type="link" onClick={() => resultsDisplay.handleUnpin(item.id)}>{t("sidepanel:rag.remove", "Remove")}</Button>
                    </div>
                  ))}
                  <Button size="small" onClick={resultsDisplay.handleClearPins}>{t("sidepanel:rag.clearPins", "Clear all")}</Button>
                </div>
              )}
            </div>
          )}

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button type="primary" onClick={() => { search.applySettings(); setOpenState(false) }}>{t("sidepanel:rag.apply", "Apply")}</Button>
            <Button type="default" onClick={() => search.runSearch({ applyFirst: true })}>{t("sidepanel:rag.applySearch", "Apply & Search")}</Button>
            <Button type="text" onClick={() => { search.setDraftSettings(search.normalizeSettings(search.storedSettings)); setOpenState(false) }}>{t("common:cancel", "Cancel")}</Button>
          </div>
        </div>
      )}

      <Modal
        open={!!resultsDisplay.previewItem}
        onCancel={() => resultsDisplay.setPreviewItem(null)}
        footer={null}
        title={resultsDisplay.previewItem?.title || t("sidepanel:rag.preview", "Preview")}
      >
        {resultsDisplay.previewItem && (
          <div className="space-y-3">
            <div className="text-xs text-text-muted">{resultsDisplay.previewItem.source || resultsDisplay.previewItem.url}</div>
            <div className="text-sm text-text whitespace-pre-wrap">{resultsDisplay.previewItem.snippet}</div>
            <div className="flex items-center gap-2">
              <Button size="small" onClick={() => { void (async () => { const resolvedPinned = await withFullMediaTextIfAvailable(resultsDisplay.previewItem!); onInsert(formatRagResult(resolvedPinned, "markdown")) })() }}>{t("sidepanel:rag.actions.insert", "Insert")}</Button>
              <Button size="small" onClick={() => resultsDisplay.handleAsk({ content: resultsDisplay.previewItem!.snippet, metadata: { url: resultsDisplay.previewItem!.url, source: resultsDisplay.previewItem!.source, title: resultsDisplay.previewItem!.title } })}>{t("sidepanel:rag.actions.ask", "Ask")}</Button>
              <Dropdown menu={resultsDisplay.copyMenu({ content: resultsDisplay.previewItem.snippet, metadata: { url: resultsDisplay.previewItem.url, source: resultsDisplay.previewItem.source, title: resultsDisplay.previewItem.title } })}>
                <Button size="small">{t("sidepanel:rag.actions.copy", "Copy")}</Button>
              </Dropdown>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default RagSearchBar
