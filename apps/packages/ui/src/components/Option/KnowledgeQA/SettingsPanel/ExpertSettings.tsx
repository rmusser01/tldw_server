/**
 * ExpertSettings - Full 150+ RAG options organized into sections
 */

import React, { useMemo, useState } from "react"
import {
  ChevronDown,
  ChevronRight,
  Search,
  Sparkles,
  Database,
  Shield,
  Zap,
  FileText,
  Brain,
  CheckCircle2,
  Quote,
  Gauge,
  SlidersHorizontal,
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useKnowledgeQA } from "../KnowledgeQAProvider"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { cn } from "@/libs/utils"
import type { RagSettings, RagTextChunkMethod } from "@/services/rag/unified-rag"
import { getRagSourceOptions } from "@/services/rag/sourceMetadata"

// Section configuration
type SectionConfig = {
  id: string
  title: string
  icon: React.ElementType
  description: string
  defaultOpen?: boolean
}

const SECTIONS: SectionConfig[] = [
  {
    id: "search",
    title: "Search",
    icon: Search,
    description: "Search mode, top-k, hybrid settings",
    defaultOpen: true,
  },
  {
    id: "query",
    title: "Query Enhancement",
    icon: Sparkles,
    description: "Expansion, spell check, intent routing",
  },
  {
    id: "retrieval",
    title: "Advanced Retrieval",
    icon: Database,
    description: "PRF, HyDE, multi-vector passages",
  },
  {
    id: "chunking",
    title: "Document Context",
    icon: FileText,
    description: "Parent expansion, siblings, chunk types",
  },
  {
    id: "agentic",
    title: "Agentic RAG",
    icon: Brain,
    description: "Query decomposition, tools, planning",
  },
  {
    id: "reranking",
    title: "Reranking",
    icon: Gauge,
    description: "Strategy, model, thresholds",
  },
  {
    id: "generation",
    title: "Answer Generation",
    icon: Sparkles,
    description: "Model, tokens, abstention",
  },
  {
    id: "citations",
    title: "Citations",
    icon: Quote,
    description: "Style, page numbers, chunk-level",
  },
  {
    id: "verification",
    title: "Verification",
    icon: CheckCircle2,
    description: "Claims, post-verification, fidelity",
  },
  {
    id: "security",
    title: "Security",
    icon: Shield,
    description: "PII detection, content policy, sanitization",
  },
  {
    id: "performance",
    title: "Performance",
    icon: Zap,
    description: "Timeout, caching, resilience",
  },
  {
    id: "all_options",
    title: "All Options",
    icon: SlidersHorizontal,
    description: "Complete key-level access for every RAG option",
  },
]

type RagKey = keyof RagSettings

const NULLABLE_STRING_KEYS = new Set<RagKey>([
  "generation_model",
  "generation_prompt",
  "user_id",
  "session_id",
  "vlm_backend",
  "agentic_vlm_backend",
  "grading_model",
  "grading_provider",
  "fast_hallucination_provider",
  "fast_hallucination_model",
  "utility_grading_provider",
  "utility_grading_model",
])

const NULLABLE_NUMBER_KEYS = new Set<RagKey>([
  "accumulation_time_budget_sec",
  "subquery_time_budget_sec",
  "subquery_doc_budget",
])

const SETTING_ENUM_OPTIONS: Partial<Record<RagKey, string[]>> = {
  strategy: ["standard", "agentic"],
  search_mode: ["fts", "vector", "hybrid"],
  fts_level: ["media", "chunk"],
  table_method: ["markdown", "html", "hybrid"],
  claim_extractor: ["aps", "claimify", "ner", "auto"],
  claim_verifier: ["nli", "llm", "hybrid"],
  reranking_strategy: [
    "flashrank",
    "cross_encoder",
    "hybrid",
    "llama_cpp",
    "llm_scoring",
    "two_tier",
    "none",
  ],
  citation_style: ["apa", "mla", "chicago", "harvard", "ieee"],
  abstention_behavior: ["continue", "ask", "decline"],
  content_policy_mode: ["redact", "drop", "annotate"],
  low_confidence_behavior: ["continue", "ask", "decline"],
  numeric_fidelity_behavior: ["continue", "ask", "decline", "retry"],
  numeric_precision_mode: ["standard", "strict", "academic"],
  web_fallback_merge_strategy: ["prepend", "append", "interleave"],
  sensitivity_level: ["public", "internal", "confidential", "restricted"],
}

const AUTO_OPTION_EXCLUDED_KEYS = new Set<RagKey>(["query"])

const AUTO_OPTION_KEYS = (settings: RagSettings): RagKey[] =>
  (Object.keys(settings) as RagKey[])
    .filter((key) => !AUTO_OPTION_EXCLUDED_KEYS.has(key))
    .sort((a, b) => a.localeCompare(b))

export function ExpertSettings() {
  const { t } = useTranslation(["sidepanel"])
  const { settings, updateSetting } = useKnowledgeQA()
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(SECTIONS.filter((s) => s.defaultOpen).map((s) => s.id))
  )
  const sourceOptions = useMemo(
    () => getRagSourceOptions((key, fallback) => t(key, fallback)),
    [t]
  )

  const toggleSection = (id: string) => {
    const newSet = new Set(openSections)
    if (newSet.has(id)) {
      newSet.delete(id)
    } else {
      newSet.add(id)
    }
    setOpenSections(newSet)
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-text-muted pb-1">
        Advanced settings for fine-tuning search behavior. Expand a section to adjust its options.
      </p>
      {SECTIONS.map((section) => (
        <SettingsSection
          key={section.id}
          config={section}
          isOpen={openSections.has(section.id)}
          onToggle={() => toggleSection(section.id)}
          settings={settings}
          updateSetting={updateSetting}
        />
      ))}
    </div>
  )
}

type SettingsSectionProps = {
  config: SectionConfig
  isOpen: boolean
  onToggle: () => void
  settings: RagSettings
  updateSetting: <K extends keyof RagSettings>(key: K, value: RagSettings[K]) => void
}

function SettingsSection({
  config,
  isOpen,
  onToggle,
  settings,
  updateSetting,
}: SettingsSectionProps) {
  const Icon = config.icon
  const contentId = `section-content-${config.id}`

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isOpen}
        aria-controls={contentId}
        className="flex items-center gap-3 w-full px-4 py-3 hover:bg-muted/50 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-text-muted" />
        ) : (
          <ChevronRight className="w-4 h-4 text-text-muted" />
        )}
        <Icon className="w-4 h-4 text-primary" />
        <div className="flex-1 text-left">
          <div className="font-medium text-sm">{config.title}</div>
          <div className="text-xs text-text-muted">{config.description}</div>
        </div>
      </button>

      {/* Content */}
      {isOpen && (
        <div id={contentId} className="px-4 pb-4 pt-2 border-t border-border space-y-4">
          <SectionContent
            sectionId={config.id}
            settings={settings}
            updateSetting={updateSetting}
          />
        </div>
      )}
    </div>
  )
}

type SectionSettingsProps = {
  settings: RagSettings
  updateSetting: <K extends keyof RagSettings>(key: K, value: RagSettings[K]) => void
}

type SectionContentProps = SectionSettingsProps & {
  sectionId: string
}

function SectionContent({ sectionId, settings, updateSetting }: SectionContentProps) {
  switch (sectionId) {
    case "search":
      return <SearchSection settings={settings} updateSetting={updateSetting} />
    case "query":
      return <QuerySection settings={settings} updateSetting={updateSetting} />
    case "retrieval":
      return <RetrievalSection settings={settings} updateSetting={updateSetting} />
    case "chunking":
      return <ChunkingSection settings={settings} updateSetting={updateSetting} />
    case "agentic":
      return <AgenticSection settings={settings} updateSetting={updateSetting} />
    case "reranking":
      return <RerankingSection settings={settings} updateSetting={updateSetting} />
    case "generation":
      return <GenerationSection settings={settings} updateSetting={updateSetting} />
    case "citations":
      return <CitationsSection settings={settings} updateSetting={updateSetting} />
    case "verification":
      return <VerificationSection settings={settings} updateSetting={updateSetting} />
    case "security":
      return <SecuritySection settings={settings} updateSetting={updateSetting} />
    case "performance":
      return <PerformanceSection settings={settings} updateSetting={updateSetting} />
    case "all_options":
      return <AllOptionsSection settings={settings} updateSetting={updateSetting} />
    default:
      return <div className="text-sm text-text-muted">Section not found</div>
  }
}

// Reusable form components
function SettingToggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description?: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  // Generate a unique ID for accessibility
  const labelId = `toggle-${label.toLowerCase().replace(/\s+/g, '-')}`

  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        <div id={labelId} className="text-sm font-medium">{label}</div>
        {description && <div className="text-xs text-text-muted">{description}</div>}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        aria-labelledby={labelId}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-5 w-9 items-center rounded-full transition-colors flex-shrink-0",
          checked ? "bg-primary" : "bg-muted"
        )}
      >
        <span
          className={cn(
            "inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform",
            checked ? "translate-x-5" : "translate-x-1"
          )}
        />
      </button>
    </div>
  )
}

function SettingSlider({
  label,
  description,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string
  description?: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">{label}</div>
        <div className="text-sm text-text-muted">{value}</div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-primary"
      />
      {description && <div className="text-xs text-text-muted">{description}</div>}
    </div>
  )
}

function SettingSelect({
  label,
  description,
  value,
  onChange,
  options,
}: {
  label: string
  description?: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-sm font-medium">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-1.5 text-sm rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {description && <div className="text-xs text-text-muted">{description}</div>}
    </div>
  )
}

function SettingInput({
  label,
  description,
  value,
  onChange,
  type = "text",
  min,
  max,
  step,
  placeholder,
}: {
  label: string
  description?: string
  value: string | number
  onChange: (value: string | number) => void
  type?: "text" | "number"
  min?: number
  max?: number
  step?: number
  placeholder?: string
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-sm font-medium">{label}</div>
      <input
        type={type}
        value={value}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        onChange={(e) =>
          onChange(type === "number" ? Number(e.target.value) : e.target.value)
        }
        className="w-full px-3 py-1.5 text-sm rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
      />
      {description && <div className="text-xs text-text-muted">{description}</div>}
    </div>
  )
}

const ARRAY_VALUE_KIND_BY_KEY: Partial<Record<RagKey, "string" | "number">> = {
  include_media_ids: "number",
  include_note_ids: "string",
  expansion_strategies: "string",
  chunk_type_filter: "string",
  content_policy_types: "string",
  html_allowed_tags: "string",
  html_allowed_attrs: "string",
  batch_queries: "string",
  ground_truth_doc_ids: "string",
}

function formatArrayDraft(values: unknown[]): string {
  return values.map((entry) => String(entry)).join(", ")
}

function parseArrayDraft(
  key: RagKey,
  draft: string
): { parsed: unknown[]; error?: string } {
  const normalized = draft.trim()
  if (!normalized) {
    return { parsed: [] }
  }
  const chunks = normalized
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)

  const kind = ARRAY_VALUE_KIND_BY_KEY[key]
  if (kind === "number") {
    const numbers = chunks.map((part) => Number(part))
    if (numbers.some((value) => !Number.isFinite(value))) {
      return { parsed: [], error: "Use comma-separated numbers only." }
    }
    return { parsed: numbers }
  }

  return { parsed: chunks }
}

function AutoArrayInput({
  fieldKey,
  values,
  onChange,
}: {
  fieldKey: RagKey
  values: unknown[]
  onChange: (next: unknown[]) => void
}) {
  const [draft, setDraft] = useState(() => formatArrayDraft(values))
  const [error, setError] = useState<string | null>(null)

  React.useEffect(() => {
    setDraft(formatArrayDraft(values))
  }, [values])

  const commitDraft = () => {
    const parsed = parseArrayDraft(fieldKey, draft)
    if (parsed.error) {
      setError(parsed.error)
      return
    }
    setError(null)
    onChange(parsed.parsed)
  }

  return (
    <div className="space-y-1">
      <input
        type="text"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commitDraft}
        className={cn(
          "w-full px-2.5 py-1.5 text-sm rounded-md border bg-surface focus:outline-none focus:ring-2 focus:ring-primary",
          error ? "border-danger" : "border-border"
        )}
      />
      <p className="text-[11px] text-text-muted">Comma-separated values</p>
      {error ? <p className="text-[11px] text-danger">{error}</p> : null}
    </div>
  )
}

function AutoOptionRow({
  fieldKey,
  settings,
  updateSetting,
}: {
  fieldKey: RagKey
  settings: RagSettings
  updateSetting: <K extends keyof RagSettings>(key: K, value: RagSettings[K]) => void
}) {
  const value = settings[fieldKey]
  const setValue = (next: unknown) =>
    updateSetting(fieldKey, next as RagSettings[typeof fieldKey])
  const enumValues = SETTING_ENUM_OPTIONS[fieldKey]

  let control: React.ReactNode = (
    <div className="text-xs text-text-muted">Unsupported value type</div>
  )

  if (typeof value === "boolean") {
    const labelId = `auto-setting-${fieldKey}`
    control = (
      <button
        role="switch"
        aria-checked={value}
        aria-labelledby={labelId}
        onClick={() => setValue(!value)}
        className={cn(
          "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
          value ? "bg-primary" : "bg-muted"
        )}
      >
        <span
          className={cn(
            "inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform",
            value ? "translate-x-5" : "translate-x-1"
          )}
        />
      </button>
    )
    return (
      <div className="p-3 flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div id={labelId} className="text-sm font-medium font-mono break-all">
            {fieldKey}
          </div>
          <div className="text-[11px] text-text-muted">boolean</div>
        </div>
        {control}
      </div>
    )
  }

  if (typeof value === "number" || (value === null && NULLABLE_NUMBER_KEYS.has(fieldKey))) {
    const step = typeof value === "number" && Number.isInteger(value) ? 1 : 0.01
    control = (
      <input
        type="number"
        value={typeof value === "number" && Number.isFinite(value) ? value : ""}
        placeholder={NULLABLE_NUMBER_KEYS.has(fieldKey) ? "null" : undefined}
        step={step}
        onChange={(event) => {
          const raw = event.target.value
          if (NULLABLE_NUMBER_KEYS.has(fieldKey) && raw.trim() === "") {
            setValue(null)
            return
          }
          const next = Number(raw)
          if (Number.isFinite(next)) {
            setValue(next)
          }
        }}
        className="w-44 px-2.5 py-1.5 text-sm rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
      />
    )
  } else if (Array.isArray(value)) {
    if (fieldKey === "sources") {
      const selectedSources = value.filter(
        (entry): entry is string => typeof entry === "string"
      )
      control = (
        <div className="w-60 space-y-2 rounded-md border border-border bg-surface p-2">
          {sourceOptions.map((source) => (
            <label key={source.value} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={selectedSources.includes(source.value)}
                onChange={(event) => {
                  const nextSources = event.target.checked
                    ? [...selectedSources, source.value]
                    : selectedSources.filter((entry) => entry !== source.value)
                  setValue(nextSources)
                }}
                className="rounded"
              />
              <span>{source.label}</span>
            </label>
          ))}
        </div>
      )
    } else {
      control = (
        <div className="w-60">
          <AutoArrayInput
            fieldKey={fieldKey}
            values={value}
            onChange={(next) => setValue(next)}
          />
        </div>
      )
    }
  } else if (typeof value === "string" || value === null) {
    if (enumValues && enumValues.length > 0) {
      control = (
        <select
          value={value ?? ""}
          onChange={(event) => setValue(event.target.value)}
          className="w-60 px-2.5 py-1.5 text-sm rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
        >
          {enumValues.map((enumValue) => (
            <option key={enumValue} value={enumValue}>
              {enumValue}
            </option>
          ))}
        </select>
      )
    } else {
      const acceptsNull = NULLABLE_STRING_KEYS.has(fieldKey)
      control = (
        <input
          type="text"
          value={value ?? ""}
          placeholder={acceptsNull ? "null" : ""}
          onChange={(event) => {
            const next = event.target.value
            setValue(acceptsNull && next.trim() === "" ? null : next)
          }}
          className="w-60 px-2.5 py-1.5 text-sm rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
        />
      )
    }
  }

  const valueType = Array.isArray(value)
    ? "array"
    : value === null
      ? "null|string"
      : typeof value

  return (
    <div className="p-3 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <div className="text-sm font-medium font-mono break-all">{fieldKey}</div>
        <div className="text-[11px] text-text-muted">{valueType}</div>
      </div>
      {control}
    </div>
  )
}

function AllOptionsSection({ settings, updateSetting }: SectionSettingsProps) {
  const [keyFilter, setKeyFilter] = useState("")
  const normalizedFilter = keyFilter.trim().toLowerCase()
  const allKeys = useMemo(() => AUTO_OPTION_KEYS(settings), [settings])
  const filteredKeys = useMemo(
    () =>
      normalizedFilter
        ? allKeys.filter((key) => key.toLowerCase().includes(normalizedFilter))
        : allKeys,
    [allKeys, normalizedFilter]
  )

  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <label htmlFor="all-options-filter" className="text-xs font-medium text-text-muted">
          Filter option keys ({filteredKeys.length}/{allKeys.length})
        </label>
        <input
          id="all-options-filter"
          type="text"
          value={keyFilter}
          onChange={(event) => setKeyFilter(event.target.value)}
          placeholder="Type part of an option key (e.g. adaptive_, table_, agentic_)"
          className="w-full px-3 py-2 text-sm rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>
      <div className="border border-border rounded-md divide-y divide-border max-h-[30rem] overflow-y-auto">
        {filteredKeys.length === 0 ? (
          <div className="p-4 text-sm text-text-muted">No option keys match this filter.</div>
        ) : (
          filteredKeys.map((fieldKey) => (
            <AutoOptionRow
              key={fieldKey}
              fieldKey={fieldKey}
              settings={settings}
              updateSetting={updateSetting}
            />
          ))
        )}
      </div>
    </div>
  )
}

// Section implementations
function SearchSection({ settings, updateSetting }: SectionSettingsProps) {
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const showTextLateChunking =
    (settings.search_mode === "fts" || settings.search_mode === "hybrid") &&
    settings.fts_level === "chunk"
  const showTextLateChunkingKnobs =
    showTextLateChunking && settings.enable_text_late_chunking
  const textLateChunkMethodOptions: { value: RagTextChunkMethod; label: string }[] = [
    { value: "sentences", label: "Sentences" },
    { value: "words", label: "Words" },
    { value: "paragraphs", label: "Paragraphs" },
    { value: "tokens", label: "Tokens" },
    { value: "semantic", label: "Semantic" },
    { value: "propositions", label: "Propositions" },
    { value: "ebook_chapters", label: "Ebook Chapters" },
    { value: "json", label: "JSON" },
  ]

  return (
    <div className="space-y-4">
      <SettingSelect
        label="Search Mode"
        description="FTS: keyword matching. Vector: semantic similarity. Hybrid: combines both for best results."
        value={settings.search_mode}
        onChange={(v) => updateSetting("search_mode", v as typeof settings.search_mode)}
        options={[
          { value: "hybrid", label: "Hybrid (FTS + Vector)" },
          { value: "vector", label: "Vector Only" },
          { value: "fts", label: "Full-Text Only" },
        ]}
      />
      <SettingSelect
        label="FTS Level"
        description="Media-level searches full documents. Chunk-level searches individual text segments for finer matches."
        value={settings.fts_level}
        onChange={(v) => updateSetting("fts_level", v as typeof settings.fts_level)}
        options={[
          { value: "media", label: "Media-level" },
          { value: "chunk", label: "Chunk-level" },
        ]}
      />
      {showTextLateChunking && (
        <SettingToggle
          label="Text late chunking"
          description="For this query, bypass stored text chunks and rechunk matched media in memory."
          checked={settings.enable_text_late_chunking}
          onChange={(v) => updateSetting("enable_text_late_chunking", v)}
        />
      )}
      {showTextLateChunkingKnobs && (
        <div className="pl-4 border-l-2 border-primary/20 space-y-3">
          <SettingSelect
            label="Late Chunk Method"
            description="Chunk matched media in memory with this method for the current query."
            value={settings.chunk_method}
            onChange={(v) => updateSetting("chunk_method", v as typeof settings.chunk_method)}
            options={textLateChunkMethodOptions}
          />
          <SettingInput
            type="number"
            label="Late Chunk Size"
            description="Maximum unit count per transient chunk."
            value={settings.chunk_size}
            min={1}
            step={1}
            onChange={(v) => updateSetting("chunk_size", Math.max(1, Math.round(Number(v))))}
          />
          <SettingInput
            type="number"
            label="Late Chunk Overlap"
            description="Overlap between transient chunks. Must stay below chunk size."
            value={settings.chunk_overlap}
            min={0}
            max={Math.max(0, settings.chunk_size - 1)}
            step={1}
            onChange={(v) =>
              updateSetting(
                "chunk_overlap",
                Math.max(
                  0,
                  Math.min(Math.round(Number(v)), Math.max(0, settings.chunk_size - 1))
                )
              )
            }
          />
          <SettingInput
            label="Late Chunk Language"
            description="Optional language override such as en or es."
            value={settings.chunk_language}
            placeholder="Auto"
            onChange={(v) => updateSetting("chunk_language", String(v))}
          />
        </div>
      )}
      <SettingSlider
        label="Hybrid Alpha"
        description="Balance between search modes. 0 = pure keyword (FTS), 1 = pure semantic (Vector). Start at 0.5 for balanced results."
        value={settings.hybrid_alpha}
        onChange={(v) => updateSetting("hybrid_alpha", v)}
        min={0}
        max={1}
        step={0.1}
      />
      <SettingSlider
        label="Top-K"
        description="Number of documents to retrieve. Higher values provide more context but slower responses. 5-10 is a good default."
        value={settings.top_k}
        onChange={(v) => updateSetting("top_k", v)}
        min={1}
        max={50}
      />
      <SettingSlider
        label="Min Score"
        description="Minimum relevance threshold. Documents below this score are excluded. Lower values include more results; 0.3-0.5 is typical."
        value={settings.min_score}
        onChange={(v) => updateSetting("min_score", v)}
        min={0}
        max={1}
        step={0.05}
      />
      <SettingToggle
        label="Web Search Fallback"
        description="Use web results when local relevance is low (requires server provider configuration)"
        checked={settings.enable_web_fallback}
        onChange={(v) => updateSetting("enable_web_fallback", v)}
      />
      {settings.enable_web_fallback && (
        <div className="space-y-3">
          <SettingSlider
            label="Web Fallback Threshold"
            description="Trigger web search when relevance drops below this"
            value={settings.web_fallback_threshold}
            onChange={(v) => updateSetting("web_fallback_threshold", v)}
            min={0}
            max={1}
            step={0.05}
          />
          <SettingSlider
            label="Web Results"
            description="Number of web results to fetch"
            value={settings.web_fallback_result_count}
            onChange={(v) => updateSetting("web_fallback_result_count", v)}
            min={1}
            max={20}
          />
          <SettingSelect
            label="Web Search Engine"
            description="Provider used for web fallback"
            value={settings.web_search_engine}
            onChange={(v) => updateSetting("web_search_engine", v)}
            options={[
              { value: "duckduckgo", label: "DuckDuckGo" },
              { value: "brave", label: "Brave" },
              { value: "bing", label: "Bing" },
              { value: "google", label: "Google" },
              { value: "tavily", label: "Tavily" },
              { value: "serper", label: "Serper" },
            ]}
          />
          <SettingSelect
            label="Web Merge Strategy"
            description="How web results combine with local docs"
            value={settings.web_fallback_merge_strategy}
            onChange={(v) => updateSetting("web_fallback_merge_strategy", v as typeof settings.web_fallback_merge_strategy)}
            options={[
              { value: "prepend", label: "Prepend" },
              { value: "append", label: "Append" },
              { value: "interleave", label: "Interleave" },
            ]}
          />
        </div>
      )}
      {settings.enable_web_fallback &&
        !capsLoading &&
        capabilities &&
        !capabilities.hasWebSearch && (
          <div className="text-xs text-warn">
            Web search isn’t configured on this server, so fallback may not return results.
          </div>
        )}
      <SettingToggle
        label="Intent Routing"
        description="Analyze query intent to adjust retrieval"
        checked={settings.enable_intent_routing}
        onChange={(v) => updateSetting("enable_intent_routing", v)}
      />
    </div>
  )
}

function QuerySection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Query Expansion"
        description="Expand query with synonyms and related terms"
        checked={settings.expand_query}
        onChange={(v) => updateSetting("expand_query", v)}
      />
      {settings.expand_query && (
        <div className="pl-4 border-l-2 border-primary/20 space-y-2">
          <div className="text-xs font-medium text-text-muted">Expansion Strategies</div>
          {["acronym", "synonym", "semantic", "domain", "entity"].map((strategy) => (
            <label key={strategy} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={settings.expansion_strategies.includes(strategy as typeof settings.expansion_strategies[number])}
                onChange={(e) => {
                  const newStrategies = e.target.checked
                    ? [...settings.expansion_strategies, strategy as typeof settings.expansion_strategies[number]]
                    : settings.expansion_strategies.filter((s) => s !== strategy)
                  updateSetting("expansion_strategies", newStrategies)
                }}
                className="rounded"
              />
              <span className="capitalize">{strategy}</span>
            </label>
          ))}
        </div>
      )}
      <SettingToggle
        label="Spell Check"
        description="Correct spelling errors in query"
        checked={settings.spell_check}
        onChange={(v) => updateSetting("spell_check", v)}
      />
    </div>
  )
}

function RetrievalSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Multi-Vector Passages"
        description="ColBERT-style fine-grained matching. Splits documents into overlapping spans for more precise retrieval at higher compute cost."
        checked={settings.enable_multi_vector_passages}
        onChange={(v) => updateSetting("enable_multi_vector_passages", v)}
      />
      {settings.enable_multi_vector_passages && (
        <div className="pl-4 border-l-2 border-primary/20 space-y-3">
          <SettingSlider
            label="Span Characters"
            description="Character length of each passage span."
            value={settings.mv_span_chars}
            onChange={(v) => updateSetting("mv_span_chars", v)}
            min={100}
            max={2000}
          />
          <SettingSlider
            label="Stride"
            description="Step size between spans. Smaller values produce more overlapping spans."
            value={settings.mv_stride}
            onChange={(v) => updateSetting("mv_stride", v)}
            min={50}
            max={1000}
          />
          <SettingSlider
            label="Max Spans"
            description="Maximum number of spans per document to consider."
            value={settings.mv_max_spans}
            onChange={(v) => updateSetting("mv_max_spans", v)}
            min={1}
            max={50}
          />
        </div>
      )}
      <SettingToggle
        label="Numeric Table Boost"
        description="Boost documents with numeric tables"
        checked={settings.enable_numeric_table_boost}
        onChange={(v) => updateSetting("enable_numeric_table_boost", v)}
      />
    </div>
  )
}

function ChunkingSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Parent Expansion"
        description="Include surrounding context"
        checked={settings.enable_parent_expansion}
        onChange={(v) => updateSetting("enable_parent_expansion", v)}
      />
      {settings.enable_parent_expansion && (
        <SettingSlider
          label="Parent Context Size"
          value={settings.parent_context_size}
          onChange={(v) => updateSetting("parent_context_size", v)}
          min={100}
          max={2000}
          step={100}
        />
      )}
      <SettingToggle
        label="Include Sibling Chunks"
        description="Include adjacent chunks"
        checked={settings.include_sibling_chunks}
        onChange={(v) => updateSetting("include_sibling_chunks", v)}
      />
      {settings.include_sibling_chunks && (
        <SettingSlider
          label="Sibling Window"
          value={settings.sibling_window}
          onChange={(v) => updateSetting("sibling_window", v)}
          min={0}
          max={5}
        />
      )}
      <SettingToggle
        label="Include Parent Document"
        description="Include full parent doc metadata"
        checked={settings.include_parent_document}
        onChange={(v) => updateSetting("include_parent_document", v)}
      />
    </div>
  )
}

function AgenticSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingSelect
        label="Strategy"
        description="Standard uses pre-indexed chunks. Agentic dynamically plans retrieval at query time for complex questions."
        value={settings.strategy}
        onChange={(v) => updateSetting("strategy", v as typeof settings.strategy)}
        options={[
          { value: "standard", label: "Standard (pre-chunked)" },
          { value: "agentic", label: "Agentic (query-time)" },
        ]}
      />
      {settings.strategy === "agentic" && (
        <>
          <SettingSlider
            label="Top-K Documents"
            description="Number of top documents the agent considers per sub-query."
            value={settings.agentic_top_k_docs}
            onChange={(v) => updateSetting("agentic_top_k_docs", v)}
            min={1}
            max={20}
          />
          <SettingSlider
            label="Window Characters"
            description="Context window size in characters for each agentic retrieval step."
            value={settings.agentic_window_chars}
            onChange={(v) => updateSetting("agentic_window_chars", v)}
            min={200}
            max={5000}
            step={100}
          />
          <SettingToggle
            label="Enable Tools"
            description="Allow agentic tool calls"
            checked={settings.agentic_enable_tools}
            onChange={(v) => updateSetting("agentic_enable_tools", v)}
          />
          <SettingToggle
            label="Query Decomposition"
            description="Break complex queries into sub-questions"
            checked={settings.agentic_enable_query_decomposition}
            onChange={(v) => updateSetting("agentic_enable_query_decomposition", v)}
          />
          <SettingToggle
            label="LLM Planner"
            description="Use LLM for retrieval planning"
            checked={settings.agentic_use_llm_planner}
            onChange={(v) => updateSetting("agentic_use_llm_planner", v)}
          />
        </>
      )}
    </div>
  )
}

function RerankingSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Enable Reranking"
        description="Re-scores retrieved results using a second model pass for better relevance ordering. Adds latency but significantly improves answer quality."
        checked={settings.enable_reranking}
        onChange={(v) => updateSetting("enable_reranking", v)}
      />
      {settings.enable_reranking && (
        <>
          <SettingSelect
            label="Strategy"
            description="FlashRank is fastest. Cross-Encoder is most accurate. LLM Scoring uses your LLM to judge relevance. Two-Tier combines fast pre-filter with accurate re-score."
            value={settings.reranking_strategy}
            onChange={(v) => updateSetting("reranking_strategy", v as typeof settings.reranking_strategy)}
            options={[
              { value: "flashrank", label: "FlashRank (fast)" },
              { value: "cross_encoder", label: "Cross-Encoder" },
              { value: "hybrid", label: "Hybrid" },
              { value: "llm_scoring", label: "LLM Scoring" },
              { value: "two_tier", label: "Two-Tier" },
            ]}
          />
          <SettingSlider
            label="Rerank Top-K"
            description="How many top results to keep after reranking. Lower values focus on the most relevant passages."
            value={settings.rerank_top_k}
            onChange={(v) => updateSetting("rerank_top_k", v)}
            min={1}
            max={100}
          />
          <SettingSlider
            label="Min Relevance Probability"
            description="Minimum reranker confidence to keep a result. Higher = stricter filtering, fewer but more relevant results."
            value={settings.rerank_min_relevance_prob}
            onChange={(v) => updateSetting("rerank_min_relevance_prob", v)}
            min={0}
            max={1}
            step={0.05}
          />
        </>
      )}
    </div>
  )
}

function GenerationSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Enable Generation"
        description="Generate synthesized answer"
        checked={settings.enable_generation}
        onChange={(v) => updateSetting("enable_generation", v)}
      />
      {settings.enable_generation && (
        <>
          <SettingSlider
            label="Max Tokens"
            description="Maximum length of the generated answer. Longer answers use more tokens and cost more."
            value={settings.max_generation_tokens}
            onChange={(v) => updateSetting("max_generation_tokens", v)}
            min={50}
            max={2000}
            step={50}
          />
          <SettingToggle
            label="Strict Extractive"
            description="Only quote from sources (no free-form)"
            checked={settings.strict_extractive}
            onChange={(v) => updateSetting("strict_extractive", v)}
          />
          <SettingToggle
            label="Enable Abstention"
            description="Decline to answer if unsure"
            checked={settings.enable_abstention}
            onChange={(v) => updateSetting("enable_abstention", v)}
          />
          {settings.enable_abstention && (
            <SettingSelect
              label="Abstention Behavior"
              value={settings.abstention_behavior}
              onChange={(v) => updateSetting("abstention_behavior", v as typeof settings.abstention_behavior)}
              options={[
                { value: "continue", label: "Continue anyway" },
                { value: "ask", label: "Ask for clarification" },
                { value: "decline", label: "Decline to answer" },
              ]}
            />
          )}
          <SettingToggle
            label="Multi-Turn Synthesis"
            description="Use iterative refinement"
            checked={settings.enable_multi_turn_synthesis}
            onChange={(v) => updateSetting("enable_multi_turn_synthesis", v)}
          />
        </>
      )}
    </div>
  )
}

function CitationsSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Enable Citations"
        description="Generate inline citations"
        checked={settings.enable_citations}
        onChange={(v) => updateSetting("enable_citations", v)}
      />
      {settings.enable_citations && (
        <>
          <SettingSelect
            label="Citation Style"
            description="Formatting standard used for inline citations in the generated answer."
            value={settings.citation_style}
            onChange={(v) => updateSetting("citation_style", v as typeof settings.citation_style)}
            options={[
              { value: "apa", label: "APA" },
              { value: "mla", label: "MLA" },
              { value: "chicago", label: "Chicago" },
              { value: "harvard", label: "Harvard" },
              { value: "ieee", label: "IEEE" },
            ]}
          />
          <SettingToggle
            label="Include Page Numbers"
            description="Add page numbers to citations"
            checked={settings.include_page_numbers}
            onChange={(v) => updateSetting("include_page_numbers", v)}
          />
          <SettingToggle
            label="Chunk-Level Citations"
            description="Cite specific chunks vs documents"
            checked={settings.enable_chunk_citations}
            onChange={(v) => updateSetting("enable_chunk_citations", v)}
          />
        </>
      )}
    </div>
  )
}

function VerificationSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Enable Claims Extraction"
        description="Extract and verify factual claims"
        checked={settings.enable_claims}
        onChange={(v) => updateSetting("enable_claims", v)}
      />
      {settings.enable_claims && (
        <>
          <SettingSelect
            label="Claim Extractor"
            description="Method for identifying factual claims. Auto selects the best approach for your content."
            value={settings.claim_extractor}
            onChange={(v) => updateSetting("claim_extractor", v as typeof settings.claim_extractor)}
            options={[
              { value: "auto", label: "Auto" },
              { value: "aps", label: "APS" },
              { value: "claimify", label: "Claimify" },
              { value: "ner", label: "NER" },
            ]}
          />
          <SettingSelect
            label="Claim Verifier"
            description="NLI uses natural language inference. LLM uses your model to judge claims. Hybrid combines both."
            value={settings.claim_verifier}
            onChange={(v) => updateSetting("claim_verifier", v as typeof settings.claim_verifier)}
            options={[
              { value: "nli", label: "NLI" },
              { value: "llm", label: "LLM" },
              { value: "hybrid", label: "Hybrid" },
            ]}
          />
          <SettingSlider
            label="Confidence Threshold"
            description="Minimum confidence to accept a claim as verified. Higher values reject more unverified claims."
            value={settings.claims_conf_threshold}
            onChange={(v) => updateSetting("claims_conf_threshold", v)}
            min={0}
            max={1}
            step={0.05}
          />
        </>
      )}
      <SettingToggle
        label="Post-Verification"
        description="Verify answer after generation"
        checked={settings.enable_post_verification}
        onChange={(v) => updateSetting("enable_post_verification", v)}
      />
      <SettingToggle
        label="Require Hard Citations"
        description="Every claim must have source spans"
        checked={settings.require_hard_citations}
        onChange={(v) => updateSetting("require_hard_citations", v)}
      />
      <SettingToggle
        label="Numeric Fidelity Check"
        description="Verify numbers appear in sources"
        checked={settings.enable_numeric_fidelity}
        onChange={(v) => updateSetting("enable_numeric_fidelity", v)}
      />
    </div>
  )
}

function SecuritySection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingToggle
        label="Security Filter"
        description="Enable security filtering"
        checked={settings.enable_security_filter}
        onChange={(v) => updateSetting("enable_security_filter", v)}
      />
      <SettingToggle
        label="Detect PII"
        description="Detect personally identifiable information"
        checked={settings.detect_pii}
        onChange={(v) => updateSetting("detect_pii", v)}
      />
      <SettingToggle
        label="Redact PII"
        description="Automatically redact PII from results"
        checked={settings.redact_pii}
        onChange={(v) => updateSetting("redact_pii", v)}
      />
      <SettingSelect
        label="Sensitivity Level"
        description="Controls access classification. Public allows all content; Restricted limits to highest-clearance material only."
        value={settings.sensitivity_level}
        onChange={(v) => updateSetting("sensitivity_level", v as typeof settings.sensitivity_level)}
        options={[
          { value: "public", label: "Public" },
          { value: "internal", label: "Internal" },
          { value: "confidential", label: "Confidential" },
          { value: "restricted", label: "Restricted" },
        ]}
      />
      <SettingToggle
        label="Content Policy Filter"
        description="Filter content by policy"
        checked={settings.enable_content_policy_filter}
        onChange={(v) => updateSetting("enable_content_policy_filter", v)}
      />
      <SettingToggle
        label="HTML Sanitizer"
        description="Sanitize HTML in responses"
        checked={settings.enable_html_sanitizer}
        onChange={(v) => updateSetting("enable_html_sanitizer", v)}
      />
    </div>
  )
}

function PerformanceSection({ settings, updateSetting }: SectionSettingsProps) {
  return (
    <div className="space-y-4">
      <SettingSlider
        label="Timeout (seconds)"
        description="Maximum time to wait for the full RAG pipeline to complete before aborting."
        value={settings.timeout_seconds}
        onChange={(v) => updateSetting("timeout_seconds", v)}
        min={1}
        max={60}
      />
      <SettingToggle
        label="Enable Cache"
        description="Cache semantic search results"
        checked={settings.enable_cache}
        onChange={(v) => updateSetting("enable_cache", v)}
      />
      {settings.enable_cache && (
        <>
          <SettingSlider
            label="Cache Threshold"
            description="Similarity threshold for cache hits"
            value={settings.cache_threshold}
            onChange={(v) => updateSetting("cache_threshold", v)}
            min={0.5}
            max={1}
            step={0.05}
          />
          <SettingToggle
            label="Adaptive Cache"
            description="Adjust cache based on query patterns"
            checked={settings.adaptive_cache}
            onChange={(v) => updateSetting("adaptive_cache", v)}
          />
        </>
      )}
      <SettingToggle
        label="Enable Resilience"
        description="Retry on transient failures"
        checked={settings.enable_resilience}
        onChange={(v) => updateSetting("enable_resilience", v)}
      />
      {settings.enable_resilience && (
        <SettingSlider
          label="Retry Attempts"
          description="Number of retries on transient errors before giving up."
          value={settings.retry_attempts}
          onChange={(v) => updateSetting("retry_attempts", v)}
          min={1}
          max={5}
        />
      )}
      <SettingToggle
        label="Circuit Breaker"
        description="Prevent cascading failures"
        checked={settings.circuit_breaker}
        onChange={(v) => updateSetting("circuit_breaker", v)}
      />
    </div>
  )
}
