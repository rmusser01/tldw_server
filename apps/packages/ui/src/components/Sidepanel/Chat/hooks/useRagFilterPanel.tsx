import React from "react"
import {
  Input,
  InputNumber,
  Select,
  Switch,
  Collapse
} from "antd"
import type { RagSettings } from "@/services/rag/unified-rag"
import { getRagSourceOptions } from "@/services/rag/sourceMetadata"

export const SOURCE_OPTIONS = getRagSourceOptions()

export const STRATEGY_OPTIONS = [
  { label: "Standard", value: "standard" },
  { label: "Agentic", value: "agentic" }
]

export const SEARCH_MODE_OPTIONS = [
  { label: "FTS", value: "fts" },
  { label: "Vector", value: "vector" },
  { label: "Hybrid", value: "hybrid" }
]

export const FTS_LEVEL_OPTIONS = [
  { label: "Media", value: "media" },
  { label: "Chunk", value: "chunk" }
]

export const EXPANSION_OPTIONS = [
  { label: "Acronym", value: "acronym" },
  { label: "Synonym", value: "synonym" },
  { label: "Semantic", value: "semantic" },
  { label: "Domain", value: "domain" },
  { label: "Entity", value: "entity" }
]

export const SENSITIVITY_OPTIONS = [
  { label: "Public", value: "public" },
  { label: "Internal", value: "internal" },
  { label: "Confidential", value: "confidential" },
  { label: "Restricted", value: "restricted" }
]

export const TABLE_METHOD_OPTIONS = [
  { label: "Markdown", value: "markdown" },
  { label: "HTML", value: "html" },
  { label: "Hybrid", value: "hybrid" }
]

export const CHUNK_TYPE_OPTIONS = [
  { label: "Text", value: "text" },
  { label: "Code", value: "code" },
  { label: "Table", value: "table" },
  { label: "List", value: "list" }
]

export const CLAIM_EXTRACTOR_OPTIONS = [
  { label: "Auto", value: "auto" },
  { label: "APS", value: "aps" },
  { label: "Claimify", value: "claimify" },
  { label: "NER", value: "ner" }
]

export const CLAIM_VERIFIER_OPTIONS = [
  { label: "Hybrid", value: "hybrid" },
  { label: "NLI", value: "nli" },
  { label: "LLM", value: "llm" }
]

export const RERANK_STRATEGY_OPTIONS = [
  { label: "FlashRank", value: "flashrank" },
  { label: "Cross-encoder", value: "cross_encoder" },
  { label: "Hybrid", value: "hybrid" },
  { label: "llama.cpp", value: "llama_cpp" },
  { label: "LLM scoring", value: "llm_scoring" },
  { label: "Two-tier", value: "two_tier" },
  { label: "None", value: "none" }
]

export const CITATION_STYLE_OPTIONS = [
  { label: "APA", value: "apa" },
  { label: "MLA", value: "mla" },
  { label: "Chicago", value: "chicago" },
  { label: "Harvard", value: "harvard" },
  { label: "IEEE", value: "ieee" }
]

export const ABSTENTION_OPTIONS = [
  { label: "Continue", value: "continue" },
  { label: "Ask", value: "ask" },
  { label: "Decline", value: "decline" }
]

export const CONTENT_POLICY_TYPES = [
  { label: "PII", value: "pii" },
  { label: "PHI", value: "phi" }
]

export const CONTENT_POLICY_MODES = [
  { label: "Redact", value: "redact" },
  { label: "Drop", value: "drop" },
  { label: "Annotate", value: "annotate" }
]

export const NUMERIC_FIDELITY_OPTIONS = [
  { label: "Continue", value: "continue" },
  { label: "Ask", value: "ask" },
  { label: "Decline", value: "decline" },
  { label: "Retry", value: "retry" }
]

export const LOW_CONFIDENCE_OPTIONS = [
  { label: "Continue", value: "continue" },
  { label: "Ask", value: "ask" },
  { label: "Decline", value: "decline" }
]

export const parseNumericIdList = (value: string) =>
  value
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((num) => Number.isFinite(num) && num > 0)

export const parseStringIdList = (value: string) =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)

export const stringifyIdList = (value: Array<string | number>) => value.join(", ")

export interface UseRagFilterPanelDeps {
  draftSettings: RagSettings
  updateSetting: <K extends keyof RagSettings>(
    key: K,
    value: RagSettings[K],
    options?: { transient?: boolean }
  ) => void
  t: (key: string, fallback?: string) => string
}

export function useRagFilterPanel(deps: UseRagFilterPanelDeps) {
  const { draftSettings, updateSetting, t } = deps

  const [advancedOpen, setAdvancedOpen] = React.useState(false)
  const [advancedSearch, setAdvancedSearch] = React.useState("")

  const advancedSearchLower = advancedSearch.trim().toLowerCase()
  const hasSettingsFilter = advancedOpen && advancedSearchLower.length > 0

  const matchesAdvancedSearch = React.useCallback(
    (label: string) =>
      !hasSettingsFilter || label.toLowerCase().includes(advancedSearchLower),
    [hasSettingsFilter, advancedSearchLower]
  )

  const matchesAny = React.useCallback(
    (...labels: string[]) => labels.some((l) => matchesAdvancedSearch(l)),
    [matchesAdvancedSearch]
  )

  const renderNumberInput = (
    label: string,
    value: number,
    onChange: (next: number) => void,
    options?: { min?: number; max?: number; step?: number; helper?: string }
  ) => (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-text">{label}</span>
      <InputNumber
        min={options?.min}
        max={options?.max}
        step={options?.step}
        value={value}
        aria-label={label}
        onChange={(next) => {
          if (next === null || next === undefined) return
          const parsed = Number(next)
          if (!Number.isFinite(parsed)) return
          onChange(parsed)
        }}
      />
      {options?.helper && (
        <span className="text-[11px] text-text-muted">{options.helper}</span>
      )}
    </div>
  )

  const renderTextInput = (
    label: string,
    value: string,
    onChange: (next: string) => void,
    options?: { placeholder?: string; helper?: string }
  ) => (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-text">{label}</span>
      <Input
        value={value}
        placeholder={options?.placeholder}
        aria-label={label}
        onChange={(e) => onChange(e.target.value)}
      />
      {options?.helper && (
        <span className="text-[11px] text-text-muted">{options.helper}</span>
      )}
    </div>
  )

  const renderSelect = (
    label: string,
    value: string,
    onChange: (next: string) => void,
    options: { label: string; value: string }[],
    helper?: string
  ) => (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-text">{label}</span>
      <Select
        value={value}
        onChange={(next) => onChange(String(next))}
        options={options}
        aria-label={label}
      />
      {helper && <span className="text-[11px] text-text-muted">{helper}</span>}
    </div>
  )

  const renderMultiSelect = (
    label: string,
    value: string[],
    onChange: (next: string[]) => void,
    options: { label: string; value: string }[],
    helper?: string
  ) => (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-text">{label}</span>
      <Select
        mode="multiple"
        value={value}
        onChange={(next) => onChange(next as string[])}
        options={options}
        aria-label={label}
      />
      {helper && <span className="text-[11px] text-text-muted">{helper}</span>}
    </div>
  )

  return {
    advancedOpen,
    setAdvancedOpen,
    advancedSearch,
    setAdvancedSearch,
    hasSettingsFilter,
    matchesAdvancedSearch,
    matchesAny,
    renderNumberInput,
    renderTextInput,
    renderSelect,
    renderMultiSelect,
  }
}
