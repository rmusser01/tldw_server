import React from "react"
import { Button, Input, InputNumber, Select, Switch } from "antd"
import { useQuery } from "@tanstack/react-query"
import { ChevronDown } from "lucide-react"
import { useTranslation } from "react-i18next"
import {
  ensureSelectOption,
  getAdvancedFieldSelectOptions
} from "@/components/Common/QuickIngest/advanced-field-options"
import { QUICK_INGEST_SCHEMA_FALLBACK } from "@/services/tldw/fallback-schemas"
import { fetchChatModels, getEmbeddingModels } from "@/services/tldw-server"

const primaryFields = new Set([
  "api_name",
  "perform_analysis",
  "perform_chunking",
  "overwrite_existing",
  "diarize"
])
const advancedFields = QUICK_INGEST_SCHEMA_FALLBACK.filter(
  (field) => !primaryFields.has(field.name)
)
type AdvancedField = (typeof advancedFields)[number]

type AdvancedSettingsProps = {
  values: Record<string, unknown> | undefined
  onChange: (name: string, value: unknown) => void
}

export function QuickIngestAdvancedSettings({ values, onChange }: AdvancedSettingsProps) {
  const { t } = useTranslation("settings")
  const id = React.useId()
  const [expanded, setExpanded] = React.useState(false)
  const [search, setSearch] = React.useState("")
  const { data: chatModels, isLoading: chatModelsLoading } = useQuery({
    queryKey: ["playground:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: expanded
  })
  const { data: embeddingModels, isLoading: embeddingModelsLoading } = useQuery({
    queryKey: ["embedding-models"],
    queryFn: getEmbeddingModels,
    enabled: expanded
  })
  const query = search.trim().toLowerCase()
  const matchingFields = advancedFields.filter((field) =>
    [field.name, field.title, field.description].some((part) =>
      part?.toLowerCase().includes(query)
    )
  )

  function renderControl(field: AdvancedField, fieldId: string) {
    const value = values?.[field.name]
    const isSet = value !== undefined && value !== null && value !== ""
    const setValue = (next: unknown) => onChange(field.name, next)
    const fieldProps = {
      id: fieldId,
      "aria-describedby": field.description ? `${fieldId}-description` : undefined
    }
    const options = getAdvancedFieldSelectOptions({
      fieldName: field.name,
      fieldEnum: field.enum,
      t,
      chatModels,
      embeddingModels
    }) ?? field.enum?.map((entry) => ({ value: String(entry), label: String(entry) }))
    if (options) {
      const selected = isSet ? String(value) : undefined
      const isChat = field.name === "contextual_llm_model"
      const isEmbedding = field.name === "embedding_model"
      return (
        <Select
          {...fieldProps}
          allowClear
          className="w-full min-w-0"
          loading={(isChat && chatModelsLoading) || (isEmbedding && embeddingModelsLoading)}
          showSearch={isChat || isEmbedding || options.length > 6}
          options={ensureSelectOption(options, selected)}
          value={selected}
          onChange={(next) => setValue(next ?? undefined)}
        />
      )
    }
    if (field.type === "boolean") {
      return (
        <div className="flex items-center gap-3">
          <Switch {...fieldProps} checked={Boolean(value)} onChange={setValue} />
          <Button
            size="small"
            disabled={!isSet}
            aria-label={t("quickIngestSettings.unsetField", {
              defaultValue: "Unset {{field}}",
              field: field.title || field.name
            })}
            onClick={() => setValue(undefined)}
          >
            {t("quickIngestSettings.unset", "Unset")}
          </Button>
        </div>
      )
    }
    if (field.type === "integer" || field.type === "number") {
      const numeric = isSet ? Number(value) : NaN
      return (
        <InputNumber
          {...fieldProps}
          className="!w-full min-w-0"
          value={Number.isFinite(numeric) ? numeric : null}
          onChange={(next) => setValue(next ?? undefined)}
        />
      )
    }
    return (
      <Input
        {...fieldProps}
        allowClear
        className="w-full min-w-0"
        value={isSet ? String(value) : ""}
        onChange={(event) => setValue(event.target.value || undefined)}
      />
    )
  }

  return (
    <div className="min-w-0 border-t border-border">
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={`${id}-fields`}
        className="flex min-h-11 w-full items-center justify-between gap-3 rounded text-start text-sm font-semibold text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
        onClick={() => setExpanded(!expanded)}
      >
        {t("quickIngestSettings.advancedHeading", "Advanced options")}
        <ChevronDown aria-hidden="true" size={16} className={expanded ? "rotate-180" : ""} />
      </button>
      <div id={`${id}-fields`} hidden={!expanded}>
        {expanded && (
          <>
            <label htmlFor={`${id}-search`} className="mb-1 block text-sm font-medium text-text">
              {t("quickIngestSettings.advancedSearch", "Search advanced fields")}
            </label>
            <Input
              id={`${id}-search`}
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            {matchingFields.length === 0 && (
              <p role="status" className="mt-4 text-sm text-text-muted">
                {t("quickIngestSettings.advancedNoMatches", "No matching advanced fields.")}
              </p>
            )}
            <div className="mt-2 divide-y divide-border">
              {matchingFields.map((field) => {
                const fieldId = `${id}-${field.name}`
                return (
                  <div key={field.name} className="grid min-w-0 grid-cols-1 gap-2 py-3 sm:grid-cols-2 sm:items-center sm:gap-4">
                    <div className="min-w-0 break-words">
                      <label htmlFor={fieldId} className="font-medium text-text">
                        {field.title || field.name}
                      </label>
                      {field.description && (
                        <p id={`${fieldId}-description`} className="mt-1 text-xs text-text-muted">
                          {field.description}
                        </p>
                      )}
                    </div>
                    {renderControl(field, fieldId)}
                  </div>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
