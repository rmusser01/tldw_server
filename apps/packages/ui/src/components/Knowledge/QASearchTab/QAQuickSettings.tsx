import React from "react"
import { AutoComplete, Input, Select } from "antd"
import { useTranslation } from "react-i18next"
import type { RagPresetName, RagSource } from "@/services/rag/unified-rag"
import { SourceChips } from "../SearchTab/SourceChips"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { getProviderDisplayName } from "@/utils/provider-registry"

type LlmProviderConfig = {
  name: string
  display_name?: string
  models?: string[]
  models_info?: Array<{
    id?: string
    name?: string
    model_id?: string
    display_name?: string
  }>
  default_model?: string | null
}

type LlmProvidersResponse = {
  providers?: LlmProviderConfig[]
  default_provider?: string | null
}

const SERVER_DEFAULT_PROVIDER_VALUE = "__server_default__"

type QAQuickSettingsProps = {
  preset: RagPresetName
  onPresetChange: (preset: RagPresetName) => void
  strategy: "standard" | "agentic"
  onStrategyChange: (strategy: "standard" | "agentic") => void
  selectedSources: RagSource[]
  onSourcesChange: (sources: RagSource[]) => void
  generationProvider: string | null
  onGenerationProviderChange: (provider: string | null) => void
  generationModel: string
  onGenerationModelChange: (model: string) => void
  disabled?: boolean
}

/**
 * Quick settings row for QA Search tab.
 * Shows preset dropdown, strategy selector, and source chips inline.
 */
export const QAQuickSettings: React.FC<QAQuickSettingsProps> = ({
  preset,
  onPresetChange,
  strategy,
  onStrategyChange,
  selectedSources,
  onSourcesChange,
  generationProvider,
  onGenerationProviderChange,
  generationModel,
  onGenerationModelChange,
  disabled = false
}) => {
  const { t } = useTranslation(["sidepanel"])
  const [providerCatalog, setProviderCatalog] =
    React.useState<LlmProvidersResponse | null>(null)

  React.useEffect(() => {
    let cancelled = false

    void (async () => {
      try {
        await tldwClient.initialize()
        const response = (await tldwClient.getProviders()) as LlmProvidersResponse
        if (!cancelled) {
          setProviderCatalog(response)
        }
      } catch {
        if (!cancelled) {
          setProviderCatalog(null)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  const presetOptions = React.useMemo(
    () => [
      { label: t("sidepanel:rag.presets.fast", "Fast"), value: "fast" as const },
      {
        label: t("sidepanel:rag.presets.balanced", "Balanced"),
        value: "balanced" as const
      },
      {
        label: t("sidepanel:rag.presets.thorough", "Thorough"),
        value: "thorough" as const
      },
      {
        label: t("sidepanel:rag.presets.custom", "Custom"),
        value: "custom" as const
      }
    ],
    [t]
  )

  const strategyOptions = React.useMemo(
    () => [
      {
        label: t("sidepanel:qaSearch.strategy.standard", "Standard"),
        value: "standard" as const
      },
      {
        label: t("sidepanel:qaSearch.strategy.agentic", "Agentic"),
        value: "agentic" as const
      }
    ],
    [t]
  )

  const providerEntries = React.useMemo(
    () => providerCatalog?.providers ?? [],
    [providerCatalog]
  )

  const providerOptions = React.useMemo(
    () => [
      {
        label: t("sidepanel:qaSearch.answerProviderDefault", "Server default"),
        value: SERVER_DEFAULT_PROVIDER_VALUE,
      },
      ...providerEntries.map((provider) => ({
        label:
          provider.display_name ||
          getProviderDisplayName(provider.name) ||
          provider.name,
        value: provider.name,
      })),
    ],
    [providerEntries, t]
  )

  const effectiveProviderKey = generationProvider || null

  const selectedProviderConfig = React.useMemo(
    () =>
      providerEntries.find((provider) => provider.name === effectiveProviderKey) ??
      null,
    [effectiveProviderKey, providerEntries]
  )

  const modelOptions = React.useMemo(() => {
    const seen = new Set<string>()
    const options: { value: string; label: string }[] = []
    const pushModel = (value?: string | null, label?: string | null) => {
      const normalized = String(value || "").trim()
      if (!normalized || seen.has(normalized)) return
      seen.add(normalized)
      options.push({
        value: normalized,
        label: String(label || normalized).trim() || normalized,
      })
    }

    selectedProviderConfig?.models?.forEach((modelId) => pushModel(modelId, modelId))
    selectedProviderConfig?.models_info?.forEach((model) =>
      pushModel(
        model.model_id || model.id || model.name,
        model.display_name || model.name || model.model_id || model.id
      )
    )
    pushModel(selectedProviderConfig?.default_model, selectedProviderConfig?.default_model)

    return options
  }, [selectedProviderConfig])

  const modelPlaceholder = React.useMemo(() => {
    const defaultModel = selectedProviderConfig?.default_model?.trim()
    if (defaultModel) {
      return t(
        "sidepanel:qaSearch.answerModelPlaceholderDefault",
        "Model override (default: {{model}})",
        { model: defaultModel }
      )
    }
    return t(
      "sidepanel:qaSearch.answerModelPlaceholder",
      "Model override (optional)"
    )
  }, [selectedProviderConfig?.default_model, t])

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Select
          value={preset}
          onChange={onPresetChange}
          options={presetOptions}
          size="small"
          className="w-28 flex-shrink-0"
          disabled={disabled}
        />
        <Select
          value={strategy}
          onChange={onStrategyChange}
          options={strategyOptions}
          size="small"
          className="w-28 flex-shrink-0"
          disabled={disabled}
        />
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-[11rem_minmax(0,1fr)]">
        <div className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-wide text-text-muted">
            {t("sidepanel:qaSearch.answerProvider", "Answer provider")}
          </span>
          <Select
            value={generationProvider ?? SERVER_DEFAULT_PROVIDER_VALUE}
            onChange={(value) => {
              const provider = value === SERVER_DEFAULT_PROVIDER_VALUE ? null : String(value)
              onGenerationProviderChange(provider)
              onGenerationModelChange(
                providerEntries.find((entry) => entry.name === provider)?.default_model?.trim() || ""
              )
            }}
            options={providerOptions}
            size="small"
            className="w-full"
            disabled={disabled}
            aria-label={t("sidepanel:qaSearch.answerProvider", "Answer provider")}
          />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-wide text-text-muted">
            {t("sidepanel:qaSearch.answerModel", "Answer model")}
          </span>
          <AutoComplete
            value={generationModel}
            options={modelOptions}
            onChange={(value) => onGenerationModelChange(value)}
            disabled={disabled}
            className="w-full"
            filterOption={(inputValue, option) =>
              String(option?.value ?? "")
                .toLowerCase()
                .includes(inputValue.toLowerCase())
            }
          >
            <Input
              size="small"
              aria-label={t("sidepanel:qaSearch.answerModel", "Answer model")}
              placeholder={modelPlaceholder}
            />
          </AutoComplete>
        </div>
      </div>
      <SourceChips
        selectedSources={selectedSources}
        onSourcesChange={onSourcesChange}
        disabled={disabled}
      />
    </div>
  )
}
