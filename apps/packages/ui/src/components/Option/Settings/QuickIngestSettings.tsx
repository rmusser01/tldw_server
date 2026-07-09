import React from "react"
import { Button, Collapse, Input, InputNumber, Select, Switch } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import {
  PRESET_META,
  resolvePresetMap,
  type PresetMap
} from "@/components/Common/QuickIngest/presets"
import { getDefaultOcrLanguage, ocrLanguages } from "@/data/ocr-language"
import {
  ensureSelectOption,
  getAdvancedFieldSelectOptions
} from "@/components/Common/QuickIngest/advanced-field-options"
import type { IngestPreset } from "@/components/Common/QuickIngest/types"
import { QUICK_INGEST_SCHEMA_FALLBACK } from "@/services/tldw/fallback-schemas"
import { fetchChatModels, getEmbeddingModels } from "@/services/tldw-server"

const PRESET_KEYS: Array<Exclude<IngestPreset, "custom">> = [
  "quick",
  "standard",
  "deep"
]
const EXCLUDED_ADVANCED_FIELDS = new Set([
  "perform_analysis",
  "perform_chunking",
  "overwrite_existing",
  "diarize"
])

type ToggleRowProps = {
  label: string
  description?: string
  checked: boolean
  onChange: (value: boolean) => void
}

const ToggleRow = ({ label, description, checked, onChange }: ToggleRowProps) => (
  <div className="flex flex-wrap items-center justify-between gap-3">
    <div className="min-w-[220px]">
      <span className="text-text">{label}</span>
      {description ? (
        <div className="text-xs text-text-muted mt-1">{description}</div>
      ) : null}
    </div>
    <Switch checked={checked} onChange={onChange} aria-label={label} />
  </div>
)

export const QuickIngestSettings = () => {
  const { t } = useTranslation(["settings", "option"])
  const qi = React.useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`option:quickIngest.${key}`, { defaultValue, ...options })
        : t(`option:quickIngest.${key}`, defaultValue),
    [t]
  )
  const [presetConfigs, setPresetConfigs] = useStorage<PresetMap>(
    "quickIngestPresetConfigs",
    resolvePresetMap()
  )
  const [defaultOCRLanguage, setDefaultOCRLanguage] = useStorage(
    "defaultOCRLanguage",
    getDefaultOcrLanguage()
  )
  const [enableOcrAssets, setEnableOcrAssets] = useStorage(
    "enableOcrAssets",
    false
  )
  const { data: chatModels = [], isLoading: chatModelsLoading } = useQuery({
    queryKey: ["playground:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: true
  })
  const { data: embeddingModels = [], isLoading: embeddingModelsLoading } =
    useQuery({
      queryKey: ["embedding-models"],
      queryFn: () => getEmbeddingModels(),
      enabled: true
    })
  const presets = React.useMemo(
    () => resolvePresetMap(presetConfigs),
    [presetConfigs]
  )
  const advancedFields = React.useMemo(
    () =>
      QUICK_INGEST_SCHEMA_FALLBACK.filter(
        (field) => !EXCLUDED_ADVANCED_FIELDS.has(field.name)
      ),
    []
  )

  const updatePreset = React.useCallback(
    (
      preset: Exclude<IngestPreset, "custom">,
      updater: (config: PresetMap["quick"]) => PresetMap["quick"]
    ) => {
      setPresetConfigs((prev) => {
        const resolved = resolvePresetMap(prev)
        return {
          ...resolved,
          [preset]: updater(resolved[preset])
        }
      })
    },
    [setPresetConfigs]
  )
  const updatePresetAdvancedValue = React.useCallback(
    (
      preset: Exclude<IngestPreset, "custom">,
      fieldName: string,
      value: unknown
    ) => {
      updatePreset(preset, (config) => {
        const nextAdvanced = { ...(config.advancedValues ?? {}) }
        if (value === undefined || value === null || value === "") {
          delete nextAdvanced[fieldName]
        } else {
          nextAdvanced[fieldName] = value
        }
        return {
          ...config,
          advancedValues: nextAdvanced
        }
      })
    },
    [updatePreset]
  )

  const resetPreset = React.useCallback(
    (preset: Exclude<IngestPreset, "custom">) => {
      setPresetConfigs((prev) => {
        const resolved = resolvePresetMap(prev)
        const defaults = resolvePresetMap()
        return {
          ...resolved,
          [preset]: defaults[preset]
        }
      })
    },
    [setPresetConfigs]
  )

  const resetAll = React.useCallback(() => {
    setPresetConfigs(resolvePresetMap())
  }, [setPresetConfigs])

  return (
    <div className="flex flex-col space-y-6 text-sm">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("quickIngestSettings.title", "Quick Ingest settings")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "quickIngestSettings.subtitle",
            "Customize the presets used in the Quick Ingest Options tab."
          )}
        </p>
        <div className="border-b border-border mt-3" />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-text-muted">
          {t(
            "quickIngestSettings.presetHint",
            "Changes apply the next time you open Quick Ingest."
          )}
        </p>
        <Button size="small" onClick={resetAll}>
          {t("quickIngestSettings.resetAll", "Reset all presets")}
        </Button>
      </div>

      <section className="rounded-md border border-border bg-surface p-4 space-y-3">
        <div>
          <h3 className="text-sm font-semibold text-text">
            {t("quickIngestSettings.ocrDefaults.title", "OCR defaults")}
          </h3>
          <p className="mt-1 text-xs text-text-muted">
            {t(
              "quickIngestSettings.ocrDefaults.description",
              "Used by ingest flows that extract text from images or scanned documents."
            )}
          </p>
        </div>
        <ToggleRow
          checked={enableOcrAssets}
          label={t("generalSettings.settings.enableOcrAssets.label")}
          onChange={setEnableOcrAssets}
        />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-text">
            {t("generalSettings.settings.ocrLanguage.label")}
          </span>
          <Select
            aria-label={t("generalSettings.settings.ocrLanguage.label")}
            className="w-full sm:w-[240px]"
            onChange={setDefaultOCRLanguage}
            options={ocrLanguages}
            showSearch
            value={defaultOCRLanguage}
          />
        </div>
      </section>

      {PRESET_KEYS.map((presetKey) => {
        const preset = presets[presetKey]
        const meta = PRESET_META[presetKey]
        const label = qi(
          meta.labelKey,
          presetKey.charAt(0).toUpperCase() + presetKey.slice(1)
        )
        const description = qi(meta.descriptionKey, "")

        return (
          <div
            key={presetKey}
            className="rounded-md border border-border bg-surface p-4 space-y-4"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-start gap-2">
                <span aria-hidden="true" className="text-lg">
                  {meta.icon}
                </span>
                <div>
                  <div className="text-sm font-semibold text-text">{label}</div>
                  {description ? (
                    <div className="text-xs text-text-muted">
                      {description}
                    </div>
                  ) : null}
                </div>
              </div>
              <Button size="small" onClick={() => resetPreset(presetKey)}>
                {qi("preset.reset", "Reset to defaults")}
              </Button>
            </div>

            <div className="space-y-3">
              <ToggleRow
                label={qi("analysisLabel", "Analysis")}
                checked={preset.common.perform_analysis}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    common: { ...config.common, perform_analysis: value }
                  }))
                }
              />
              <ToggleRow
                label={qi("chunkingLabel", "Chunking")}
                checked={preset.common.perform_chunking}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    common: { ...config.common, perform_chunking: value }
                  }))
                }
              />
              <ToggleRow
                label={qi("overwriteLabel", "Overwrite existing")}
                checked={preset.common.overwrite_existing}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    common: { ...config.common, overwrite_existing: value }
                  }))
                }
              />
              <ToggleRow
                label={qi("storeRemote", "Store to remote DB")}
                checked={preset.storeRemote}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    storeRemote: value,
                    reviewBeforeStorage: value
                      ? config.reviewBeforeStorage
                      : false
                  }))
                }
              />
              <ToggleRow
                label={qi("reviewBeforeStorage", "Review before saving")}
                checked={preset.reviewBeforeStorage}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    reviewBeforeStorage: value,
                    storeRemote: value ? true : config.storeRemote
                  }))
                }
              />
              <ToggleRow
                label={qi("audioDiarizationLabel", "Diarization")}
                checked={preset.typeDefaults?.audio?.diarize ?? false}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    typeDefaults: {
                      ...config.typeDefaults,
                      audio: {
                        ...config.typeDefaults?.audio,
                        diarize: value
                      }
                    }
                  }))
                }
              />
              <ToggleRow
                label={qi("documentOcrLabel", "OCR")}
                checked={preset.typeDefaults?.document?.ocr ?? false}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    typeDefaults: {
                      ...config.typeDefaults,
                      document: {
                        ...config.typeDefaults?.document,
                        ocr: value
                      }
                    }
                  }))
                }
              />
              <ToggleRow
                label={qi("videoCaptionsLabel", "Captions")}
                checked={preset.typeDefaults?.video?.captions ?? false}
                onChange={(value) =>
                  updatePreset(presetKey, (config) => ({
                    ...config,
                    typeDefaults: {
                      ...config.typeDefaults,
                      video: {
                        ...config.typeDefaults?.video,
                        captions: value
                      }
                    }
                  }))
                }
              />
            </div>

            <Collapse
              className="bg-transparent border-0"
              items={[
                {
                  key: "advanced",
                  label: t(
                    "quickIngestSettings.advancedHeading",
                    "Advanced options"
                  ),
                  className: "!border-0",
                  children: (
                    <>
                      <p className="text-xs text-text-subtle mb-4">
                        {t(
                          "quickIngestSettings.advancedHint",
                          "Configure advanced ingest fields for this preset."
                        )}
                      </p>
                      <div className="space-y-4">
                        {advancedFields.map((field) => {
                          const value = preset.advancedValues?.[field.name]
                          const labelText = field.title || field.name
                          const descriptionText = field.description
                          const setValue = (next: unknown) =>
                            updatePresetAdvancedValue(presetKey, field.name, next)
                          const selectOptions = getAdvancedFieldSelectOptions({
                            fieldName: field.name,
                            fieldEnum: field.enum,
                            t,
                            chatModels,
                            embeddingModels
                          })
                          const fallbackEnumOptions = Array.isArray(field.enum)
                            ? field.enum.map((entry: unknown) => ({
                                value: String(entry),
                                label: String(entry)
                              }))
                            : null
                          const selectValue =
                            value === undefined || value === null || value === ""
                              ? undefined
                              : String(value)
                          const resolvedSelectOptions = selectOptions
                            ? ensureSelectOption(selectOptions, selectValue)
                            : fallbackEnumOptions
                              ? ensureSelectOption(fallbackEnumOptions, selectValue)
                              : null
                          const isContextualModel =
                            field.name === "contextual_llm_model"
                          const isEmbeddingModel =
                            field.name === "embedding_model"
                          const shouldShowSearch =
                            isContextualModel ||
                            isEmbeddingModel ||
                            (resolvedSelectOptions?.length ?? 0) > 6

                          if (resolvedSelectOptions) {
                            return (
                              <div
                                key={field.name}
                                className="flex flex-wrap items-center justify-between gap-3"
                              >
                                <div className="min-w-[220px]">
                                  <span className="text-text">{labelText}</span>
                                  {descriptionText ? (
                                    <div className="text-xs text-text-muted mt-1">
                                      {descriptionText}
                                    </div>
                                  ) : null}
                                </div>
                                <Select
                                  className="min-w-[220px]"
                                  allowClear
                                  showSearch={shouldShowSearch}
                                  loading={
                                    (isContextualModel && chatModelsLoading) ||
                                    (isEmbeddingModel && embeddingModelsLoading)
                                  }
                                  value={selectValue}
                                  onChange={(next) => setValue(next ?? undefined)}
                                  options={resolvedSelectOptions}
                                  aria-label={labelText}
                                />
                              </div>
                            )
                          }

                          if (field.type === "boolean") {
                            const isSet = value !== undefined
                            return (
                              <div
                                key={field.name}
                                className="flex flex-wrap items-center justify-between gap-3"
                              >
                                <div className="min-w-[220px]">
                                  <span className="text-text">{labelText}</span>
                                  {descriptionText ? (
                                    <div className="text-xs text-text-muted mt-1">
                                      {descriptionText}
                                    </div>
                                  ) : null}
                                </div>
                                <div className="flex items-center gap-2">
                                  <Switch
                                    checked={Boolean(value)}
                                    onChange={(next) => setValue(next)}
                                    aria-label={labelText}
                                  />
                                  <Button
                                    size="small"
                                    onClick={() => setValue(undefined)}
                                    disabled={!isSet}
                                  >
                                    {qi("unset", "Unset")}
                                  </Button>
                                </div>
                              </div>
                            )
                          }

                          if (field.type === "integer" || field.type === "number") {
                            const numericValue =
                              typeof value === "number"
                                ? value
                                : value === null ||
                                    value === undefined ||
                                    value === ""
                                  ? undefined
                                  : Number(value)
                            return (
                              <div
                                key={field.name}
                                className="flex flex-wrap items-center justify-between gap-3"
                              >
                                <div className="min-w-[220px]">
                                  <span className="text-text">{labelText}</span>
                                  {descriptionText ? (
                                    <div className="text-xs text-text-muted mt-1">
                                      {descriptionText}
                                    </div>
                                  ) : null}
                                </div>
                                <InputNumber
                                  className="min-w-[220px]"
                                  value={
                                    Number.isNaN(numericValue)
                                      ? undefined
                                      : numericValue
                                  }
                                  onChange={(next) => setValue(next ?? undefined)}
                                  aria-label={labelText}
                                  placeholder={labelText}
                                />
                              </div>
                            )
                          }

                          return (
                            <div
                              key={field.name}
                              className="flex flex-wrap items-center justify-between gap-3"
                            >
                              <div className="min-w-[220px]">
                                <span className="text-text">{labelText}</span>
                                {descriptionText ? (
                                  <div className="text-xs text-text-muted mt-1">
                                    {descriptionText}
                                  </div>
                                ) : null}
                              </div>
                              <Input
                                className="min-w-[220px]"
                                allowClear
                                value={value === undefined ? "" : String(value)}
                                onChange={(event) => {
                                  const nextValue = event.target.value
                                  setValue(nextValue === "" ? undefined : nextValue)
                                }}
                                aria-label={labelText}
                                placeholder={labelText}
                              />
                            </div>
                          )
                        })}
                      </div>
                    </>
                  )
                }
              ]}
            />
          </div>
        )
      })}
    </div>
  )
}

export default QuickIngestSettings
