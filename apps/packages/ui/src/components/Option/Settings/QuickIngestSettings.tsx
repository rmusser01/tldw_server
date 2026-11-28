import React from "react"
import { AutoComplete, Button, Select, Switch } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import {
  DEFAULT_PRESET,
  PRESET_META,
  type PresetMap
} from "@/components/Common/QuickIngest/presets"
import { getDefaultOcrLanguage, ocrLanguages } from "@/data/ocr-language"
import { getQuickIngestAnalysisProviderWarning } from "@/services/tldw/quick-ingest-batch"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  getChatModelsCacheGeneration,
  subscribeChatModelsInvalidation
} from "@/services/chat-models-invalidation"
import {
  presetDefaults as factory,
  useQuickIngestPresetSettings
} from "./useQuickIngestPresetSettings"
import { QuickIngestAdvancedSettings } from "./QuickIngestAdvancedSettings"

type NamedPreset = keyof PresetMap
const presetKeys: NamedPreset[] = ["quick", "standard", "deep"]

// Inactive advanced overrides are stored too. They must survive a reset and Undo.
const atFactoryDefaults = (map: PresetMap, key: NamedPreset) =>
  JSON.stringify(map[key]) === JSON.stringify(factory[key])

function ToggleRow({
  label,
  checked,
  onChange
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  const id = React.useId()
  return (
    <label
      htmlFor={id}
      className="flex min-h-11 cursor-pointer items-center justify-between gap-4 text-text"
    >
      <span>{label}</span>
      <Switch id={id} aria-label={label} checked={checked} onChange={onChange} />
    </label>
  )
}

export function QuickIngestSettings() {
  const { t } = useTranslation(["settings", "option"])
  const { presets, isLoading, write } = useQuickIngestPresetSettings()
  const [ocrLanguage, setOcrLanguage] = useStorage("defaultOCRLanguage", getDefaultOcrLanguage())
  const [ocrAssets, setOcrAssets] = useStorage("enableOcrAssets", false)
  const [writeError, setWriteError] = React.useState("")
  const [resetPending, setResetPending] = React.useState(false)
  const [selected, setSelected] = React.useState<NamedPreset>(DEFAULT_PRESET)
  const [lastReset, setLastReset] = React.useState<{
    before: PresetMap
    keys: NamedPreset[]
    message: string
  } | null>(null)
  const [restoredMessage, setRestoredMessage] = React.useState("")
  const undoButton = React.useRef<HTMLButtonElement>(null)
  const resetTrigger = React.useRef<HTMLElement | null>(null)
  const selectedRadio = React.useRef<HTMLInputElement>(null)
  const focusAfterReset = React.useRef(false)
  const focusAfterUndo = React.useRef(false)
  const id = React.useId()
  const preset = presets[selected]
  const presetName = (key: NamedPreset) =>
    t("option:quickIngest." + PRESET_META[key].labelKey, {
      defaultValue: key.charAt(0).toUpperCase() + key.slice(1)
    })
  const name = presetName(selected)
  const destination = preset.reviewBeforeStorage
    ? "draft"
    : preset.storeRemote
      ? "media"
      : "extract"
  const missingProvider = Boolean(getQuickIngestAnalysisProviderWarning(preset))
  const provider =
    typeof preset.advancedValues?.api_name === "string" ? preset.advancedValues.api_name : ""
  const generation = React.useSyncExternalStore(
    subscribeChatModelsInvalidation,
    getChatModelsCacheGeneration,
    getChatModelsCacheGeneration
  )
  const { data: providers } = useQuery({
    queryKey: ["playground:providersStatus", generation],
    queryFn: () => tldwClient.getProvidersStatus(),
    enabled: preset.common.perform_analysis,
    staleTime: 60_000,
    retry: false
  })
  const providerNames = new Set(
    providers?.providers
      ?.filter((entry) => entry.configured)
      .map((entry) => entry.name.trim())
      .filter(Boolean)
  )
  const resettable = presetKeys.filter((key) => !atFactoryDefaults(presets, key))
  const canUndo = lastReset?.keys.some((key) => atFactoryDefaults(presets, key))

  React.useEffect(() => {
    if (focusAfterReset.current && canUndo && undoButton.current) {
      focusAfterReset.current = false
      undoButton.current?.focus()
    } else if (focusAfterUndo.current) {
      focusAfterUndo.current = false
      const trigger = resetTrigger.current
      if (trigger && !trigger.matches(":disabled")) trigger.focus()
      else selectedRadio.current?.focus()
    }
  }, [lastReset, canUndo])

  function updatePreset(updater: (previous: PresetMap[NamedPreset]) => PresetMap[NamedPreset]) {
    setRestoredMessage("")
    setWriteError("")
    void write((current) => ({ ...current, [selected]: updater(current[selected]) }))
      .then(() => {
        setLastReset((previous) => {
          if (!previous?.keys.includes(selected)) return previous
          const keys = previous.keys.filter((key) => key !== selected)
          return keys.length ? { ...previous, keys } : null
        })
      })
      .catch(() => {
        setWriteError(
          t("quickIngestSettings.saveFailed", "Could not save the preset change. Try again.")
        )
      })
  }

  function updateAdvanced(field: string, value: unknown) {
    updatePreset((current) => {
      const advancedValues = { ...current.advancedValues }
      if (value == null || value === "") delete advancedValues[field]
      else advancedValues[field] = value
      return { ...current, advancedValues }
    })
  }

  async function reset(keys: NamedPreset[], message: string, trigger: HTMLElement) {
    setResetPending(true)
    setWriteError("")
    try {
      const before = await write((current) => {
        const next = { ...current }
        for (const key of keys) next[key] = factory[key]
        return next
      })
      resetTrigger.current = trigger
      focusAfterReset.current = true
      setRestoredMessage("")
      setLastReset({ before, keys, message })
    } catch {
      setWriteError(t("quickIngestSettings.resetFailed", "Could not reset the presets. Try again."))
    } finally {
      setResetPending(false)
    }
  }

  async function undoReset() {
    if (!lastReset) return
    setResetPending(true)
    setWriteError("")
    try {
      await write((current) => {
        const next = { ...current }
        for (const key of lastReset.keys) {
          // Preserve any edits to this preset made since the reset, including from another tab.
          if (atFactoryDefaults(current, key)) next[key] = lastReset.before[key]
        }
        return next
      })
      focusAfterUndo.current = true
      setLastReset(null)
      setRestoredMessage(t("quickIngestSettings.restored", "Previous preset settings restored."))
    } catch {
      focusAfterReset.current = true
      setWriteError(
        t("quickIngestSettings.undoFailed", "Could not restore the presets. Try Undo again.")
      )
    } finally {
      setResetPending(false)
    }
  }

  const selectClass =
    "min-h-11 w-full rounded-md border border-border bg-surface px-3 py-2 text-text focus-visible:outline-primary"
  return (
    <div className="min-w-0 space-y-6 text-sm text-text">
      <fieldset role="radiogroup" aria-labelledby={id + "-choice"}>
        <legend id={id + "-choice"} className="mb-2 text-base font-semibold">
          {t("quickIngestSettings.presetChoice", "Preset to edit")}
        </legend>
        <div className="inline-grid w-full grid-cols-3 gap-1 rounded-lg border border-border bg-surface p-1 sm:max-w-sm">
          {presetKeys.map((key) => (
            <label key={key} className="min-w-0 cursor-pointer">
              <input
                type="radio"
                className="peer sr-only"
                name={id + "-preset"}
                value={key}
                ref={selected === key ? selectedRadio : undefined}
                checked={selected === key}
                onChange={() => setSelected(key)}
              />
              <span className="flex min-h-11 items-center justify-center rounded-md px-2 text-center text-text-muted peer-checked:bg-primary/15 peer-checked:font-semibold peer-checked:text-primary peer-checked:ring-1 peer-checked:ring-primary/50 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-primary">
                {presetName(key)}
              </span>
            </label>
          ))}
        </div>
        <p className="mt-2 text-sm text-text-muted">
          {t(
            "quickIngestSettings.defaultsHint",
            "Edits are saved for future ingests that use this preset."
          )}
        </p>
      </fieldset>

      <section
        aria-labelledby={id + "-heading"}
        className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5"
      >
        <h2 id={id + "-heading"} className="mb-4 text-lg font-semibold">
          {t("quickIngestSettings.editorTitle", {
            defaultValue: "{{preset}} preset",
            preset: name
          })}
        </h2>
        <fieldset disabled={isLoading || resetPending} className="min-w-0 space-y-4">
          <div>
            <label className="mb-1 block font-medium" htmlFor={id + "-destination"}>
              {t("quickIngestSettings.destination", "Destination")}
            </label>
            <select
              id={id + "-destination"}
              className={selectClass}
              value={destination}
              aria-describedby={id + "-destination-help"}
              onChange={(event) => {
                const next = event.currentTarget.value
                updatePreset((current) => ({
                  ...current,
                  storeRemote: next !== "extract",
                  reviewBeforeStorage: next === "draft"
                }))
              }}
            >
              <option value="media">{t("quickIngestSettings.saveToMedia", "Save to Media")}</option>
              <option value="draft">
                {t("quickIngestSettings.createDrafts", "Create review drafts")}
              </option>
              <option value="extract">
                {t("quickIngestSettings.extractOnly", "Extract without saving")}
              </option>
            </select>
            <p id={id + "-destination-help"} className="mt-1 text-xs text-text-muted">
              {destination === "draft"
                ? t(
                    "quickIngestSettings.draftHelp",
                    "Review extracted content in this browser before saving it to Media."
                  )
                : destination === "extract"
                  ? t(
                      "quickIngestSettings.extractHelp",
                      "Read the extracted content in the result without creating a Media record."
                    )
                  : t(
                      "quickIngestSettings.mediaHelp",
                      "Completed sources are saved directly to Media."
                    )}
            </p>
          </div>

          <div className="border-y border-border py-1">
            <ToggleRow
              label={t("option:quickIngest.analysisLabel", "Analysis")}
              checked={preset.common.perform_analysis}
              onChange={(value) =>
                updatePreset((current) => ({
                  ...current,
                  common: { ...current.common, perform_analysis: value }
                }))
              }
            />
            {preset.common.perform_analysis && (
              <div className="mb-3">
                <label htmlFor={id + "-provider"} className="mb-1 block font-medium">
                  {t("option:quickIngest.analysisProvider.label", "Analysis provider")}
                </label>
                <AutoComplete
                  id={id + "-provider"}
                  className="w-full [&_.ant-select-placeholder]:!text-text-muted [&_.ant-select-selection-placeholder]:!text-text-muted"
                  size="large"
                  allowClear
                  aria-label={t("option:quickIngest.analysisProvider.label", "Analysis provider")}
                  aria-required="true"
                  aria-invalid={missingProvider}
                  aria-describedby={id + "-provider-help"}
                  placeholder={t(
                    "option:quickIngest.analysisProvider.placeholder",
                    "Choose or enter a provider"
                  )}
                  value={provider}
                  options={Array.from(providerNames, (value) => ({ value }))}
                  filterOption={(query, option) =>
                    (option?.value ?? "").toLowerCase().includes(query.toLowerCase())
                  }
                  onChange={(value) => updateAdvanced("api_name", value)}
                  onBlur={() => {
                    if (provider !== provider.trim()) updateAdvanced("api_name", provider.trim())
                  }}
                />
                <p
                  id={id + "-provider-help"}
                  aria-live="polite"
                  className={"mt-1 text-xs " + (missingProvider ? "text-warn" : "text-text-muted")}
                >
                  {missingProvider
                    ? t(
                        "quickIngestSettings.providerNeeded",
                        "Choose an analysis provider or turn off Analysis before using this preset."
                      )
                    : t(
                        "quickIngestSettings.providerHelp",
                        "Choose a configured provider or enter its name."
                      )}
                </p>
              </div>
            )}
            <ToggleRow
              label={t("option:quickIngest.chunkingLabel", "Chunking")}
              checked={preset.common.perform_chunking}
              onChange={(value) =>
                updatePreset((current) => ({
                  ...current,
                  common: { ...current.common, perform_chunking: value }
                }))
              }
            />
            {destination === "media" && (
              <ToggleRow
                label={t("option:quickIngest.overwriteLabel", "Overwrite existing")}
                checked={preset.common.overwrite_existing}
                onChange={(value) =>
                  updatePreset((current) => ({
                    ...current,
                    common: { ...current.common, overwrite_existing: value }
                  }))
                }
              />
            )}
          </div>

          <fieldset>
            <legend className="font-semibold">
              {t("quickIngestSettings.sourceFormats", "Source formats")}
            </legend>
            <ToggleRow
              label={t("quickIngestSettings.speakers", "Identify speakers in audio")}
              checked={preset.typeDefaults.audio?.diarize ?? false}
              onChange={(value) =>
                updatePreset((current) => ({
                  ...current,
                  typeDefaults: {
                    ...current.typeDefaults,
                    audio: { ...current.typeDefaults.audio, diarize: value }
                  }
                }))
              }
            />
            <ToggleRow
              label={t("quickIngestSettings.scans", "Read text from scanned documents")}
              checked={preset.typeDefaults.document?.ocr ?? false}
              onChange={(value) =>
                updatePreset((current) => ({
                  ...current,
                  typeDefaults: {
                    ...current.typeDefaults,
                    document: { ...current.typeDefaults.document, ocr: value }
                  }
                }))
              }
            />
            <ToggleRow
              label={t("quickIngestSettings.captions", "Use video captions")}
              checked={preset.typeDefaults.video?.captions ?? false}
              onChange={(value) =>
                updatePreset((current) => ({
                  ...current,
                  typeDefaults: {
                    ...current.typeDefaults,
                    video: { ...current.typeDefaults.video, captions: value }
                  }
                }))
              }
            />
          </fieldset>
          <QuickIngestAdvancedSettings
            key={selected}
            values={preset.advancedValues}
            onChange={updateAdvanced}
          />
        </fieldset>
      </section>

      <div className="space-y-2">
        {writeError && (
          <p role="alert" className="text-danger">
            {writeError}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <Button
            size="large"
            className="min-h-11"
            disabled={isLoading || resetPending || !resettable.includes(selected)}
            onClick={(event) =>
              reset(
                [selected],
                t("quickIngestSettings.resetOneNotice", {
                  defaultValue: "{{preset}} reset to defaults.",
                  preset: name
                }),
                event.currentTarget
              )
            }
          >
            {t("quickIngestSettings.resetOne", {
              defaultValue: "Reset {{preset}} preset",
              preset: name
            })}
          </Button>
          <Button
            size="large"
            className="min-h-11"
            disabled={isLoading || resetPending || !resettable.length}
            onClick={(event) =>
              reset(
                resettable,
                t("quickIngestSettings.resetManyNotice", {
                  defaultValue: "{{presets}} reset to defaults.",
                  presets: resettable.map(presetName).join(", ")
                }),
                event.currentTarget
              )
            }
          >
            {t("quickIngestSettings.resetAll", "Reset all presets")}
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-x-3">
          <p role="status" className="text-sm text-text-muted">
            {lastReset?.message ?? restoredMessage}
          </p>
          {canUndo && (
            <Button
              ref={undoButton}
              className="min-h-11"
              disabled={resetPending}
              onClick={undoReset}
            >
              {t("quickIngestSettings.undo", "Undo")}
            </Button>
          )}
        </div>
      </div>

      <section aria-labelledby={id + "-ocr"} className="space-y-2 border-t border-border pt-4">
        <h2 id={id + "-ocr"} className="text-base font-semibold">
          {t("quickIngestSettings.ocrDefaults.title", "OCR defaults")}
        </h2>
        <p className="text-sm text-text-muted">
          {t(
            "quickIngestSettings.sharedOcr",
            "These text recognition settings apply across presets."
          )}
        </p>
        <ToggleRow
          checked={ocrAssets ?? false}
          label={t("generalSettings.settings.enableOcrAssets.label")}
          onChange={setOcrAssets}
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <label htmlFor={id + "-language"}>
            {t("generalSettings.settings.ocrLanguage.label")}
          </label>
          <Select
            id={id + "-language"}
            aria-label={t("generalSettings.settings.ocrLanguage.label")}
            size="large"
            className="w-full sm:w-64"
            onChange={setOcrLanguage}
            options={ocrLanguages}
            showSearch={{ optionFilterProp: ["label", "value"] }}
            value={ocrLanguage}
          />
        </div>
      </section>
    </div>
  )
}

export default QuickIngestSettings
