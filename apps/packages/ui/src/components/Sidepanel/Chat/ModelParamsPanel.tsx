import { Checkbox, Input, InputNumber, Select, Tooltip } from "antd"
import {
  ChevronDown,
  ChevronUp,
  Settings2,
  AlertCircle,
  Server
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { classNames } from "@/libs/class-name"
import { useStoreChatModelSettings } from "@/store/model"
import { useUiModeStore } from "@/store/ui-mode"
import { LlamaCppAdvancedControls } from "@/components/Common/Settings/LlamaCppAdvancedControls"
import { ModelParamSliderField } from "./ModelParamField"
import {
  useInheritedSamplingDefaults,
  useIsSamplingSetHere
} from "./useInheritedSamplingDefaults"

const { TextArea } = Input

interface ModelParamsPanelProps {
  onOpenFullSettings?: () => void
  className?: string
  selectedModel?: string | null
}

const PROVIDER_OPTIONS = [
  { value: "", label: "Default" },
  { value: "llama.cpp", label: "llama.cpp" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "azure", label: "Azure OpenAI" },
  { value: "custom", label: "Custom" }
]

// Validate JSON string, returns error message or null if valid
const validateJson = (value: string): string | null => {
  if (!value || value.trim() === "") return null
  try {
    JSON.parse(value)
    return null
  } catch (e) {
    return "Invalid JSON format"
  }
}

const ModelParamsPanelBase: React.FC<ModelParamsPanelProps> = ({
  onOpenFullSettings,
  className,
  selectedModel
}) => {
  const { t } = useTranslation(["sidepanel", "playground"])
  const mode = useUiModeStore((state) => state.mode)
  const [paramsCollapsed, setParamsCollapsed] = React.useState(true)
  const [providerCollapsed, setProviderCollapsed] = React.useState(true)
  const [advancedCollapsed, setAdvancedCollapsed] = React.useState(true)
  const [headersError, setHeadersError] = React.useState<string | null>(null)
  const [bodyError, setBodyError] = React.useState<string | null>(null)

  // Basic params
  const temperature = useStoreChatModelSettings((s) => s.temperature)
  const setTemperature = useStoreChatModelSettings((s) => s.setTemperature)
  const topP = useStoreChatModelSettings((s) => s.topP)
  const setTopP = useStoreChatModelSettings((s) => s.setTopP)
  const numPredict = useStoreChatModelSettings((s) => s.numPredict)
  const setNumPredict = useStoreChatModelSettings((s) => s.setNumPredict)
  const seed = useStoreChatModelSettings((s) => s.seed)
  const setSeed = useStoreChatModelSettings((s) => s.setSeed)

  // Additional params
  const frequencyPenalty = useStoreChatModelSettings((s) => s.frequencyPenalty)
  const setFrequencyPenalty = useStoreChatModelSettings(
    (s) => s.setFrequencyPenalty
  )
  const presencePenalty = useStoreChatModelSettings((s) => s.presencePenalty)
  const setPresencePenalty = useStoreChatModelSettings(
    (s) => s.setPresencePenalty
  )
  const topK = useStoreChatModelSettings((s) => s.topK)
  const updateSetting = useStoreChatModelSettings((s) => s.updateSetting)
  // Resetting clears this model's value. An unset Temperature, Top P or
  // penalty is then sent as `inherited` (saved defaults, else ChatTldw's
  // fallback), and an unset Min P is not sent at all.
  const resetSetting = (
    key: "temperature" | "topP" | "frequencyPenalty" | "presencePenalty" | "minP"
  ) => updateSetting(key, undefined)
  const inherited = useInheritedSamplingDefaults()
  const isSetHere = useIsSamplingSetHere()
  const minP = useStoreChatModelSettings((s) => s.minP)
  const setMinP = useStoreChatModelSettings((s) => s.setMinP)

  // Provider & BYOK params
  const apiProvider = useStoreChatModelSettings((s) => s.apiProvider)
  const setApiProvider = useStoreChatModelSettings((s) => s.setApiProvider)
  const extraHeaders = useStoreChatModelSettings((s) => s.extraHeaders)
  const setExtraHeaders = useStoreChatModelSettings((s) => s.setExtraHeaders)
  const extraBody = useStoreChatModelSettings((s) => s.extraBody)
  const setExtraBody = useStoreChatModelSettings((s) => s.setExtraBody)

  // Response format
  const jsonMode = useStoreChatModelSettings((s) => s.jsonMode)
  const setJsonMode = useStoreChatModelSettings((s) => s.setJsonMode)

  // Only show in Pro mode
  if (mode !== "pro") {
    return null
  }

  const handleExtraHeadersChange = (value: string) => {
    const error = validateJson(value)
    setHeadersError(error)
    setExtraHeaders(value)
  }

  const handleExtraBodyChange = (value: string) => {
    const error = validateJson(value)
    setBodyError(error)
    setExtraBody(value)
  }

  return (
    <div
      className={classNames("border-t border-border bg-surface", className)}
      data-testid="model-params-panel"
    >
      {/* Provider & API Section */}
      <button
        type="button"
        onClick={() => setProviderCollapsed(!providerCollapsed)}
        className="w-full flex items-center justify-between px-3 py-2 text-sm text-text-muted hover:text-text hover:bg-surface2 transition-colors"
        aria-expanded={!providerCollapsed}
        aria-controls="provider-content"
        title={t("sidepanel:modelParams.providerTitle", "Provider & API")}
      >
        <span className="flex items-center gap-2">
          <Server className="size-3.5" />
          {t("sidepanel:modelParams.providerTitle", "Provider & API")}
        </span>
        {providerCollapsed ? (
          <ChevronDown className="size-4" />
        ) : (
          <ChevronUp className="size-4" />
        )}
      </button>

      {!providerCollapsed && (
        <div id="provider-content" className="px-3 pb-3 space-y-3 max-h-48 overflow-y-auto">
          {/* Provider Select */}
          <div className="space-y-1">
            <Tooltip
              title={t(
                'sidepanel:modelParams.providerTooltip',
                'Override the API provider for this conversation. New chats keep this choice with the selected model.'
              )}
            >
              <label className="text-xs text-text-muted cursor-help">
                {t("sidepanel:modelParams.provider", "Provider")}
              </label>
            </Tooltip>
            <Select
              size="small"
              value={apiProvider || ""}
              onChange={(value) => setApiProvider(value)}
              options={PROVIDER_OPTIONS}
              className="w-full"
              placeholder="Default"
            />
          </div>

          {/* Extra Headers */}
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Tooltip
                title={t(
                  "sidepanel:modelParams.extraHeadersTooltip",
                  "Custom HTTP headers as JSON object (e.g., for API keys)"
                )}
              >
                <label className="text-xs text-text-muted cursor-help">
                  {t("sidepanel:modelParams.extraHeaders", "Extra Headers")}
                </label>
              </Tooltip>
              {headersError && (
                <Tooltip title={headersError}>
                  <AlertCircle className="size-3 text-danger" />
                </Tooltip>
              )}
            </div>
            <TextArea
              size="small"
              value={extraHeaders || ""}
              onChange={(e) => handleExtraHeadersChange(e.target.value)}
              placeholder='{"Authorization": "Bearer ..."}'
              rows={2}
              className={classNames(
                "font-mono text-xs",
                headersError && "border-danger"
              )}
            />
          </div>

          {/* Extra Body */}
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Tooltip
                title={t(
                  "sidepanel:modelParams.extraBodyTooltip",
                  "Additional request body parameters as JSON"
                )}
              >
                <label className="text-xs text-text-muted cursor-help">
                  {t("sidepanel:modelParams.extraBody", "Extra Body")}
                </label>
              </Tooltip>
              {bodyError && (
                <Tooltip title={bodyError}>
                  <AlertCircle className="size-3 text-danger" />
                </Tooltip>
              )}
            </div>
            <TextArea
              size="small"
              value={extraBody || ""}
              onChange={(e) => handleExtraBodyChange(e.target.value)}
              placeholder='{"custom_param": "value"}'
              rows={2}
              className={classNames(
                "font-mono text-xs",
                bodyError && "border-danger"
              )}
            />
          </div>

          <LlamaCppAdvancedControls selectedModel={selectedModel} />

          {/* JSON Mode Toggle */}
          <div className="flex items-center justify-between gap-2 pt-2 border-t border-border/50">
            <Tooltip
              title={t(
                "sidepanel:modelParams.jsonModeTooltip",
                "Force the model to output valid JSON. Only works with models that support structured output."
              )}
            >
              <label className="text-xs text-text-muted cursor-help">
                {t("sidepanel:modelParams.jsonMode", "JSON Mode")}
              </label>
            </Tooltip>
            <Checkbox
              checked={jsonMode ?? false}
              onChange={(e) => setJsonMode(e.target.checked)}
            />
          </div>
        </div>
      )}

      {/* Model Parameters Section */}
      <button
        type="button"
        onClick={() => setParamsCollapsed(!paramsCollapsed)}
        className="w-full flex items-center justify-between px-3 py-2 text-sm text-text-muted hover:text-text hover:bg-surface2 transition-colors border-t border-border"
        aria-expanded={!paramsCollapsed}
        aria-controls="model-params-content"
        title={t("sidepanel:modelParams.title", "Model Parameters")}
      >
        <span className="flex items-center gap-2">
          <Settings2 className="size-3.5" />
          {t("sidepanel:modelParams.title", "Model Parameters")}
        </span>
        {paramsCollapsed ? (
          <ChevronDown className="size-4" />
        ) : (
          <ChevronUp className="size-4" />
        )}
      </button>

      {!paramsCollapsed && (
        <div id="model-params-content" className="px-3 pb-3 space-y-3 max-h-64 overflow-y-auto">
          <ModelParamSliderField
            label={t("sidepanel:modelParams.temperature", "Temperature")}
            tooltip={t(
              "sidepanel:modelParams.temperatureTooltip",
              "Controls randomness. Lower = more focused, higher = more creative"
            )}
            value={temperature}
            min={0}
            max={2}
            step={0.1}
            precision={1}
            restingValue={1}
            onChange={(value) => setTemperature(value)}
            defaultValue={inherited.temperature}
            isSetHere={isSetHere("temperature")}
            onReset={() => resetSetting("temperature")}
          />
          <ModelParamSliderField
            label={t("sidepanel:modelParams.topP", "Top P")}
            tooltip={t(
              "sidepanel:modelParams.topPTooltip",
              "Nucleus sampling threshold. Lower = more focused output"
            )}
            value={topP}
            min={0}
            max={1}
            step={0.05}
            precision={2}
            restingValue={1}
            onChange={(value) => setTopP(value)}
            defaultValue={inherited.topP}
            isSetHere={isSetHere("topP")}
            onReset={() => resetSetting("topP")}
          />
          <ModelParamSliderField
            label={t("sidepanel:modelParams.frequencyPenalty", "Frequency Penalty")}
            tooltip={t(
              "sidepanel:modelParams.frequencyPenaltyTooltip",
              "Penalizes repeated tokens based on frequency. Higher = less repetition"
            )}
            value={frequencyPenalty}
            min={-2}
            max={2}
            step={0.1}
            precision={1}
            restingValue={0}
            onChange={(value) => setFrequencyPenalty(value)}
            defaultValue={inherited.frequencyPenalty}
            isSetHere={isSetHere("frequencyPenalty")}
            onReset={() => resetSetting("frequencyPenalty")}
          />
          <ModelParamSliderField
            label={t("sidepanel:modelParams.presencePenalty", "Presence Penalty")}
            tooltip={t(
              "sidepanel:modelParams.presencePenaltyTooltip",
              "Penalizes repeated tokens based on presence. Higher = more topic diversity"
            )}
            value={presencePenalty}
            min={-2}
            max={2}
            step={0.1}
            precision={1}
            restingValue={0}
            onChange={(value) => setPresencePenalty(value)}
            defaultValue={inherited.presencePenalty}
            isSetHere={isSetHere("presencePenalty")}
            onReset={() => resetSetting("presencePenalty")}
          />

          {/* Top K */}
          <div className="flex items-center justify-between gap-2" data-param-field>
            <Tooltip
              title={t(
                "sidepanel:modelParams.topKTooltip",
                "Limits sampling to top K tokens. Lower = more focused"
              )}
            >
              <label className="text-xs text-text-muted cursor-help whitespace-nowrap">
                {t("sidepanel:modelParams.topK", "Top K")}
              </label>
            </Tooltip>
            <InputNumber
              size="small"
              min={1}
              max={100}
              value={topK}
              onChange={(value) => updateSetting("topK", value ?? undefined)}
              placeholder={t("sidepanel:modelParams.default", "Default")}
              className="w-24"
            />
          </div>

          <ModelParamSliderField
            label={t("sidepanel:modelParams.minP", "Min P")}
            tooltip={t(
              "sidepanel:modelParams.minPTooltip",
              "Minimum probability threshold for token selection"
            )}
            value={minP}
            min={0}
            max={1}
            step={0.05}
            precision={2}
            restingValue={0}
            onChange={(value) => setMinP(value)}
            isSetHere={isSetHere("minP")}
            onReset={() => resetSetting("minP")}
          />

          {/* Max Tokens / numPredict */}
          <div className="flex items-center justify-between gap-2">
            <Tooltip
              title={t(
                "sidepanel:modelParams.maxTokensTooltip",
                "Maximum number of tokens to generate"
              )}
            >
              <label className="text-xs text-text-muted cursor-help whitespace-nowrap">
                {t("sidepanel:modelParams.maxTokens", "Max Tokens")}
              </label>
            </Tooltip>
            <InputNumber
              size="small"
              min={1}
              max={128000}
              value={numPredict}
              onChange={(value) => setNumPredict(value ?? undefined)}
              placeholder="Auto"
              className="w-24"
            />
          </div>

          {/* Seed */}
          <div className="flex items-center justify-between gap-2">
            <Tooltip
              title={t(
                "sidepanel:modelParams.seedTooltip",
                "Random seed for reproducible outputs. Leave empty for random."
              )}
            >
              <label className="text-xs text-text-muted cursor-help">
                {t("sidepanel:modelParams.seed", "Seed")}
              </label>
            </Tooltip>
            <InputNumber
              size="small"
              min={0}
              value={seed}
              onChange={(value) => setSeed(value ?? undefined)}
              placeholder="Random"
              className="w-24"
            />
          </div>

          {/* More Settings Link */}
          {onOpenFullSettings && (
            <button
              type="button"
              onClick={onOpenFullSettings}
              className="text-xs text-primary hover:text-primary-hover underline"
              title={t("sidepanel:modelParams.moreSettings", "More settings...")}
            >
              {t("sidepanel:modelParams.moreSettings", "More settings...")}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export const ModelParamsPanel = React.memo(ModelParamsPanelBase)
ModelParamsPanel.displayName = "ModelParamsPanel"
