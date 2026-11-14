import React from "react"
import { useTranslation } from "react-i18next"
import { Collapse, Slider, InputNumber, Switch, Select, Input } from "antd"
import {
  Settings2,
  Thermometer,
  Layers,
  BookText,
  Database,
  Sparkles
} from "lucide-react"
import { useStoreChatModelSettings } from "@/store/model"
import { useMessageOption } from "@/hooks/useMessageOption"
import {
  ParameterPresetsDropdown,
  JsonModeToggle
} from "@/components/Option/Playground/playground-features"

/**
 * ParametersSidebar - Model parameters and RAG settings sidebar
 *
 * Shows all configuration options for:
 * - Model parameters (temperature, top_p, max_tokens, etc.)
 * - RAG settings (search type, top_k, reranking, etc.)
 * - System prompt
 */
export const ParametersSidebar: React.FC = () => {
  const { t } = useTranslation(["playground", "sidepanel", "common"])

  // Model settings from store
  const temperature = useStoreChatModelSettings((s) => s.temperature)
  const setTemperature = useStoreChatModelSettings((s) => s.setTemperature)
  const topP = useStoreChatModelSettings((s) => s.topP)
  const setTopP = useStoreChatModelSettings((s) => s.setTopP)
  const topK = useStoreChatModelSettings((s) => s.topK)
  const setTopK = useStoreChatModelSettings((s) => s.setTopK)
  const frequencyPenalty = useStoreChatModelSettings((s) => s.frequencyPenalty)
  const setFrequencyPenalty = useStoreChatModelSettings((s) => s.setFrequencyPenalty)
  const presencePenalty = useStoreChatModelSettings((s) => s.presencePenalty)
  const setPresencePenalty = useStoreChatModelSettings((s) => s.setPresencePenalty)
  const systemPrompt = useStoreChatModelSettings((s) => s.systemPrompt)
  const setSystemPrompt = useStoreChatModelSettings((s) => s.setSystemPrompt)

  // RAG settings from message option
  const {
    ragSearchMode,
    setRagSearchMode,
    ragTopK,
    setRagTopK,
    ragEnableCitations,
    setRagEnableCitations
  } = useMessageOption({ forceCompareEnabled: true })

  const collapseItems = [
    {
      key: "presets",
      label: (
        <div className="flex items-center gap-2 text-sm font-medium">
          <Sparkles className="h-4 w-4 text-primary" />
          {t("playground:workspace.presets", "Presets")}
        </div>
      ),
      children: (
        <div className="space-y-3">
          <ParameterPresetsDropdown />
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">
              {t("playground:jsonMode.title", "JSON Mode")}
            </span>
            <JsonModeToggle />
          </div>
        </div>
      )
    },
    {
      key: "model-params",
      label: (
        <div className="flex items-center gap-2 text-sm font-medium">
          <Settings2 className="h-4 w-4 text-primary" />
          {t("playground:workspace.modelParams", "Model Parameters")}
        </div>
      ),
      children: (
        <div className="space-y-4">
          {/* Temperature */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 text-xs text-text-muted">
                <Thermometer className="h-3.5 w-3.5" />
                {t("playground:params.temperature", "Temperature")}
              </label>
              <InputNumber
                size="small"
                min={0}
                max={2}
                step={0.1}
                value={temperature ?? 0.7}
                onChange={(v) => setTemperature(v ?? 0.7)}
                className="w-16"
              />
            </div>
            <Slider
              min={0}
              max={2}
              step={0.1}
              value={temperature ?? 0.7}
              onChange={setTemperature}
              tooltip={{ formatter: (v) => v?.toFixed(1) }}
            />
          </div>

          {/* Top P */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 text-xs text-text-muted">
                {t("playground:params.topP", "Top P")}
              </label>
              <InputNumber
                size="small"
                min={0}
                max={1}
                step={0.05}
                value={topP ?? 0.9}
                onChange={(v) => setTopP(v ?? 0.9)}
                className="w-16"
              />
            </div>
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={topP ?? 0.9}
              onChange={setTopP}
            />
          </div>

          {/* Top K */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 text-xs text-text-muted">
                {t("playground:params.topK", "Top K")}
              </label>
              <InputNumber
                size="small"
                min={1}
                max={100}
                step={1}
                value={topK ?? 40}
                onChange={(v) => setTopK(v ?? 40)}
                className="w-16"
              />
            </div>
            <Slider min={1} max={100} step={1} value={topK ?? 40} onChange={setTopK} />
          </div>

          {/* Frequency Penalty */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs text-text-muted">
                {t("playground:params.frequencyPenalty", "Frequency Penalty")}
              </label>
              <InputNumber
                size="small"
                min={-2}
                max={2}
                step={0.1}
                value={frequencyPenalty ?? 0}
                onChange={(v) => setFrequencyPenalty(v ?? 0)}
                className="w-16"
              />
            </div>
            <Slider
              min={-2}
              max={2}
              step={0.1}
              value={frequencyPenalty ?? 0}
              onChange={setFrequencyPenalty}
            />
          </div>

          {/* Presence Penalty */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs text-text-muted">
                {t("playground:params.presencePenalty", "Presence Penalty")}
              </label>
              <InputNumber
                size="small"
                min={-2}
                max={2}
                step={0.1}
                value={presencePenalty ?? 0}
                onChange={(v) => setPresencePenalty(v ?? 0)}
                className="w-16"
              />
            </div>
            <Slider
              min={-2}
              max={2}
              step={0.1}
              value={presencePenalty ?? 0}
              onChange={setPresencePenalty}
            />
          </div>
        </div>
      )
    },
    {
      key: "system-prompt",
      label: (
        <div className="flex items-center gap-2 text-sm font-medium">
          <BookText className="h-4 w-4 text-primary" />
          {t("playground:workspace.systemPrompt", "System Prompt")}
        </div>
      ),
      children: (
        <div className="space-y-2">
          <Input.TextArea
            value={systemPrompt || ""}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder={t(
              "playground:workspace.systemPromptPlaceholder",
              "Enter a system prompt to guide the model's behavior..."
            )}
            autoSize={{ minRows: 4, maxRows: 12 }}
            className="text-sm"
          />
          <p className="text-[10px] text-text-muted">
            {t(
              "playground:workspace.systemPromptHint",
              "The system prompt provides context and instructions that shape how the model responds."
            )}
          </p>
        </div>
      )
    },
    {
      key: "rag-settings",
      label: (
        <div className="flex items-center gap-2 text-sm font-medium">
          <Database className="h-4 w-4 text-primary" />
          {t("playground:workspace.ragSettings", "RAG Settings")}
        </div>
      ),
      children: (
        <div className="space-y-4">
          {/* Search Type */}
          <div className="space-y-1">
            <label className="text-xs text-text-muted">
              {t("sidepanel:rag.settings.searchType", "Search Type")}
            </label>
            <Select
              size="small"
              value={ragSearchMode || "hybrid"}
              onChange={(v) => setRagSearchMode(v as "hybrid" | "vector" | "fts")}
              options={[
                { value: "hybrid", label: "Hybrid" },
                { value: "vector", label: "Vector" },
                { value: "fts", label: "Full-text" }
              ]}
              className="w-full"
            />
          </div>

          {/* Top K Results */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs text-text-muted">
                {t("sidepanel:rag.settings.topK", "Top K Results")}
              </label>
              <InputNumber
                size="small"
                min={1}
                max={50}
                value={ragTopK ?? 10}
                onChange={(v) => setRagTopK(v ?? 10)}
                className="w-16"
              />
            </div>
          </div>

          <p className="text-xs text-text-muted">
            {t(
              "playground:workspace.chatRetrievalGeneration",
              "Chat retrieves sources, then the selected model generates one answer."
            )}
          </p>

          {/* Include Citations */}
          <div className="flex items-center justify-between">
            <label className="text-xs text-text-muted">
              {t("sidepanel:rag.settings.citations", "Include Citations")}
            </label>
            <Switch
              size="small"
              checked={ragEnableCitations}
              onChange={setRagEnableCitations}
            />
          </div>
        </div>
      )
    }
  ]

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Layers className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-text">
          {t("playground:workspace.configuration", "Configuration")}
        </h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <Collapse
          defaultActiveKey={["presets", "model-params"]}
          ghost
          items={collapseItems}
          className="workspace-sidebar-collapse"
        />
      </div>
    </div>
  )
}
