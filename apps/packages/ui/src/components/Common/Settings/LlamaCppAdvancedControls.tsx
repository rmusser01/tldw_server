import React from "react"
import { Button, Input, InputNumber, Select, Typography } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import { useStoreChatModelSettings } from "@/store/model"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  tldwLlamaGrammars,
  type LlamaGrammarRecord
} from "@/services/tldw/TldwLlamaGrammars"
import { LlamaGrammarLibraryModal } from "./LlamaGrammarLibraryModal"

const { TextArea } = Input
const { Text } = Typography

type LlamaCppControlsMetadata = {
  grammar?: {
    supported?: boolean
    effective_reason?: string | null
  }
  thinking_budget?: {
    supported?: boolean
    request_key?: string | null
    effective_reason?: string | null
  }
  reserved_extra_body_keys?: string[]
}

type LlamaGrammarMode = "none" | "library" | "inline"
type LlamaControlField =
  | "llamaThinkingBudgetTokens"
  | "llamaGrammarMode"
  | "llamaGrammarId"
  | "llamaGrammarInline"
  | "llamaGrammarOverride"

type Props = {
  selectedModel?: string | null
  resolvedProvider?: string | null
  className?: string
  thinkingBudget?: number
  grammarMode?: LlamaGrammarMode
  grammarId?: string
  grammarInline?: string
  grammarOverride?: string
  extraBody?: string
  onChange?: (
    key: LlamaControlField,
    value: number | string | undefined
  ) => void
}

const normalizeResolvedProvider = (value: string | null | undefined) => {
  const normalized = String(value || "").trim().toLowerCase()
  if (normalized === "llamacpp") return "llama.cpp"
  return normalized || null
}

const parseJsonRecord = (value: string | null | undefined): Record<string, unknown> | null => {
  if (!value || !value.trim()) return null
  try {
    const parsed = JSON.parse(value)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    return null
  }
  return null
}

const extractLlamaCppControls = (payload: any): LlamaCppControlsMetadata | null => {
  const providers = payload?.providers
  if (!providers) return null
  if (!Array.isArray(providers)) {
    return (
      providers?.["llama.cpp"]?.llama_cpp_controls ??
      providers?.llamacpp?.llama_cpp_controls ??
      null
    )
  }
  const matched = providers.find((entry: any) => {
    const key = String(entry?.provider || entry?.id || entry?.name || "")
      .trim()
      .toLowerCase()
    return key === "llama.cpp" || key === "llamacpp"
  })
  return matched?.llama_cpp_controls ?? null
}

export function LlamaCppAdvancedControls({
  selectedModel,
  resolvedProvider,
  className,
  thinkingBudget,
  grammarMode,
  grammarId,
  grammarInline,
  grammarOverride,
  extraBody,
  onChange
}: Props) {
  const { t } = useTranslation(["common", "sidepanel"])
  const apiProviderOverride = useStoreChatModelSettings((state) => state.apiProvider)
  const updateSetting = useStoreChatModelSettings((state) => state.updateSetting)
  const storeThinkingBudget = useStoreChatModelSettings(
    (state) => state.llamaThinkingBudgetTokens
  )
  const storeGrammarMode = useStoreChatModelSettings((state) => state.llamaGrammarMode)
  const storeGrammarId = useStoreChatModelSettings((state) => state.llamaGrammarId)
  const storeGrammarInline = useStoreChatModelSettings((state) => state.llamaGrammarInline)
  const storeGrammarOverride = useStoreChatModelSettings(
    (state) => state.llamaGrammarOverride
  )
  const storeExtraBody = useStoreChatModelSettings((state) => state.extraBody)
  const [libraryOpen, setLibraryOpen] = React.useState(false)
  const [derivedProvider, setDerivedProvider] = React.useState<string | null>(
    normalizeResolvedProvider(resolvedProvider)
  )
  const useControlledValues = typeof onChange === "function"
  const currentThinkingBudget = useControlledValues
    ? thinkingBudget
    : storeThinkingBudget
  const currentGrammarMode = useControlledValues ? grammarMode : storeGrammarMode
  const currentGrammarId = useControlledValues ? grammarId : storeGrammarId
  const currentGrammarInline = useControlledValues
    ? grammarInline
    : storeGrammarInline
  const currentGrammarOverride = useControlledValues
    ? grammarOverride
    : storeGrammarOverride
  const currentExtraBody = useControlledValues ? extraBody : storeExtraBody

  const updateLlamaField = React.useCallback(
    (key: LlamaControlField, value: number | string | undefined) => {
      if (onChange) {
        onChange(key, value)
        return
      }
      updateSetting(key, value as never)
    },
    [onChange, updateSetting]
  )

  React.useEffect(() => {
    const direct = normalizeResolvedProvider(resolvedProvider)
    if (direct) {
      setDerivedProvider(direct)
      return
    }

    let cancelled = false
    void resolveApiProviderForModel({
      modelId: selectedModel,
      explicitProvider: apiProviderOverride
    }).then((nextProvider) => {
      if (!cancelled) {
        setDerivedProvider(normalizeResolvedProvider(nextProvider))
      }
    })

    return () => {
      cancelled = true
    }
  }, [apiProviderOverride, resolvedProvider, selectedModel])

  const isLlamaCpp = derivedProvider === "llama.cpp"

  const providersQuery = useQuery({
    queryKey: ["tldw:llm-providers"],
    queryFn: () => tldwClient.getLlmProviders(),
    enabled: isLlamaCpp
  })

  const grammarsQuery = useQuery({
    queryKey: ["tldw:llama-grammars"],
    queryFn: () => tldwLlamaGrammars.list(),
    enabled: isLlamaCpp
  })

  const controls = extractLlamaCppControls(providersQuery.data)
  const grammarSupported = controls?.grammar?.supported !== false
  const thinkingSupported = controls?.thinking_budget?.supported === true
  const conflictingExtraBodyKeys = React.useMemo(() => {
    const reservedKeys = controls?.reserved_extra_body_keys ?? []
    if (!reservedKeys.length) return []
    const parsedExtraBody = parseJsonRecord(currentExtraBody)
    if (!parsedExtraBody) return []
    return reservedKeys.filter((key) =>
      Object.prototype.hasOwnProperty.call(parsedExtraBody, key)
    )
  }, [controls?.reserved_extra_body_keys, currentExtraBody])
  const grammarOptions =
    grammarsQuery.data?.items?.map((item: LlamaGrammarRecord) => ({
      label: item.name,
      value: item.id
    })) ?? []

  if (!isLlamaCpp) {
    return null
  }

  return (
    <div className={className}>
      <div className="rounded-xl border border-border/70 bg-surface2/60 p-3 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="space-y-0.5">
            <Text strong>
              {t("sidepanel:llamaControls.title", "llama.cpp advanced controls")}
            </Text>
            <div className="text-xs text-text-muted">
              {t(
                "sidepanel:llamaControls.subtitle",
                "Grammar-constrained output and optional reasoning budget."
              )}
            </div>
          </div>
          <Button size="small" onClick={() => setLibraryOpen(true)}>
            {t("sidepanel:llamaControls.manageGrammars", "Manage grammars")}
          </Button>
        </div>

        {!grammarSupported && controls?.grammar?.effective_reason ? (
          <Alert
            variant="info"
            title={controls.grammar.effective_reason}
          />
        ) : null}

        {conflictingExtraBodyKeys.length > 0 ? (
          <Alert
            variant="warning"
            title={t(
              "sidepanel:llamaControls.extraBodyConflictTitle",
              "First-class llama.cpp controls override reserved raw extra body keys."
            )}
          >
            {conflictingExtraBodyKeys.join(", ")}
          </Alert>
        ) : null}

        <div className="space-y-1">
          <label className="text-xs text-text-muted">
            {t("sidepanel:llamaControls.thinkingBudget", "Thinking budget")}
          </label>
          <InputNumber
            min={0}
            disabled={!thinkingSupported}
            value={currentThinkingBudget}
            onChange={(value) =>
              updateLlamaField(
                "llamaThinkingBudgetTokens",
                typeof value === "number" ? value : undefined
              )
            }
            className="w-full"
            placeholder={t(
              "sidepanel:llamaControls.thinkingBudgetPlaceholder",
              "Disabled for this deployment"
            )}
          />
          {!thinkingSupported && controls?.thinking_budget?.effective_reason ? (
            <div className="text-xs text-text-muted">
              {controls.thinking_budget.effective_reason}
            </div>
          ) : null}
        </div>

        <div className="space-y-1">
          <label className="text-xs text-text-muted">
            {t("sidepanel:llamaControls.grammarSource", "Grammar source")}
          </label>
          <Select
            value={currentGrammarMode || "none"}
            onChange={(value: LlamaGrammarMode) => {
              updateLlamaField(
                "llamaGrammarMode",
                value === "none" ? undefined : value
              )
              if (value === "none") {
                updateLlamaField("llamaGrammarId", undefined)
                updateLlamaField("llamaGrammarInline", undefined)
                updateLlamaField("llamaGrammarOverride", undefined)
              }
              if (value === "inline") {
                updateLlamaField("llamaGrammarId", undefined)
              }
              if (value === "library") {
                updateLlamaField("llamaGrammarInline", undefined)
              }
            }}
            options={[
              { label: t("common:none", "None"), value: "none" },
              { label: t("sidepanel:llamaControls.savedGrammar", "Saved grammar"), value: "library" },
              { label: t("sidepanel:llamaControls.inlineGrammar", "Inline grammar"), value: "inline" }
            ]}
          />
        </div>

        {currentGrammarMode === "library" ? (
          <div className="space-y-2">
            <Select
              value={currentGrammarId}
              options={grammarOptions}
              onChange={(value) => updateLlamaField("llamaGrammarId", value)}
              placeholder={t(
                "sidepanel:llamaControls.selectSavedGrammar",
                "Select a saved grammar"
              )}
              loading={grammarsQuery.isLoading}
              allowClear
            />
            <TextArea
              value={currentGrammarOverride || ""}
              onChange={(event) =>
                updateLlamaField(
                  "llamaGrammarOverride",
                  event.target.value || undefined
                )
              }
              rows={5}
              className="font-mono text-xs"
              placeholder={t(
                "sidepanel:llamaControls.overridePlaceholder",
                "Optional per-chat override"
              )}
            />
          </div>
        ) : null}

        {currentGrammarMode === "inline" ? (
          <TextArea
            value={currentGrammarInline || ""}
            onChange={(event) =>
              updateLlamaField(
                "llamaGrammarInline",
                event.target.value || undefined
              )
            }
            rows={8}
            className="font-mono text-xs"
            placeholder={'root ::= "ok"'}
          />
        ) : null}
      </div>

      <LlamaGrammarLibraryModal
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        selectedGrammarId={currentGrammarId}
        onSelectGrammar={(grammar) => {
          updateLlamaField("llamaGrammarMode", "library")
          updateLlamaField("llamaGrammarId", grammar.id)
          setLibraryOpen(false)
        }}
      />
    </div>
  )
}
