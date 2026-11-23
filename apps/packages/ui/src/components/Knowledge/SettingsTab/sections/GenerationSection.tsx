import React from "react"
import { useTranslation } from "react-i18next"
import type { TFunction } from "i18next"
import type { RagSettings } from "@/services/rag/unified-rag"
import { SettingField } from "../shared/SettingField"
import { CollapsibleSection } from "../shared/CollapsibleSection"

type GenerationSectionProps = {
  settings: RagSettings
  onUpdate: <K extends keyof RagSettings>(key: K, value: RagSettings[K]) => void
  searchFilter?: string
}

const ABSTENTION_BEHAVIOR_VALUES: Record<
  RagSettings["abstention_behavior"],
  true
> = {
  continue: true,
  ask: true,
  decline: true
}

const isAbstentionBehavior = (
  value: string
): value is RagSettings["abstention_behavior"] =>
  Object.prototype.hasOwnProperty.call(ABSTENTION_BEHAVIOR_VALUES, value)

export const getGenerationSectionVisible = (
  searchFilter: string,
  t: TFunction
) => {
  const normalizedFilter = searchFilter.trim().toLowerCase()
  if (!normalizedFilter) return true
  const abstentionBehaviorOptions = [
    t("sidepanel:rag.abstentionBehaviorContinue", "Continue"),
    t("sidepanel:rag.abstentionBehaviorAskUser", "Ask user"),
    t("sidepanel:rag.abstentionBehaviorDecline", "Decline")
  ]
  const labels = [
    t("sidepanel:rag.generation", "Answer Generation"),
    t("sidepanel:rag.enableGeneration", "Enable generation"),
    t("sidepanel:rag.strictExtractive", "Strict extractive"),
    t("sidepanel:rag.generationModel", "Generation model"),
    t("sidepanel:rag.generationPrompt", "Generation prompt"),
    t("sidepanel:rag.maxGenerationTokens", "Max tokens"),
    t("sidepanel:rag.abstention", "Abstention"),
    t("sidepanel:rag.enableAbstention", "Enable abstention"),
    t("sidepanel:rag.abstentionBehavior", "Behavior"),
    ...abstentionBehaviorOptions,
    "continue",
    "ask",
    "decline",
    t("sidepanel:rag.synthesis", "Multi-turn Synthesis"),
    t("sidepanel:rag.enableSynthesis", "Enable synthesis"),
    t("sidepanel:rag.synthesisTimeBudget", "Time budget (s)"),
    t("sidepanel:rag.synthesisDraftTokens", "Draft tokens"),
    t("sidepanel:rag.synthesisRefineTokens", "Refine tokens")
  ]
  return labels.some((label) => label.toLowerCase().includes(normalizedFilter))
}

/**
 * Generation section - answer generation, abstention, and synthesis
 */
export const GenerationSection: React.FC<GenerationSectionProps> = ({
  settings,
  onUpdate,
  searchFilter = ""
}) => {
  const { t, i18n } = useTranslation(["sidepanel"])
  const generationTitle = t("sidepanel:rag.generation", "Answer Generation")
  const enableGenerationLabel = t(
    "sidepanel:rag.enableGeneration",
    "Enable generation"
  )
  // QA search answers only on request (CC-10); this switch feeds chat's
  // answers from your sources (useKnowledgeSettings -> ragEnableGeneration).
  const enableGenerationHelper = t(
    "sidepanel:rag.enableGenerationHelper",
    "Chat answers from your sources. QA search writes one only when you click Generate answer."
  )
  const strictExtractiveLabel = t(
    "sidepanel:rag.strictExtractive",
    "Strict extractive"
  )
  const strictExtractiveHelper = t(
    "sidepanel:rag.strictExtractiveHelper",
    "Only use verbatim quotes from sources"
  )
  const generationModelLabel = t(
    "sidepanel:rag.generationModel",
    "Generation model"
  )
  const generationModelPlaceholder = t(
    "sidepanel:rag.generationModelPlaceholder",
    "Default"
  )
  const generationPromptLabel = t(
    "sidepanel:rag.generationPrompt",
    "Generation prompt"
  )
  const generationPromptPlaceholder = t(
    "sidepanel:rag.generationPromptPlaceholder",
    "Custom prompt template..."
  )
  const maxGenerationTokensLabel = t(
    "sidepanel:rag.maxGenerationTokens",
    "Max tokens"
  )
  const abstentionLabel = t("sidepanel:rag.abstention", "Abstention")
  const enableAbstentionLabel = t(
    "sidepanel:rag.enableAbstention",
    "Enable abstention"
  )
  const enableAbstentionHelper = t(
    "sidepanel:rag.enableAbstentionHelper",
    "Allow declining when uncertain"
  )
  const abstentionBehaviorLabel = t(
    "sidepanel:rag.abstentionBehavior",
    "Behavior"
  )
  const synthesisLabel = t("sidepanel:rag.synthesis", "Multi-turn Synthesis")
  const enableSynthesisLabel = t(
    "sidepanel:rag.enableSynthesis",
    "Enable synthesis"
  )
  const enableSynthesisHelper = t(
    "sidepanel:rag.enableSynthesisHelper",
    "Draft and refine answers in multiple passes"
  )
  const synthesisTimeBudgetLabel = t(
    "sidepanel:rag.synthesisTimeBudget",
    "Time budget (s)"
  )
  const synthesisDraftTokensLabel = t(
    "sidepanel:rag.synthesisDraftTokens",
    "Draft tokens"
  )
  const synthesisRefineTokensLabel = t(
    "sidepanel:rag.synthesisRefineTokens",
    "Refine tokens"
  )

  const abstentionBehaviorOptions = React.useMemo<
    { label: string; value: RagSettings["abstention_behavior"] }[]
  >(
    () => [
      {
        label: t("sidepanel:rag.abstentionBehaviorContinue", "Continue"),
        value: "continue"
      },
      {
        label: t("sidepanel:rag.abstentionBehaviorAskUser", "Ask user"),
        value: "ask"
      },
      {
        label: t("sidepanel:rag.abstentionBehaviorDecline", "Decline"),
        value: "decline"
      }
    ],
    [t, i18n.language]
  )

  const handleEnableGeneration = React.useCallback(
    (val: boolean) => onUpdate("enable_generation", val),
    [onUpdate]
  )
  const handleStrictExtractive = React.useCallback(
    (val: boolean) => onUpdate("strict_extractive", val),
    [onUpdate]
  )
  const handleGenerationModel = React.useCallback(
    (val: string) => onUpdate("generation_model", val || null),
    [onUpdate]
  )
  const handleGenerationPrompt = React.useCallback(
    (val: string) => onUpdate("generation_prompt", val || null),
    [onUpdate]
  )
  const handleMaxGenerationTokens = React.useCallback(
    (val: number) => onUpdate("max_generation_tokens", val),
    [onUpdate]
  )
  const handleEnableAbstention = React.useCallback(
    (val: boolean) => onUpdate("enable_abstention", val),
    [onUpdate]
  )
  const handleAbstentionBehavior = React.useCallback(
    (val: string) => {
      if (isAbstentionBehavior(val)) {
        onUpdate("abstention_behavior", val)
      }
    },
    [onUpdate]
  )
  const handleEnableSynthesis = React.useCallback(
    (val: boolean) => onUpdate("enable_multi_turn_synthesis", val),
    [onUpdate]
  )
  const handleSynthesisTimeBudget = React.useCallback(
    (val: number) => onUpdate("synthesis_time_budget_sec", val),
    [onUpdate]
  )
  const handleSynthesisDraftTokens = React.useCallback(
    (val: number) => onUpdate("synthesis_draft_tokens", val),
    [onUpdate]
  )
  const handleSynthesisRefineTokens = React.useCallback(
    (val: number) => onUpdate("synthesis_refine_tokens", val),
    [onUpdate]
  )

  const normalizedFilter = searchFilter.trim().toLowerCase()
  const matchesFilter = (label: string) =>
    !normalizedFilter || label.toLowerCase().includes(normalizedFilter)
  const matchesAny = (...labels: string[]) => labels.some(matchesFilter)
  const abstentionFilterTerms = React.useMemo(
    () =>
      abstentionBehaviorOptions.flatMap((option) => [
        option.label,
        option.value
      ]),
    [abstentionBehaviorOptions]
  )

  const sectionVisible = getGenerationSectionVisible(searchFilter, t)
  // The answer fields stay visible with chat generation off: QA search's
  // Generate answer uses them too (CC-10 review).
  const showAbstentionSection =
    matchesAny(
      abstentionLabel,
      enableAbstentionLabel,
      abstentionBehaviorLabel,
      ...abstentionFilterTerms
    )
  const showSynthesisSection =
    matchesAny(
      synthesisLabel,
      enableSynthesisLabel,
      synthesisTimeBudgetLabel,
      synthesisDraftTokensLabel,
      synthesisRefineTokensLabel
    )

  return (
    <CollapsibleSection
      title={generationTitle}
      defaultExpanded={false}
      visible={sectionVisible}
    >
      {/* Enable Generation */}
      {matchesFilter(enableGenerationLabel) && (
        <SettingField
          type="switch"
          label={enableGenerationLabel}
          value={settings.enable_generation}
          onChange={handleEnableGeneration}
          helper={enableGenerationHelper}
        />
      )}

      {/* Strict Extractive */}
      {matchesFilter(strictExtractiveLabel) && (
        <SettingField
          type="switch"
          label={strictExtractiveLabel}
          value={settings.strict_extractive}
          onChange={handleStrictExtractive}
          helper={strictExtractiveHelper}
        />
      )}

      {/* Generation Model */}
      {matchesFilter(generationModelLabel) && (
        <SettingField
          type="text"
          label={generationModelLabel}
          value={settings.generation_model || ""}
          onChange={handleGenerationModel}
          placeholder={generationModelPlaceholder}
        />
      )}

      {/* Generation Prompt */}
      {matchesFilter(generationPromptLabel) && (
        <div className="col-span-2">
          <SettingField
            type="textarea"
            label={generationPromptLabel}
            value={settings.generation_prompt || ""}
            onChange={handleGenerationPrompt}
            placeholder={generationPromptPlaceholder}
            rows={3}
          />
        </div>
      )}

      {/* Max Generation Tokens */}
      {matchesFilter(maxGenerationTokensLabel) && (
        <SettingField
          type="number"
          label={maxGenerationTokensLabel}
          value={settings.max_generation_tokens}
          onChange={handleMaxGenerationTokens}
          min={100}
          max={4000}
          step={100}
        />
      )}

      {/* Abstention Section */}
      {showAbstentionSection && (
        <div className="col-span-2 border-t border-border pt-3 mt-2">
          <span className="text-xs font-medium text-text mb-2 block">
            {abstentionLabel}
          </span>

          <div className="grid gap-3 md:grid-cols-2">
            <SettingField
              type="switch"
              label={enableAbstentionLabel}
              value={settings.enable_abstention}
              onChange={handleEnableAbstention}
              helper={enableAbstentionHelper}
            />

            {settings.enable_abstention && (
              <SettingField
                type="select"
                label={abstentionBehaviorLabel}
                value={settings.abstention_behavior}
                onChange={handleAbstentionBehavior}
                options={abstentionBehaviorOptions}
              />
            )}
          </div>
        </div>
      )}

      {/* Multi-turn Synthesis Section */}
      {showSynthesisSection && (
        <div className="col-span-2 border-t border-border pt-3 mt-2">
          <span className="text-xs font-medium text-text mb-2 block">
            {synthesisLabel}
          </span>

          <div className="grid gap-3 md:grid-cols-2">
            <SettingField
              type="switch"
              label={enableSynthesisLabel}
              value={settings.enable_multi_turn_synthesis}
              onChange={handleEnableSynthesis}
              helper={enableSynthesisHelper}
            />

            {settings.enable_multi_turn_synthesis && (
              <>
                <SettingField
                  type="number"
                  label={synthesisTimeBudgetLabel}
                  value={settings.synthesis_time_budget_sec}
                  onChange={handleSynthesisTimeBudget}
                  min={10}
                  max={180}
                />

                <SettingField
                  type="number"
                  label={synthesisDraftTokensLabel}
                  value={settings.synthesis_draft_tokens}
                  onChange={handleSynthesisDraftTokens}
                  min={100}
                  max={2000}
                  step={100}
                />

                <SettingField
                  type="number"
                  label={synthesisRefineTokensLabel}
                  value={settings.synthesis_refine_tokens}
                  onChange={handleSynthesisRefineTokens}
                  min={100}
                  max={2000}
                  step={100}
                />
              </>
            )}
          </div>
        </div>
      )}
    </CollapsibleSection>
  )
}
