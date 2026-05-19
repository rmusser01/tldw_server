/**
 * Panel for generating a complete character from a concept
 */

import React from "react"
import { Button, Input, Alert, Dropdown, Spin, Modal, Progress } from "antd"
import type { MenuProps } from "antd"
import { Sparkles, ChevronDown, X, Check, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { fetchChatModels } from "@/services/tldw-server"
import { useStorage } from "@plasmohq/storage/hook"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { resolveStartupSelectedModel } from "@/utils/model-startup-selection"
import { ProviderIcons } from "@/components/Common/ProviderIcon"
import type { GeneratedCharacter } from "@/services/character-generation"

type ChatModel = {
  model: string
  nickname?: string
  provider?: string
  details?: {
    capabilities?: string[]
  }
}

/**
 * Generation progress steps
 */
type GenerationStep = 'analyzing' | 'generating' | 'finalizing'

const GENERATION_STEPS: GenerationStep[] = ['analyzing', 'generating', 'finalizing']

/**
 * Map error codes/patterns to user-friendly messages with actionable guidance
 */
function getErrorMessage(error: string, t: (key: string, opts?: any) => string): { message: string; action?: string } {
  const errorLower = error.toLowerCase()

  // Timeout errors
  if (errorLower.includes('timeout') || errorLower.includes('timed out') || errorLower.includes('took too long')) {
    return {
      message: t("settings:manageCharacters.generate.errors.timeout", {
        defaultValue: "Generation took too long to complete."
      }),
      action: t("settings:manageCharacters.generate.errors.timeoutAction", {
        defaultValue: "Try a simpler concept or check your connection."
      })
    }
  }

  // Auth/API key errors
  if (errorLower.includes('401') || errorLower.includes('unauthorized') || errorLower.includes('api key') || errorLower.includes('authentication')) {
    return {
      message: t("settings:manageCharacters.generate.errors.auth", {
        defaultValue: "Model access denied."
      }),
      action: t("settings:manageCharacters.generate.errors.authAction", {
        defaultValue: "Check your API key in settings."
      })
    }
  }

  // Rate limit errors
  if (errorLower.includes('429') || errorLower.includes('rate limit') || errorLower.includes('too many requests') || errorLower.includes('quota')) {
    return {
      message: t("settings:manageCharacters.generate.errors.rateLimit", {
        defaultValue: "Too many requests."
      }),
      action: t("settings:manageCharacters.generate.errors.rateLimitAction", {
        defaultValue: "Please wait 30 seconds and try again."
      })
    }
  }

  // Model unavailable
  if (errorLower.includes('model') && (errorLower.includes('not found') || errorLower.includes('unavailable') || errorLower.includes('does not exist'))) {
    return {
      message: t("settings:manageCharacters.generate.errors.modelUnavailable", {
        defaultValue: "Selected model is unavailable."
      }),
      action: t("settings:manageCharacters.generate.errors.modelUnavailableAction", {
        defaultValue: "Try selecting a different model."
      })
    }
  }

  // Network errors
  if (errorLower.includes('network') || errorLower.includes('connection') || errorLower.includes('fetch') || errorLower.includes('econnrefused')) {
    return {
      message: t("settings:manageCharacters.generate.errors.network", {
        defaultValue: "Network connection error."
      }),
      action: t("settings:manageCharacters.generate.errors.networkAction", {
        defaultValue: "Check your internet connection and try again."
      })
    }
  }

  // Content filter/safety
  if (errorLower.includes('content') && (errorLower.includes('filter') || errorLower.includes('policy') || errorLower.includes('safety'))) {
    return {
      message: t("settings:manageCharacters.generate.errors.contentFilter", {
        defaultValue: "Content was filtered by the model."
      }),
      action: t("settings:manageCharacters.generate.errors.contentFilterAction", {
        defaultValue: "Try rephrasing your concept."
      })
    }
  }

  // Generic error with original message
  return {
    message: error,
    action: t("settings:manageCharacters.generate.errors.genericAction", {
      defaultValue: "Please try again or select a different model."
    })
  }
}

interface GenerateCharacterPanelProps {
  /** Whether generation is in progress */
  isGenerating: boolean
  /** Error message if any */
  error: string | null
  /** Callback when user wants to generate a character */
  onGenerate: (concept: string, model: string, apiProvider?: string) => void
  /** Callback to cancel generation */
  onCancel: () => void
  /** Callback to clear error */
  onClearError: () => void
  /** Callback when generation produces a result (for preview modal) */
  onPreviewResult?: (result: GeneratedCharacter) => void
}

export const GenerateCharacterPanel: React.FC<GenerateCharacterPanelProps> = ({
  isGenerating,
  error,
  onGenerate,
  onCancel,
  onClearError
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [concept, setConcept] = React.useState("")
  const [selectedModel, setSelectedModel, selectedModelMeta] = useStorage<string | null>(
    "characterGenModel",
    null
  )

  // Progress tracking for generation steps
  const [currentStep, setCurrentStep] = React.useState<GenerationStep>('analyzing')
  const [stepProgress, setStepProgress] = React.useState(0)
  const progressIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(null)

  // Simulate progress through steps while generating
  React.useEffect(() => {
    if (isGenerating) {
      setCurrentStep('analyzing')
      setStepProgress(0)

      let elapsed = 0
      progressIntervalRef.current = setInterval(() => {
        elapsed += 500

        // Progress through steps based on time
        if (elapsed < 3000) {
          setCurrentStep('analyzing')
          setStepProgress(Math.min(33, (elapsed / 3000) * 33))
        } else if (elapsed < 8000) {
          setCurrentStep('generating')
          setStepProgress(33 + Math.min(34, ((elapsed - 3000) / 5000) * 34))
        } else {
          setCurrentStep('finalizing')
          setStepProgress(67 + Math.min(28, ((elapsed - 8000) / 4000) * 28)) // Cap at 95%
        }
      }, 500)

      return () => {
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current)
        }
      }
    } else {
      // Reset on completion
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current)
      }
      setStepProgress(0)
    }
  }, [isGenerating])

  // Fetch available models
  const { data: models, isLoading: modelsLoading } = useQuery<ChatModel[]>({
    queryKey: ["getAllModelsForGeneration"],
    queryFn: () => fetchChatModels({ returnEmpty: true })
  })

  const hasModels = Array.isArray(models) && models.length > 0
  const selectedModelData = React.useMemo(() => {
    if (!selectedModel || !models) return null
    return models.find((m) => m.model === selectedModel)
  }, [selectedModel, models])

  // Auto-select first model if none selected
  React.useEffect(() => {
    const nextModel = resolveStartupSelectedModel({
      currentModel: selectedModel,
      models,
      isCurrentModelHydrating: selectedModelMeta.isLoading
    })
    if (nextModel) {
      setSelectedModel(nextModel)
    }
  }, [models, selectedModel, selectedModelMeta.isLoading, setSelectedModel])

  const handleGenerate = () => {
    if (!concept.trim()) return
    if (!selectedModel) return
    const provider = selectedModelData?.provider
    onGenerate(concept.trim(), selectedModel, provider)
  }

  const handleRetry = () => {
    onClearError()
    // Refocus the input for retry
    const input = document.querySelector('textarea[placeholder*="Describe"]') as HTMLTextAreaElement
    input?.focus()
  }

  const modelMenuItems = React.useMemo<NonNullable<MenuProps["items"]>>(() => {
    if (!hasModels || !models) return []

    // Group models by provider
    const groups = new Map<string, ChatModel[]>()
    for (const model of models) {
      const provider = model.provider || "other"
      if (!groups.has(provider)) groups.set(provider, [])
      groups.get(provider)!.push(model)
    }

    const items: NonNullable<MenuProps["items"]> = []

    for (const [provider, providerModels] of groups.entries()) {
      const providerLabel = getProviderDisplayName(provider)
      items.push({
        type: "group",
        key: `group-${provider}`,
        label: (
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-text-subtle">
            <ProviderIcons provider={provider} className="h-3 w-3" />
            <span>{providerLabel}</span>
          </div>
        ),
        children: providerModels.map((model) => ({
          key: model.model,
          label: (
            <div className="flex items-center justify-between gap-2">
              <span className="truncate">{model.nickname || model.model}</span>
              {selectedModel === model.model && (
                <Check className="h-4 w-4 text-primary flex-shrink-0" />
              )}
            </div>
          ),
          onClick: () => setSelectedModel(model.model)
        }))
      })
    }

    return items
  }, [hasModels, models, selectedModel, setSelectedModel])

  const selectedModelDisplay = React.useMemo(() => {
    if (!selectedModelData) {
      return t("settings:manageCharacters.generate.selectModel", {
        defaultValue: "Select model"
      })
    }
    const provider = getProviderDisplayName(selectedModelData.provider || "")
    const name = selectedModelData.nickname || selectedModelData.model
    const shortName =
      name.length > 25 ? name.substring(0, 23) + "..." : name
    return `${provider} - ${shortName}`
  }, [selectedModelData, t])

  // Get step display text
  const getStepText = (step: GenerationStep) => {
    switch (step) {
      case 'analyzing':
        return t("settings:manageCharacters.generate.steps.analyzing", {
          defaultValue: "Analyzing concept..."
        })
      case 'generating':
        return t("settings:manageCharacters.generate.steps.generating", {
          defaultValue: "Generating fields..."
        })
      case 'finalizing':
        return t("settings:manageCharacters.generate.steps.finalizing", {
          defaultValue: "Finalizing character..."
        })
    }
  }

  // Parse error message
  const parsedError = error ? getErrorMessage(error, t) : null

  if (modelsLoading) {
    return (
      <div className="flex items-center gap-2 text-text-subtle text-sm">
        <Spin size="small" />
        <span>
          {t("settings:manageCharacters.generate.loadingModels", {
            defaultValue: "Loading models..."
          })}
        </span>
      </div>
    )
  }

  if (!hasModels) {
    return (
      <Alert
        type="info"
        showIcon
        title={t("settings:manageCharacters.generate.noModelsTitle", {
          defaultValue: "No AI generation model available"
        })}
        description={
          <span>
            {t("settings:manageCharacters.generate.noModelsDesc", {
              defaultValue:
                "AI character generation needs a configured LLM model. Saved characters remain available for browsing, editing, and chat once a chat model is configured."
            })}{" "}
            <Link to="/settings/model" className="text-primary hover:underline">
              {t("settings:manageCharacters.generate.goToSettings", {
                defaultValue: "Go to model settings"
              })}
            </Link>
          </span>
        }
        className="mb-4"
      />
    )
  }

  return (
    <div
      className="mb-4 rounded-lg border border-dashed border-primary/30 bg-primary/5 p-3 space-y-3"
      aria-busy={isGenerating}
      aria-live="polite"
    >
      <div className="flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-primary" aria-hidden="true" />
        <span className="text-sm font-medium text-text">
          {t("settings:manageCharacters.generate.title", {
            defaultValue: "Generate character with AI"
          })}
        </span>
      </div>

      {/* Error display with actionable guidance */}
      {parsedError && (
        <Alert
          type="error"
          showIcon
          title={parsedError.message}
          description={
            <div className="flex items-center justify-between gap-2 mt-1">
              <span className="text-sm">{parsedError.action}</span>
              <Button
                size="small"
                icon={<RefreshCw className="w-3 h-3" />}
                onClick={handleRetry}>
                {t("common:tryAgain", { defaultValue: "Try again" })}
              </Button>
            </div>
          }
          closable
          onClose={onClearError}
          className="mb-2"
        />
      )}

      {/* Generation progress indicator */}
      {isGenerating && (
        <div className="space-y-2" role="status" aria-label={getStepText(currentStep)}>
          <div className="flex items-center gap-3">
            <Progress
              percent={Math.round(stepProgress)}
              showInfo={false}
              size="small"
              strokeColor={{
                '0%': "rgb(var(--color-primary))",
                '100%': "rgb(var(--color-accent))"
              }}
              className="flex-1"
            />
            <span className="text-xs text-text-subtle whitespace-nowrap">
              {Math.round(stepProgress)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            {GENERATION_STEPS.map((step, index) => {
              const isActive = step === currentStep
              const isComplete = GENERATION_STEPS.indexOf(currentStep) > index
              return (
                <div
                  key={step}
                  className={`flex items-center gap-1.5 text-xs ${
                    isActive
                      ? 'text-primary font-medium'
                      : isComplete
                        ? 'text-success'
                        : 'text-text-subtle'
                  }`}
                >
                  {isComplete ? (
                    <Check className="w-3 h-3" aria-hidden="true" />
                  ) : isActive ? (
                    <Spin size="small" />
                  ) : (
                    <span className="w-3 h-3 rounded-full border border-current opacity-50" aria-hidden="true" />
                  )}
                  <span>{getStepText(step).replace('...', '')}</span>
                  {index < GENERATION_STEPS.length - 1 && (
                    <span className="text-text-subtle mx-1" aria-hidden="true">→</span>
                  )}
                </div>
              )
            })}
          </div>
          {/* Screen reader announcement */}
          <div className="sr-only" aria-live="assertive">
            {t("settings:manageCharacters.generate.aria.generating", {
              defaultValue: "Generating character. {{step}}",
              step: getStepText(currentStep)
            })}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <Input.TextArea
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          placeholder={t("settings:manageCharacters.generate.conceptPlaceholder", {
            defaultValue:
              "Describe your character idea... (e.g., 'a wise old wizard who runs a magical library')"
          })}
          autoSize={{ minRows: 2, maxRows: 4 }}
          disabled={isGenerating}
          aria-label={t("settings:manageCharacters.generate.conceptLabel", {
            defaultValue: "Character concept"
          })}
        />

        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            {/* Model selector */}
            <Dropdown
              menu={{ items: modelMenuItems }}
              trigger={["click"]}
              disabled={isGenerating}
              placement="bottomLeft">
              <Button
                size="small"
                className="flex items-center gap-1 text-xs"
                disabled={isGenerating}
                aria-label={t("settings:manageCharacters.generate.selectModelLabel", {
                  defaultValue: "Select AI model for generation"
                })}>
                {selectedModelData?.provider && (
                  <ProviderIcons
                    provider={selectedModelData.provider}
                    className="h-3 w-3"
                  />
                )}
                <span className="max-w-[180px] truncate">
                  {selectedModelDisplay}
                </span>
                <ChevronDown className="h-3 w-3" aria-hidden="true" />
              </Button>
            </Dropdown>
          </div>

          <div className="flex items-center gap-2">
            {isGenerating && (
              <Button
                size="small"
                danger
                onClick={onCancel}
                icon={<X className="w-3 h-3" />}>
                {t("common:cancel", { defaultValue: "Cancel" })}
              </Button>
            )}
            <Button
              type="primary"
              size="small"
              onClick={handleGenerate}
              disabled={!concept.trim() || !selectedModel || isGenerating}
              loading={isGenerating}
              icon={!isGenerating && <Sparkles className="w-3 h-3" />}>
              {isGenerating
                ? t("settings:manageCharacters.generate.generating", {
                    defaultValue: "Generating..."
                  })
                : t("settings:manageCharacters.generate.generateBtn", {
                    defaultValue: "Generate"
                  })}
            </Button>
          </div>
        </div>
      </div>

      <p className="text-xs text-text-subtle">
        {t("settings:manageCharacters.generate.hint", {
          defaultValue:
            "AI will generate all character fields based on your concept. You can edit the results before saving."
        })}
      </p>
    </div>
  )
}

/**
 * Preview modal for generated character data
 */
interface GenerationPreviewModalProps {
  open: boolean
  generatedData: GeneratedCharacter | null
  onApply: () => void
  onCancel: () => void
  /** Field name if single field, or null for full character */
  fieldName?: string | null
}

export const GenerationPreviewModal: React.FC<GenerationPreviewModalProps> = ({
  open,
  generatedData,
  onApply,
  onCancel,
  fieldName
}) => {
  const { t } = useTranslation(["settings", "common"])

  const title = fieldName
    ? t("settings:manageCharacters.generate.previewFieldTitle", {
        defaultValue: "Generated {{field}}",
        field: fieldName
      })
    : t("settings:manageCharacters.generate.previewTitle", {
        defaultValue: "Generated character"
      })

  const previewContent = React.useMemo(() => {
    if (!generatedData) return null

    if (fieldName && typeof fieldName === "string") {
      const value = (generatedData as any)[fieldName]
      if (Array.isArray(value)) {
        return (
          <ul className="list-disc list-inside text-sm">
            {value.map((v, i) => (
              <li key={i}>{v}</li>
            ))}
          </ul>
        )
      }
      return <p className="text-sm whitespace-pre-wrap">{value}</p>
    }

    // Full character preview
    return (
      <div className="space-y-3 text-sm max-h-[60vh] overflow-y-auto">
        {generatedData.name && (
          <div>
            <strong className="text-text-muted">Name:</strong>{" "}
            {generatedData.name}
          </div>
        )}
        {generatedData.description && (
          <div>
            <strong className="text-text-muted">Description:</strong>
            <p className="mt-1 whitespace-pre-wrap">{generatedData.description}</p>
          </div>
        )}
        {generatedData.personality && (
          <div>
            <strong className="text-text-muted">Personality:</strong>
            <p className="mt-1 whitespace-pre-wrap">{generatedData.personality}</p>
          </div>
        )}
        {generatedData.scenario && (
          <div>
            <strong className="text-text-muted">Scenario:</strong>
            <p className="mt-1 whitespace-pre-wrap">{generatedData.scenario}</p>
          </div>
        )}
        {generatedData.system_prompt && (
          <div>
            <strong className="text-text-muted">System prompt:</strong>
            <p className="mt-1 whitespace-pre-wrap bg-surface2 rounded p-2">
              {generatedData.system_prompt}
            </p>
          </div>
        )}
        {generatedData.first_message && (
          <div>
            <strong className="text-text-muted">First message:</strong>
            <p className="mt-1 whitespace-pre-wrap bg-surface2 rounded p-2">
              {generatedData.first_message}
            </p>
          </div>
        )}
        {generatedData.tags && generatedData.tags.length > 0 && (
          <div>
            <strong className="text-text-muted">Tags:</strong>{" "}
            {generatedData.tags.join(", ")}
          </div>
        )}
      </div>
    )
  }, [generatedData, fieldName])

  return (
    <Modal
      title={title}
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          {t("common:cancel", { defaultValue: "Cancel" })}
        </Button>,
        <Button key="apply" type="primary" onClick={onApply}>
          {t("settings:manageCharacters.generate.applyBtn", {
            defaultValue: "Apply to form"
          })}
        </Button>
      ]}
      destroyOnHidden
      rootClassName="characters-motion-modal"
      aria-describedby="generation-preview-content">
      <div id="generation-preview-content">
        {previewContent}
        {/* Screen reader announcement when modal opens */}
        <div className="sr-only" aria-live="polite">
          {t("settings:manageCharacters.generate.aria.previewReady", {
            defaultValue: "Character generated. Review the details before applying to the form."
          })}
        </div>
      </div>
    </Modal>
  )
}
