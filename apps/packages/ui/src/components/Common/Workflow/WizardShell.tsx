import React, { useCallback, useMemo } from "react"
import { Steps, Card, Button, Progress } from "antd"
import { ArrowLeft, ArrowRight, X, Loader2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { useWorkflowsStore } from "@/store/workflows"
import type { WizardShellProps } from "@/types/workflows"

/**
 * WizardShell
 *
 * A reusable wizard container that provides:
 * - Step indicator (Ant Design Steps)
 * - Navigation (back/next/cancel)
 * - Processing state with progress bar
 * - Error handling
 *
 * The actual step content is rendered via children.
 */
export const WizardShell: React.FC<WizardShellProps> = ({
  workflow,
  children,
  canAdvance = true,
  onComplete
}) => {
  const { t } = useTranslation(["workflows", "common"])

  // Store state
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const isProcessing = useWorkflowsStore((s) => s.isProcessing)
  const processingProgress = useWorkflowsStore((s) => s.processingProgress)
  const processingMessage = useWorkflowsStore((s) => s.processingMessage)
  const error = useWorkflowsStore((s) => s.error)

  // Store actions
  const setWorkflowStep = useWorkflowsStore((s) => s.setWorkflowStep)
  const cancelWorkflow = useWorkflowsStore((s) => s.cancelWorkflow)
  const completeWorkflow = useWorkflowsStore((s) => s.completeWorkflow)
  const setError = useWorkflowsStore((s) => s.setError)

  const currentStepIndex = activeWorkflow?.currentStepIndex ?? 0
  const totalSteps = workflow.steps.length

  // Step configuration with translations
  const steps = useMemo(
    () =>
      workflow.steps.map((step) => ({
        key: step.id,
        title: t(step.labelToken, step.id),
        description: step.descriptionToken
          ? t(step.descriptionToken, "")
          : undefined
      })),
    [workflow.steps, t]
  )

  // Navigation helpers
  const canGoBack = currentStepIndex > 0 && !isProcessing
  const canGoNext =
    canAdvance && currentStepIndex < totalSteps - 1 && !isProcessing
  const isLastStep = currentStepIndex === totalSteps - 1

  const goNext = useCallback(() => {
    if (currentStepIndex < totalSteps - 1) {
      setWorkflowStep(currentStepIndex + 1)
      setError(null)
    }
  }, [currentStepIndex, totalSteps, setWorkflowStep, setError])

  const goBack = useCallback(() => {
    if (currentStepIndex > 0) {
      setWorkflowStep(currentStepIndex - 1)
      setError(null)
    }
  }, [currentStepIndex, setWorkflowStep, setError])

  const handleComplete = useCallback(() => {
    if (onComplete) {
      onComplete()
    } else {
      completeWorkflow()
    }
  }, [onComplete, completeWorkflow])

  const handleCancel = useCallback(() => {
    cancelWorkflow()
  }, [cancelWorkflow])

  return (
    <div className="flex flex-col h-full">
      {/* Header with title and close button */}
      <div className="flex items-center justify-between mb-4 px-1">
        <h2 className="text-lg font-semibold text-text">
          {t(workflow.labelToken, workflow.id)}
        </h2>
        <Button
          type="text"
          size="small"
          icon={<X className="h-4 w-4" />}
          onClick={handleCancel}
          disabled={isProcessing}
          aria-label={t("common:cancel", "Cancel")}
        />
      </div>

      {/* Steps indicator */}
      <Steps
        current={currentStepIndex}
        size="small"
        className="mb-4"
        items={steps.map((step) => ({
          key: step.key,
          title: step.title
        }))}
      />

      {/* Error alert */}
      {error && (
        <DesignSystemAlert
          title={t("workflows:error", "Error")}
          variant="error"
          dismissible
          dismissLabel={t(
            "workflows:dismissWorkflowError",
            "Dismiss workflow error"
          )}
          onDismiss={() => setError(null)}
          className="mb-4"
        >
          {error}
        </DesignSystemAlert>
      )}

      {/* Processing indicator */}
      {isProcessing && (
        <div className="mb-4 px-1">
          <div className="flex items-center gap-2 mb-2">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span className="text-sm text-text-muted">
              {processingMessage || t("workflows:processing", "Processing...")}
            </span>
          </div>
          <Progress
            percent={processingProgress}
            size="small"
            showInfo={false}
            strokeColor="var(--color-primary)"
          />
        </div>
      )}

      {/* Step content */}
      <Card className="flex-1 overflow-auto mb-4">{children}</Card>

      {/* Navigation buttons */}
      <div className="flex items-center justify-between">
        <div>
          {canGoBack && (
            <Button icon={<ArrowLeft className="h-4 w-4" />} onClick={goBack}>
              {t("common:back", "Back")}
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={handleCancel} disabled={isProcessing}>
            {t("common:cancel", "Cancel")}
          </Button>
          {isLastStep ? (
            <Button
              type="primary"
              onClick={handleComplete}
              disabled={!canAdvance || isProcessing}
              loading={isProcessing}
            >
              {t("workflows:complete", "Complete")}
            </Button>
          ) : (
            <Button
              type="primary"
              onClick={goNext}
              disabled={!canGoNext}
              loading={isProcessing}
            >
              {t("common:next", "Next")}
              <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
