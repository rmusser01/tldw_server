import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { Check, Circle, Loader2 } from "lucide-react"
import type { WizardStep } from "./types"
import { useIngestWizard } from "./IngestWizardContext"

// ---------------------------------------------------------------------------
// Step metadata
// ---------------------------------------------------------------------------

type StepMeta = {
  step: WizardStep
  labelKey: string
  defaultLabel: string
  /** Abbreviated label for narrow viewports. */
  shortLabel: string
}

const STEPS: StepMeta[] = [
  { step: 1, labelKey: "wizard.step.add", defaultLabel: "Add", shortLabel: "Add" },
  { step: 2, labelKey: "wizard.step.configure", defaultLabel: "Configure", shortLabel: "Config" },
  { step: 3, labelKey: "wizard.step.review", defaultLabel: "Review", shortLabel: "Review" },
  { step: 4, labelKey: "wizard.step.processing", defaultLabel: "Processing", shortLabel: "Proc." },
  { step: 5, labelKey: "wizard.step.results", defaultLabel: "Results", shortLabel: "Results" },
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const IngestWizardStepper: React.FC = () => {
  const { t } = useTranslation(["option"])
  const { state, goToStep } = useIngestWizard()

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  const { currentStep, highestStep, queueItems, selectedPreset, processingState } = state

  const finishedCount = useMemo(
    () => processingState.perItemProgress.filter(
      item => ["complete", "failed", "cancelled"].includes(item.status)
    ).length,
    [processingState.perItemProgress]
  )

  // Build summary text shown on completed steps
  const getSummary = useCallback(
    (step: WizardStep): string | null => {
      if (step >= currentStep) return null
      switch (step) {
        case 1: {
          const count = queueItems.length
          if (count === 0) return null
          return qi("wizard.summary.addCount", "{count, plural, one {# item} other {# items}}", { count })
        }
        case 2: {
          const presetLabel = selectedPreset.charAt(0).toUpperCase() + selectedPreset.slice(1)
          return qi("wizard.summary.preset", "{{preset}}", { preset: presetLabel })
        }
        case 3:
          return qi("wizard.summary.reviewed", "Reviewed")
        default:
          return null
      }
    },
    [currentStep, queueItems.length, selectedPreset, qi]
  )

  const handleStepClick = useCallback(
    (step: WizardStep) => {
      // Only allow clicking on completed steps (backward navigation)
      if (step < currentStep && step <= highestStep) {
        goToStep(step)
      }
    },
    [currentStep, highestStep, goToStep]
  )

  return (
    <nav
      aria-label={qi("wizard.stepper.ariaLabel", "Ingest wizard progress")}
      className="flex items-center border-b border-border px-2 py-2 sm:px-4"
    >
      <ol className="flex w-full items-center gap-0">
        {STEPS.map((meta, idx) => {
          const isCompleted = meta.step < currentStep
          const isCurrent = meta.step === currentStep
          const isFuture = meta.step > currentStep
          const isClickable = isCompleted && meta.step <= highestStep
          const isProcessingStep = meta.step === 4 && isCurrent && processingState.status === "running"
          const summary = getSummary(meta.step)

          return (
            <React.Fragment key={meta.step}>
              {/* Connector line between steps */}
              {idx > 0 && (
                <div
                  className={`mx-1 hidden h-px flex-1 sm:block ${
                    meta.step <= currentStep ? "bg-primary" : "bg-border"
                  }`}
                  aria-hidden="true"
                />
              )}

              <li className="flex items-center">
                <button
                  type="button"
                  onClick={() => handleStepClick(meta.step)}
                  disabled={!isClickable}
                  aria-current={isCurrent ? "step" : undefined}
                  aria-label={
                    isCompleted
                      ? qi("wizard.stepper.completedStep", "Step {{step}}: {{label}} (completed)", {
                          step: meta.step,
                          label: meta.defaultLabel,
                        })
                      : isCurrent
                        ? qi("wizard.stepper.currentStep", "Step {{step}}: {{label}} (current)", {
                            step: meta.step,
                            label: meta.defaultLabel,
                          })
                        : qi("wizard.stepper.futureStep", "Step {{step}}: {{label}}", {
                            step: meta.step,
                            label: meta.defaultLabel,
                          })
                  }
                  className={`group flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium transition-colors sm:text-sm ${
                    isClickable
                      ? "cursor-pointer hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
                      : "cursor-default"
                  } ${
                    isCurrent
                      ? "text-primary"
                      : isCompleted
                        ? "text-text"
                        : "text-text-muted opacity-50"
                  }`}
                >
                  {/* Step indicator icon */}
                  <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center">
                    {isCompleted ? (
                      <Check
                        className="h-4 w-4 text-primary"
                        aria-hidden="true"
                        strokeWidth={2.5}
                      />
                    ) : isProcessingStep ? (
                      <Loader2
                        className="h-4 w-4 animate-spin text-primary"
                        aria-hidden="true"
                      />
                    ) : isCurrent ? (
                      <Circle
                        className="h-3 w-3 fill-primary text-primary"
                        aria-hidden="true"
                      />
                    ) : (
                      <Circle
                        className="h-3 w-3 text-text-muted"
                        aria-hidden="true"
                      />
                    )}
                  </span>

                  {/* Label: full on sm+, abbreviated on narrow */}
                  <span className="hidden sm:inline">
                    {qi(meta.labelKey, meta.defaultLabel)}
                  </span>
                  <span className="inline sm:hidden">
                    {meta.shortLabel}
                  </span>

                  {/* Summary badge on completed steps */}
                  {isCompleted && summary && (
                    <span className="hidden whitespace-nowrap rounded-full bg-surface2 px-1.5 py-0.5 text-[10px] text-text-muted sm:inline">
                      {summary}
                    </span>
                  )}

                  {/* Confirmed finished-item count on the processing step */}
                  {isProcessingStep && (
                    <span className="whitespace-nowrap text-[10px] tabular-nums text-primary">
                      {qi("wizard.summary.finishedItems", "{{done}}/{{total}} finished", {
                        done: finishedCount,
                        total: processingState.perItemProgress.length,
                      })}
                    </span>
                  )}
                </button>
              </li>
            </React.Fragment>
          )
        })}
      </ol>
    </nav>
  )
}

export default IngestWizardStepper
