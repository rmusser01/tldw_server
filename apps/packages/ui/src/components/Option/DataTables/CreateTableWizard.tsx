import React, { useCallback, useMemo } from "react"
import { Steps, Card, Button } from "antd"
import { ArrowLeft, ArrowRight, Sparkles } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useDataTablesStore } from "@/store/data-tables"
import { Alert } from "@/components/ui/primitives"
import { SourceSelector } from "./SourceSelector"
import { GenerationPanel } from "./GenerationPanel"
import { TablePreview } from "./TablePreview"
import { SaveTablePanel } from "./SaveTablePanel"

/**
 * CreateTableWizard
 *
 * Multi-step wizard for creating data tables from sources.
 * Steps: Select Sources -> Describe Table -> Generate & Preview -> Save
 */
export const CreateTableWizard: React.FC = () => {
  const { t } = useTranslation(["dataTables", "common"])

  // Store state
  const wizardStep = useDataTablesStore((s) => s.wizardStep)
  const selectedSources = useDataTablesStore((s) => s.selectedSources)
  const tableName = useDataTablesStore((s) => s.tableName)
  const prompt = useDataTablesStore((s) => s.prompt)
  const generatedTable = useDataTablesStore((s) => s.generatedTable)
  const isGenerating = useDataTablesStore((s) => s.isGenerating)

  // Store actions
  const setWizardStep = useDataTablesStore((s) => s.setWizardStep)
  const resetWizard = useDataTablesStore((s) => s.resetWizard)

  // Step configuration
  const steps = useMemo(
    () => [
      {
        key: "sources",
        title: t("dataTables:wizard.sources", "Select Sources"),
        description: t("dataTables:wizard.sourcesDesc", "Choose data sources")
      },
      {
        key: "prompt",
        title: t("dataTables:wizard.prompt", "Describe Table"),
        description: t("dataTables:wizard.promptDesc", "Write your prompt")
      },
      {
        key: "preview",
        title: t("dataTables:wizard.preview", "Preview"),
        description: t("dataTables:wizard.previewDesc", "Review generated table")
      },
      {
        key: "save",
        title: t("dataTables:wizard.save", "Save"),
        description: t("dataTables:wizard.saveDesc", "Save to library")
      }
    ],
    [t]
  )

  const currentStepIndex = steps.findIndex((s) => s.key === wizardStep)

  // Navigation helpers
  const canGoNext = () => {
    switch (wizardStep) {
      case "sources":
        return selectedSources.length > 0
      case "prompt":
        return tableName.trim().length > 0 && prompt.trim().length > 0
      case "preview":
        return generatedTable !== null
      case "save":
        return false
      default:
        return false
    }
  }

  const canGoBack = () => {
    return currentStepIndex > 0
  }

  const goNext = useCallback(() => {
    const nextIndex = currentStepIndex + 1
    if (nextIndex < steps.length) {
      setWizardStep(steps[nextIndex].key as typeof wizardStep)
    }
  }, [currentStepIndex, setWizardStep, steps, wizardStep])

  const goBack = useCallback(() => {
    const prevIndex = currentStepIndex - 1
    if (prevIndex >= 0) {
      setWizardStep(steps[prevIndex].key as typeof wizardStep)
    }
  }, [currentStepIndex, setWizardStep, steps, wizardStep])

  // Render step content
  const renderStepContent = () => {
    switch (wizardStep) {
      case "sources":
        return <SourceSelector />
      case "prompt":
        return <GenerationPanel />
      case "preview":
        return <TablePreview />
      case "save":
        return <SaveTablePanel />
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Steps indicator */}
      <Steps
        current={currentStepIndex}
        className="mb-8"
        items={steps.map((step) => ({
          key: step.key,
          title: step.title,
          description: step.description
        }))}
      />

      {/* Intro tip */}
      {wizardStep === "sources" && (
        <Alert
          variant="info"
          icon={<Sparkles className="size-4" />}
          title={t("dataTables:wizard.tip", "Tip")}
          className="mb-4"
        >
          {t(
            "dataTables:wizard.tipText",
            "Select chats, documents, or search your knowledge base to extract structured data. The more specific your sources, the better the results."
          )}
        </Alert>
      )}

      {/* Step content */}
      <Card className="min-h-[400px]">
        {renderStepContent()}
      </Card>

      {/* Navigation buttons */}
      <div className="flex items-center justify-between">
        <div>
          {canGoBack() && (
            <Button
              icon={<ArrowLeft className="h-4 w-4" />}
              onClick={goBack}
              disabled={isGenerating}
            >
              {t("common:back", "Back")}
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={resetWizard} disabled={isGenerating}>
            {t("dataTables:wizard.reset", "Start Over")}
          </Button>
          {wizardStep !== "save" && (
            <Button
              type="primary"
              onClick={goNext}
              disabled={!canGoNext() || isGenerating}
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
