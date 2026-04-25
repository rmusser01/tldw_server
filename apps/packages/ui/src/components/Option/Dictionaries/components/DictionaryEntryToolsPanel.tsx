import React from "react"
import { Button, Collapse, Switch, Tooltip } from "antd"
import { useTranslation } from "react-i18next"
import { LabelWithHelp } from "@/components/Common/LabelWithHelp"
import { DictionaryValidationPanel } from "./DictionaryValidationPanel"
import { DictionaryPreviewPanel } from "./DictionaryPreviewPanel"

type PreviewDiffSegment = {
  type: "unchanged" | "removed" | "added"
  text: string
}

type DictionaryEntryToolsPanelProps = {
  entriesCount: number
  toolsPanelKeys: string[]
  onToolsPanelKeysChange: (keys: string[]) => void
  validationStrict: boolean
  onValidationStrictChange: (value: boolean) => void
  onRunValidation: () => void
  validating: boolean
  validationError: string | null
  validationReport: any | null
  onJumpToValidationEntry: (field: unknown) => void
  onRunPreview: () => void
  previewing: boolean
  previewText: string
  onPreviewTextChange: (value: string) => void
  previewCaseName: string
  onPreviewCaseNameChange: (value: string) => void
  onSavePreviewCase: () => void
  previewCaseError: string | null
  savedPreviewCases: Array<{ id: string; name: string; text: string }>
  onLoadPreviewCase: (caseId: string) => void
  onDeletePreviewCase: (caseId: string) => void
  previewTokenBudget: number | null
  onPreviewTokenBudgetChange: (value: number | null) => void
  previewMaxIterations: number | null
  onPreviewMaxIterationsChange: (value: number | null) => void
  previewError: string | null
  previewResult: any | null
  previewHasDiffChanges: boolean
  previewDiffSegments: PreviewDiffSegment[]
  previewProcessedText: string
  previewEntriesUsed: Array<string | number>
}

export const DictionaryEntryToolsPanel: React.FC<DictionaryEntryToolsPanelProps> = ({
  entriesCount,
  toolsPanelKeys,
  onToolsPanelKeysChange,
  validationStrict,
  onValidationStrictChange,
  onRunValidation,
  validating,
  validationError,
  validationReport,
  onJumpToValidationEntry,
  onRunPreview,
  previewing,
  previewText,
  onPreviewTextChange,
  previewCaseName,
  onPreviewCaseNameChange,
  onSavePreviewCase,
  previewCaseError,
  savedPreviewCases,
  onLoadPreviewCase,
  onDeletePreviewCase,
  previewTokenBudget,
  onPreviewTokenBudgetChange,
  previewMaxIterations,
  onPreviewMaxIterationsChange,
  previewError,
  previewResult,
  previewHasDiffChanges,
  previewDiffSegments,
  previewProcessedText,
  previewEntriesUsed,
}) => {
  const { t } = useTranslation(["common", "option"])
  const toggleOnLabel = t("common:on", "On")
  const toggleOffLabel = t("common:off", "Off")

  const openToolsPanel = React.useCallback(
    (panelKey: "validate" | "preview") => {
      if (toolsPanelKeys.includes(panelKey)) return
      onToolsPanelKeysChange([...toolsPanelKeys, panelKey])
    },
    [onToolsPanelKeysChange, toolsPanelKeys]
  )

  return (
    <>
      <div className="rounded-lg border border-border bg-surface2/40 px-3 py-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs text-text-muted">
            {t(
              "option:dictionariesTools.actionsHelp",
              "Run validation or preview without opening accordion sections first."
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2">
              <Switch
                checked={validationStrict}
                onChange={onValidationStrictChange}
                checkedChildren={toggleOnLabel}
                unCheckedChildren={toggleOffLabel}
                aria-label={t(
                  "option:dictionariesTools.strictLabel",
                  "Strict validation"
                )}
              />
              <LabelWithHelp
                label={t("option:dictionariesTools.strictLabel", "Strict validation")}
                help={t(
                  "option:dictionariesTools.strictHelp",
                  "When on, checks additional rules like regex safety and pattern conflicts. When off, only checks basic format."
                )}
              />
            </div>
            <Tooltip
              title={t(
                "option:dictionariesTools.validateShortcut",
                "Ctrl+Shift+V"
              )}
            >
              <Button
                size="small"
                onClick={() => {
                  openToolsPanel("validate")
                  onRunValidation()
                }}
                loading={validating}
                disabled={entriesCount === 0}>
                {t("option:dictionariesTools.validateButton", "Run validation")}
              </Button>
            </Tooltip>
            <Button
              size="small"
              type="primary"
              onClick={() => {
                openToolsPanel("preview")
                onRunPreview()
              }}
              loading={previewing}>
              {t("option:dictionariesTools.previewButton", "Run preview")}
            </Button>
          </div>
        </div>
        {entriesCount === 0 && (
          <div className="mt-2 text-xs text-text-muted">
            {t(
              "option:dictionariesTools.validateEmpty",
              "Add at least one entry to validate."
            )}
          </div>
        )}
      </div>

      <Collapse
        ghost
        className="rounded-lg border border-border bg-surface2/40"
        activeKey={toolsPanelKeys}
        onChange={(nextKeys) => {
          if (Array.isArray(nextKeys)) {
            onToolsPanelKeysChange(nextKeys.map((key) => String(key)))
            return
          }
          if (nextKeys) {
            onToolsPanelKeysChange([String(nextKeys)])
            return
          }
          onToolsPanelKeysChange([])
        }}
        items={[
          {
            key: "validate",
            label: t(
              "option:dictionariesTools.validateTitle",
              "Validate dictionary"
            ),
            children: (
              <DictionaryValidationPanel
                entriesLength={entriesCount}
                validationError={validationError}
                validationReport={validationReport}
                onJumpToValidationEntry={onJumpToValidationEntry}
              />
            ),
          },
          {
            key: "preview",
            label: t(
              "option:dictionariesTools.previewTitle",
              "Preview transforms"
            ),
            children: (
              <DictionaryPreviewPanel
                previewText={previewText}
                onPreviewTextChange={onPreviewTextChange}
                previewCaseName={previewCaseName}
                onPreviewCaseNameChange={onPreviewCaseNameChange}
                onSavePreviewCase={onSavePreviewCase}
                previewCaseError={previewCaseError}
                savedPreviewCases={savedPreviewCases}
                onLoadPreviewCase={onLoadPreviewCase}
                onDeletePreviewCase={onDeletePreviewCase}
                previewTokenBudget={previewTokenBudget}
                onPreviewTokenBudgetChange={onPreviewTokenBudgetChange}
                previewMaxIterations={previewMaxIterations}
                onPreviewMaxIterationsChange={onPreviewMaxIterationsChange}
                onRunPreview={onRunPreview}
                previewing={previewing}
                previewError={previewError}
                previewResult={previewResult}
                previewProcessedText={previewProcessedText}
                previewEntriesUsed={previewEntriesUsed.map((value) => String(value))}
                previewDiffSegments={previewDiffSegments}
                previewHasDiffChanges={previewHasDiffChanges}
              />
            ),
          },
        ]}
      />
    </>
  )
}
