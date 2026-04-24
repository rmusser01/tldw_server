import { Button, Descriptions, Input, InputNumber, Tag } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"
import { LabelWithHelp } from "@/components/Common/LabelWithHelp"

type TextDiffSegment = {
  type: "unchanged" | "removed" | "added"
  text: string
}

type SavedPreviewCase = {
  id: string
  name: string
  text: string
}

type DictionaryPreviewPanelProps = {
  previewText: string
  onPreviewTextChange: (value: string) => void
  previewCaseName: string
  onPreviewCaseNameChange: (value: string) => void
  onSavePreviewCase: () => void
  previewCaseError: string | null
  savedPreviewCases: SavedPreviewCase[]
  onLoadPreviewCase: (caseId: string) => void
  onDeletePreviewCase: (caseId: string) => void
  previewTokenBudget: number | null
  onPreviewTokenBudgetChange: (value: number | null) => void
  previewMaxIterations: number | null
  onPreviewMaxIterationsChange: (value: number | null) => void
  onRunPreview: () => void
  previewing: boolean
  previewError: string | null
  previewResult: any | null
  previewProcessedText: string
  previewEntriesUsed: string[]
  previewDiffSegments: TextDiffSegment[]
  previewHasDiffChanges: boolean
}

export const DictionaryPreviewPanel: React.FC<DictionaryPreviewPanelProps> = ({
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
  onRunPreview,
  previewing,
  previewError,
  previewResult,
  previewProcessedText,
  previewEntriesUsed,
  previewDiffSegments,
  previewHasDiffChanges,
}) => {
  const { t } = useTranslation(["common", "option"])

  return (
    <div
      className="space-y-3"
      role="region"
      aria-label="Dictionary preview panel"
      data-testid="dictionary-preview-panel">
      <p className="text-xs text-text-muted">
        {t(
          "option:dictionariesTools.previewHelp",
          "Test how this dictionary rewrites sample text."
        )}
      </p>
      <div className="space-y-2">
        <div className="text-xs font-medium text-text">
          {t("option:dictionariesTools.sampleTextLabel", "Sample text")}
        </div>
        <Input.TextArea
          rows={4}
          value={previewText}
          onChange={(event) => onPreviewTextChange(event.target.value)}
          placeholder={t(
            "option:dictionariesTools.sampleTextPlaceholder",
            "Paste text to preview dictionary substitutions."
          )}
        />
      </div>
      <div className="space-y-2 rounded-md border border-border bg-surface2/40 px-3 py-2">
        <div className="text-xs font-medium text-text">
          {t("option:dictionariesTools.savedCasesLabel", "Saved test cases")}
        </div>
        <div
          data-testid="dictionary-preview-case-controls"
          className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Input
            size="small"
            value={previewCaseName}
            onChange={(event) => onPreviewCaseNameChange(event.target.value)}
            placeholder={t(
              "option:dictionariesTools.caseNamePlaceholder",
              "Case name (optional)"
            )}
            aria-label={t("option:dictionariesTools.caseNameAria", "Test case name")}
          />
          <Button size="small" onClick={onSavePreviewCase}>
            {t("option:dictionariesTools.saveCaseButton", "Save test case")}
          </Button>
        </div>
        {previewCaseError && (
          <div className="text-xs text-danger">{previewCaseError}</div>
        )}
        {savedPreviewCases.length > 0 ? (
          <div className="space-y-1">
            {savedPreviewCases.map((savedCase) => (
              <div
                key={savedCase.id}
                className="flex items-center justify-between gap-2 rounded border border-border bg-surface px-2 py-1">
                <div className="truncate text-xs text-text">{savedCase.name}</div>
                <div className="flex items-center gap-1">
                  <Button
                    size="small"
                    onClick={() => onLoadPreviewCase(savedCase.id)}
                    aria-label={`Load test case ${savedCase.name}`}>
                    {t("option:dictionariesTools.loadCaseButton", "Load")}
                  </Button>
                  <Button
                    size="small"
                    danger
                    onClick={() => onDeletePreviewCase(savedCase.id)}
                    aria-label={`Delete test case ${savedCase.name}`}>
                    {t("option:dictionariesTools.deleteCaseButton", "Delete")}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-text-muted">
            {t(
              "option:dictionariesTools.noSavedCases",
              "No saved test cases for this dictionary yet."
            )}
          </div>
        )}
      </div>
      <div
        data-testid="dictionary-preview-controls-grid"
        className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <div className="space-y-1 min-w-0">
          <div className="text-xs font-medium text-text">
            <LabelWithHelp
              label={t("option:dictionariesTools.tokenBudgetLabel", "Processing limit")}
              help={t(
                "option:dictionariesTools.tokenBudgetHelp",
                "Maximum amount of text processed. Leave empty for no limit."
              )}
            />
          </div>
          <InputNumber
            min={0}
            style={{ width: "100%" }}
            value={previewTokenBudget ?? undefined}
            onChange={(value) =>
              onPreviewTokenBudgetChange(typeof value === "number" ? value : null)
            }
          />
        </div>
        <div className="space-y-1 min-w-0">
          <div className="text-xs font-medium text-text">
            <LabelWithHelp
              label={t("option:dictionariesTools.maxIterationsLabel", "Max passes")}
              help={t(
                "option:dictionariesTools.maxIterationsHelp",
                "Number of passes over the text. Usually 1 is enough. Increase if rules can trigger other rules."
              )}
            />
          </div>
          <InputNumber
            min={1}
            style={{ width: "100%" }}
            value={previewMaxIterations ?? undefined}
            onChange={(value) =>
              onPreviewMaxIterationsChange(typeof value === "number" ? value : null)
            }
          />
        </div>
      </div>
      <Button
        size="small"
        type="primary"
        onClick={onRunPreview}
        loading={previewing}
        disabled={!previewText.trim()}>
        {t("option:dictionariesTools.previewButton", "Run preview")}
      </Button>
      {previewError && <div className="text-xs text-danger">{previewError}</div>}
      {previewResult && (
        <div className="space-y-2 rounded-md border border-border bg-surface px-3 py-2">
          <div className="space-y-1">
            <div className="text-xs font-medium text-text">
              {t("option:dictionariesTools.diffPreviewLabel", "Diff preview")}
            </div>
            {previewHasDiffChanges ? (
              <div className="grid gap-2 sm:grid-cols-2">
                <div className="rounded border border-border bg-surface2/50 p-2">
                  <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-text-muted">
                    {t(
                      "option:dictionariesTools.originalDiffLabel",
                      "Original (with removals)"
                    )}
                  </div>
                  <p className="text-xs leading-relaxed whitespace-pre-wrap break-words">
                    {previewDiffSegments
                      .filter((segment) => segment.type !== "added")
                      .map((segment, index) => (
                        <span
                          key={`diff-original-${index}`}
                          className={
                            segment.type === "removed"
                              ? "rounded-sm bg-danger/15 px-0.5 text-danger line-through"
                              : ""
                          }>
                          {segment.text}
                        </span>
                      ))}
                  </p>
                </div>
                <div className="rounded border border-border bg-surface2/50 p-2">
                  <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-text-muted">
                    {t(
                      "option:dictionariesTools.processedDiffLabel",
                      "Processed (with additions)"
                    )}
                  </div>
                  <p className="text-xs leading-relaxed whitespace-pre-wrap break-words">
                    {previewDiffSegments
                      .filter((segment) => segment.type !== "removed")
                      .map((segment, index) => (
                        <span
                          key={`diff-processed-${index}`}
                          className={
                            segment.type === "added"
                              ? "rounded-sm bg-success/15 px-0.5 text-success"
                              : ""
                          }>
                          {segment.text}
                        </span>
                      ))}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-xs text-text-muted">
                {t(
                  "option:dictionariesTools.noDiffChanges",
                  "No differences detected between original and processed text."
                )}
              </div>
            )}
          </div>
          <div className="space-y-1">
            <div className="text-xs font-medium text-text">
              {t("option:dictionariesTools.processedTextLabel", "Processed text")}
            </div>
            <Input.TextArea rows={4} value={previewProcessedText || ""} readOnly />
          </div>
          <Descriptions size="small" column={1} bordered>
            <Descriptions.Item
              label={t("option:dictionariesTools.replacementsLabel", "Replacements")}>
              {previewResult.replacements ?? 0}
            </Descriptions.Item>
            <Descriptions.Item
              label={t("option:dictionariesTools.iterationsLabel", "Iterations")}>
              {previewResult.iterations ?? 0}
            </Descriptions.Item>
            <Descriptions.Item
              label={t("option:dictionariesTools.entriesUsedLabel", "Entries used")}>
              {previewEntriesUsed.length > 0 ? previewEntriesUsed.join(", ") : "—"}
            </Descriptions.Item>
          </Descriptions>
          {previewResult.token_budget_exceeded && (
            <Tag color="red">
              {t(
                "option:dictionariesTools.tokenBudgetExceeded",
                "Token budget exceeded"
              )}
            </Tag>
          )}
        </div>
      )}
    </div>
  )
}
