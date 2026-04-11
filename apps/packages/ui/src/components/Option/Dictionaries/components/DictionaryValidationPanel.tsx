import { Descriptions } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"
import { humanizeValidationCode } from "./dictionaryEntryUtils"

type DictionaryValidationPanelProps = {
  entriesLength: number
  validationError: string | null
  validationReport: any | null
  onJumpToValidationEntry: (field: unknown) => void
}

function isEntryFieldPath(field: unknown): field is string {
  return typeof field === "string" && /^entries\[\d+\]/.test(field)
}

export const DictionaryValidationPanel: React.FC<DictionaryValidationPanelProps> = ({
  entriesLength,
  validationError,
  validationReport,
  onJumpToValidationEntry,
}) => {
  const { t } = useTranslation(["common", "option"])

  const validationErrors = Array.isArray(validationReport?.errors)
    ? validationReport.errors
    : []
  const validationWarnings = Array.isArray(validationReport?.warnings)
    ? validationReport.warnings
    : []
  const entryStats = validationReport?.entry_stats || null

  return (
    <div
      className="space-y-3"
      role="region"
      aria-label="Dictionary validation panel"
      data-testid="dictionary-validation-panel">
      <p className="text-xs text-text-muted">
        {t(
          "option:dictionariesTools.validateHelp",
          "Check schema, regex safety, and template syntax for this dictionary."
        )}
      </p>
      {entriesLength === 0 && (
        <div className="text-xs text-text-muted">
          {t(
            "option:dictionariesTools.validateEmpty",
            "Add at least one entry to validate."
          )}
        </div>
      )}
      {validationError && (
        <div className="text-xs text-danger">{validationError}</div>
      )}
      {validationReport && (
        <div className="space-y-3 rounded-md border border-border bg-surface px-3 py-2">
          <Descriptions size="small" column={1} bordered>
            <Descriptions.Item
              label={t("option:dictionariesTools.validationOk", "Valid")}>
              {validationReport.ok ? "Yes" : "No"}
            </Descriptions.Item>
            {entryStats && (
              <Descriptions.Item
                label={t("option:dictionariesTools.entryStats", "Entry stats")}>
                {`${entryStats.total ?? 0} total · ${entryStats.literal ?? 0} literal · ${entryStats.regex ?? 0} regex`}
              </Descriptions.Item>
            )}
          </Descriptions>
          <div>
            <div className="text-xs font-medium text-text">
              {t("option:dictionariesTools.errorsLabel", "Errors")}
            </div>
            {validationErrors.length > 0 ? (
              <ul className="list-disc pl-4 text-xs text-text-muted">
                {validationErrors.map((err: any, idx: number) => (
                  <li key={`err-${idx}`}>
                    <button
                      type="button"
                      className={
                        isEntryFieldPath(err?.field)
                          ? "w-full text-left hover:text-text hover:underline"
                          : "w-full cursor-default text-left"
                      }
                      onClick={() => onJumpToValidationEntry(err?.field)}
                      disabled={!isEntryFieldPath(err?.field)}
                    >
                      {(() => {
                        const humanized = humanizeValidationCode(err?.code || "error")
                        return (
                          <>
                            <span className="font-medium text-text">{humanized.label}:</span>{" "}
                            {err?.message || String(err)}
                            {err?.field ? ` (${err.field})` : ""}
                            {humanized.fix && (
                              <span className="block text-text-muted mt-0.5">
                                {t("option:dictionariesTools.tipLabel", "Tip")}: {humanized.fix}
                              </span>
                            )}
                          </>
                        )
                      })()}
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-xs text-text-muted">
                {t("option:dictionariesTools.noErrors", "No errors found.")}
              </div>
            )}
          </div>
          <div>
            <div className="text-xs font-medium text-text">
              {t("option:dictionariesTools.warningsLabel", "Warnings")}
            </div>
            {validationWarnings.length > 0 ? (
              <ul className="list-disc pl-4 text-xs text-text-muted">
                {validationWarnings.map((warn: any, idx: number) => (
                  <li key={`warn-${idx}`}>
                    <button
                      type="button"
                      className={
                        isEntryFieldPath(warn?.field)
                          ? "w-full text-left hover:text-text hover:underline"
                          : "w-full cursor-default text-left"
                      }
                      onClick={() => onJumpToValidationEntry(warn?.field)}
                      disabled={!isEntryFieldPath(warn?.field)}
                    >
                      {(() => {
                        const humanized = humanizeValidationCode(warn?.code || "warning")
                        return (
                          <>
                            <span className="font-medium text-text">{humanized.label}:</span>{" "}
                            {warn?.message || String(warn)}
                            {warn?.field ? ` (${warn.field})` : ""}
                            {humanized.fix && (
                              <span className="block text-text-muted mt-0.5">
                                {t("option:dictionariesTools.tipLabel", "Tip")}: {humanized.fix}
                              </span>
                            )}
                          </>
                        )
                      })()}
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-xs text-text-muted">
                {t("option:dictionariesTools.noWarnings", "No warnings found.")}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
