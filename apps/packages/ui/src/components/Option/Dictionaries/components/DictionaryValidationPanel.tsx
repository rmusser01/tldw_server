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
  const localize = React.useCallback(
    (key: string, fallback: string) => t(key, fallback),
    [t]
  )

  const validationErrors = Array.isArray(validationReport?.errors)
    ? validationReport.errors
    : []
  const validationWarnings = Array.isArray(validationReport?.warnings)
    ? validationReport.warnings
    : []
  const entryStats = validationReport?.entry_stats || null

  const renderValidationItems = React.useCallback(
    (items: any[], kind: "error" | "warning") => {
      if (items.length === 0) {
        return (
          <div className="text-xs text-text-muted">
            {kind === "error"
              ? t("option:dictionariesTools.noErrors", "No errors found.")
              : t("option:dictionariesTools.noWarnings", "No warnings found.")}
          </div>
        )
      }

      return (
        <ul className="list-disc pl-4 text-xs text-text-muted">
          {items.map((item: any, idx: number) => {
            const humanized = humanizeValidationCode(
              item?.code || kind,
              localize
            )
            return (
              <li key={`${kind}-${idx}`}>
                <button
                  type="button"
                  className={
                    isEntryFieldPath(item?.field)
                      ? "w-full text-left hover:text-text hover:underline"
                      : "w-full cursor-default text-left"
                  }
                  onClick={() => onJumpToValidationEntry(item?.field)}
                  disabled={!isEntryFieldPath(item?.field)}
                >
                  <span className="font-medium text-text">{humanized.label}:</span>{" "}
                  {item?.message || String(item)}
                  {item?.field ? ` (${item.field})` : ""}
                  {humanized.fix && (
                    <span className="block text-text-muted mt-0.5">
                      {t("option:dictionariesTools.tipLabel", "Tip")}: {humanized.fix}
                    </span>
                  )}
                </button>
              </li>
            )
          })}
        </ul>
      )
    },
    [localize, onJumpToValidationEntry, t]
  )

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
              {validationReport.ok ? t("common:yes", "Yes") : t("common:no", "No")}
            </Descriptions.Item>
            {entryStats && (
              <Descriptions.Item
                label={t("option:dictionariesTools.entryStats", "Entry stats")}>
                {`${entryStats.total ?? 0} ${t("option:dictionariesTools.entryStatsTotal", "total")} · ${entryStats.literal ?? 0} ${t("option:dictionariesTools.entryStatsLiteral", "literal")} · ${entryStats.regex ?? 0} ${t("option:dictionariesTools.entryStatsRegex", "regex")}`}
              </Descriptions.Item>
            )}
          </Descriptions>
          <div>
            <div className="text-xs font-medium text-text">
              {t("option:dictionariesTools.errorsLabel", "Errors")}
            </div>
            {renderValidationItems(validationErrors, "error")}
          </div>
          <div>
            <div className="text-xs font-medium text-text">
              {t("option:dictionariesTools.warningsLabel", "Warnings")}
            </div>
            {renderValidationItems(validationWarnings, "warning")}
          </div>
        </div>
      )}
    </div>
  )
}
