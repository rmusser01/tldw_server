import React from "react"
import { Button } from "antd"
import { useTranslation } from "react-i18next"
import { WORLD_BOOK_STARTER_TEMPLATES } from "./worldBookFormUtils"

type WorldBookEmptyStateProps = {
  onCreateNew: () => void
  onCreateFromTemplate: (key: string) => void
  onImport: () => void
}

export const WorldBookEmptyState: React.FC<WorldBookEmptyStateProps> = ({
  onCreateNew,
  onCreateFromTemplate,
  onImport
}) => {
  const { t } = useTranslation(["option"])

  return (
    <div className="mx-auto max-w-xl rounded-2xl border border-border bg-surface p-7 text-sm text-text">
      <div className="space-y-5">
        <h2 className="text-lg font-semibold text-text">
          {t("worldBooks.emptyState.title", {
            defaultValue: "World Books"
          })}
        </h2>

        <p className="text-sm text-text-muted">
          {t("worldBooks.emptyState.description", {
            defaultValue:
              "World books inject background knowledge into every message. Define facts, rules, or lore once and the AI references them automatically when keywords match."
          })}
        </p>

        {/* 3-step visual flow */}
        <ol className="space-y-2 text-sm text-text-muted">
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              1
            </span>
            <span>
              {t("worldBooks.emptyState.step1", {
                defaultValue: "Create a world book to hold related knowledge"
              })}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              2
            </span>
            <span>
              {t("worldBooks.emptyState.step2", {
                defaultValue: "Add entries with keywords that trigger injection"
              })}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              3
            </span>
            <span>
              {t("worldBooks.emptyState.step3", {
                defaultValue: "Attach to a character or chat to activate"
              })}
            </span>
          </li>
        </ol>

        {/* Concrete example */}
        <div className="rounded-lg bg-surface2/60 px-4 py-3 text-xs text-text-muted">
          {t("worldBooks.emptyState.example", {
            defaultValue:
              "Example: An entry with the keyword \"magic system\" will automatically inject its content whenever someone mentions magic system in conversation, giving the AI the context it needs without you repeating yourself."
          })}
        </div>

        {/* Primary CTA */}
        <div>
          <Button type="primary" onClick={onCreateNew}>
            {t("worldBooks.emptyState.createFirst", {
              defaultValue: "Create your first world book"
            })}
          </Button>
        </div>

        {/* Template quick-start buttons */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-text-muted">
            {t("worldBooks.emptyState.quickStart", {
              defaultValue: "Or start from a template:"
            })}
          </p>
          <div className="flex flex-wrap gap-2">
            {WORLD_BOOK_STARTER_TEMPLATES.map((template) => (
              <Button
                key={template.key}
                size="small"
                onClick={() => onCreateFromTemplate(template.key)}
              >
                {template.label}
              </Button>
            ))}
          </div>
        </div>

        {/* Import button */}
        <div>
          <Button type="default" onClick={onImport}>
            {t("worldBooks.emptyState.import", {
              defaultValue: "Import from JSON"
            })}
          </Button>
        </div>
      </div>
    </div>
  )
}
