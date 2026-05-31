import React from "react"
import { BookOpen, FileText, Link, Plus, Upload } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/Common/Button"
import { EmptyState } from "@/components/ui/feedback/EmptyState"
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
  const title = t("worldBooks.emptyState.title", {
    defaultValue: "World Books"
  })
  const description = t("worldBooks.emptyState.description", {
    defaultValue:
      "World books inject background knowledge into every message. Define facts, rules, or lore once and the AI references them automatically when keywords match."
  })
  const step1 = t("worldBooks.emptyState.step1", {
    defaultValue: "Create a world book to hold related knowledge"
  })
  const step2 = t("worldBooks.emptyState.step2", {
    defaultValue: "Add entries with keywords that trigger injection"
  })
  const step3 = t("worldBooks.emptyState.step3", {
    defaultValue: "Attach to a character or chat to activate"
  })
  const example = t("worldBooks.emptyState.example", {
    defaultValue:
      "Example: An entry with the keyword \"magic system\" will automatically inject its content whenever someone mentions magic system in conversation, giving the AI the context it needs without you repeating yourself."
  })
  const createLabel = t("worldBooks.emptyState.createFirst", {
    defaultValue: "Create your first world book"
  })
  const quickStartLabel = t("worldBooks.emptyState.quickStart", {
    defaultValue: "Or start from a template:"
  })
  const importLabel = t("worldBooks.emptyState.import", {
    defaultValue: "Import from JSON"
  })

  return (
    <EmptyState
      title={title}
      description={description}
      icon={BookOpen}
      iconClassName="text-text-muted"
      size="lg"
      variant="card"
      className="text-sm text-text"
      steps={[
        { icon: BookOpen, text: step1 },
        { icon: FileText, text: step2 },
        { icon: Link, text: step3 }
      ]}
      primaryAction={{
        label: createLabel,
        icon: <Plus className="h-4 w-4" />,
        onClick: onCreateNew
      }}
      secondaryAction={{
        label: importLabel,
        icon: <Upload className="h-4 w-4" />,
        onClick: onImport
      }}
    >
      <div className="space-y-5 pt-2">
        <div className="rounded-lg bg-surface2/60 px-4 py-3 text-left text-xs text-text-muted">
          {example}
        </div>

        <div className="space-y-2">
          <p className="text-xs font-medium text-text-muted">
            {quickStartLabel}
          </p>
          <div className="flex flex-wrap justify-center gap-2">
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
      </div>
    </EmptyState>
  )
}
