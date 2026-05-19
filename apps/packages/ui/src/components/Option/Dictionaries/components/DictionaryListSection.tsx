import React from "react"
import { Button, Input, Select, Skeleton, Table } from "antd"
import { useTranslation } from "react-i18next"
import { ChevronDown, ChevronUp, Plus } from "lucide-react"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"

const HowItWorksDictionaries: React.FC = () => {
  const { t } = useTranslation(["option"])
  const [expanded, setExpanded] = React.useState(true)
  return (
    <div className="mb-4 rounded-lg border border-border bg-surface p-4">
      <button
        type="button"
        className="flex w-full items-center justify-between text-sm font-medium text-text"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        {t("option:dictionariesList.howItWorksTitle", "How it works")}
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {expanded && (
        <ol className="mt-3 list-inside list-decimal space-y-2 text-xs text-text-muted">
          <li>
            <span className="font-medium text-text">
              {t("option:dictionariesList.stepCreateTitle", "Create a dictionary")}
            </span>{" "}
            {t(
              "option:dictionariesList.stepCreateBody",
              "— a set of find-and-replace rules (or start from a template)."
            )}
          </li>
          <li>
            <span className="font-medium text-text">
              {t("option:dictionariesList.stepEntriesTitle", "Add entries")}
            </span>{" "}
            {t(
              "option:dictionariesList.stepEntriesBody",
              "— each entry has a pattern to find and text to replace it with."
            )}
          </li>
          <li>
            <span className="font-medium text-text">
              {t("option:dictionariesList.stepPreviewTitle", "Test with preview")}
            </span>{" "}
            {t(
              "option:dictionariesList.stepPreviewBody",
              "— paste sample text to see the rules in action before going live."
            )}
          </li>
          <li>
            <span className="font-medium text-text">
              {t("option:dictionariesList.stepAssignTitle", "Assign to chats")}
            </span>{" "}
            {t(
              "option:dictionariesList.stepAssignBody",
              "— link the dictionary to chat sessions so rules apply automatically."
            )}
          </li>
        </ol>
      )}
    </div>
  )
}

type DictionaryListSectionProps = {
  dictionarySearch: string
  onDictionarySearchChange: (value: string) => void
  categoryFilter: string
  onCategoryFilterChange: (value: string) => void
  tagFilters: string[]
  onTagFiltersChange: (value: string[]) => void
  categoryFilterOptions: string[]
  tagFilterOptions: string[]
  onOpenImport: () => void
  onOpenCreate: () => void
  status: "pending" | "success" | "error" | string
  dictionariesUnsupported: boolean
  unsupportedTitle: string
  unsupportedDescription: string
  unsupportedPrimaryActionLabel: string
  onOpenHealthDiagnostics: () => void
  data: any[] | undefined
  filteredDictionaries: any[]
  columns: any[]
  error: unknown
  onRetry: () => void
}

export const DictionaryListSection: React.FC<DictionaryListSectionProps> = ({
  dictionarySearch,
  onDictionarySearchChange,
  categoryFilter,
  onCategoryFilterChange,
  tagFilters,
  onTagFiltersChange,
  categoryFilterOptions,
  tagFilterOptions,
  onOpenImport,
  onOpenCreate,
  status,
  dictionariesUnsupported,
  unsupportedTitle,
  unsupportedDescription,
  unsupportedPrimaryActionLabel,
  onOpenHealthDiagnostics,
  data,
  filteredDictionaries,
  columns,
  error,
  onRetry
}) => {
  const { t } = useTranslation(["option"])

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Input
            value={dictionarySearch}
            onChange={(event) => onDictionarySearchChange(event.target.value)}
            allowClear
            className="w-full min-w-[220px] md:w-72"
            placeholder={t(
              "option:dictionariesList.searchPlaceholder",
              "Search dictionaries by name, description, category, or tags"
            )}
            aria-label={t("option:dictionariesList.searchAria", "Search dictionaries")}
          />
          <Select
            allowClear
            value={categoryFilter || undefined}
            onChange={(value) => onCategoryFilterChange(value || "")}
            options={categoryFilterOptions.map((category) => ({
              label: category,
              value: category,
            }))}
            placeholder={t("option:dictionariesList.allCategories", "All categories")}
            className="w-40"
            aria-label={t(
              "option:dictionariesList.categoryFilterAria",
              "Filter dictionaries by category"
            )}
          />
          <Select
            mode="multiple"
            allowClear
            value={tagFilters}
            onChange={(value) => onTagFiltersChange(value)}
            options={tagFilterOptions.map((tag) => ({
              label: tag,
              value: tag,
            }))}
            placeholder={t("option:dictionariesList.tagsPlaceholder", "Filter by tags")}
            className="w-44"
            maxTagCount="responsive"
            aria-label={t(
              "option:dictionariesList.tagsFilterAria",
              "Filter dictionaries by tags"
            )}
          />
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={onOpenImport}>
            {t("option:dictionariesList.importButton", "Import")}
          </Button>
          <Button type="primary" icon={<Plus className="w-4 h-4" />} onClick={onOpenCreate}>
            {t("option:dictionariesList.newDictionary", "New Dictionary")}
          </Button>
        </div>
      </div>
      <div className="rounded border border-border bg-surface-secondary px-3 py-2">
        <p className="text-xs text-text-muted">
          {t(
            "option:dictionariesList.processingOrderHint",
            "Processing order for active dictionaries uses Priority (alphabetical by dictionary name), then each dictionary's entry order."
          )}
        </p>
      </div>
      {status === "pending" && <Skeleton active paragraph={{ rows: 6 }} />}
      {status === "success" && dictionariesUnsupported && (
        <FeatureEmptyState
          title={unsupportedTitle}
          description={unsupportedDescription}
          primaryActionLabel={unsupportedPrimaryActionLabel}
          onPrimaryAction={onOpenHealthDiagnostics}
        />
      )}
      {status === "success" && !dictionariesUnsupported && (
        Array.isArray(data) && data.length === 0 ? (
          <>
            <HowItWorksDictionaries />
            <FeatureEmptyState
              title={t("option:dictionariesList.emptyTitle", "No dictionaries yet")}
              description={t(
                "option:dictionariesList.emptyDescription",
                "Create your first dictionary to transform text consistently across chats."
              )}
              examples={[
                t(
                  "option:dictionariesList.emptyExampleMedical",
                  "Medical abbreviations (e.g., BP -> blood pressure)"
                ),
                t(
                  "option:dictionariesList.emptyExampleTerminology",
                  "Custom terminology (e.g., internal product names)"
                ),
                t(
                  "option:dictionariesList.emptyExampleRoleplay",
                  "Roleplay language style mappings"
                )
              ]}
              primaryActionLabel={t(
                "option:dictionariesList.emptyPrimaryAction",
                "Create your first dictionary"
              )}
              onPrimaryAction={onOpenCreate}
              secondaryActionLabel={t(
                "option:dictionariesList.emptySecondaryAction",
                "Import dictionary"
              )}
              onSecondaryAction={onOpenImport}
            />
          </>
        ) : (
          <Table
            rowKey={(record: any) => record.id}
            dataSource={filteredDictionaries}
            columns={columns as any}
            pagination={{
              pageSize: 20,
              showSizeChanger: true,
              pageSizeOptions: [10, 20, 50, 100],
              showTotal: (total, range) =>
                `${range[0]}-${range[1]} ${t("option:dictionariesList.ofTotal", "of")} ${total}`
            }}
          />
        )
      )}
      {status === "error" && !dictionariesUnsupported && (
        <FeatureEmptyState
          title={t("option:dictionariesList.errorTitle", "Unable to load dictionaries")}
          description={
            error instanceof Error
              ? t(
                  "option:dictionariesList.errorDescriptionWithMessage",
                  {
                    defaultValue: "Could not load dictionaries: {{message}}",
                    message: error.message,
                  }
                )
              : t(
                  "option:dictionariesList.errorDescription",
                  "Could not load dictionaries right now. Check your server connection and try again."
                )
          }
          primaryActionLabel={t("option:dictionariesList.retry", "Retry")}
          onPrimaryAction={onRetry}
          secondaryActionLabel={t(
            "option:dictionariesList.emptySecondaryAction",
            "Import dictionary"
          )}
          onSecondaryAction={onOpenImport}
        />
      )}
    </>
  )
}
