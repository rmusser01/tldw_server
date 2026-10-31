import React from "react"
import { Input, Select } from "antd"
import { Download, Plus, UploadCloud } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { PromptListQueryState } from "./prompt-workspace-types"

type PromptListToolbarV1Props = {
  mode?: "v1"
  query: PromptListQueryState
  allTags: string[]
  disabled?: boolean
  onQueryChange: (patch: Partial<PromptListQueryState>) => void
  onCreatePrompt?: () => void
  onImportPrompts?: () => void
  onExportPrompts?: () => void
}

type PromptListToolbarLegacyProps = {
  mode: "legacy"
  children: React.ReactNode
  className?: string
  testId?: string
}

type PromptListToolbarProps = PromptListToolbarV1Props | PromptListToolbarLegacyProps

export const PromptListToolbar: React.FC<PromptListToolbarProps> = (props) => {
  const { t } = useTranslation(["settings"])
  if (props.mode === "legacy") {
    return (
      <div
        data-testid={props.testId || "prompts-list-toolbar-legacy"}
        className={props.className || "flex flex-wrap items-start justify-between gap-3"}
      >
        {props.children}
      </div>
    )
  }

  const {
    query,
    allTags,
    disabled = false,
    onQueryChange,
    onCreatePrompt,
    onImportPrompts,
    onExportPrompts
  } = props

  return (
    <div
      data-testid="prompts-list-toolbar-scaffold"
      className="flex flex-wrap items-start justify-between gap-3"
    >
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={onCreatePrompt}
          className="inline-flex items-center gap-2 rounded-md border border-transparent bg-primary px-3 py-2 text-sm font-medium text-white hover:bg-primaryStrong disabled:opacity-50"
        >
          <Plus className="size-4" />
          New prompt
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={onImportPrompts}
          className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
        >
          <UploadCloud className="size-4" />
          Import
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={onExportPrompts}
          className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
        >
          <Download className="size-4" />
          Export
        </button>
      </div>

      <div className="flex w-full flex-wrap items-center gap-2 md:w-auto">
        <Input
          allowClear
          value={query.searchText}
          onChange={(event) => onQueryChange({ searchText: event.target.value, page: 1 })}
          placeholder="Search title/content/tag..."
          disabled={disabled}
          style={{ width: 260 }}
        />
        <Select
          value={query.typeFilter}
          onChange={(value) => onQueryChange({ typeFilter: value, page: 1 })}
          disabled={disabled}
          style={{ width: 130 }}
          options={[
            { label: "All types", value: "all" },
            { label: "System", value: "system" },
            { label: "Quick", value: "quick" },
            { label: "Mixed", value: "mixed" },
            {
              label: t("managePrompts.recipe.filters.all", {
                defaultValue: "Recipes"
              }),
              value: "recipe"
            },
            {
              label: t("managePrompts.recipe.filters.system", {
                defaultValue: "System recipes"
              }),
              value: "recipe_system"
            },
            {
              label: t("managePrompts.recipe.filters.user", {
                defaultValue: "User recipes"
              }),
              value: "recipe_user"
            }
          ]}
        />
        <Select
          value={query.syncFilter}
          onChange={(value) => onQueryChange({ syncFilter: value, page: 1 })}
          disabled={disabled}
          style={{ width: 130 }}
          options={[
            { label: "All sync", value: "all" },
            { label: "Local", value: "local" },
            { label: "Pending", value: "pending" },
            { label: "Synced", value: "synced" },
            { label: "Conflict", value: "conflict" }
          ]}
        />
        <Select
          mode="multiple"
          allowClear
          value={query.tagFilter}
          onChange={(value) => onQueryChange({ tagFilter: value, page: 1 })}
          disabled={disabled}
          style={{ width: 200 }}
          placeholder="Tags"
          options={allTags.map((tag) => ({ label: tag, value: tag }))}
        />
      </div>
    </div>
  )
}
