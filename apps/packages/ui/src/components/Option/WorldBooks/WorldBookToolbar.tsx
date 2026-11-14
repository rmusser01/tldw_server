import React from "react"
import { useTranslation } from "react-i18next"
import { Button, Dropdown, Input, Select } from "antd"
import type { MenuProps } from "antd"
import { Link, useInRouterContext } from "react-router-dom"
import {
  Wrench,
  FlaskConical,
  Network,
  BarChart3,
  FileDown,
  FileUp,
  Bug
} from "lucide-react"
import { LOREBOOK_DEBUG_ENTRYPOINT_HREF } from "./worldBookManagerUtils"

type WorldBookToolbarProps = {
  listSearch: string
  onSearchChange: (value: string) => void
  enabledFilter: "all" | "enabled" | "disabled"
  onEnabledFilterChange: (value: "all" | "enabled" | "disabled") => void
  attachmentFilter: "all" | "attached" | "unattached"
  onAttachmentFilterChange: (value: "all" | "attached" | "unattached") => void
  onNewWorldBook: () => void
  onOpenTestMatching: () => void
  onOpenMatrix: () => void
  onOpenGlobalStats: () => void
  onImport: () => void
  onExportAll: () => void
  onExportSelected?: () => void
  hasWorldBooks: boolean
  hasSelection: boolean
  globalStatsFetching: boolean
  bulkExportAllLoading: boolean
  bulkExportSelectedLoading?: boolean
  compact?: boolean
}

export const WorldBookToolbar: React.FC<WorldBookToolbarProps> = ({
  listSearch,
  onSearchChange,
  enabledFilter,
  onEnabledFilterChange,
  attachmentFilter,
  onAttachmentFilterChange,
  onNewWorldBook,
  onOpenTestMatching,
  onOpenMatrix,
  onOpenGlobalStats,
  onImport,
  onExportAll,
  onExportSelected,
  hasWorldBooks,
  hasSelection,
  globalStatsFetching,
  bulkExportAllLoading,
  bulkExportSelectedLoading,
  compact
}) => {
  const inRouterContext = useInRouterContext()
  const { t } = useTranslation(["option"])

  const toolsMenuItems: MenuProps["items"] = [
    // Group 1: Analysis
    {
      key: "test-matching",
      label: "Test Matching",
      icon: <FlaskConical size={14} />,
      disabled: !hasWorldBooks,
      onClick: onOpenTestMatching
    },
    {
      key: "relationship-matrix",
      label: "Relationship Matrix",
      icon: <Network size={14} />,
      disabled: !hasWorldBooks,
      onClick: onOpenMatrix
    },
    {
      key: "global-statistics",
      label: "Global Statistics",
      icon: <BarChart3 size={14} />,
      disabled: !hasWorldBooks || globalStatsFetching,
      onClick: onOpenGlobalStats
    },
    { type: "divider" },
    // Group 2: I/O
    {
      key: "import-json",
      label: <span data-testid="world-books-import-button">Import JSON</span>,
      icon: <FileUp size={14} />,
      onClick: onImport
    },
    {
      key: "export-all",
      label: bulkExportAllLoading ? "Exporting..." : "Export All",
      icon: <FileDown size={14} />,
      disabled: !hasWorldBooks || bulkExportAllLoading,
      onClick: onExportAll
    },
    ...(hasSelection
      ? [
          {
            key: "export-selected",
            label: bulkExportSelectedLoading
              ? "Exporting..."
              : "Export Selected",
            icon: <FileDown size={14} />,
            disabled: bulkExportSelectedLoading,
            onClick: onExportSelected
          } as NonNullable<MenuProps["items"]>[number]
        ]
      : []),
    { type: "divider" },
    // Group 3: Debug
    {
      key: "chat-injection-panel",
      label: inRouterContext ? (
        <Link
          to={LOREBOOK_DEBUG_ENTRYPOINT_HREF}
          onClick={(e) => e.stopPropagation()}
        >
          Chat Injection Panel
        </Link>
      ) : (
        <a
          href={LOREBOOK_DEBUG_ENTRYPOINT_HREF}
          onClick={(e) => e.stopPropagation()}
        >
          Chat Injection Panel
        </a>
      ),
      icon: <Bug size={14} />
    }
  ]

  const filters = (
    <>
      <Select
        data-testid="world-books-enabled-filter"
        value={enabledFilter}
        onChange={onEnabledFilterChange}
        aria-label="Filter by enabled status"
        className={compact ? "w-full" : "w-40"}
        options={[
          { label: "All statuses", value: "all" },
          { label: "Enabled", value: "enabled" },
          { label: "Disabled", value: "disabled" }
        ]}
      />
      <Select
        data-testid="world-books-attachment-filter"
        value={attachmentFilter}
        onChange={onAttachmentFilterChange}
        aria-label="Filter by attachment state"
        className={compact ? "w-full" : "w-44"}
        options={[
          { label: "All attachments", value: "all" },
          { label: "Has attachments", value: "attached" },
          { label: "Unattached only", value: "unattached" }
        ]}
      />
    </>
  )

  if (compact) {
    return (
      <div className="space-y-2" data-testid="world-books-toolbar">
        <Input
          data-testid="world-books-search-input"
          allowClear
          placeholder={"Search world books\u2026"}
          aria-label="Search world books"
          value={listSearch}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full"
        />
        <details className="rounded-md border border-border">
          <summary
            data-testid="world-books-filters-toggle"
            className="min-h-[44px] cursor-pointer px-3 py-3 text-sm font-medium text-text focus-visible:ring-2 focus-visible:ring-focus"
          >
            {t("worldBooks.filtersToggle", "Filters")}
          </summary>
          <div className="space-y-2 border-t border-border p-3">{filters}</div>
        </details>
        <div className="flex items-center justify-between gap-2">
          <Dropdown menu={{ items: toolsMenuItems }} trigger={["click"]}>
            <Button icon={<Wrench size={14} />} aria-label="Tools" />
          </Dropdown>
          <Button
            type="primary"
            data-testid="world-books-new-button"
            onClick={onNewWorldBook}
          >
            New World Book
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex flex-wrap items-center justify-between gap-2"
      data-testid="world-books-toolbar"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Input
          data-testid="world-books-search-input"
          allowClear
          placeholder={"Search world books\u2026"}
          aria-label="Search world books"
          value={listSearch}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full min-w-[220px] md:w-72"
        />
        {filters}
      </div>
      <div className="flex items-center gap-2">
        <Dropdown menu={{ items: toolsMenuItems }} trigger={["click"]}>
          <Button icon={<Wrench size={14} />}>Tools</Button>
        </Dropdown>
        <Button
          type="primary"
          data-testid="world-books-new-button"
          onClick={onNewWorldBook}
        >
          New World Book
        </Button>
      </div>
    </div>
  )
}
