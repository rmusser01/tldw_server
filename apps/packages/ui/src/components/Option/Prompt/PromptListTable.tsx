import React from "react"
import { Table } from "antd"
import type { TablePaginationConfig, TableProps } from "antd"
import type { SorterResult } from "antd/es/table/interface"
import {
  buildPromptTableColumns,
  type PromptTableColumnLabels
} from "./prompt-table-columns"
import type {
  PromptListQueryState,
  PromptListSortKey,
  PromptListSortOrder,
  PromptRowVM
} from "./prompt-workspace-types"

export type PromptTableDensity = "comfortable" | "compact" | "dense"

type PromptListTableV1Props = {
  mode?: "v1"
  rows: PromptRowVM[]
  total: number
  loading?: boolean
  isOnline: boolean
  isCompactViewport: boolean
  query: PromptListQueryState
  selectedIds: string[]
  onQueryChange: (patch: Partial<PromptListQueryState>) => void
  onSelectionChange: (ids: string[]) => void
  onRowOpen: (id: string) => void
  onEdit?: (id: string) => void
  onToggleFavorite?: (id: string, nextFavorite: boolean) => void
  onOpenConflictResolution?: (id: string) => void
  renderActions?: (row: PromptRowVM) => React.ReactNode
  renderTitleMeta?: (row: PromptRowVM) => React.ReactNode
  favoriteButtonTestId?: (row: PromptRowVM) => string
  columnLabels?: Partial<PromptTableColumnLabels>
  formatRelativeTime?: (timestamp?: number) => string
  selectionDisabled?: boolean
  tableTestId?: string
  tableShellTestId?: string
  scrollContainerTestId?: string
  overflowIndicatorTestId?: string
  rowTestIdPrefix?: string
  paginationShowTotal?: (total: number, range: [number, number]) => React.ReactNode
  tableDensity?: PromptTableDensity
}

type PromptListTableLegacyProps = {
  mode: "legacy"
  isCompactViewport: boolean
  children: React.ReactNode
  tableShellTestId?: string
  scrollContainerTestId?: string
  overflowIndicatorTestId?: string
}

type PromptListTableProps = PromptListTableV1Props | PromptListTableLegacyProps

const parseSortFromSorter = (
  sorter: SorterResult<PromptRowVM> | SorterResult<PromptRowVM>[]
) => {
  const activeSorter = Array.isArray(sorter) ? sorter[0] : sorter
  const rawKey = activeSorter?.columnKey
  const key: PromptListSortKey =
    rawKey === "title" || rawKey === "modifiedAt" ? rawKey : null
  const order = (activeSorter?.order as PromptListSortOrder) || null
  return {
    key,
    order
  }
}

export const PromptListTable: React.FC<PromptListTableProps> = (props) => {
  if (props.mode === "legacy") {
    const {
      isCompactViewport,
      children,
      tableShellTestId = "prompts-table-shell",
      scrollContainerTestId = "prompts-table-scroll-container",
      overflowIndicatorTestId = "prompts-table-overflow-indicator"
    } = props

    return (
      <div className="relative" data-testid={tableShellTestId}>
        <div className="overflow-x-auto pb-1" data-testid={scrollContainerTestId}>
          {children}
        </div>
        {isCompactViewport && (
          <div
            data-testid={overflowIndicatorTestId}
            className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-bg to-transparent sm:hidden"
            aria-hidden="true"
          />
        )}
      </div>
    )
  }

  const {
    rows,
    total,
    loading = false,
    isOnline,
    isCompactViewport,
    query,
    selectedIds,
    onQueryChange,
    onSelectionChange,
    onRowOpen,
    onEdit,
    onToggleFavorite,
    onOpenConflictResolution,
    renderActions,
    renderTitleMeta,
    favoriteButtonTestId,
    columnLabels,
    formatRelativeTime,
    selectionDisabled = false,
    tableTestId = "prompts-table",
    tableShellTestId = "prompts-table-shell",
    scrollContainerTestId = "prompts-table-scroll-container",
    overflowIndicatorTestId = "prompts-table-overflow-indicator",
    rowTestIdPrefix = "prompt-row-",
    paginationShowTotal,
    tableDensity = "comfortable"
  } = props

  const tableClassName = `prompts-table prompts-table-density-${tableDensity}`

  const columns = React.useMemo(
    () =>
      buildPromptTableColumns({
        isOnline,
        isCompactViewport,
        sortKey: query.sort.key,
        sortOrder: query.sort.order,
        onToggleFavorite: (row, nextFavorite) =>
          onToggleFavorite?.(row.id, nextFavorite),
        onEdit: (row) => onEdit?.(row.id),
        onOpenConflictResolution: (row) =>
          onOpenConflictResolution?.(row.id),
        renderActions,
        renderTitleMeta,
        favoriteButtonTestId,
        labels: columnLabels,
        formatRelativeTime
      }),
    [
      columnLabels,
      favoriteButtonTestId,
      formatRelativeTime,
      isOnline,
      isCompactViewport,
      onEdit,
      onOpenConflictResolution,
      onToggleFavorite,
      query.sort.key,
      query.sort.order,
      renderTitleMeta,
      renderActions
    ]
  )

  const rowSelection: TableProps<PromptRowVM>["rowSelection"] = {
    selectedRowKeys: selectedIds,
    getCheckboxProps: () => ({
      disabled: selectionDisabled
    }),
    onChange: (keys) => {
      onSelectionChange(keys.map((key) => String(key)))
    }
  }

  const handleTableChange: TableProps<PromptRowVM>["onChange"] = (
    pagination,
    _filters,
    sorter
  ) => {
    const nextPage = (pagination as TablePaginationConfig)?.current || query.page
    const nextPageSize =
      (pagination as TablePaginationConfig)?.pageSize || query.pageSize
    const sort = parseSortFromSorter(sorter as SorterResult<PromptRowVM>)

    onQueryChange({
      page: nextPageSize !== query.pageSize ? 1 : nextPage,
      pageSize: nextPageSize,
      sort
    })
  }

  return (
    <div className="relative" data-testid={tableShellTestId}>
      <div className="overflow-x-auto pb-1" data-testid={scrollContainerTestId}>
        <Table<PromptRowVM>
          className={tableClassName}
          size={tableDensity === "comfortable" ? "middle" : "small"}
          data-testid={tableTestId}
          loading={loading}
          columns={columns}
          dataSource={rows}
          rowKey={(record) => record.id}
          pagination={{
            current: query.page,
            pageSize: query.pageSize,
            total,
            showSizeChanger: true,
            pageSizeOptions: ["10", "20", "50", "100"],
            showTotal: paginationShowTotal
          }}
          onChange={handleTableChange}
          rowSelection={rowSelection}
          scroll={isCompactViewport ? { x: 980 } : undefined}
          onRow={(record) =>
            ({
              "data-testid": `${rowTestIdPrefix}${record.id}`,
              tabIndex: 0,
              role: "row",
              className:
                "cursor-pointer focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary",
              onClick: (event: React.MouseEvent<HTMLTableRowElement>) => {
                const target = event.target as HTMLElement | null
                if (
                  target?.closest(
                    "button, a, input, textarea, select, label, [role='button'], .ant-select, .ant-checkbox-wrapper, .ant-dropdown, .ant-pagination"
                  )
                ) {
                  return
                }
                onRowOpen(record.id)
              },
              onDoubleClick: () => onEdit?.(record.id),
              onKeyDown: (event: React.KeyboardEvent<HTMLTableRowElement>) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onRowOpen(record.id)
                }
                if (event.key === "e" || event.key === "E") {
                  event.preventDefault()
                  onEdit?.(record.id)
                }
              }
            }) as React.HTMLAttributes<HTMLTableRowElement>
          }
        />
      </div>
      {isCompactViewport && (
        <div
          data-testid={overflowIndicatorTestId}
          className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-bg to-transparent sm:hidden"
          aria-hidden="true"
        />
      )}
    </div>
  )
}
