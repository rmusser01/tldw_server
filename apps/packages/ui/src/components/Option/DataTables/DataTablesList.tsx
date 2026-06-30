import React, { Suspense } from "react"
import {
  Button,
  Card,
  Empty,
  Input,
  Modal,
  Pagination,
  Skeleton,
  Spin,
  Table,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Download,
  Eye,
  RefreshCw,
  Search,
  Trash2,
  FileSpreadsheet
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useDataTablesStore } from "@/store/data-tables"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { DataTableSummary } from "@/types/data-tables"
import { StatePanel } from "@/components/ui/state"
import { ExportMenu } from "./ExportMenu"
const TableDetailModal = React.lazy(() =>
  import("./TableDetailModal").then((module) => ({
    default: module.TableDetailModal
  }))
)

/**
 * DataTablesList
 *
 * Displays a list of saved data tables with search, pagination, and actions.
 */
export const DataTablesList: React.FC = () => {
  const { t } = useTranslation(["dataTables", "common"])
  const queryClient = useQueryClient()

  // Store state
  const tablesPage = useDataTablesStore((s) => s.tablesPage)
  const tablesPageSize = useDataTablesStore((s) => s.tablesPageSize)
  const tablesSearch = useDataTablesStore((s) => s.tablesSearch)
  const selectedTableId = useDataTablesStore((s) => s.selectedTableId)
  const tableDetailOpen = useDataTablesStore((s) => s.tableDetailOpen)
  const deleteConfirmOpen = useDataTablesStore((s) => s.deleteConfirmOpen)
  const deleteTargetId = useDataTablesStore((s) => s.deleteTargetId)

  // Store actions
  const setTablesPage = useDataTablesStore((s) => s.setTablesPage)
  const setTablesSearch = useDataTablesStore((s) => s.setTablesSearch)
  const openTableDetail = useDataTablesStore((s) => s.openTableDetail)
  const closeTableDetail = useDataTablesStore((s) => s.closeTableDetail)
  const openDeleteConfirm = useDataTablesStore((s) => s.openDeleteConfirm)
  const closeDeleteConfirm = useDataTablesStore((s) => s.closeDeleteConfirm)

  const {
    data,
    isLoading,
    isFetching,
    error,
    refetch
  } = useQuery({
    queryKey: ["dataTables", tablesPage, tablesPageSize, tablesSearch],
    queryFn: () =>
      tldwClient.listDataTables({
        page: tablesPage,
        page_size: tablesPageSize,
        search: tablesSearch || undefined
      }),
    placeholderData: keepPreviousData,
    staleTime: 30_000
  })

  const tables = data?.tables ?? []
  const tablesTotal = data?.total ?? 0
  const tablesError =
    error instanceof Error ? error.message : error ? "Failed to load tables" : null

  // Handle delete
  const handleDelete = async () => {
    if (!deleteTargetId) return

    try {
      await tldwClient.deleteDataTable(deleteTargetId)
      await queryClient.invalidateQueries({ queryKey: ["dataTables"] })
      message.success(t("dataTables:deleteSuccess", "Table deleted successfully"))
      closeDeleteConfirm()
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to delete table"
      message.error(errorMessage)
    }
  }

  // Format date
  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric"
      })
    } catch {
      return dateStr
    }
  }

  // Table columns
  const columns: ColumnsType<DataTableSummary> = [
    {
      title: t("dataTables:columns.name", "Name"),
      dataIndex: "name",
      key: "name",
      render: (name: string, record: DataTableSummary) => (
        <button
          className="text-left text-primary hover:text-primaryStrong font-medium"
          onClick={() => openTableDetail(record.id)}
        >
          {name}
        </button>
      )
    },
    {
      title: t("dataTables:columns.rows", "Rows"),
      dataIndex: "row_count",
      key: "row_count",
      width: 80,
      align: "center"
    },
    {
      title: t("dataTables:columns.columns", "Columns"),
      dataIndex: "column_count",
      key: "column_count",
      width: 80,
      align: "center"
    },
    {
      title: t("dataTables:columns.sources", "Sources"),
      dataIndex: "source_count",
      key: "source_count",
      width: 80,
      align: "center"
    },
    {
      title: t("dataTables:columns.created", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 120,
      render: (date: string) => formatDate(date)
    },
    {
      title: t("dataTables:columns.actions", "Actions"),
      key: "actions",
      width: 150,
      render: (_: any, record: DataTableSummary) => (
        <div className="flex items-center gap-2">
          <Tooltip title={t("dataTables:view", "View")}>
            <Button
              type="text"
              size="small"
              aria-label={t("dataTables:view", "View")}
              className="min-h-[44px] min-w-[44px] md:min-h-[32px] md:min-w-[32px]"
              icon={<Eye className="h-4 w-4" />}
              onClick={() => openTableDetail(record.id)}
            />
          </Tooltip>
          <ExportMenu tableId={record.id} tableName={record.name} />
          <Tooltip title={t("dataTables:delete", "Delete")}>
            <Button
              type="text"
              size="small"
              danger
              aria-label={t("dataTables:delete", "Delete")}
              className="min-h-[44px] min-w-[44px] md:min-h-[32px] md:min-w-[32px]"
              icon={<Trash2 className="h-4 w-4" />}
              onClick={() => openDeleteConfirm(record.id)}
            />
          </Tooltip>
        </div>
      )
    }
  ]

  return (
    <div className="space-y-4">
      {/* Search and refresh */}
      <div className="flex items-center justify-between gap-4">
        <Input
          placeholder={t("dataTables:searchPlaceholder", "Search tables...")}
          prefix={<Search className="h-4 w-4 text-text-subtle" />}
          value={tablesSearch}
          onChange={(e) => setTablesSearch(e.target.value)}
          className="max-w-xs"
          allowClear
        />
        <Button
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={() => refetch()}
          loading={isFetching}
        >
          {t("common:refresh", "Refresh")}
        </Button>
      </div>

      {/* Error state */}
      {tablesError && (
        <StatePanel
          state="unavailable"
          title={t("dataTables:loadErrorTitle", "Data tables could not load")}
          message={t(
            "dataTables:loadErrorBody",
            "Check diagnostics or try again after confirming the server is reachable."
          )}
          diagnostics={[
            {
              label: t("dataTables:loadErrorDetailsLabel", "Details"),
              value: tablesError
            }
          ]}
          primaryAction={{
            label: t("common:tryAgain", "Try again"),
            onClick: () => {
              void refetch()
            },
            loading: isFetching
          }}
        />
      )}

      {/* Loading state */}
      {isLoading && tables.length === 0 && (
        <div className="flex justify-center py-12">
          <Spin size="large" />
        </div>
      )}

      {/* Empty state */}
      {!isLoading && tables.length === 0 && !tablesError && (
        <Empty
          image={<FileSpreadsheet className="h-16 w-16 mx-auto text-text-subtle" />}
          description={
            <span className="text-text-muted">
              {tablesSearch
                ? t("dataTables:noSearchResults", "No tables found matching your search")
                : t("dataTables:noTables", "No tables yet. Create your first table!")}
            </span>
          }
        />
      )}

      {/* Tables list */}
      {tables.length > 0 && (
        <>
          <Table
            dataSource={tables}
            columns={columns}
            rowKey="id"
            loading={isFetching}
            pagination={false}
            size="middle"
          />

          {/* Pagination */}
          {tablesTotal > tablesPageSize && (
            <div className="flex justify-end">
              <Pagination
                current={tablesPage}
                pageSize={tablesPageSize}
                total={tablesTotal}
                onChange={(page) => setTablesPage(page)}
                showSizeChanger={false}
                showTotal={(total) =>
                  t("dataTables:totalTables", "{{total}} tables", { total })
                }
              />
            </div>
          )}
        </>
      )}

      {/* Table detail modal */}
      {tableDetailOpen && (
        <Suspense
          fallback={
            <Card className="bg-surface">
              <Skeleton active title paragraph={{ rows: 2 }} />
            </Card>
          }
        >
          <TableDetailModal
            open={tableDetailOpen}
            tableId={selectedTableId}
            onClose={closeTableDetail}
          />
        </Suspense>
      )}

      {/* Delete confirmation modal */}
      <Modal
        title={t("dataTables:deleteConfirmTitle", "Delete Table")}
        open={deleteConfirmOpen}
        onOk={handleDelete}
        onCancel={closeDeleteConfirm}
        okText={t("common:delete", "Delete")}
        cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ danger: true }}
      >
        <p>
          {t(
            "dataTables:deleteConfirmMessage",
            "Are you sure you want to delete this table? This action cannot be undone."
          )}
        </p>
      </Modal>
    </div>
  )
}
