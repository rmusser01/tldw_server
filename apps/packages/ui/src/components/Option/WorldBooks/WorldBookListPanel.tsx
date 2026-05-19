import React from "react"
import { Button, Dropdown, Table, Tag, Tooltip } from "antd"
import type { MenuProps } from "antd"
import {
  Pen,
  MoreHorizontal,
  CircleCheck,
  CirclePause
} from "lucide-react"
import { formatWorldBookLastModified } from "./worldBookListUtils"

type WorldBookListPanelProps = {
  worldBooks: any[]
  selectedWorldBookId: number | null
  onSelectWorldBook: (id: number) => void
  selectedRowKeys: React.Key[]
  onSelectedRowKeysChange: (keys: React.Key[]) => void
  pendingDeleteIds: number[]
  onEditWorldBook: (record: any) => void
  onRowAction: (action: string, record: any) => void
  tableSort: { field?: string; order?: "ascend" | "descend" | null }
  onTableSortChange: (_: any, __: any, sorter: any) => void
  loading: boolean
  collapsible?: boolean
}

const getOverflowMenuItems = (record: any): MenuProps["items"] => [
  { key: "entries", label: "Manage Entries" },
  { key: "duplicate", label: "Duplicate" },
  { key: "attach", label: "Quick Attach Characters" },
  { key: "export", label: "Export JSON" },
  { key: "stats", label: "Statistics" },
  { type: "divider" },
  { key: "delete", label: "Delete", danger: true }
]

export const WorldBookListPanel: React.FC<WorldBookListPanelProps> = ({
  worldBooks,
  selectedWorldBookId,
  onSelectWorldBook,
  selectedRowKeys,
  onSelectedRowKeysChange,
  pendingDeleteIds,
  onEditWorldBook,
  onRowAction,
  tableSort,
  onTableSortChange,
  loading,
  collapsible
}) => {
  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      sorter: (a: any, b: any) =>
        String(a?.name || "").localeCompare(String(b?.name || "")),
      sortOrder: tableSort.field === "name" ? tableSort.order : null,
      render: (value: string, record: any) => (
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span>{value}</span>
            {pendingDeleteIds.includes(Number(record?.id)) && (
              <Tag color="orange">Pending delete</Tag>
            )}
          </div>
          {record?.description && (
            <span className="text-xs text-text-muted line-clamp-1">
              {record.description}
            </span>
          )}
        </div>
      )
    },
    {
      title: "Entries",
      dataIndex: "entry_count",
      key: "entry_count",
      sorter: (a: any, b: any) =>
        Number(a?.entry_count || 0) - Number(b?.entry_count || 0),
      sortOrder: tableSort.field === "entry_count" ? tableSort.order : null
    },
    {
      title: "Status",
      dataIndex: "enabled",
      key: "enabled",
      sorter: (a: any, b: any) =>
        Number(Boolean(a?.enabled)) - Number(Boolean(b?.enabled)),
      sortOrder: tableSort.field === "enabled" ? tableSort.order : null,
      render: (v: boolean) =>
        v ? (
          <Tag color="green">
            <CircleCheck className="w-3 h-3 inline mr-1" />
            Enabled
          </Tag>
        ) : (
          <Tag color="volcano">
            <CirclePause className="w-3 h-3 inline mr-1" />
            Disabled
          </Tag>
        )
    },
    {
      title: "Last Modified",
      dataIndex: "last_modified",
      key: "last_modified",
      render: (v: unknown) => {
        const { relative, absolute } = formatWorldBookLastModified(v)
        if (absolute) {
          return <Tooltip title={absolute}>{relative}</Tooltip>
        }
        return <span>{relative}</span>
      }
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: any) => {
        const bookName = record?.name || "world book"
        return (
          <div
            className="flex items-center gap-1"
            onClick={(e) => e.stopPropagation()}
          >
            <Button
              type="text"
              size="small"
              icon={<Pen className="w-4 h-4" />}
              aria-label={`Edit ${bookName}`}
              onClick={() => onEditWorldBook(record)}
            />
            <Dropdown
              menu={{
                items: getOverflowMenuItems(record),
                onClick: ({ key }) => onRowAction(key, record)
              }}
              trigger={["click"]}
            >
              <Button
                type="text"
                size="small"
                icon={<MoreHorizontal className="w-4 h-4" />}
                aria-label={`More actions for ${bookName}`}
              />
            </Dropdown>
          </div>
        )
      }
    }
  ]

  const tableJsx = (
    <Table
      rowKey={(r: any) => r.id}
      dataSource={worldBooks}
      columns={columns as any}
      loading={loading}
      rowSelection={{
        selectedRowKeys,
        onChange: (keys) => onSelectedRowKeysChange(keys)
      }}
      onRow={(record: any) => ({
        onClick: () => onSelectWorldBook(record.id),
        style: { cursor: "pointer" }
      })}
      onChange={onTableSortChange}
      rowClassName={(record: any) =>
        record.id === selectedWorldBookId
          ? "bg-primary/5 ring-1 ring-primary/20"
          : ""
      }
      pagination={false}
    />
  )

  if (collapsible && selectedWorldBookId != null) {
    const selectedName =
      worldBooks.find((b: any) => b.id === selectedWorldBookId)?.name ||
      "Selected"
    return (
      <nav aria-label="World books list">
        <details className="rounded border border-border">
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
            World Books — {selectedName}
          </summary>
          <div className="border-t border-border">{tableJsx}</div>
        </details>
      </nav>
    )
  }

  return <nav aria-label="World books list">{tableJsx}</nav>
}
