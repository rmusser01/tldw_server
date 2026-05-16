import React from "react"
import { Button, Card, Empty, Space, Typography } from "antd"

import { getDesignSystemState } from "@/design-system"
import type { IngestionSourceItem } from "@/types/ingestion-sources"

export type SourceItemsFilter = "all" | "detached" | "degraded"

type SourceItemsTableProps = {
  items: IngestionSourceItem[]
  filter: SourceItemsFilter
  onFilterChange: (filter: SourceItemsFilter) => void
  onReattach: (itemId: string) => void
  isReattaching?: boolean
}

export const SourceItemsTable: React.FC<SourceItemsTableProps> = ({
  items,
  filter,
  onFilterChange,
  onReattach,
  isReattaching = false
}) => {
  const degradedState = getDesignSystemState("degraded")

  return (
    <div className="space-y-4">
      <Space wrap>
        <Button type={filter === "all" ? "primary" : "default"} onClick={() => onFilterChange("all")}>
          All items
        </Button>
        <Button
          type={filter === "detached" ? "primary" : "default"}
          onClick={() => onFilterChange("detached")}>
          Detached
        </Button>
        <Button
          type={filter === "degraded" ? "primary" : "default"}
          onClick={() => onFilterChange("degraded")}>
          {degradedState.label}
        </Button>
      </Space>

      {items.length === 0 ? (
        <Empty description="No tracked items match this filter." />
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <Card key={item.id} size="small">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="space-y-1">
                  <Typography.Text strong>{item.normalized_relative_path}</Typography.Text>
                  <div>
                    <Typography.Text type="secondary">{item.sync_status}</Typography.Text>
                  </div>
                </div>
                {item.sync_status === "conflict_detached" ? (
                  <Button loading={isReattaching} onClick={() => onReattach(item.id)}>
                    Reattach
                  </Button>
                ) : null}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
