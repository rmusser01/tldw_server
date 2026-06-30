import React from "react"
import { Tag } from "antd"

import { getDesignSystemState } from "@/design-system"
import {
  Badge,
  getBadgeVariantForDesignSystemSeverity
} from "@/components/ui/primitives"
import type { IngestionSourceSummary } from "@/types/ingestion-sources"

type SourceStatusPanelsProps = {
  source: IngestionSourceSummary
}

export const SourceStatusPanels: React.FC<SourceStatusPanelsProps> = ({ source }) => {
  const summary = source.last_successful_sync_summary
  if (!summary) {
    return (
      <div className="flex flex-wrap gap-2">
        <Tag>{source.last_sync_status || "Unknown status"}</Tag>
      </div>
    )
  }

  const degradedState =
    summary.degraded_count > 0 ? getDesignSystemState("degraded") : null

  return (
    <div className="flex flex-wrap gap-2">
      <Tag color="blue">Changed {summary.changed_count}</Tag>
      {degradedState ? (
        <Badge
          variant={getBadgeVariantForDesignSystemSeverity(
            degradedState.severity
          )}
          size="md"
        >
          {degradedState.label} {summary.degraded_count}
        </Badge>
      ) : null}
      {summary.conflict_count > 0 && <Tag color="volcano">Detached {summary.conflict_count}</Tag>}
      <Tag>{source.last_sync_status || "Unknown status"}</Tag>
    </div>
  )
}
