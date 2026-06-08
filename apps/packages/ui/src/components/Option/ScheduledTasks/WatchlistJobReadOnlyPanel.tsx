import React from "react"
import { Button, Card, Descriptions, Tag, Typography } from "antd"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import { buildWatchlistTaskLinks } from "./scheduled-task-status"

type WatchlistJobReadOnlyPanelProps = {
  task: ScheduledTask
  onOpenManageUrl: (task: ScheduledTask) => void
}

export const WatchlistJobReadOnlyPanel: React.FC<WatchlistJobReadOnlyPanelProps> = ({
  task,
  onOpenManageUrl
}) => {
  const manageUrl = buildWatchlistTaskLinks(task).settingsUrl ?? "/watchlists?tab=jobs"

  return (
    <Card title={task.title}>
      <div style={{ display: "flex", flexDirection: "column", gap: 16, width: "100%" }}>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
          This task is managed from Watchlists and is read-only here.
        </Typography.Paragraph>
        <Descriptions bordered size="small" column={1}>
          <Descriptions.Item label="Schedule">{task.schedule_summary || "Manual"}</Descriptions.Item>
          <Descriptions.Item label="Timezone">{task.timezone || "—"}</Descriptions.Item>
          <Descriptions.Item label="Status">
            <Tag color={task.enabled ? "green" : "default"}>{task.status}</Tag>
          </Descriptions.Item>
        </Descriptions>
        <Button href={manageUrl} onClick={() => onOpenManageUrl(task)}>
          Manage in Watchlists
        </Button>
      </div>
    </Card>
  )
}

export default WatchlistJobReadOnlyPanel
