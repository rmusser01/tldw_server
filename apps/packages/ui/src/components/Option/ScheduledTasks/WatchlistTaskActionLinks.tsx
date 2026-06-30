import React from "react"
import { Button, Space } from "antd"
import type { ButtonProps } from "antd"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import {
  buildWatchlistTaskLinks,
  type WatchlistTaskLinks
} from "./scheduled-task-status"

type WatchlistTaskActionLinksProps = {
  task: ScheduledTask
  size?: ButtonProps["size"]
  getAriaLabel?: (label: string, task: ScheduledTask) => string
}

const WATCHLIST_ACTION_LABELS = {
  settingsUrl: "Open monitor settings",
  activityUrl: "Open activity",
  reportsUrl: "Open reports",
  latestRunUrl: "Open latest run",
  latestOutputUrl: "Open latest report"
} as const

export const WatchlistTaskActionLinks: React.FC<WatchlistTaskActionLinksProps> = ({
  task,
  size,
  getAriaLabel
}) => {
  const links = buildWatchlistTaskLinks(task)

  return (
    <Space wrap>
      {Object.entries(WATCHLIST_ACTION_LABELS).map(([key, label]) => {
        const href = links[key as keyof WatchlistTaskLinks]
        if (!href) return null

        return (
          <Button
            key={key}
            size={size}
            type="link"
            href={href}
            target="_self"
            aria-label={getAriaLabel?.(label, task)}
          >
            {label}
          </Button>
        )
      })}
    </Space>
  )
}

export default WatchlistTaskActionLinks
