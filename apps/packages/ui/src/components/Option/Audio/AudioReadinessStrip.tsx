import React from "react"
import { Tag, Tooltip, Typography } from "antd"

import type { MetadataSource, ReadinessItem, ReadinessState } from "./audio-readiness"

const { Text } = Typography

const STATE_COLOR: Record<ReadinessState, string | undefined> = {
  ready: "success",
  warning: "warning",
  blocked: "error",
  unknown: undefined
}

const STATE_LABEL: Record<ReadinessState, string> = {
  ready: "Ready",
  warning: "Needs review",
  blocked: "Blocked",
  unknown: "Unknown"
}

const SOURCE_LABEL: Record<MetadataSource, string> = {
  health: "model health",
  static_catalog: "static catalog",
  provider: "provider metadata",
  response_schema: "response schema",
  unknown: "unknown source"
}

export function AudioReadinessStrip({
  items,
  label = "Audio readiness"
}: {
  items: ReadinessItem[]
  label?: string
}) {
  if (items.length === 0) return null

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded border border-border bg-background-subtle px-3 py-2"
      role="status"
      aria-label={label}
    >
      <Text className="text-xs font-medium">{label}</Text>
      {items.map((item) => {
        const stateLabel = STATE_LABEL[item.state]
        const detail = item.source
          ? `${item.detail} Source: ${SOURCE_LABEL[item.source]}.`
          : item.detail
        return (
          <Tooltip key={item.id} title={detail}>
            <Tag
              color={STATE_COLOR[item.state]}
              className="m-0 max-w-full whitespace-normal"
              aria-label={`${item.label}: ${stateLabel}. ${detail}`}
            >
              {item.label}: {stateLabel}
            </Tag>
          </Tooltip>
        )
      })}
    </div>
  )
}
