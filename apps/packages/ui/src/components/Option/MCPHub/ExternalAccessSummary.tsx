import { Empty, Space, Typography } from "antd"
import { ProductStateBadge as Tag } from "@/components/Option/productStatePrimitives"

import type { McpHubEffectiveExternalAccess } from "@/services/tldw/mcp-hub"

import { getExternalBlockedReasonLabel } from "./policyHelpers"

type ExternalAccessSummaryProps = {
  summary?: McpHubEffectiveExternalAccess | null
  emptyText?: string
}

export const ExternalAccessSummary = ({
  summary,
  emptyText = "No external server access is currently configured."
}: ExternalAccessSummaryProps) => {
  if (!summary || summary.servers.length === 0) {
    return <Empty description={emptyText} />
  }

  return (
    <Space orientation="vertical" size="small" style={{ width: "100%" }}>
      {summary.servers.map((server) => {
        const blockedReason = getExternalBlockedReasonLabel(server.blocked_reason)
        const showBlockedReason =
          blockedReason && !(server.disabled_by_assignment && blockedReason === "Disabled by assignment")
        return (
          <Space key={server.server_id} orientation="vertical" size={4} style={{ width: "100%" }}>
            <Space wrap size="small">
              <Typography.Text strong>{server.server_name || server.server_id}</Typography.Text>
              {server.granted_by ? <Tag color="blue">{`granted by ${server.granted_by}`}</Tag> : null}
              {server.disabled_by_assignment ? <Tag color="orange">Disabled by assignment</Tag> : null}
              {server.secret_available ? <Tag color="green">secret available</Tag> : <Tag>no secret</Tag>}
              {server.runtime_executable ? <Tag color="green">runtime executable</Tag> : <Tag>not executable</Tag>}
              {showBlockedReason ? <Tag color="red">{blockedReason}</Tag> : null}
            </Space>
            {server.slots?.length ? (
              <Space wrap size="small">
                {server.slots.map((slot) => {
                  const slotBlockedReason = getExternalBlockedReasonLabel(slot.blocked_reason)
                  return (
                    <Space key={`${server.server_id}-${slot.slot_name}`} wrap size={4}>
                      <Tag color="cyan">{slot.display_name || slot.slot_name}</Tag>
                      {slot.granted_by ? <Tag>{`slot via ${slot.granted_by}`}</Tag> : null}
                      {slot.secret_available ? <Tag color="green">slot secret</Tag> : <Tag>slot missing</Tag>}
                      {slot.runtime_usable ? <Tag color="green">usable</Tag> : <Tag>blocked</Tag>}
                      {slot.disabled_by_assignment ? <Tag color="orange">slot disabled</Tag> : null}
                      {slotBlockedReason ? <Tag color="red">{slotBlockedReason}</Tag> : null}
                    </Space>
                  )
                })}
              </Space>
            ) : null}
          </Space>
        )
      })}
    </Space>
  )
}
