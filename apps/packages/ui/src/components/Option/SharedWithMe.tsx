import React from "react"
import { CopyOutlined, ExportOutlined } from "@ant-design/icons"
import { Button, Card, Empty, Space, Spin, Tag, Typography, message } from "antd"
import { useNavigate } from "react-router-dom"

import { useCloneWorkspace, useSharedWithMe } from "@/hooks/useSharing"
import {
  ACCESS_LEVEL_COLORS,
  ACCESS_LEVEL_LABELS,
  type AccessLevel
} from "@/types/sharing"

const { Paragraph, Text } = Typography

const isValidAccessLevel = (accessLevel: string): accessLevel is AccessLevel =>
  Object.prototype.hasOwnProperty.call(ACCESS_LEVEL_LABELS, accessLevel) &&
  Object.prototype.hasOwnProperty.call(ACCESS_LEVEL_COLORS, accessLevel)

export const SharedWithMe: React.FC = () => {
  const { data, isLoading, error } = useSharedWithMe()
  const cloneWorkspace = useCloneWorkspace()
  const navigate = useNavigate()
  const shares = data?.items ?? []
  const [messageApi, messageContext] = message.useMessage()

  const _cloneWorkspace = (shareId: number, workspaceName: string) => {
    cloneWorkspace.mutate(
      {
        shareId,
        new_name: `${workspaceName} (Copy)`
      },
      {
        onSuccess: () => {
          messageApi.success(`Cloned "${workspaceName}" into your workspace list.`)
        },
        onError: (cloneError) => {
          messageApi.error(cloneError.message || "Failed to clone shared workspace.")
        }
      }
    )
  }

  if (isLoading) {
    return (
      <div className="flex min-h-[240px] items-center justify-center">
        <Spin size="large" />
      </div>
    )
  }

  if (error) {
    return (
      <Card>
        <Paragraph type="danger">
          {error.message || "Failed to load workspaces shared with you."}
        </Paragraph>
      </Card>
    )
  }

  if (!shares.length) {
    return (
      <Card>
        <Empty description="No shared workspaces available yet." />
      </Card>
    )
  }

  return (
    <Card title="Shared With Me">
      {messageContext}
      <ul
        aria-label="Shared workspaces"
        className="m-0 list-none divide-y divide-border p-0"
      >
        {shares.map((share) => {
          const workspaceLabel = share.workspace_name?.trim() || share.workspace_id
          const accessLevel = String(share.access_level || "")
          const hasValidAccessLevel = isValidAccessLevel(accessLevel)
          const accessLevelColor = hasValidAccessLevel
            ? ACCESS_LEVEL_COLORS[accessLevel]
            : "default"
          const accessLevelLabel = hasValidAccessLevel
            ? ACCESS_LEVEL_LABELS[accessLevel]
            : accessLevel || "Unknown access"

          return (
            <li
              key={share.share_id}
              className="flex min-w-0 flex-col gap-3 py-3 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="min-w-0 flex-1">
                <Space size="small" wrap>
                  <span>{workspaceLabel}</span>
                  <Tag color={accessLevelColor}>{accessLevelLabel}</Tag>
                </Space>
                <Space orientation="vertical" size={2} className="mt-1">
                  {share.workspace_description ? (
                    <Text type="secondary">{share.workspace_description}</Text>
                  ) : null}
                  <Text type="secondary">
                    {`Shared by workspace owner (account ${share.owner_user_id})`}
                  </Text>
                </Space>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Button
                  type="link"
                  onClick={() =>
                    navigate(`/research-workspace?shared=${share.share_id}`)
                  }
                  aria-label={`Open ${workspaceLabel}`}
                  icon={<ExportOutlined />}
                >
                  Open
                </Button>
                <Button
                  aria-label={`Clone ${workspaceLabel}`}
                  disabled={!share.allow_clone}
                  icon={<CopyOutlined />}
                  loading={
                    cloneWorkspace.isPending &&
                    cloneWorkspace.variables?.shareId === share.share_id
                  }
                  onClick={() => _cloneWorkspace(share.share_id, workspaceLabel)}
                >
                  Clone
                </Button>
              </div>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}
