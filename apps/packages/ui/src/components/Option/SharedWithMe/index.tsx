import React from "react"
import { Card, Tag, Button, Empty, Spin, message, Modal, Input } from "antd"
import { ExternalLink, Copy, Users } from "lucide-react"
import { useSharedWithMe, useCloneWorkspace } from "@/hooks/useSharing"
import {
  ACCESS_LEVEL_LABELS,
  ACCESS_LEVEL_COLORS,
  type AccessLevel,
} from "@/types/sharing"

export const SharedWithMe: React.FC = () => {
  const { data, isLoading, error } = useSharedWithMe()
  const cloneMutation = useCloneWorkspace()
  const [cloneTarget, setCloneTarget] = React.useState<{
    shareId: number
    name: string
  } | null>(null)
  const [cloneName, setCloneName] = React.useState("")

  const handleClone = async () => {
    if (!cloneTarget) return
    try {
      await cloneMutation.mutateAsync({
        shareId: cloneTarget.shareId,
        new_name: cloneName || undefined,
      })
      message.success("Clone job started")
      setCloneTarget(null)
      setCloneName("")
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Failed to clone workspace")
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spin size="large" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Empty description="Failed to load shared workspaces" />
      </div>
    )
  }

  const items = data?.items || []

  return (
    <div className="mx-auto max-w-5xl p-6">
      <div className="mb-6 flex items-center gap-3">
        <Users className="h-6 w-6 text-text-muted" />
        <h1 className="text-2xl font-semibold text-text">Shared With Me</h1>
        <Tag className="ml-2">{items.length}</Tag>
      </div>

      {items.length === 0 ? (
        <Empty
          description="No workspaces have been shared with you yet"
          className="mt-16"
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <Card
              key={item.share_id}
              className="transition hover:shadow-md"
              actions={[
                <Button
                  key="open"
                  type="link"
                  size="small"
                  icon={<ExternalLink className="h-3.5 w-3.5" />}
                  href={`/research-workspace?shared=${item.share_id}`}
                >
                  Open
                </Button>,
                ...(item.allow_clone
                  ? [
                      <Button
                        key="clone"
                        type="link"
                        size="small"
                        icon={<Copy className="h-3.5 w-3.5" />}
                        onClick={() =>
                          setCloneTarget({
                            shareId: item.share_id,
                            name: item.workspace_name || item.workspace_id,
                          })
                        }
                      >
                        Clone
                      </Button>,
                    ]
                  : []),
              ]}
            >
              <Card.Meta
                title={
                  <span className="text-sm font-medium">
                    {item.workspace_name || item.workspace_id}
                  </span>
                }
                description={
                  <div className="space-y-1">
                    <div className="text-xs text-text-muted">
                      Owner: #{item.owner_user_id}
                      {item.owner_username && ` (${item.owner_username})`}
                    </div>
                    <Tag
                      color={
                        ACCESS_LEVEL_COLORS[item.access_level] || "default"
                      }
                    >
                      {ACCESS_LEVEL_LABELS[item.access_level as AccessLevel] ||
                        item.access_level}
                    </Tag>
                    {item.shared_at && (
                      <div className="text-xs text-text-muted">
                        Shared{" "}
                        {new Date(item.shared_at).toLocaleDateString()}
                      </div>
                    )}
                  </div>
                }
              />
            </Card>
          ))}
        </div>
      )}

      <Modal
        title="Clone Workspace"
        open={!!cloneTarget}
        onCancel={() => setCloneTarget(null)}
        onOk={handleClone}
        confirmLoading={cloneMutation.isPending}
        okText="Clone"
      >
        <p className="mb-3 text-sm text-text-muted">
          Create a personal copy of this workspace. Your clone will be
          independent from the original.
        </p>
        <Input
          placeholder="Clone name (optional)"
          value={cloneName}
          onChange={(e) => setCloneName(e.target.value)}
        />
      </Modal>
    </div>
  )
}

export default SharedWithMe
