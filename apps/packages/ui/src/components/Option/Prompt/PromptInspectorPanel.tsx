import React from "react"
import { Button, Drawer, Tag, Typography } from "antd"
import { Copy, MessageCircle, Pen, Trash2 } from "lucide-react"
import { SyncStatusBadge } from "./SyncStatusBadge"
import type { PromptRowVM } from "./prompt-workspace-types"

const { Paragraph, Text, Title } = Typography

type PromptInspectorPanelProps = {
  open: boolean
  prompt: PromptRowVM | null
  onClose: () => void
  onEdit?: (id: string) => void
  onUseInChat?: (id: string) => void
  onDuplicate?: (id: string) => void
  onDelete?: (id: string) => void
}

export const PromptInspectorPanel: React.FC<PromptInspectorPanelProps> = ({
  open,
  prompt,
  onClose,
  onEdit,
  onUseInChat,
  onDuplicate,
  onDelete
}) => {
  return (
    <Drawer
      title="Prompt details"
      placement="right"
      size="default"
      open={open}
      onClose={onClose}
      data-testid="prompts-inspector-panel-scaffold"
    >
      {prompt ? (
        <div className="space-y-4">
          <div className="space-y-1">
            <Title level={5} className="!mb-0">
              {prompt.title}
            </Title>
            {prompt.author ? (
              <Text type="secondary" className="text-xs">
                by {prompt.author}
              </Text>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <SyncStatusBadge
              syncStatus={prompt.syncStatus}
              sourceSystem={prompt.sourceSystem}
              serverId={prompt.serverId}
              compact={false}
            />
            <Tag>{prompt.usageCount} uses</Tag>
            {prompt.keywords.slice(0, 3).map((keyword) => (
              <Tag key={`${prompt.id}-${keyword}`}>{keyword}</Tag>
            ))}
            {prompt.keywords.length > 3 ? (
              <Tag>+{prompt.keywords.length - 3}</Tag>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="primary"
              icon={<MessageCircle className="size-4" />}
              onClick={() => onUseInChat?.(prompt.id)}
            >
              Use
            </Button>
            <Button
              icon={<Pen className="size-4" />}
              onClick={() => onEdit?.(prompt.id)}
            >
              Edit
            </Button>
            <Button
              icon={<Copy className="size-4" />}
              onClick={() => onDuplicate?.(prompt.id)}
            >
              Duplicate
            </Button>
            <Button
              danger
              icon={<Trash2 className="size-4" />}
              onClick={() => onDelete?.(prompt.id)}
            >
              Delete
            </Button>
          </div>

          {prompt.previewSystem ? (
            <div className="space-y-1">
              <Text strong className="text-xs uppercase tracking-wide text-text-muted">
                System prompt
              </Text>
              <Paragraph className="!mb-0 whitespace-pre-wrap rounded border border-border bg-surface2 p-3 text-sm">
                {prompt.previewSystem}
              </Paragraph>
            </div>
          ) : null}

          {prompt.previewUser ? (
            <div className="space-y-1">
              <Text strong className="text-xs uppercase tracking-wide text-text-muted">
                User prompt
              </Text>
              <Paragraph className="!mb-0 whitespace-pre-wrap rounded border border-border bg-surface2 p-3 text-sm">
                {prompt.previewUser}
              </Paragraph>
            </div>
          ) : null}

          {prompt.details ? (
            <div className="space-y-1">
              <Text strong className="text-xs uppercase tracking-wide text-text-muted">
                Notes
              </Text>
              <Paragraph className="!mb-0 whitespace-pre-wrap text-sm text-text-muted">
                {prompt.details}
              </Paragraph>
            </div>
          ) : null}
        </div>
      ) : (
        <Text type="secondary">Select a prompt to inspect details.</Text>
      )}
    </Drawer>
  )
}
