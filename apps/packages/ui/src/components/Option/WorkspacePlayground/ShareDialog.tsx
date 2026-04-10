import React, { useState } from "react"
import {
  Modal,
  Form,
  Select,
  Radio,
  Switch,
  Input,
  InputNumber,
  Button,
  Table,
  Tag,
  Tooltip,
  Tabs,
  message,
  Space,
  DatePicker,
} from "antd"
import { Copy, Link2, Trash2 } from "lucide-react"
import {
  useWorkspaceShares,
  useShareWorkspace,
  useRevokeShare,
  useShareTokens,
  useCreateToken,
  useRevokeToken,
} from "@/hooks/useSharing"
import {
  ACCESS_LEVEL_LABELS,
  ACCESS_LEVEL_COLORS,
  type AccessLevel,
  type ShareScopeType,
} from "@/types/sharing"

interface ShareDialogProps {
  workspaceId: string
  open: boolean
  onClose: () => void
}

export const ShareDialog: React.FC<ShareDialogProps> = ({
  workspaceId,
  open,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState("team")

  return (
    <Modal
      title="Share Workspace"
      open={open}
      onCancel={onClose}
      footer={null}
      width={640}
      destroyOnHidden
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: "team",
            label: "Team / Org",
            children: (
              <TeamShareTab workspaceId={workspaceId} />
            ),
          },
          {
            key: "link",
            label: "Share Link",
            children: (
              <LinkShareTab workspaceId={workspaceId} />
            ),
          },
          {
            key: "active",
            label: "Active Shares",
            children: (
              <ActiveSharesTab workspaceId={workspaceId} />
            ),
          },
        ]}
      />
    </Modal>
  )
}

// ── Team Share Tab ──

const TeamShareTab: React.FC<{ workspaceId: string }> = ({ workspaceId }) => {
  const [form] = Form.useForm()
  const shareMutation = useShareWorkspace(workspaceId)

  const onFinish = async (values: {
    scope_type: ShareScopeType
    scope_id: number
    access_level: AccessLevel
    allow_clone: boolean
  }) => {
    try {
      await shareMutation.mutateAsync({
        share_scope_type: values.scope_type,
        share_scope_id: values.scope_id,
        access_level: values.access_level,
        allow_clone: values.allow_clone,
      })
      message.success("Workspace shared successfully")
      form.resetFields()
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Failed to share workspace")
    }
  }

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onFinish}
      initialValues={{
        scope_type: "team",
        access_level: "view_chat",
        allow_clone: true,
      }}
    >
      <Form.Item
        name="scope_type"
        label="Share with"
        rules={[{ required: true }]}
      >
        <Select>
          <Select.Option value="team">Team</Select.Option>
          <Select.Option value="org">Organization</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item
        name="scope_id"
        label="Team / Org ID"
        rules={[{ required: true, message: "Enter the team or org ID" }]}
      >
        <InputNumber style={{ width: "100%" }} min={1} placeholder="Enter ID" />
      </Form.Item>

      <Form.Item name="access_level" label="Access Level">
        <Radio.Group>
          <Radio.Button value="view_chat">{ACCESS_LEVEL_LABELS.view_chat}</Radio.Button>
          <Radio.Button value="view_chat_add">{ACCESS_LEVEL_LABELS.view_chat_add}</Radio.Button>
          <Radio.Button value="full_edit">{ACCESS_LEVEL_LABELS.full_edit}</Radio.Button>
        </Radio.Group>
      </Form.Item>

      <Form.Item name="allow_clone" label="Allow Cloning" valuePropName="checked">
        <Switch />
      </Form.Item>

      <Button
        type="primary"
        htmlType="submit"
        loading={shareMutation.isPending}
        block
      >
        Share
      </Button>
    </Form>
  )
}

// ── Link Share Tab ──

const LinkShareTab: React.FC<{ workspaceId: string }> = ({ workspaceId }) => {
  const [form] = Form.useForm()
  const [generatedLink, setGeneratedLink] = useState<string | null>(null)
  const createToken = useCreateToken()

  const onFinish = async (values: {
    access_level: AccessLevel
    allow_clone: boolean
    password?: string
    max_uses?: number
    expires_at?: any
  }) => {
    try {
      const result = await createToken.mutateAsync({
        resource_type: "workspace",
        resource_id: workspaceId,
        access_level: values.access_level,
        allow_clone: values.allow_clone,
        password: values.password || undefined,
        max_uses: values.max_uses || undefined,
        expires_at: values.expires_at
          ? values.expires_at.toISOString()
          : undefined,
      })
      if (result.raw_token) {
        const link = `${window.location.origin}/share/${result.raw_token}`
        setGeneratedLink(link)
        message.success("Share link created")
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : "Failed to create share link")
    }
  }

  const copyLink = () => {
    if (generatedLink) {
      navigator.clipboard.writeText(generatedLink)
      message.success("Link copied to clipboard")
    }
  }

  return (
    <div>
      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
        initialValues={{
          access_level: "view_chat",
          allow_clone: true,
        }}
      >
        <Form.Item name="access_level" label="Access Level">
          <Radio.Group>
            <Radio.Button value="view_chat">{ACCESS_LEVEL_LABELS.view_chat}</Radio.Button>
            <Radio.Button value="view_chat_add">{ACCESS_LEVEL_LABELS.view_chat_add}</Radio.Button>
            <Radio.Button value="full_edit">{ACCESS_LEVEL_LABELS.full_edit}</Radio.Button>
          </Radio.Group>
        </Form.Item>

        <Form.Item name="allow_clone" label="Allow Cloning" valuePropName="checked">
          <Switch />
        </Form.Item>

        <Form.Item name="password" label="Password (optional)">
          <Input.Password placeholder="Leave empty for no password" />
        </Form.Item>

        <Form.Item name="max_uses" label="Max Uses (optional)">
          <InputNumber
            style={{ width: "100%" }}
            min={1}
            max={10000}
            placeholder="Unlimited"
          />
        </Form.Item>

        <Form.Item name="expires_at" label="Expires At (optional)">
          <DatePicker
            showTime
            style={{ width: "100%" }}
            placeholder="No expiration"
          />
        </Form.Item>

        <Button
          type="primary"
          htmlType="submit"
          loading={createToken.isPending}
          icon={<Link2 className="h-4 w-4" />}
          block
        >
          Generate Link
        </Button>
      </Form>

      {generatedLink && (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-border bg-surface p-3">
          <Input value={generatedLink} readOnly className="flex-1" />
          <Tooltip title="Copy link">
            <Button
              icon={<Copy className="h-4 w-4" />}
              onClick={copyLink}
            />
          </Tooltip>
        </div>
      )}
    </div>
  )
}

// ── Active Shares Tab ──

const ActiveSharesTab: React.FC<{ workspaceId: string }> = ({
  workspaceId,
}) => {
  const { data, isLoading } = useWorkspaceShares(workspaceId)
  const revokeMutation = useRevokeShare()
  const { data: tokensData } = useShareTokens()
  const revokeTokenMutation = useRevokeToken()

  const shareColumns = [
    {
      title: "Scope",
      dataIndex: "share_scope_type",
      key: "scope",
      render: (type: string, record: any) => (
        <span>
          {type === "team" ? "Team" : "Org"} #{record.share_scope_id}
        </span>
      ),
    },
    {
      title: "Access",
      dataIndex: "access_level",
      key: "access",
      render: (level: string) => (
        <Tag color={ACCESS_LEVEL_COLORS[level] || "default"}>
          {ACCESS_LEVEL_LABELS[level as AccessLevel] || level}
        </Tag>
      ),
    },
    {
      title: "Clone",
      dataIndex: "allow_clone",
      key: "clone",
      render: (v: boolean) => (v ? "Yes" : "No"),
    },
    {
      title: "",
      key: "actions",
      render: (_: any, record: any) => (
        <Button
          type="text"
          danger
          size="small"
          icon={<Trash2 className="h-3.5 w-3.5" />}
          loading={revokeMutation.isPending}
          onClick={() => {
            revokeMutation.mutate(record.id)
          }}
        >
          Revoke
        </Button>
      ),
    },
  ]

  const tokenColumns = [
    {
      title: "Prefix",
      dataIndex: "token_prefix",
      key: "prefix",
      render: (p: string) => <code>{p}...</code>,
    },
    {
      title: "Uses",
      key: "uses",
      render: (_: any, record: any) => (
        <span>
          {record.use_count}
          {record.max_uses ? ` / ${record.max_uses}` : ""}
        </span>
      ),
    },
    {
      title: "Password",
      dataIndex: "is_password_protected",
      key: "pw",
      render: (v: boolean) => (v ? "Yes" : "No"),
    },
    {
      title: "",
      key: "actions",
      render: (_: any, record: any) =>
        !record.is_revoked && (
          <Button
            type="text"
            danger
            size="small"
            icon={<Trash2 className="h-3.5 w-3.5" />}
            loading={revokeTokenMutation.isPending}
            onClick={() => {
              revokeTokenMutation.mutate(record.id)
            }}
          >
            Revoke
          </Button>
        ),
    },
  ]

  const workspaceTokens = (tokensData?.tokens || []).filter(
    (t) => t.resource_id === workspaceId && !t.is_revoked
  )

  return (
    <div className="space-y-4">
      <div>
        <h4 className="mb-2 text-sm font-medium text-text-muted">Team/Org Shares</h4>
        <Table
          dataSource={data?.shares || []}
          columns={shareColumns}
          rowKey="id"
          loading={isLoading}
          size="small"
          pagination={false}
          locale={{ emptyText: "No active shares" }}
        />
      </div>

      <div>
        <h4 className="mb-2 text-sm font-medium text-text-muted">Share Links</h4>
        <Table
          dataSource={workspaceTokens}
          columns={tokenColumns}
          rowKey="id"
          size="small"
          pagination={false}
          locale={{ emptyText: "No share links" }}
        />
      </div>
    </div>
  )
}
