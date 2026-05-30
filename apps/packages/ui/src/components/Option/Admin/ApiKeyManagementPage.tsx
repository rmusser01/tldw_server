import React, { useState, useRef, useCallback, useEffect } from "react"
import {
  Card,
  Table,
  Button,
  Input,
  Modal,
  Form,
  Tag,
  Space,
  Select,
  Popconfirm,
  message
} from "antd"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { Alert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const ApiKeyManagementPage: React.FC = () => {
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // User selection state
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)
  const [users, setUsers] = useState<any[]>([])
  const [usersLoading, setUsersLoading] = useState(false)

  // API keys state
  const [keys, setKeys] = useState<any[]>([])
  const [keysLoading, setKeysLoading] = useState(false)
  const [keysError, setKeysError] = useState<string | null>(null)

  // Create key modal
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)

  // New key display (shown after creation with the raw key value)
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null)

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // Load users for the selector
  const loadUsers = useCallback(async () => {
    setUsersLoading(true)
    try {
      const result = await tldwClient.listAdminUsers({ limit: 100 })
      setUsers(result.users || [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setUsersLoading(false)
    }
  }, [markAdminGuardFromError])

  // Load API keys for selected user
  const loadKeys = useCallback(async (userId: number) => {
    setKeysLoading(true)
    setKeysError(null)
    try {
      const result = await tldwClient.listUserApiKeys(userId)
      setKeys(Array.isArray(result) ? result : [])
    } catch (err: any) {
      markAdminGuardFromError(err)
      setKeysError(sanitizeAdminErrorMessage(err, "Failed to load API keys"))
    } finally {
      setKeysLoading(false)
    }
  }, [markAdminGuardFromError])

  // Initial load
  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadUsers()
  }, [loadUsers])

  // Load keys when user selected
  useEffect(() => {
    if (selectedUserId) {
      void loadKeys(selectedUserId)
    } else {
      setKeys([])
    }
  }, [selectedUserId, loadKeys])

  // Create key handler
  const handleCreateKey = async () => {
    if (!selectedUserId) return
    try {
      const values = await createForm.validateFields()
      setCreating(true)
      const result = await tldwClient.createUserApiKey(selectedUserId, {
        name: values.name?.trim() || undefined,
        rate_limit: values.rate_limit || undefined,
      })
      // Show the new key value (only visible once)
      if (result?.key || result?.api_key) {
        setNewKeyValue(result.key || result.api_key)
      }
      createForm.resetFields()
      setCreateModalOpen(false)
      await loadKeys(selectedUserId)
      message.success("API key created")
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to create key"))
    } finally {
      setCreating(false)
    }
  }

  // Revoke key handler
  const handleRevokeKey = async (keyId: number) => {
    if (!selectedUserId) return
    try {
      await tldwClient.revokeUserApiKey(selectedUserId, keyId)
      message.success("API key revoked")
      await loadKeys(selectedUserId)
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to revoke key"))
    }
  }

  // Rotate key handler
  const handleRotateKey = async (keyId: number) => {
    if (!selectedUserId) return
    try {
      const result = await tldwClient.rotateUserApiKey(selectedUserId, keyId)
      if (result?.key || result?.api_key) {
        setNewKeyValue(result.key || result.api_key)
      }
      message.success("API key rotated")
      await loadKeys(selectedUserId)
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to rotate key"))
    }
  }

  // Table columns
  const keyColumns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (name: string) => name || "\u2014",
    },
    {
      title: "Key Preview",
      dataIndex: "key_preview",
      key: "key_preview",
      render: (_: any, record: any) => {
        const preview = record.key_preview || record.prefix || record.key_prefix
        return preview ? <code>{preview}...</code> : "\u2014"
      },
    },
    {
      title: "Rate Limit",
      dataIndex: "rate_limit",
      key: "rate_limit",
      render: (val: number | null) => val ? `${val}/min` : "Default",
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleDateString() : "\u2014",
    },
    {
      title: "Status",
      dataIndex: "is_active",
      key: "is_active",
      render: (active: boolean) => (
        <Tag color={active !== false ? "green" : "red"}>
          {active !== false ? "Active" : "Revoked"}
        </Tag>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Popconfirm title="Rotate this key?" onConfirm={() => handleRotateKey(record.id)}>
            <Button size="small">Rotate</Button>
          </Popconfirm>
          <Popconfirm title="Revoke this key? This cannot be undone." onConfirm={() => handleRevokeKey(record.id)}>
            <Button size="small" danger>Revoke</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  // Render
  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title="Access Denied">
        You don't have permission to manage API keys.
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title="Not Available">
        API key management is not available on this server.
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h2 style={{ marginBottom: 16 }}>API Key Management</h2>

      {/* New key alert */}
      {newKeyValue && (
        <Alert
          variant="success"
          title="New API Key Created"
          dismissible
          onDismiss={() => setNewKeyValue(null)}
          className="mb-4"
        >
          <div>
            <p>Copy this key now -- it will not be shown again:</p>
            <code className="block break-all rounded border border-border bg-surface2 px-3 py-2 font-mono text-sm text-foreground">
              {newKeyValue}
            </code>
          </div>
        </Alert>
      )}

      {/* User selector */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <span>Select User:</span>
          <Select
            showSearch
            placeholder="Search users..."
            style={{ width: 300 }}
            loading={usersLoading}
            value={selectedUserId}
            onChange={(val) => setSelectedUserId(val)}
            optionFilterProp="label"
            options={users.map((u: any) => ({
              value: u.id,
              label: `${u.username} (${u.email || "no email"})`,
            }))}
          />
        </Space>
      </Card>

      {/* Keys table */}
      {selectedUserId && (
        <Card
          title="API Keys"
          extra={
            <Button type="primary" onClick={() => setCreateModalOpen(true)}>
              Create Key
            </Button>
          }
        >
          {keysError && (
            <Alert variant="error" className="mb-3">
              {keysError}
            </Alert>
          )}
          <Table
            dataSource={keys}
            columns={keyColumns}
            rowKey="id"
            loading={keysLoading}
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {/* Create key modal */}
      <Modal
        title="Create API Key"
        open={createModalOpen}
        onOk={handleCreateKey}
        onCancel={() => setCreateModalOpen(false)}
        confirmLoading={creating}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item name="name" label="Key Name (optional)">
            <Input placeholder="e.g. Production Key" />
          </Form.Item>
          <Form.Item name="rate_limit" label="Rate Limit (requests/minute, optional)">
            <Input type="number" placeholder="Default" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default ApiKeyManagementPage
