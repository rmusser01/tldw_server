import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
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
  const { t } = useTranslation(["settings", "common"])
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // User selection state
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)
  const [users, setUsers] = useState<any[]>([])
  const [usersLoading, setUsersLoading] = useState(false)
  const [usersError, setUsersError] = useState<string | null>(null)

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
    setUsersError(null)
    try {
      const result = await tldwClient.listAdminUsers({ limit: 100 })
      const loaded = result.users || []
      setUsers(loaded)
      // Single-user servers have exactly one account — select it directly
      // instead of asking the operator to search for themselves.
      if (loaded.length === 1) {
        setSelectedUserId((current) => current ?? loaded[0].id)
      }
    } catch (err) {
      markAdminGuardFromError(err)
      setUsersError(
        sanitizeAdminErrorMessage(err, t("settings:adminApiKeys.usersLoadFailed", "Failed to load the user list."))
      )
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
      setKeysError(sanitizeAdminErrorMessage(err, t("settings:adminApiKeys.keysLoadFailed", "Failed to load API keys")))
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
      message.success(t("settings:adminApiKeys.created", "API key created"))
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminApiKeys.createFailed", "Failed to create key")))
    } finally {
      setCreating(false)
    }
  }

  // Revoke key handler
  const handleRevokeKey = async (keyId: number) => {
    if (!selectedUserId) return
    try {
      await tldwClient.revokeUserApiKey(selectedUserId, keyId)
      message.success(t("settings:adminApiKeys.revoked", "API key revoked"))
      await loadKeys(selectedUserId)
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminApiKeys.revokeFailed", "Failed to revoke key")))
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
      message.success(t("settings:adminApiKeys.rotated", "API key rotated"))
      await loadKeys(selectedUserId)
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminApiKeys.rotateFailed", "Failed to rotate key")))
    }
  }

  // Table columns
  const keyColumns = [
    {
      title: t("settings:adminApiKeys.colName", "Name"),
      dataIndex: "name",
      key: "name",
      render: (name: string) => name || "\u2014",
    },
    {
      title: t("settings:adminApiKeys.colPreview", "Key Preview"),
      dataIndex: "key_preview",
      key: "key_preview",
      render: (_: any, record: any) => {
        const preview = record.key_preview || record.prefix || record.key_prefix
        return preview ? <code>{preview}...</code> : "\u2014"
      },
    },
    {
      title: t("settings:adminApiKeys.colRateLimit", "Rate Limit"),
      dataIndex: "rate_limit",
      key: "rate_limit",
      render: (val: number | null) => val ? `${val}/min` : t("settings:adminApiKeys.rateLimitDefault", "Default"),
    },
    {
      title: t("settings:adminApiKeys.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleDateString() : "\u2014",
    },
    {
      title: t("settings:adminApiKeys.colStatus", "Status"),
      dataIndex: "is_active",
      key: "is_active",
      render: (active: boolean) => (
        <Tag color={active !== false ? "green" : "red"}>
          {active !== false ? t("settings:adminApiKeys.statusActive", "Active") : t("settings:adminApiKeys.statusRevoked", "Revoked")}
        </Tag>
      ),
    },
    {
      title: t("settings:adminApiKeys.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Popconfirm title={t("settings:adminApiKeys.rotateConfirm", "Rotate this key?")} onConfirm={() => handleRotateKey(record.id)}>
            <Button size="small">{t("settings:adminApiKeys.rotate", "Rotate")}</Button>
          </Popconfirm>
          <Popconfirm title={t("settings:adminApiKeys.revokeConfirm", "Revoke this key? This cannot be undone.")} onConfirm={() => handleRevokeKey(record.id)}>
            <Button size="small" danger>{t("settings:adminApiKeys.revoke", "Revoke")}</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  // Render
  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminApiKeys.forbiddenTitle", "Access Denied")}>
        {t("settings:adminApiKeys.forbiddenBody", "You don't have permission to manage API keys.")}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminApiKeys.notFoundTitle", "Not Available")}>
        {t("settings:adminApiKeys.notFoundBody", "API key management is not available on this server.")}
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 4, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminApiKeys.title", "API Key Management")}</h1>
      <p style={{ marginBottom: 16, color: "var(--color-text-secondary, #888)" }}>
        {t(
          "settings:adminApiKeys.description",
          "Create, rotate, and revoke the API keys a user presents to authenticate against this server."
        )}
      </p>

      {usersError && (
        <Alert variant="error" title={t("settings:adminApiKeys.usersErrorTitle", "Unable to load users")} className="mb-4">
          <Space orientation="vertical" size="small">
            <span>{usersError}</span>
            <Button size="small" onClick={() => void loadUsers()}>
              {t("common:retry", "Retry")}
            </Button>
          </Space>
        </Alert>
      )}

      {/* New key alert */}
      {newKeyValue && (
        <Alert
          variant="success"
          title={t("settings:adminApiKeys.newKeyTitle", "New API Key Created")}
          dismissible
          onDismiss={() => setNewKeyValue(null)}
          className="mb-4"
        >
          <div>
            <p>{t("settings:adminApiKeys.newKeyCopyHint", "Copy this key now -- it will not be shown again:")}</p>
            <code className="block break-all rounded border border-border bg-surface2 px-3 py-2 font-mono text-sm text-foreground">
              {newKeyValue}
            </code>
          </div>
        </Alert>
      )}

      {/* User selector */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <span>{t("settings:adminApiKeys.selectUser", "Select User:")}</span>
          <Select
            showSearch
            placeholder={t("settings:adminApiKeys.searchUsers", "Search users...")}
            style={{ width: 300 }}
            loading={usersLoading}
            value={selectedUserId}
            onChange={(val) => setSelectedUserId(val)}
            optionFilterProp="label"
            options={users.map((u: any) => ({
              value: u.id,
              label: `${u.username} (${u.email || t("settings:adminApiKeys.noEmail", "no email")})`,
            }))}
          />
        </Space>
      </Card>

      {/* Pre-selection guidance (multi-user servers with several accounts) */}
      {!selectedUserId && !usersLoading && !usersError && (
        <Card size="small">
          <p style={{ margin: 0, color: "var(--color-text-secondary, #888)" }}>
            {users.length === 0
              ? t("settings:adminApiKeys.noUsers", "No users were found on this server.")
              : t("settings:adminApiKeys.selectUserHint", "Select a user above to view and manage their API keys.")}
          </p>
        </Card>
      )}

      {/* Keys table */}
      {selectedUserId && (
        <Card
          title={t("settings:adminApiKeys.keysCardTitle", "API Keys")}
          extra={
            <Button type="primary" onClick={() => setCreateModalOpen(true)}>
              {t("settings:adminApiKeys.createKey", "Create Key")}
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
        title={t("settings:adminApiKeys.createModalTitle", "Create API Key")}
        open={createModalOpen}
        onOk={handleCreateKey}
        onCancel={() => setCreateModalOpen(false)}
        confirmLoading={creating}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item name="name" label={t("settings:adminApiKeys.keyNameLabel", "Key Name (optional)")}>
            <Input placeholder={t("settings:adminApiKeys.keyNamePlaceholder", "e.g. Production Key")} />
          </Form.Item>
          <Form.Item name="rate_limit" label={t("settings:adminApiKeys.rateLimitLabel", "Rate Limit (requests/minute, optional)")}>
            <Input type="number" placeholder={t("settings:adminApiKeys.rateLimitDefault", "Default")} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default ApiKeyManagementPage
