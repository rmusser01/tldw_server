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
import { formatAdminDateTime } from "./admin-format"
import { Alert } from "@/components/ui/primitives"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"

import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { useAdminRequestScope, useAdminMutation } from "./admin-request-scope"

const ApiKeyManagementPage: React.FC = () => {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const { t } = useTranslation("settings")
  if (!config || authorityLoading) {
    return (
      <p role="status">
        {t("admin.resolvingConnection", "Resolving admin connection…")}
      </p>
    )
  }
  if (config.authMode === "single-user") {
    return (
      <Alert
        variant="info"
        title={t("adminApiKeys.singleUserTitle", "Single-user API key")}
      >
        {t(
          "adminApiKeys.singleUserBody",
          "This server accepts only the primary API key configured on the server. Change that key in the server configuration. Per-user key creation and rotation are available only in multi-user mode."
        )}
      </Alert>
    )
  }
  return (
    <ApiKeyManagementWorkspace
      key={deriveConnectionAuthorityId(config)}
      config={config}
    />
  )
}

const ApiKeyManagementWorkspace = ({ config }: { config: TldwConfig }) => {
  const { t } = useTranslation(["settings", "common"])
  const captureScope = useAdminRequestScope(config)
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(
    null
  )
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)
  const [users, setUsers] = useState<any[]>([])
  const [usersLoading, setUsersLoading] = useState(false)
  const [usersError, setUsersError] = useState<string | null>(null)
  const usersRequest = useRef(0)

  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const loadUsers = useCallback(async (search?: string) => {
    const scope = captureScope()
    if (!scope) return
    const request = ++usersRequest.current
    const current = () =>
      !scope.abortSignal.aborted && request === usersRequest.current
    setUsersLoading(true)
    setUsersError(null)
    try {
      const result = await tldwClient.listAdminUsers({ limit: 20, ...(search ? { search } : {}) }, scope)
      if (!current()) return
      const loaded = result.users || []
      setUsers(loaded)
      if (!search && loaded.length === 1)
        setSelectedUserId((value) => value ?? loaded[0].id)
    } catch (err) {
      if (!current()) return
      const guard = deriveAdminGuardFromError(err)
      if (guard) setAdminGuard(guard)
      setUsersError(
        sanitizeAdminErrorMessage(
          err,
          t(
            "settings:adminApiKeys.usersLoadFailed",
            "Failed to load the user list."
          )
        )
      )
    } finally {
      if (current()) setUsersLoading(false)
    }
  }, [captureScope, t])
  // Remote search so accounts beyond the first 20 stay reachable on large
  // servers; the Select filters nothing locally (filterOption={false}).
  const handleUserSearch = useCallback(
    (term: string) => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
      searchTimerRef.current = setTimeout(() => {
        void loadUsers(term.trim() || undefined)
      }, 300)
    },
    [loadUsers]
  )

  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    }
  }, [])

  useEffect(() => {
    void loadUsers()
  }, [loadUsers])
  if (adminGuard === "forbidden") {
    return (
      <Alert
        variant="error"
        title={t("settings:adminApiKeys.forbiddenTitle", "Access Denied")}
      >
        {t(
          "settings:adminApiKeys.forbiddenBody",
          "You don't have permission to manage API keys."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert
        variant="warning"
        title={t("settings:adminApiKeys.notFoundTitle", "Not Available")}
      >
        {t(
          "settings:adminApiKeys.notFoundBody",
          "API key management is not available on this server."
        )}
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 4, fontSize: "1.5rem", fontWeight: 600 }}>
        {t("settings:adminApiKeys.title", "API Key Management")}
      </h1>
      <p
        style={{ marginBottom: 16, color: "var(--color-text-secondary, #888)" }}
      >
        {t(
          "settings:adminApiKeys.description",
          "Create, rotate, and revoke the API keys a user presents to authenticate against this server."
        )}
      </p>

      {usersError && (
        <Alert
          variant="error"
          title={t(
            "settings:adminApiKeys.usersErrorTitle",
            "Unable to load users"
          )}
          className="mb-4"
        >
          <Space orientation="vertical" size="small">
            <span>{usersError}</span>
            <Button size="small" onClick={() => void loadUsers()}>
              {t("common:retry", "Retry")}
            </Button>
          </Space>
        </Alert>
      )}

      {/* User selector */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <span>{t("settings:adminApiKeys.selectUser", "Select User:")}</span>
          <Select
            showSearch
            placeholder={t(
              "settings:adminApiKeys.searchUsers",
              "Search users..."
            )}
            style={{ width: 300 }}
            loading={usersLoading}
            value={selectedUserId}
            onChange={(val) => setSelectedUserId(val)}
            optionFilterProp="label"
            options={users.map((u: any) => ({
              value: u.id,
              label: `${u.username} (${u.email || t("settings:adminApiKeys.noEmail", "no email")})`
            }))}
          />
        </Space>
      </Card>

      {/* Pre-selection guidance (multi-user servers with several accounts) */}
      {!selectedUserId && !usersLoading && !usersError && (
        <Card size="small">
          <p style={{ margin: 0, color: "var(--color-text-secondary, #888)" }}>
            {users.length === 0
              ? t(
                  "settings:adminApiKeys.noUsers",
                  "No users were found on this server."
                )
              : t(
                  "settings:adminApiKeys.selectUserHint",
                  "Select a user above to view and manage their API keys."
                )}
          </p>
        </Card>
      )}

      {selectedUserId !== null && (
        <UserApiKeys
          key={selectedUserId}
          selectedUserId={selectedUserId}
          config={config}
        />
      )}
    </div>
  )
}

const UserApiKeys = ({
  selectedUserId,
  config
}: {
  selectedUserId: number
  config: TldwConfig
}) => {
  const { t } = useTranslation(["settings", "common"])
  const captureScope = useAdminRequestScope(config)
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(
    null
  )
  const [keys, setKeys] = useState<any[]>([])
  const [keysLoading, setKeysLoading] = useState(false)
  const [keysError, setKeysError] = useState<string | null>(null)
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [createForm] = Form.useForm()
  const { runMutation, mutating } = useAdminMutation(captureScope)
  const keysRequest = useRef(0)
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null)
  const [newKeyRevealed, setNewKeyRevealed] = useState(true)

  const loadKeys = useCallback(
    async (scope = captureScope()) => {
      if (!scope || scope.abortSignal.aborted) return
      const request = ++keysRequest.current
      const current = () =>
        !scope.abortSignal.aborted && request === keysRequest.current
      setKeysLoading(true)
      setKeysError(null)
      try {
        const result = await tldwClient.listUserApiKeys(selectedUserId, scope)
        if (current()) setKeys(Array.isArray(result) ? result : [])
      } catch (err) {
        if (!current()) return
        const guard = deriveAdminGuardFromError(err)
        if (guard) setAdminGuard(guard)
        setKeysError(
          sanitizeAdminErrorMessage(
            err,
            t("settings:adminApiKeys.keysLoadFailed", "Failed to load API keys")
          )
        )
      } finally {
        if (current()) setKeysLoading(false)
      }
    },
    [captureScope, selectedUserId, t]
  )

  useEffect(() => {
    void loadKeys()
  }, [loadKeys])

  const handleCreateKey = () =>
    runMutation(
      t("settings:adminApiKeys.createFailed", "Failed to create key"),
      async (scope) => {
        const values = await createForm.validateFields()
        if (scope.abortSignal.aborted) return
        const result = await tldwClient.createUserApiKey(
          selectedUserId,
          {
            name: values.name?.trim() || undefined
          },
          scope
        )
        if (scope.abortSignal.aborted) return
        if (result?.key || result?.api_key) {
          setNewKeyValue(result.key || result.api_key)
          setNewKeyRevealed(true)
        }
        createForm.resetFields()
        setCreateModalOpen(false)
        await loadKeys(scope)
        if (!scope.abortSignal.aborted)
          message.success(t("settings:adminApiKeys.created", "API key created"))
      }
    )

  const handleRevokeKey = (keyId: number) =>
    runMutation(
      t("settings:adminApiKeys.revokeFailed", "Failed to revoke key"),
      async (scope) => {
        await tldwClient.revokeUserApiKey(selectedUserId, keyId, scope)
        if (scope.abortSignal.aborted) return
        message.success(t("settings:adminApiKeys.revoked", "API key revoked"))
        await loadKeys(scope)
      }
    )

  const handleRotateKey = (keyId: number) =>
    runMutation(
      t("settings:adminApiKeys.rotateFailed", "Failed to rotate key"),
      async (scope) => {
        const result = await tldwClient.rotateUserApiKey(
          selectedUserId,
          keyId,
          scope
        )
        if (scope.abortSignal.aborted) return
        if (result?.key || result?.api_key) {
          setNewKeyValue(result.key || result.api_key)
          setNewKeyRevealed(true)
        }
        message.success(t("settings:adminApiKeys.rotated", "API key rotated"))
        await loadKeys(scope)
      }
    )
  // Table columns
  const keyColumns = [
    {
      title: t("settings:adminApiKeys.colName", "Name"),
      dataIndex: "name",
      key: "name",
      render: (name: string) => name || "\u2014"
    },
    {
      title: t("settings:adminApiKeys.colPreview", "Key Preview"),
      dataIndex: "key_preview",
      key: "key_preview",
      render: (_: any, record: any) => {
        const preview = record.key_preview || record.prefix || record.key_prefix
        return preview ? <code>{preview}...</code> : "\u2014"
      }
    },
    {
      title: t("settings:adminApiKeys.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? formatAdminDateTime(val) : "\u2014",
    },
    {
      title: t("settings:adminApiKeys.colStatus", "Status"),
      dataIndex: "is_active",
      key: "is_active",
      render: (active: boolean) => (
        <Tag color={active !== false ? "green" : "red"}>
          {active !== false
            ? t("settings:adminApiKeys.statusActive", "Active")
            : t("settings:adminApiKeys.statusRevoked", "Revoked")}
        </Tag>
      )
    },
    {
      title: t("settings:adminApiKeys.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Popconfirm
            title={t("settings:adminApiKeys.rotateConfirm", "Rotate this key?")}
            okText={t("settings:adminApiKeys.rotateOk", "Rotate key")}
            onConfirm={() => handleRotateKey(record.id)}
          >
            <Button size="small" disabled={mutating}>
              {t("settings:adminApiKeys.rotate", "Rotate")}
            </Button>
          </Popconfirm>
          <Popconfirm
            title={t(
              "settings:adminApiKeys.revokeConfirm",
              "Revoke this key? This cannot be undone."
            )}
            okText={t("settings:adminApiKeys.revokeOk", "Revoke key")}
            okButtonProps={{ danger: true }}
            onConfirm={() => handleRevokeKey(record.id)}
          >
            <Button size="small" danger disabled={mutating}>
              {t("settings:adminApiKeys.revoke", "Revoke")}
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  if (adminGuard === "forbidden") {
    return (
      <Alert
        variant="error"
        title={t("settings:adminApiKeys.forbiddenTitle", "Access Denied")}
      >
        {t(
          "settings:adminApiKeys.forbiddenBody",
          "You don't have permission to manage API keys."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert
        variant="warning"
        title={t("settings:adminApiKeys.notFoundTitle", "Not Available")}
      >
        {t(
          "settings:adminApiKeys.notFoundBody",
          "API key management is not available on this server."
        )}
      </Alert>
    )
  }

  return (
    <>
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
            <p>
              {t(
                "settings:adminApiKeys.newKeyCopyHint",
                "Copy this key now -- it will not be shown again:"
              )}
            </p>
            <code className="block break-all rounded border border-border bg-surface2 px-3 py-2 font-mono text-sm text-foreground">
              {newKeyRevealed
                ? newKeyValue
                : `${newKeyValue.slice(0, 12)}${"•".repeat(12)}`}
            </code>
            <Space className="mt-2">
              <Button
                size="small"
                type="primary"
                onClick={() => {
                  const scope = captureScope()
                  if (!scope) return
                  void (async () => {
                    try {
                      if (!navigator.clipboard?.writeText) {
                        throw new Error("Clipboard unavailable")
                      }
                      await navigator.clipboard.writeText(newKeyValue)
                      if (scope.abortSignal.aborted) return
                      message.success(
                        t("settings:adminApiKeys.newKeyCopied", "Key copied")
                      )
                      // Mask only once the copy actually succeeded - a denied
                      // clipboard must never hide the only visible credential.
                      setNewKeyRevealed(false)
                    } catch {
                      if (scope.abortSignal.aborted) return
                      message.error(
                        t(
                          "settings:adminApiKeys.newKeyCopyFailed",
                          "Could not access the clipboard - select and copy the key manually."
                        )
                      )
                    }
                  })()
                }}
              >
                {t("settings:adminApiKeys.newKeyCopy", "Copy key")}
              </Button>
              <Button size="small" onClick={() => setNewKeyRevealed((v) => !v)}>
                {newKeyRevealed
                  ? t("settings:adminApiKeys.newKeyHide", "Hide")
                  : t("settings:adminApiKeys.newKeyReveal", "Reveal")}
              </Button>
            </Space>
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
            onSearch={handleUserSearch}
            filterOption={false}
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
            <Button
              type="primary"
              disabled={mutating}
              onClick={() => setCreateModalOpen(true)}
            >
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
        okText={t("settings:adminApiKeys.createModalOk", "Create key")}
        onCancel={() => {
          if (!mutating) setCreateModalOpen(false)
        }}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item
            name="name"
            label={t(
              "settings:adminApiKeys.keyNameLabel",
              "Key Name (optional)"
            )}
          >
            <Input
              placeholder={t(
                "settings:adminApiKeys.keyNamePlaceholder",
                "e.g. Production Key"
              )}
            />
          </Form.Item>
          <p>
            <a href="/admin/rate-limiting">
              {t(
                "settings:adminApiKeys.rateLimitCrossLink",
                "Baseline limits and endpoint coverage live in Rate Limiting."
              )}
            </a>
          </p>
        </Form>
      </Modal>
    </>
  )
}

export default ApiKeyManagementPage
