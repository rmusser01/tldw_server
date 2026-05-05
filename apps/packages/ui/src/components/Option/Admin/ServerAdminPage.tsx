import React from "react"
import {
  Typography,
  Card,
  Descriptions,
  Button,
  Space,
  Alert,
  Table,
  Tag,
  Select,
  Switch,
  Divider,
  Input,
  Popconfirm,
  Form
} from "antd"
import { useTranslation } from "react-i18next"
import {
  tldwClient,
  type TldwConfig,
  type AdminUserListResponse,
  type AdminUserSummary,
  type AdminRole,
  type MediaIngestionBudgetDiagnostics
} from "@/services/tldw/TldwApiClient"
import { PageShell } from "@/components/Common/PageShell"
import { isTimeoutLikeError } from "@/utils/request-timeout"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { AdminAudioInstallerCard } from "./AdminAudioInstallerCard"
import {
  PermissionNotice,
  RecoveryCallout,
  StatePanel
} from "@/components/ui/state"

const { Title, Text } = Typography
const SYSTEM_STATS_TIMEOUT_MS = 10_000
const LEGACY_STORAGE_MAX_REASONABLE_MB = 10 * 1024 * 1024
const TLDW_SERVER_DOCUMENTATION_URL =
  "https://github.com/rmusser01/tldw_server#documentation--resources"

const formatBytesForAdmin = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "–"
  if (value < 1024) return `${value} B`

  const units = ["KiB", "MiB", "GiB", "TiB"]
  let normalized = value / 1024
  let unitIndex = 0
  while (normalized >= 1024 && unitIndex < units.length - 1) {
    normalized /= 1024
    unitIndex += 1
  }
  const rounded = normalized.toFixed(1).replace(/\.0$/, "")
  return `${rounded} ${units[unitIndex]}`
}

const formatMegabytesForAdmin = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "–"

  // Keep a narrow fallback for legacy backends that leaked raw bytes into *_mb fields.
  // Legitimate *_mb values can be large, so only treat values above a 10 TiB-equivalent
  // megabyte total as raw bytes. This still catches sub-1 GiB legacy byte payloads.
  const bytesValue =
    value > LEGACY_STORAGE_MAX_REASONABLE_MB ? value : value * 1024 * 1024
  return formatBytesForAdmin(bytesValue)
}

const formatRetryAfterForAdmin = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "–"
  if (value < 60) {
    return `~${Math.round(value)}s`
  }
  if (value < 3600) {
    return `~${Math.round(value / 60)}m`
  }

  const totalMinutes = Math.round(value / 60)
  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  if (hours >= 24) {
    const days = Math.floor(hours / 24)
    const remainingHours = hours % 24
    if (remainingHours > 0) {
      return `~${days}d ${remainingHours}h`
    }
    return `~${days}d`
  }
  return `~${hours}h ${minutes}m`
}

export const ServerAdminPage: React.FC = () => {
  const { t } = useTranslation(["option", "settings"])
  const [config, setConfig] = React.useState<TldwConfig | null>(null)
  const [stats, setStats] = React.useState<any | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [adminGuard, setAdminGuard] = React.useState<"forbidden" | "notFound" | null>(null)
  const [usersData, setUsersData] = React.useState<AdminUserListResponse | null>(null)
  const [usersLoading, setUsersLoading] = React.useState(false)
  const [usersError, setUsersError] = React.useState<string | null>(null)
  const [roles, setRoles] = React.useState<AdminRole[]>([])
  const [rolesLoading, setRolesLoading] = React.useState(false)
  const [rolesError, setRolesError] = React.useState<string | null>(null)
  const [mediaBudget, setMediaBudget] = React.useState<MediaIngestionBudgetDiagnostics | null>(null)
  const [mediaBudgetLoading, setMediaBudgetLoading] = React.useState(false)
  const [mediaBudgetError, setMediaBudgetError] = React.useState<string | null>(null)
  const [mediaBudgetUserId, setMediaBudgetUserId] = React.useState<number | null>(null)
  const [mediaBudgetPolicyId, setMediaBudgetPolicyId] = React.useState("media.default")
  const [userRoleFilter, setUserRoleFilter] = React.useState<string | undefined>(undefined)
  const [userActiveFilter, setUserActiveFilter] = React.useState<string | undefined>(undefined)
  const [usersPage, setUsersPage] = React.useState(1)
  const [usersPageSize, setUsersPageSize] = React.useState(20)
  const [updatingUserId, setUpdatingUserId] = React.useState<number | null>(null)
  const [creatingRole, setCreatingRole] = React.useState(false)
  const [deletingRoleId, setDeletingRoleId] = React.useState<number | null>(null)
  const [roleForm] = Form.useForm()
  const initialLoadRef = React.useRef(false)
  const openAdminDocumentation = React.useCallback(() => {
    window.open(TLDW_SERVER_DOCUMENTATION_URL, "_blank", "noopener,noreferrer")
  }, [])

  const markAdminGuardFromError = React.useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) {
      setAdminGuard(guardState)
    }
  }, [])

  const loadUsers = React.useCallback(
    async (
      page: number,
      limit: number,
      role?: string,
      activeFilter?: string
    ) => {
      try {
        setUsersLoading(true)
        const is_active =
          activeFilter === "active" ? true : activeFilter === "inactive" ? false : undefined
        const data = await tldwClient.listAdminUsers({
          page,
          limit,
          role,
          is_active
        })
        setUsersData(data)
        setUsersError(null)
      } catch (e: any) {
        setUsersError(sanitizeAdminErrorMessage(e, "Failed to load users."))
        markAdminGuardFromError(e)
      } finally {
        setUsersLoading(false)
      }
    },
    [markAdminGuardFromError]
  )

  const loadRoles = React.useCallback(async () => {
    try {
      setRolesLoading(true)
      const data = await tldwClient.listAdminRoles()
      setRoles(data || [])
      setRolesError(null)
    } catch (e: any) {
      setRolesError(sanitizeAdminErrorMessage(e, "Failed to load roles."))
      markAdminGuardFromError(e)
    } finally {
      setRolesLoading(false)
    }
  }, [markAdminGuardFromError])

  const loadMediaBudget = React.useCallback(
    async (userId: number, policyId: string = "media.default") => {
      try {
        setMediaBudgetLoading(true)
        const data = await tldwClient.getMediaIngestionBudgetDiagnostics({
          userId,
          policyId: policyId || "media.default"
        })
        setMediaBudget(data)
        setMediaBudgetError(null)
    } catch (e: any) {
      setMediaBudgetError(
        sanitizeAdminErrorMessage(
          e,
          "Failed to load media ingestion budget diagnostics."
        )
      )
      markAdminGuardFromError(e)
    } finally {
      setMediaBudgetLoading(false)
      }
    },
    [markAdminGuardFromError]
  )

  const loadSystemStats = React.useCallback(async () => {
    try {
      setLoading(true)
      const data = await tldwClient.getSystemStats({
        timeoutMs: SYSTEM_STATS_TIMEOUT_MS
      })
      setStats(data)
      setError(null)
    } catch (e: any) {
      const baseError = sanitizeAdminErrorMessage(
        e,
        "Failed to load system statistics."
      )
      setError(
        isTimeoutLikeError(e)
          ? (t(
              "settings:admin.systemStatsTimeout",
              "System statistics took longer than 10 seconds. Retry to try again."
            ) as string)
          : baseError
      )
      markAdminGuardFromError(e)
    } finally {
      setLoading(false)
    }
  }, [markAdminGuardFromError, t])

  React.useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    let cancelled = false
    const load = async () => {
      try {
        const cfg = await tldwClient.getConfig()
        if (!cancelled) {
          setConfig(cfg)
        }
      } catch {
        // ignore; health checks will surface errors
      }
      if (!cancelled) {
        await loadSystemStats()
      }

      // Initial users + roles
      void loadUsers(1, usersPageSize, userRoleFilter, userActiveFilter)
      void loadRoles()
    }
    load()
    return () => {
      cancelled = true
    }
  }, [
    loadRoles,
    loadSystemStats,
    loadUsers,
    userActiveFilter,
    userRoleFilter,
    usersPageSize
  ])

  const handleRefresh = async () => {
    await loadSystemStats()
    if (mediaBudgetUserId !== null && !adminGuard) {
      void loadMediaBudget(mediaBudgetUserId, mediaBudgetPolicyId)
    }
  }

  const users = stats?.users || {}
  const storage = stats?.storage || {}
  const sessions = stats?.sessions || {}
  const mediaBudgetLimits = mediaBudget?.limits || {}
  const mediaBudgetUsage = mediaBudget?.usage || {}

  React.useEffect(() => {
    if (mediaBudgetUserId !== null) {
      return
    }
    const firstUser = usersData?.users?.[0]
    if (firstUser && typeof firstUser.id === "number" && firstUser.id > 0) {
      setMediaBudgetUserId(firstUser.id)
    }
  }, [mediaBudgetUserId, usersData])

  React.useEffect(() => {
    if (adminGuard || mediaBudgetUserId === null) {
      return
    }
    void loadMediaBudget(mediaBudgetUserId, mediaBudgetPolicyId)
  }, [adminGuard, loadMediaBudget, mediaBudgetPolicyId, mediaBudgetUserId])

  const handleUserTableChange = (pagination: any) => {
    const page = pagination.current || 1
    const pageSize = pagination.pageSize || usersPageSize
    setUsersPage(page)
    setUsersPageSize(pageSize)
    void loadUsers(page, pageSize, userRoleFilter, userActiveFilter)
  }

  const handleUserFilterChange = (nextRole?: string, nextActive?: string) => {
    const role = typeof nextRole === "string" && nextRole.length > 0 ? nextRole : undefined
    const active = typeof nextActive === "string" && nextActive.length > 0 ? nextActive : undefined
    setUserRoleFilter(role)
    setUserActiveFilter(active)
    setUsersPage(1)
    void loadUsers(1, usersPageSize, role, active)
  }

  const handleToggleUserActive = async (user: AdminUserSummary, nextActive: boolean) => {
    try {
      setUpdatingUserId(user.id)
      await tldwClient.updateAdminUser(user.id, { is_active: nextActive })
      await loadUsers(usersPage, usersPageSize, userRoleFilter, userActiveFilter)
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to update user active state", e)
    } finally {
      setUpdatingUserId(null)
    }
  }

  const handleChangeUserRole = async (user: AdminUserSummary, role: string) => {
    try {
      setUpdatingUserId(user.id)
      await tldwClient.updateAdminUser(user.id, { role })
      await loadUsers(usersPage, usersPageSize, userRoleFilter, userActiveFilter)
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to update user role", e)
    } finally {
      setUpdatingUserId(null)
    }
  }

  const handleCreateRole = async () => {
    try {
      const values = await roleForm.validateFields()
      const name = String(values.name || "").trim()
      const description = values.description ? String(values.description).trim() : undefined
      if (!name) return
      setCreatingRole(true)
      await tldwClient.createAdminRole(name, description)
      roleForm.resetFields()
      await loadRoles()
    } catch (e) {
      // validation or request error; log-only
      // eslint-disable-next-line no-console
      if (e) console.error("Failed to create role", e)
    } finally {
      setCreatingRole(false)
    }
  }

  const handleDeleteRole = async (roleId: number) => {
    try {
      setDeletingRoleId(roleId)
      await tldwClient.deleteAdminRole(roleId)
      await loadRoles()
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to delete role", e)
    } finally {
      setDeletingRoleId(null)
    }
  }

  const userRoleOptions =
    roles && roles.length > 0
      ? roles.map((r) => ({ label: r.name, value: r.name }))
      : [
          { label: "user", value: "user" },
          { label: "admin", value: "admin" },
          { label: "service", value: "service" }
        ]

  const userColumns = [
    {
      title: t("settings:admin.users.username", "Username"),
      dataIndex: "username",
      key: "username"
    },
    {
      title: t("settings:admin.users.email", "Email"),
      dataIndex: "email",
      key: "email"
    },
    {
      title: t("settings:admin.users.role", "Role"),
      dataIndex: "role",
      key: "role",
      render: (role: string, record: AdminUserSummary) => (
        <Select
          size="small"
          value={role}
          style={{ minWidth: 120 }}
          onChange={(value) => handleChangeUserRole(record, value)}
          loading={updatingUserId === record.id}
          options={userRoleOptions}
        />
      )
    },
    {
      title: t("settings:admin.users.active", "Active"),
      dataIndex: "is_active",
      key: "is_active",
      render: (value: boolean, record: AdminUserSummary) => (
        <Switch
          size="small"
          checked={Boolean(value)}
          onChange={(checked) => handleToggleUserActive(record, checked)}
          loading={updatingUserId === record.id}
        />
      )
    },
    {
      title: t("settings:admin.users.verified", "Verified"),
      dataIndex: "is_verified",
      key: "is_verified",
      render: (value: boolean) =>
        value ? (
          <Tag color="green">
            {t("settings:admin.users.verifiedLabel", "Verified")}
          </Tag>
        ) : (
          <Tag>
            {t("settings:admin.users.unverifiedLabel", "Unverified")}
          </Tag>
        )
    },
    {
      title: t("settings:admin.users.storage", "Storage"),
      key: "storage",
      render: (_: any, record: AdminUserSummary) => (
        <span>
          {formatMegabytesForAdmin(record.storage_used_mb)} /{" "}
          {formatMegabytesForAdmin(record.storage_quota_mb)}
        </span>
      )
    }
  ]

  return (
    <PageShell>
      <Space orientation="vertical" size="large" className="w-full py-6">
        {adminGuard && (
          adminGuard === "forbidden" ? (
            <PermissionNotice
              className="mb-4"
              title={t(
                "settings:admin.adminGuardForbiddenTitle",
                "Admin access required for these controls"
              )}
              message={t(
                "settings:admin.adminGuardForbiddenBody",
                "Sign in as an admin user on your tldw server to view and manage users, roles, and system statistics."
              )}
              primaryAction={{
                label: t(
                  "settings:admin.adminGuardRequestAccess",
                  "Request access"
                ),
                onClick: openAdminDocumentation
              }}
            >
              <span>
                <a
                  href={TLDW_SERVER_DOCUMENTATION_URL}
                  target="_blank"
                  rel="noopener noreferrer">
                  {t(
                    "settings:admin.adminGuardLearnMore",
                    "Learn more in the tldw server documentation."
                  )}
                </a>
              </span>
            </PermissionNotice>
          ) : (
            <RecoveryCallout
              state="blocked"
              className="mb-4"
              title={t(
                "settings:admin.adminGuardNotFoundTitle",
                "Admin APIs are not available on this server"
              )}
              message={t(
                "settings:admin.adminGuardNotFoundBody",
                "This tldw server does not expose the /admin endpoints, or they are disabled. Upgrade or reconfigure the server to enable these views."
              )}
              primaryAction={{
                label: t(
                  "settings:admin.adminGuardEnableApis",
                  "Review server configuration"
                ),
                onClick: openAdminDocumentation
              }}
            >
              <a
                href={TLDW_SERVER_DOCUMENTATION_URL}
                target="_blank"
                rel="noopener noreferrer">
                {t(
                  "settings:admin.adminGuardLearnMore",
                  "Learn more in the tldw server documentation."
                )}
              </a>
            </RecoveryCallout>
          )
        )}
        {adminGuard && (
          <Text type="secondary">
            {t(
              "settings:admin.adminGuardLimitedInfo",
              "Admin-level details and controls are hidden until admin APIs are available."
            )}
          </Text>
        )}
        <div>
          <Title level={2}>{t("option:header.adminServer", "Server Admin")}</Title>
          <Text type="secondary">
            {t(
              "settings:admin.serverIntro",
              "Monitor core stats and configuration for your connected tldw server."
            )}
          </Text>
        </div>

        {config && (
          <Card title={t("settings:admin.connectionCardTitle", "Connection")} size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label={t("settings:admin.serverUrl", "Server URL")}>
                {config.serverUrl || "–"}
              </Descriptions.Item>
              <Descriptions.Item label={t("settings:admin.authMode", "Auth mode")}>
                {config.authMode || "single-user"}
              </Descriptions.Item>
            </Descriptions>
            {adminGuard && (
              <Text type="secondary">
                {t(
                  "settings:admin.adminGuardConnectionHint",
                  "Only basic connection details are shown; admin dashboards are disabled until admin APIs are available."
                )}
              </Text>
            )}
          </Card>
        )}

        <AdminAudioInstallerCard />

        {!adminGuard && (
          <>
            <Card
              title={t("settings:admin.systemStatsTitle", "System statistics")}
              loading={loading}
              extra={
                <Button size="small" onClick={handleRefresh} disabled={loading}>
                  {t("common:refresh", "Refresh")}
                </Button>
              }>
              {error && (
                <StatePanel
                  state="error"
                  title={t("settings:admin.systemStatsError", "Unable to load system statistics")}
                  message={error}
                  className="mb-3"
                  primaryAction={{
                    label: t("common:retry", "Retry"),
                    onClick: handleRefresh,
                    disabled: loading
                  }}
                />
              )}
              {stats ? (
                <Space orientation="vertical" size="large" className="w-full">
                  <Descriptions title={t("settings:admin.userStats", "Users")} column={3} size="small">
                    <Descriptions.Item label={t("settings:admin.users.total", "Total")}>
                      {users.total ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.users.active", "Active")}>
                      {users.active ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.users.admins", "Admins")}>
                      {users.admins ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.users.verified", "Verified")}>
                      {users.verified ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.users.new30d", "New (30d)")}>
                      {users.new_last_30d ?? "–"}
                    </Descriptions.Item>
                  </Descriptions>

                  <Descriptions
                    title={t("settings:admin.storageStats", "Storage")}
                    column={3}
                    size="small">
                    <Descriptions.Item label={t("settings:admin.storage.totalUsed", "Total used")}>
                      {formatMegabytesForAdmin(storage.total_used_mb)}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.storage.totalQuota", "Total quota")}>
                      {formatMegabytesForAdmin(storage.total_quota_mb)}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.storage.averageUsed", "Average used")}>
                      {formatMegabytesForAdmin(storage.average_used_mb)}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.storage.maxUsed", "Max used")}>
                      {formatMegabytesForAdmin(storage.max_used_mb)}
                    </Descriptions.Item>
                  </Descriptions>

                  <Descriptions title={t("settings:admin.sessionStats", "Sessions")} column={2} size="small">
                    <Descriptions.Item label={t("settings:admin.sessions.active", "Active sessions")}>
                      {sessions.active ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.sessions.uniqueUsers", "Unique users")}>
                      {sessions.unique_users ?? "–"}
                    </Descriptions.Item>
                  </Descriptions>
                </Space>
              ) : !loading && !error ? (
                <StatePanel
                  state="empty"
                  title={t(
                    "settings:admin.systemStatsEmptyTitle",
                    "No system statistics"
                  )}
                  message={t("settings:admin.systemStatsEmpty", "No system statistics available yet.")}
                  primaryAction={{
                    label: t("common:refresh", "Refresh"),
                    onClick: handleRefresh
                  }}
                />
              ) : null}
            </Card>

            <Card
              title={t("settings:admin.usersAndRolesTitle", "Users & roles")}
              extra={
                <Space size="small">
                  <Button
                    size="small"
                    onClick={() =>
                      loadUsers(usersPage, usersPageSize, userRoleFilter, userActiveFilter)
                    }
                    disabled={usersLoading}>
                    {t("common:refresh", "Refresh")}
                  </Button>
                  <Button size="small" onClick={loadRoles} disabled={rolesLoading}>
                    {t("settings:admin.roles.refresh", "Refresh roles")}
                  </Button>
                </Space>
              }>
              <Space orientation="vertical" size="middle" className="w-full">
                {usersError && (
                  <Alert
                    type="error"
                    title={t("settings:admin.usersError", "Unable to load users")}
                    description={usersError}
                    showIcon
                  />
                )}
                <Space align="center" wrap>
                  <Text strong>
                    {t("settings:admin.users.filtersTitle", "Filters")}
                  </Text>
                  <Select
                    size="small"
                    allowClear
                    placeholder={t("settings:admin.users.filterRole", "Role")}
                    value={userRoleFilter}
                    style={{ minWidth: 140 }}
                    onChange={(value) => handleUserFilterChange(value, userActiveFilter)}
                    options={[
                      { label: "user", value: "user" },
                      { label: "admin", value: "admin" },
                      { label: "service", value: "service" }
                    ]}
                  />
                  <Select
                    size="small"
                    allowClear
                    placeholder={t("settings:admin.users.filterActive", "Status")}
                    value={userActiveFilter}
                    style={{ minWidth: 160 }}
                    onChange={(value) => handleUserFilterChange(userRoleFilter, value)}
                    options={[
                      {
                        label: t("settings:admin.users.filterActiveOnly", "Active only"),
                        value: "active"
                      },
                      {
                        label: t("settings:admin.users.filterInactiveOnly", "Inactive only"),
                        value: "inactive"
                      }
                    ]}
                  />
                </Space>

                <Table<AdminUserSummary>
                  size="small"
                  rowKey="id"
                  loading={usersLoading}
                  dataSource={usersData?.users || []}
                  columns={userColumns as any}
                  pagination={{
                    current: usersPage,
                    pageSize: usersPageSize,
                    total: usersData?.total || 0,
                    showSizeChanger: true
                  }}
                  onChange={handleUserTableChange}
                />
                {!usersLoading && (usersData?.users || []).length === 0 && (
                  <StatePanel
                    state="empty"
                    title={t("settings:admin.users.emptyTitle", "No users found")}
                    message={t(
                      "settings:admin.users.empty",
                      "No user diagnostics match the current filters."
                    )}
                    primaryAction={{
                      label: t("common:refresh", "Refresh"),
                      onClick: () =>
                        loadUsers(
                          usersPage,
                          usersPageSize,
                          userRoleFilter,
                          userActiveFilter
                        )
                    }}
                  />
                )}

                <Divider />

                {rolesError && (
                  <Alert
                    type="error"
                    title={t("settings:admin.rolesError", "Unable to load roles")}
                    description={rolesError}
                    showIcon
                  />
                )}

                <Space orientation="vertical" size="small" className="w-full">
                  <Text strong>
                    {t("settings:admin.roles.title", "Roles")}
                  </Text>
                  <Table<AdminRole>
                    size="small"
                    rowKey="id"
                    loading={rolesLoading}
                    dataSource={roles}
                    pagination={false}
                    columns={[
                      {
                        title: t("settings:admin.roles.name", "Name"),
                        dataIndex: "name",
                        key: "name"
                      },
                      {
                        title: t("settings:admin.roles.description", "Description"),
                        dataIndex: "description",
                        key: "description",
                        render: (value: string | null | undefined) =>
                          value || (
                            <Text type="secondary">
                              {t(
                                "settings:admin.roles.noDescription",
                                "No description provided"
                              )}
                            </Text>
                          )
                      },
                      {
                        title: t("settings:admin.roles.system", "System"),
                        dataIndex: "is_system",
                        key: "is_system",
                        render: (value: boolean) =>
                          value ? (
                            <Tag color="blue">
                              {t("settings:admin.roles.systemLabel", "System")}
                            </Tag>
                          ) : (
                            <Tag>
                              {t("settings:admin.roles.customLabel", "Custom")}
                            </Tag>
                          )
                      },
                      {
                        title: t("settings:admin.roles.actions", "Actions"),
                        key: "actions",
                        render: (_: any, record: AdminRole) =>
                          record.is_system ? null : (
                            <Popconfirm
                              title={t(
                                "settings:admin.roles.deleteConfirmTitle",
                                "Delete role?"
                              )}
                              description={t(
                                "settings:admin.roles.deleteConfirmDescription",
                                "This will remove the role from the server. Existing users will lose this role."
                              )}
                              okText={t("common:confirm", "Confirm")}
                              cancelText={t("common:cancel", "Cancel")}
                              onConfirm={() => handleDeleteRole(record.id)}>
                              <Button
                                danger
                                size="small"
                                loading={deletingRoleId === record.id}>
                                {t("common:delete", "Delete")}
                              </Button>
                            </Popconfirm>
                          )
                      }
                    ]}
                  />
                  <Form
                    form={roleForm}
                    layout="inline"
                    className="mt-2 flex flex-wrap gap-2"
                    onFinish={handleCreateRole}>
                    <Form.Item
                      name="name"
                      rules={[
                        {
                          required: true,
                          message: t(
                            "settings:admin.roles.nameRequired",
                            "Enter a role name"
                          )
                        }
                      ]}>
                      <Input
                        size="small"
                        placeholder={t(
                          "settings:admin.roles.namePlaceholder",
                          "Role name (e.g. analyst)"
                        )}
                      />
                    </Form.Item>
                    <Form.Item name="description">
                      <Input
                        size="small"
                        placeholder={t(
                          "settings:admin.roles.descriptionPlaceholder",
                          "Optional description"
                        )}
                        style={{ minWidth: 220 }}
                      />
                    </Form.Item>
                    <Form.Item>
                      <Button
                        type="primary"
                        size="small"
                        htmlType="submit"
                        loading={creatingRole}>
                        {t("settings:admin.roles.create", "Create role")}
                      </Button>
                    </Form.Item>
                  </Form>
                </Space>
              </Space>
            </Card>

            <Card
              title={t("settings:admin.mediaBudget.title", "Media ingestion budget")}
              loading={mediaBudgetLoading}
              extra={
                <Space size="small">
                  <Button
                    size="small"
                    onClick={() =>
                      mediaBudgetUserId !== null
                        ? loadMediaBudget(mediaBudgetUserId, mediaBudgetPolicyId)
                        : undefined
                    }
                    disabled={mediaBudgetUserId === null || mediaBudgetLoading}>
                    {t("common:refresh", "Refresh")}
                  </Button>
                </Space>
              }>
              <Space orientation="vertical" size="middle" className="w-full">
                <Space wrap align="center">
                  <Select
                    size="small"
                    style={{ minWidth: 220 }}
                    value={mediaBudgetUserId ?? undefined}
                    placeholder={t("settings:admin.mediaBudget.user", "User")}
                    onChange={(value) => setMediaBudgetUserId(value)}
                    options={(usersData?.users || []).map((user) => ({
                      label: `${user.username} (#${user.id})`,
                      value: user.id
                    }))}
                  />
                  <Input
                    size="small"
                    style={{ width: 180 }}
                    value={mediaBudgetPolicyId}
                    onChange={(event) => setMediaBudgetPolicyId(event.target.value || "media.default")}
                    placeholder={t("settings:admin.mediaBudget.policy", "Policy ID")}
                  />
                </Space>

                {mediaBudgetError && (
                  <Alert
                    type="error"
                    showIcon
                    title={t(
                      "settings:admin.mediaBudget.errorTitle",
                      "Unable to load media ingestion budget diagnostics"
                    )}
                    description={mediaBudgetError}
                  />
                )}

                {mediaBudget ? (
                  <Descriptions
                    size="small"
                    column={2}
                    title={t("settings:admin.mediaBudget.current", "Current diagnostics")}>
                    <Descriptions.Item label={t("settings:admin.mediaBudget.status", "Status")}>
                      {mediaBudget.status || "–"}
                    </Descriptions.Item>
                    <Descriptions.Item label={t("settings:admin.mediaBudget.entity", "Entity")}>
                      {mediaBudget.entity || "–"}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.mediaBudget.jobsLimit", "Jobs max concurrent")}>
                      {mediaBudgetLimits.jobs_max_concurrent ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.mediaBudget.jobsActive", "Jobs active")}>
                      {mediaBudgetUsage.jobs_active ?? "–"}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.mediaBudget.bytesCap", "Daily ingestion bytes cap")}>
                      {formatBytesForAdmin(mediaBudgetLimits.ingestion_bytes_daily_cap)}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.mediaBudget.bytesUsed", "Daily ingestion bytes used")}>
                      {formatBytesForAdmin(mediaBudgetUsage.ingestion_bytes_daily_used)}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.mediaBudget.bytesRemaining", "Daily ingestion bytes remaining")}>
                      {formatBytesForAdmin(
                        mediaBudgetUsage.ingestion_bytes_daily_remaining
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item
                      label={t("settings:admin.mediaBudget.retryAfter", "Retry after (seconds)")}>
                      {formatRetryAfterForAdmin(mediaBudget.retry_after)}
                    </Descriptions.Item>
                  </Descriptions>
                ) : (
                  <StatePanel
                    state="empty"
                    title={t(
                      "settings:admin.mediaBudget.emptyTitle",
                      "No media budget diagnostics"
                    )}
                    message={t(
                      "settings:admin.mediaBudget.empty",
                      "Select a user to inspect media ingestion limits and usage."
                    )}
                    primaryAction={{
                      label: t("common:refresh", "Refresh"),
                      onClick: () => {
                        if (mediaBudgetUserId !== null) {
                          void loadMediaBudget(
                            mediaBudgetUserId,
                            mediaBudgetPolicyId
                          )
                        }
                      },
                      disabled: mediaBudgetUserId === null
                    }}
                  />
                )}
              </Space>
            </Card>
          </>
        )}
      </Space>
    </PageShell>
  )
}

export default ServerAdminPage
