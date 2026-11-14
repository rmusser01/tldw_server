import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import {
  Card,
  Table,
  Tabs,
  Button,
  Input,
  Modal,
  Select,
  Space,
  Popconfirm,
  Form,
  Tag,
  Checkbox,
  message
} from "antd"
import {
  PlusOutlined,
  DeleteOutlined,
  ReloadOutlined,
  LockOutlined,
  SafetyOutlined
} from "@ant-design/icons"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  adminKeys,
  useAdminPermissions,
  useAdminRoles
} from "@/services/tldw/adminQueries"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { useAdminRequestScope, useAdminMutation } from "./admin-request-scope"

type RbacProps = { config: TldwConfig; onGuardError: (error: unknown) => void }

// ── Types ──

interface Permission {
  id: number
  name: string
  description?: string
  category?: string
}

interface Role {
  id: number
  name: string
  description?: string
  is_system?: boolean
  permission_count?: number
}

interface UserOverride {
  permission_id: number
  permission_name?: string
  effect: string
}

interface EffectivePerm {
  permission_id?: number
  permission_name?: string
  source?: string
  effect?: string
}

interface RolePermissionMatrix {
  roles: Role[]
  permissions: Permission[]
  grants: Array<{ role_id: number; permission_id: number }>
  total_roles: number
}

// ── Permission Matrix Tab ──

const PermissionMatrixTab: React.FC<RbacProps> = ({ config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [matrix, setMatrix] = useState<RolePermissionMatrix | null>(null)
  const [matrixError, setMatrixError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const matrixRequest = useRef(0)
  const categoriesRequest = useRef(0)
  const [categoriesError, setCategoriesError] = useState<string | null>(null)
  const [categories, setCategories] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  const loadMatrix = useCallback(async () => {
    const scope = captureScope()
    if (!scope) return
    const request = ++matrixRequest.current
    const current = () => !scope.abortSignal.aborted && request === matrixRequest.current
    setLoading(true)
    setMatrixError(null)
    try {
      const result = await tldwClient.getRolePermissionMatrix(scope)
      if (!current()) return
      const validId = (id: number) => Number.isSafeInteger(id) && id > 0
      if (!Array.isArray(result?.roles) || !Array.isArray(result?.permissions) || !Array.isArray(result?.grants) ||
        !result.roles.every((role: Role) => validId(role.id) && typeof role.name === "string") ||
        !result.permissions.every((permission: Permission) => validId(permission.id) && typeof permission.name === "string") ||
        !result.grants.every((grant: { role_id: number; permission_id: number }) => validId(grant.role_id) && validId(grant.permission_id))) {
        throw new Error("Invalid permission matrix response")
      }
      setMatrix(result)
    } catch (err) {
      if (!current()) return
      onGuardError(err)
      setMatrix(null)
      setMatrixError(
        sanitizeAdminErrorMessage(err, t("settings:adminRbac.matrixLoadFailed", "Failed to load the permission matrix."))
      )
    } finally {
      if (current()) setLoading(false)
    }
  }, [captureScope, onGuardError, t])

  const loadCategories = useCallback(async () => {
    const scope = captureScope()
    if (!scope) return
    const request = ++categoriesRequest.current
    const current = () => !scope.abortSignal.aborted && request === categoriesRequest.current
    setCategoriesError(null)
    try {
      const result = await tldwClient.listPermissionCategories(scope)
      if (!current()) return
      const cats = Array.isArray(result)
        ? result.map((c: any) => (typeof c === "string" ? c : c?.name ?? c?.category ?? ""))
        : []
      setCategories(cats.filter(Boolean))
    } catch (err) {
      if (!current()) return
      onGuardError(err)
      setCategories([])
      setCategoriesError(sanitizeAdminErrorMessage(err, "Permission categories unavailable"))
    }
  }, [captureScope, onGuardError])

  useEffect(() => {
    loadMatrix()
    loadCategories()
  }, [loadMatrix, loadCategories])

  const handleToggle = (roleId: number, permissionId: number, currentValue: boolean) =>
    runMutation(t("settings:adminRbac.matrixUpdateFailed", "Failed to update permission matrix"), async (scope) => {
      if (currentValue) {
        await tldwClient.revokeRolePermission(roleId, permissionId, scope)
      } else {
        await tldwClient.grantRolePermission(roleId, permissionId, scope)
      }
      if (scope.abortSignal.aborted) return
      await loadMatrix()
      if (scope.abortSignal.aborted) return
      message.success(currentValue ? t("settings:adminRbac.permissionRevoked", "Permission revoked") : t("settings:adminRbac.permissionGranted", "Permission granted"))
    })

  // `permissions` must be derived before the loading early-return below so
  // filteredPermissions can be memoized without breaking the rules of hooks.
  const permissions: Permission[] = React.useMemo(() => matrix?.permissions ?? [], [matrix?.permissions])
  // Recompute the category filter only when the matrix or the filter changes
  // (admin perf C-S3 / F16); the grid cells themselves are already O(1).
  const filteredPermissions = React.useMemo(
    () =>
      selectedCategory
        ? permissions.filter((p) => p.category === selectedCategory)
        : permissions,
    [permissions, selectedCategory]
  )

  const roles = React.useMemo(() => matrix?.roles ?? [], [matrix?.roles])
  const grants = React.useMemo(() => new Set((matrix?.grants ?? []).map(grant => `${grant.role_id}:${grant.permission_id}`)), [matrix])
  const matrixAction = useRef({ handleToggle, disabled: mutating || loading })
  matrixAction.current = { handleToggle, disabled: mutating || loading }
  const roleColumns = React.useMemo(() => roles.map(role => ({
    title: role.name, key: `role-${role.id}`, width: 120, align: "center" as const,
    render: (_: any, record: Permission) => {
      const granted = grants.has(`${role.id}:${record.id}`)
      return <Checkbox checked={granted} disabled={matrixAction.current.disabled}
        onChange={() => matrixAction.current.handleToggle(role.id, record.id, granted)} />
    }
  })), [roles, grants])

  if (!matrix) {
    if (matrixError && !loading) {
      return (
        <DesignSystemAlert variant="error" title={t("settings:adminRbac.matrixErrorTitle", "Unable to load the permission matrix")}>
          <Space orientation="vertical" size="small">
            <span>{matrixError}</span>
            <Button size="small" onClick={() => void loadMatrix()}>
              {t("common:retry", "Retry")}
            </Button>
          </Space>
        </DesignSystemAlert>
      )
    }
    return <Card loading={loading}><div style={{ minHeight: 200 }} /></Card>
  }

  const columns = [
    {
      title: t("settings:adminRbac.colPermission", "Permission"),
      dataIndex: "name",
      key: "name",
      fixed: "left" as const,
      width: 260,
      render: (name: string, record: Permission) => (
        <span title={record.description ?? ""}>
          {record.category ? <Tag style={{ marginRight: 4 }}>{record.category}</Tag> : null}
          {name}
        </span>
      )
    },
    ...roleColumns
  ]

  return (
    <Card
      title={t("settings:adminRbac.matrixCardTitle", "Permission Matrix")}
      extra={
        <Space>
          <Select
            allowClear
            placeholder={t("settings:adminRbac.filterByCategory", "Filter by category")}
            style={{ width: 200 }}
            value={selectedCategory}
            onChange={v => setSelectedCategory(v ?? null)}
            options={categories.map(c => ({ label: c, value: c }))}
          />
          <Button icon={<ReloadOutlined />} onClick={loadMatrix} loading={loading}>
            {t("common:refresh", "Refresh")}
          </Button>
        </Space>
      }
    >
      {categoriesError && (
        <DesignSystemAlert variant="error" title={t("settings:adminRbac.categoriesUnavailable", "Permission categories unavailable")}>
          <span>{categoriesError}</span>
          <Button onClick={() => void loadCategories()}>{t("common:retry", "Retry")}</Button>
        </DesignSystemAlert>
      )}
      {permissions.length === 0 && !loading ? (
        <DesignSystemAlert
          variant="info"
          title={t("settings:adminRbac.emptyCatalogTitle", "No permissions registered yet")}
        >
          {t(
            "settings:adminRbac.emptyCatalogBody",
            "This server's RBAC catalog is empty, so there is nothing to grant or revoke per role — built-in role behavior still applies. The matrix fills in once permissions are seeded on the server (see the privilege catalog configuration)."
          )}
        </DesignSystemAlert>
      ) : (
        <Table
          dataSource={filteredPermissions}
          columns={columns}
          rowKey="id"
          loading={loading}
          // Bound the mounted checkbox grid to one page of permissions
          // (admin perf C-S4 / F16). The matrix fetch already returns the
          // complete permissions catalog (the server pages only roles), so
          // client-side paging covers every permission.
          pagination={{ pageSize: 50 }}
          scroll={{ x: 260 + roles.length * 120 }}
          size="small"
          locale={{
            emptyText: selectedCategory
              ? t("settings:adminRbac.emptyCategory", "No permissions in this category.")
              : t("settings:adminRbac.emptyPermissions", "No permissions found.")
          }}
        />
      )}
    </Card>
  )
}

// ── Roles Tab ──

const RolesTab: React.FC<RbacProps> = ({ config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  // Roles are shared admin reference data (B-S4/F11): the hook dedupes the
  // fetch with Server Admin inside the stale window, and mutations below
  // invalidate the shared key so every consumer refetches.
  const rolesQuery = useAdminRoles(config)
  const roles: Role[] = Array.isArray(rolesQuery.data)
    ? rolesQuery.data
    : ((rolesQuery.data as any)?.data ?? [])
  const loading = rolesQuery.isFetching
  const [createForm] = Form.useForm()
  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const expandedRequest = useRef(new Map<number, number>())
  const [rolesError, setRolesError] = useState<string | null>(null)
  const [expandedErrors, setExpandedErrors] = useState<Record<number, string>>({})
  const [expandedPerms, setExpandedPerms] = useState<Record<number, any[]>>({})

  const { refetch: refetchRoles } = rolesQuery
  const loadRoles = useCallback(async () => { await refetchRoles() }, [refetchRoles])
  useEffect(() => {
    setRolesError(null)
    if (rolesQuery.error) {
      onGuardError(rolesQuery.error)
      setRolesError(sanitizeAdminErrorMessage(rolesQuery.error, "Roles unavailable"))
    }
  }, [rolesQuery.error, onGuardError])

  const handleCreate = () =>
    runMutation(t("settings:adminRbac.roleCreateFailed", "Failed to create role"), async (scope) => {
      const values = await createForm.validateFields()
      if (scope.abortSignal.aborted) return
      await tldwClient.createAdminRole(values.name, values.description, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminRbac.roleCreated", "Role created"))
      createForm.resetFields()
      await queryClient.invalidateQueries({ queryKey: adminKeys.roles })
    })

  const handleDelete = (roleId: number) =>
    runMutation(t("settings:adminRbac.roleDeleteFailed", "Failed to delete role"), async (scope) => {
      await tldwClient.deleteAdminRole(roleId, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminRbac.roleDeleted", "Role deleted"))
      await queryClient.invalidateQueries({ queryKey: adminKeys.roles })
    })

  const loadRolePerms = useCallback(async (roleId: number) => {
    const scope = captureScope()
    if (!scope) return
    const request = (expandedRequest.current.get(roleId) ?? 0) + 1
    expandedRequest.current.set(roleId, request)
    const current = () => !scope.abortSignal.aborted && expandedRequest.current.get(roleId) === request
    setExpandedErrors(previous => {
      const next = { ...previous }
      delete next[roleId]
      return next
    })
    try {
      const result = await tldwClient.listRolePermissions(roleId, scope)
      if (!current()) return
      setExpandedPerms(prev => ({
        ...prev,
        [roleId]: Array.isArray(result) ? result : (result as any)?.data ?? []
      }))
    } catch (err) {
      if (!current()) return
      onGuardError(err)
      setExpandedErrors(previous => ({ ...previous, [roleId]: sanitizeAdminErrorMessage(err, "Role permissions unavailable") }))
    }
  }, [captureScope, onGuardError])

  const columns = [
    { title: t("settings:adminRbac.colName", "Name"), dataIndex: "name", key: "name" },
    { title: t("settings:adminRbac.colDescription", "Description"), dataIndex: "description", key: "description" },
    {
      title: t("settings:adminRbac.colSystem", "System"),
      dataIndex: "is_system",
      key: "is_system",
      width: 80,
      render: (v: boolean) => v ? <Tag color="blue">{t("settings:adminRbac.systemTag", "System")}</Tag> : null
    },
    {
      title: t("settings:adminRbac.colPermissions", "Permissions"),
      dataIndex: "permission_count",
      key: "permission_count",
      width: 110,
      render: (v: number) => v ?? "-"
    },
    {
      title: t("settings:adminRbac.colActions", "Actions"),
      key: "actions",
      width: 100,
      render: (_: any, record: Role) =>
        record.is_system ? (
          <Tag>{t("settings:adminRbac.protectedTag", "Protected")}</Tag>
        ) : (
          <Popconfirm
            title={t("settings:adminRbac.deleteRoleConfirm", "Delete this role?")}
            onConfirm={() => handleDelete(record.id)}
          >
            <Button danger size="small" disabled={mutating || loading || Boolean(rolesError)} icon={<DeleteOutlined />} />
          </Popconfirm>
        )
    }
  ]

  return (
    <Card
      title={t("settings:adminRbac.rolesCardTitle", "Roles")}
      extra={
        <Button icon={<ReloadOutlined />} onClick={() => rolesQuery.refetch()} loading={loading}>
          {t("common:refresh", "Refresh")}
        </Button>
      }
    >
      <Card size="small" style={{ marginBottom: 16 }} title={t("settings:adminRbac.createRoleCardTitle", "Create Role")}>
        <Form form={createForm} layout="inline">
          <Form.Item name="name" rules={[{ required: true, message: t("settings:adminRbac.nameRequired", "Name required") }]}>
            <Input placeholder={t("settings:adminRbac.roleNamePlaceholder", "Role name")} />
          </Form.Item>
          <Form.Item name="description">
            <Input placeholder={t("settings:adminRbac.descriptionPlaceholder", "Description (optional)")} style={{ width: 250 }} />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={handleCreate}
              loading={mutating}
            >
              {t("settings:adminRbac.create", "Create")}
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {rolesError && (
        <DesignSystemAlert variant="error" title={t("settings:adminRbac.rolesUnavailable", "Roles unavailable")}>
          <span>{rolesError}</span><Button onClick={() => void loadRoles()}>{t("common:retry", "Retry")}</Button>
        </DesignSystemAlert>
      )}
      <Table
        dataSource={roles}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={false}
        expandable={{
          expandedRowRender: (record: Role) => {
            if (expandedErrors[record.id]) return (
              <DesignSystemAlert variant="error" title={t("settings:adminRbac.rolePermissionsUnavailable", "Role permissions unavailable")}>
                <span>{expandedErrors[record.id]}</span><Button onClick={() => void loadRolePerms(record.id)}>{t("common:retry", "Retry")}</Button>
              </DesignSystemAlert>
            )
            const perms = expandedPerms[record.id]
            if (!perms) return <div style={{ padding: 8, color: "#999" }}>{t("settings:adminRbac.loading", "Loading...")}</div>
            if (perms.length === 0) return <div style={{ padding: 8, color: "#999" }}>{t("settings:adminRbac.noPermissionsAssigned", "No permissions assigned")}</div>
            return (
              <div style={{ padding: 8 }}>
                {perms.map((p: any) => (
                  <Tag key={p.id ?? p.permission_id} style={{ margin: 2 }}>
                    {p.name ?? p.permission_name ?? `#${p.id ?? p.permission_id}`}
                  </Tag>
                ))}
              </div>
            )
          },
          onExpand: (expanded, record) => {
            if (expanded && !expandedPerms[record.id]) {
              loadRolePerms(record.id)
            }
          }
        }}
      />
    </Card>
  )
}

// ── User Permissions Tab ──

const UserPermissionsTab: React.FC<RbacProps> = ({ config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const captureScope = useAdminRequestScope(config)
  const [users, setUsers] = useState<any[]>([])
  const [usersLoading, setUsersLoading] = useState(false)
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)
  const [usersError, setUsersError] = useState<string | null>(null)
  const [search, setSearch] = useState("")
  const searchRequest = useRef(0)

  const searchUsers = useCallback(async (value: string) => {
    setSearch(value)
    const request = ++searchRequest.current
    setUsersError(null)
    if (!value) { setUsers([]); setUsersLoading(false); return }
    const scope = captureScope()
    if (!scope) return
    const current = () => !scope.abortSignal.aborted && request === searchRequest.current
    setUsersLoading(true)
    try {
      const result = await tldwClient.listAdminUsers({ search: value, limit: 20 }, scope)
      if (current()) setUsers(Array.isArray(result) ? result : (result as any)?.data ?? (result as any)?.users ?? [])
    } catch (err) {
      if (!current()) return
      onGuardError(err)
      setUsers([])
      setUsersError(sanitizeAdminErrorMessage(err, "User search unavailable"))
    } finally {
      if (current()) setUsersLoading(false)
    }
  }, [captureScope, onGuardError])

  return <>
    {usersError && <DesignSystemAlert variant="error" title={t("settings:adminRbac.usersUnavailable", "User search unavailable")}>
      <span>{usersError}</span><Button onClick={() => void searchUsers(search)}>{t("common:retry", "Retry")}</Button>
    </DesignSystemAlert>}
    <UserPermissionWorkspace key={selectedUserId ?? "none"} config={config} onGuardError={onGuardError}
      selectedUserId={selectedUserId} onSelectUser={setSelectedUserId} users={users} usersLoading={usersLoading} searchUsers={searchUsers} />
  </>
}

type UserPermissionProps = RbacProps & {
  selectedUserId: number | null
  onSelectUser: (userId: number) => void
  users: Array<{ id: number; username?: string | null; email?: string | null }>
  usersLoading: boolean
  searchUsers: (search: string) => void
}

const UserPermissionWorkspace: React.FC<UserPermissionProps> = ({ config, onGuardError, selectedUserId, onSelectUser, users, usersLoading, searchUsers }) => {
  const { t } = useTranslation(["settings", "common"])
  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const dataRequest = useRef(0)
  const [unavailable, setUnavailable] = useState<string[]>([])
  const [metaUnavailable, setMetaUnavailable] = useState<string[]>([])
  const [metaLoading, setMetaLoading] = useState(true)

  // User roles
  const [userRoles, setUserRoles] = useState<any[]>([])
  const [rolesLoading, setRolesLoading] = useState(true)

  // Overrides
  const [overrides, setOverrides] = useState<UserOverride[]>([])
  const [overridesLoading, setOverridesLoading] = useState(true)

  // Effective permissions
  const [effectivePerms, setEffectivePerms] = useState<EffectivePerm[]>([])
  const [effectiveLoading, setEffectiveLoading] = useState(true)

  // Modals
  const [addRoleModalOpen, setAddRoleModalOpen] = useState(false)
  const [addRoleId, setAddRoleId] = useState<number | null>(null)

  const [addOverrideModalOpen, setAddOverrideModalOpen] = useState(false)
  const [overrideForm] = Form.useForm()

  const loadUserData = useCallback(async (userId: number) => {
    const scope = captureScope()
    if (!scope) return
    const request = ++dataRequest.current
    const current = () => !scope.abortSignal.aborted && request === dataRequest.current
    setRolesLoading(true)
    setOverridesLoading(true)
    setEffectiveLoading(true)
    const results = await Promise.allSettled([
      tldwClient.listUserRoles(userId, scope),
      tldwClient.listUserOverrides(userId, scope),
      tldwClient.getUserEffectivePermissions(userId, scope)
    ])
    if (!current()) return
    const labels = ["Assigned Roles", "Permission Overrides", "Effective Permissions"]
    const setters = [setUserRoles, setOverrides, setEffectivePerms]
    const failed: string[] = []
    results.forEach((result, index) => {
      if (result.status === "fulfilled") {
        const value: unknown = result.value
        try {
          const fields = ["roles", "overrides", "permissions"]
          const envelope = value !== null && typeof value === "object" ? value as Record<string, unknown> : null
          const entries = Array.isArray(value) ? value : envelope?.[fields[index]] ?? envelope?.data
          if (!Array.isArray(entries)) throw new Error("Invalid user permission response")
          const normalized = index === 1 ? entries.map(entry => ({
            ...entry,
            effect: typeof entry.granted === "boolean" ? (entry.granted ? "allow" : "deny") : entry.effect
          })) : index === 2 ? entries.map(entry => typeof entry === "string" ? { permission_name: entry } : entry) : entries
          setters[index](normalized)
        } catch (err) {
          setters[index]([])
          failed.push(labels[index])
          onGuardError(err)
        }
      } else {
        setters[index]([])
        failed.push(labels[index])
        onGuardError(result.reason)
      }
    })
    setUnavailable(failed)
    setRolesLoading(false)
    setOverridesLoading(false)
    setEffectiveLoading(false)
  }, [captureScope, onGuardError])

  // All roles and permissions are shared admin reference data (B-S4/F11):
  // the hooks reuse whatever Server Admin / the Roles tab already fetched
  // inside the stale window. Failures stay non-critical here, as before.
  const rolesQuery = useAdminRoles(config)
  const permissionsQuery = useAdminPermissions(config)
  // Memoize the normalized arrays so downstream Map/Set memos keep a stable
  // dependency when react-query serves the cached array instance (the `??`
  // fallbacks would otherwise mint a fresh empty array every render).
  const allRoles: Role[] = React.useMemo(
    () =>
      Array.isArray(rolesQuery.data)
        ? rolesQuery.data
        : ((rolesQuery.data as any)?.data ?? []),
    [rolesQuery.data]
  )
  const allPermissions: Permission[] = React.useMemo(
    () =>
      Array.isArray(permissionsQuery.data)
        ? permissionsQuery.data
        : ((permissionsQuery.data as any)?.data ?? []),
    [permissionsQuery.data]
  )
  // O(1) keyed lookups for per-row rendering (admin perf C-S3 / F16): the
  // override table resolves names through the Map instead of scanning the
  // catalog with `.find` per row, and role options test membership in a Set
  // instead of `userRoles.some(...)` per option.
  const { refetch: refetchRoles } = rolesQuery
  const { refetch: refetchPermissions } = permissionsQuery
  const loadMeta = useCallback(async () => { await Promise.all([refetchRoles(), refetchPermissions()]) }, [refetchRoles, refetchPermissions])
  useEffect(() => {
    const errors = [rolesQuery.error, permissionsQuery.error]
    errors.forEach(error => { if (error) onGuardError(error) })
    setMetaUnavailable(errors.flatMap((error, i) => error ? [i === 0 ? "Role catalog" : "Permission catalog"] : []))
    setMetaLoading(rolesQuery.isLoading || permissionsQuery.isLoading)
  }, [rolesQuery.error, permissionsQuery.error, rolesQuery.isLoading, permissionsQuery.isLoading, onGuardError])
  const permissionById = React.useMemo(
    () => new Map(allPermissions.map((p) => [p.id, p])),
    [allPermissions]
  )
  const assignedRoleIds = React.useMemo(
    () => new Set(userRoles.map((ur: any) => ur.id ?? ur.role_id)),
    [userRoles]
  )
  useEffect(() => {
    if (selectedUserId !== null) void loadUserData(selectedUserId)
  }, [loadUserData, selectedUserId])

  const ready = !rolesLoading && !overridesLoading && !effectiveLoading && !metaLoading && unavailable.length === 0 && metaUnavailable.length === 0

  const handleRemoveRole = (roleId: number) => {
    if (!selectedUserId || !ready) return
    return runMutation(t("settings:adminRbac.roleRemoveFailed", "Failed to remove role"), async (scope) => {
      await tldwClient.removeUserRole(selectedUserId, roleId, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminRbac.roleRemoved", "Role removed"))
      await loadUserData(selectedUserId)
    })
  }

  const handleAddRole = () => {
    if (!selectedUserId || !addRoleId || !ready) return
    return runMutation(t("settings:adminRbac.roleAssignFailed", "Failed to assign role"), async (scope) => {
      await tldwClient.assignUserRole(selectedUserId, addRoleId, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminRbac.roleAssigned", "Role assigned"))
      setAddRoleModalOpen(false)
      setAddRoleId(null)
      await loadUserData(selectedUserId)
    })
  }

  const handleAddOverride = () => {
    if (!selectedUserId || !ready) return
    return runMutation(t("settings:adminRbac.overrideSaveFailed", "Failed to save user permission overrides"), async (scope) => {
      const values = await overrideForm.validateFields()
      if (scope.abortSignal.aborted) return
      await tldwClient.addUserOverride(selectedUserId, {
        permission_id: values.permission_id,
        effect: values.effect
      }, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminRbac.overrideAdded", "Override added"))
      setAddOverrideModalOpen(false)
      overrideForm.resetFields()
      await loadUserData(selectedUserId)
    })
  }

  const handleRemoveOverride = (permissionId: number) => {
    if (!selectedUserId || !ready) return
    return runMutation(t("settings:adminRbac.overrideRemoveFailed", "Failed to revoke user permission override"), async (scope) => {
      await tldwClient.deleteUserOverride(selectedUserId, permissionId, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminRbac.overrideRemoved", "Override removed"))
      await loadUserData(selectedUserId)
    })
  }

  const roleColumns = [
    {
      title: t("settings:adminRbac.colRole", "Role"),
      key: "name",
      render: (_: any, record: any) => record.name ?? record.role_name ?? `Role #${record.id ?? record.role_id}`
    },
    {
      title: t("settings:adminRbac.colActions", "Actions"),
      key: "actions",
      width: 100,
      render: (_: any, record: any) => (
        <Popconfirm
          title={t("settings:adminRbac.removeRoleConfirm", "Remove this role?")}
          onConfirm={() => handleRemoveRole(record.id ?? record.role_id)}
        >
          <Button danger size="small" disabled={mutating || !ready} icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  const overrideColumns = [
    {
      title: t("settings:adminRbac.colPermission", "Permission"),
      key: "permission_name",
      render: (_: any, record: UserOverride) =>
        record.permission_name ?? permissionById.get(record.permission_id)?.name ?? `#${record.permission_id}`
    },
    {
      title: t("settings:adminRbac.colEffect", "Effect"),
      dataIndex: "effect",
      key: "effect",
      width: 100,
      render: (v: string) => (
        <Tag color={v === "allow" || v === "grant" ? "green" : "red"}>{v?.toUpperCase()}</Tag>
      )
    },
    {
      title: t("settings:adminRbac.colActions", "Actions"),
      key: "actions",
      width: 100,
      render: (_: any, record: UserOverride) => (
        <Popconfirm
          title={t("settings:adminRbac.removeOverrideConfirm", "Remove this override?")}
          onConfirm={() => handleRemoveOverride(record.permission_id)}
        >
          <Button danger size="small" disabled={mutating || !ready} icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  const effectiveColumns = [
    {
      title: t("settings:adminRbac.colPermission", "Permission"),
      key: "permission_name",
      render: (_: any, record: EffectivePerm) =>
        record.permission_name ?? `#${record.permission_id}`
    },
    {
      title: t("settings:adminRbac.colSource", "Source"),
      dataIndex: "source",
      key: "source",
      width: 120,
      render: (v: string) => v ? <Tag>{v}</Tag> : null
    },
    {
      title: t("settings:adminRbac.colEffect", "Effect"),
      dataIndex: "effect",
      key: "effect",
      width: 100,
      render: (v: string) =>
        v ? <Tag color={v === "grant" ? "green" : v === "deny" ? "red" : "default"}>{v?.toUpperCase()}</Tag> : <Tag color="green">GRANT</Tag>
    }
  ]

  return (
    <Card title={t("settings:adminRbac.userPermissionsCardTitle", "User Permissions")}>
      <div style={{ marginBottom: 16 }}>
        <Select
          showSearch
          placeholder={t("settings:adminRbac.searchUserPlaceholder", "Search for a user...")}
          style={{ width: 400 }}
          loading={usersLoading}
          filterOption={false}
          onSearch={searchUsers}
          onChange={onSelectUser}
          value={selectedUserId}
          options={users.map((u: any) => ({
            label: u.username ?? u.email ?? `User #${u.id}`,
            value: u.id
          }))}
        />
      </div>

      {unavailable.length > 0 && selectedUserId !== null && (
        <DesignSystemAlert variant="error" title={t("settings:adminRbac.userDataIncomplete", "User permissions are incomplete")}>
          <span>{unavailable.join(", ")}</span>
          <Button onClick={() => void loadUserData(selectedUserId)}>{t("common:retry", "Retry")}</Button>
        </DesignSystemAlert>
      )}
      {metaUnavailable.length > 0 && (
        <DesignSystemAlert variant="error" title={t("settings:adminRbac.catalogIncomplete", "Permission catalogs are incomplete")}>
          <span>{metaUnavailable.join(", ")}</span>
          <Button onClick={() => void loadMeta()}>{t("common:retry", "Retry")}</Button>
        </DesignSystemAlert>
      )}
      {selectedUserId && (
        <div>
          {/* Assigned Roles */}
          <Card
            size="small"
            title={t("settings:adminRbac.assignedRolesCardTitle", "Assigned Roles")}
            style={{ marginBottom: 16 }}
            extra={
              <Button
                size="small"
                icon={<PlusOutlined />}
                disabled={mutating || !ready}
                onClick={() => setAddRoleModalOpen(true)}
              >
                {t("settings:adminRbac.addRole", "Add Role")}
              </Button>
            }
          >
            <Table
              dataSource={userRoles}
              locale={{ emptyText: unavailable.includes("Assigned Roles") ? t("settings:adminRbac.paneUnavailable", "Unavailable — retry to load this information") : undefined }}
              columns={roleColumns}
              rowKey={(r: any) => r.id ?? r.role_id}
              loading={rolesLoading}
              pagination={false}
              size="small"
            />
          </Card>

          {/* Permission Overrides */}
          <Card
            size="small"
            title={t("settings:adminRbac.overridesCardTitle", "Permission Overrides")}
            style={{ marginBottom: 16 }}
            extra={
              <Button
                size="small"
                icon={<PlusOutlined />}
                disabled={mutating || !ready}
                onClick={() => setAddOverrideModalOpen(true)}
              >
                {t("settings:adminRbac.addOverride", "Add Override")}
              </Button>
            }
          >
            <Table
              dataSource={overrides}
              locale={{ emptyText: unavailable.includes("Permission Overrides") ? t("settings:adminRbac.paneUnavailable", "Unavailable — retry to load this information") : undefined }}
              columns={overrideColumns}
              rowKey="permission_id"
              loading={overridesLoading}
              pagination={false}
              size="small"
            />
          </Card>

          {/* Effective Permissions */}
          <Card size="small" title={t("settings:adminRbac.effectivePermissionsCardTitle", "Effective Permissions (read-only)")}>
            <Table
              dataSource={effectivePerms}
              locale={{ emptyText: unavailable.includes("Effective Permissions") ? t("settings:adminRbac.paneUnavailable", "Unavailable — retry to load this information") : undefined }}
              columns={effectiveColumns}
              rowKey={(permission: EffectivePerm) => permission.permission_name ?? String(permission.permission_id)}
              loading={effectiveLoading}
              pagination={false}
              size="small"
            />
          </Card>
        </div>
      )}

      {/* Add Role Modal */}
      <Modal
        title={t("settings:adminRbac.assignRoleModalTitle", "Assign Role")}
        open={addRoleModalOpen}
        onOk={handleAddRole}
        onCancel={() => { if (!mutating) { setAddRoleModalOpen(false); setAddRoleId(null) } }}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
        okButtonProps={{ disabled: !addRoleId || !ready }}
      >
        <Select
          placeholder={t("settings:adminRbac.selectRolePlaceholder", "Select a role")}
          style={{ width: "100%" }}
          value={addRoleId}
          onChange={setAddRoleId}
          options={allRoles.map(r => ({
            label: r.name,
            value: r.id,
            disabled: assignedRoleIds.has(r.id)
          }))}
        />
      </Modal>

      {/* Add Override Modal */}
      <Modal
        title={t("settings:adminRbac.addOverrideModalTitle", "Add Permission Override")}
        open={addOverrideModalOpen}
        onOk={handleAddOverride}
        onCancel={() => { if (!mutating) { setAddOverrideModalOpen(false); overrideForm.resetFields() } }}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
        okButtonProps={{ disabled: !ready }}
      >
        <Form form={overrideForm} layout="vertical">
          <Form.Item
            name="permission_id"
            label={t("settings:adminRbac.permissionLabel", "Permission")}
            rules={[{ required: true, message: t("settings:adminRbac.selectPermissionRequired", "Select a permission") }]}
          >
            <Select
              showSearch
              placeholder={t("settings:adminRbac.selectPermissionPlaceholder", "Select permission")}
              filterOption={(input, option) =>
                (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) ?? false
              }
              options={allPermissions.map(p => ({
                label: p.category ? `[${p.category}] ${p.name}` : p.name,
                value: p.id
              }))}
            />
          </Form.Item>
          <Form.Item
            name="effect"
            label={t("settings:adminRbac.effectLabel", "Effect")}
            rules={[{ required: true, message: t("settings:adminRbac.selectEffectRequired", "Select an effect") }]}
          >
            <Select
              placeholder={t("settings:adminRbac.selectEffectPlaceholder", "Select effect")}
              options={[
                { label: t("settings:adminRbac.effectGrant", "Grant"), value: "allow" },
                { label: t("settings:adminRbac.effectDeny", "Deny"), value: "deny" }
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  )
}

// ── Main Page ──

const RbacEditorPage: React.FC = () => {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const { t } = useTranslation("settings")
  if (!config || authorityLoading) {
    return <p role="status">{t("admin.resolvingConnection", "Resolving admin connection…")}</p>
  }
  return <RbacEditorWorkspace key={deriveConnectionAuthorityId(config)} config={config} />
}

const RbacEditorWorkspace = ({ config }: { config: TldwConfig }) => {
  const { t } = useTranslation(["settings", "common"])
  const [guardError, setGuardError] = useState<string | null>(null)
  const guardRef = useRef(false)

  const handleGuardError = useCallback((err: unknown) => {
    const guard = deriveAdminGuardFromError(err)
    if (guard && !guardRef.current) {
      guardRef.current = true
      setGuardError(guard)
    }
  }, [])

  if (guardError) {
    return (
      <DesignSystemAlert variant="warning" title={t("settings:adminRbac.accessRestrictedTitle", "Access Restricted")} className="m-6">
        {guardError}
      </DesignSystemAlert>
    )
  }

  return (
    <div style={{ padding: 16 }}>
      <h1 style={{ marginBottom: 4, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminRbac.title", "Roles & Permissions")}</h1>
      <p style={{ marginBottom: 16, color: "var(--color-text-secondary, #888)" }}>
        {t(
          "settings:adminRbac.description",
          "Review the permission matrix, manage roles, and adjust per-user grants."
        )}
      </p>
      <Tabs
        defaultActiveKey="matrix"
        items={[
          {
            key: "matrix",
            label: (
              <span>
                <SafetyOutlined /> {t("settings:adminRbac.tabMatrix", "Permission Matrix")}
              </span>
            ),
            children: <PermissionMatrixTab config={config} onGuardError={handleGuardError} />
          },
          {
            key: "roles",
            label: (
              <span>
                <LockOutlined /> {t("settings:adminRbac.tabRoles", "Roles")}
              </span>
            ),
            children: <RolesTab config={config} onGuardError={handleGuardError} />
          },
          {
            key: "users",
            label: (
              <span>
                <LockOutlined /> {t("settings:adminRbac.tabUserPermissions", "User Permissions")}
              </span>
            ),
            children: <UserPermissionsTab config={config} onGuardError={handleGuardError} />
          }
        ]}
      />
    </div>
  )
}

export default RbacEditorPage
