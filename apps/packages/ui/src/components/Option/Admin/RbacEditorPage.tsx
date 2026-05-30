import React, { useState, useRef, useCallback, useEffect } from "react"
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
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"

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
  permission_id: number
  permission_name?: string
  source?: string
  effect?: string
}

// ── Permission Matrix Tab ──

const PermissionMatrixTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const [matrix, setMatrix] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [toggling, setToggling] = useState<string | null>(null)
  const [categories, setCategories] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  const loadMatrix = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.getRolePermissionMatrix()
      setMatrix(result)
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  const loadCategories = useCallback(async () => {
    try {
      const result = await tldwClient.listPermissionCategories()
      const cats = Array.isArray(result)
        ? result.map((c: any) => (typeof c === "string" ? c : c?.name ?? c?.category ?? ""))
        : []
      setCategories(cats.filter(Boolean))
    } catch {
      // non-critical
    }
  }, [])

  useEffect(() => {
    loadMatrix()
    loadCategories()
  }, [loadMatrix, loadCategories])

  const handleToggle = useCallback(async (roleId: number, permissionId: number, currentValue: boolean) => {
    const key = `${roleId}-${permissionId}`
    setToggling(key)
    try {
      if (currentValue) {
        await tldwClient.revokeRolePermission(roleId, permissionId)
      } else {
        await tldwClient.grantRolePermission(roleId, permissionId)
      }
      await loadMatrix()
      message.success(currentValue ? "Permission revoked" : "Permission granted")
    } catch (err: any) {
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to update permission matrix")
      )
    } finally {
      setToggling(null)
    }
  }, [loadMatrix])

  if (!matrix) {
    return <Card loading={loading}><div style={{ minHeight: 200 }} /></Card>
  }

  // matrix shape: { roles: [{id, name}], permissions: [{id, name, category}], grid: {[permId]: {[roleId]: bool}} }
  const roles: Array<{ id: number; name: string }> = matrix.roles ?? []
  const permissions: Permission[] = matrix.permissions ?? []
  const grid: Record<number, Record<number, boolean>> = matrix.grid ?? {}

  const filteredPermissions = selectedCategory
    ? permissions.filter(p => p.category === selectedCategory)
    : permissions

  const columns = [
    {
      title: "Permission",
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
    ...roles.map(role => ({
      title: role.name,
      key: `role-${role.id}`,
      width: 120,
      align: "center" as const,
      render: (_: any, record: Permission) => {
        const val = grid[record.id]?.[role.id] ?? false
        const key = `${role.id}-${record.id}`
        return (
          <Checkbox
            checked={val}
            disabled={toggling === key}
            onChange={() => handleToggle(role.id, record.id, val)}
          />
        )
      }
    }))
  ]

  return (
    <Card
      title="Permission Matrix"
      extra={
        <Space>
          <Select
            allowClear
            placeholder="Filter by category"
            style={{ width: 200 }}
            value={selectedCategory}
            onChange={v => setSelectedCategory(v ?? null)}
            options={categories.map(c => ({ label: c, value: c }))}
          />
          <Button icon={<ReloadOutlined />} onClick={loadMatrix} loading={loading}>
            Refresh
          </Button>
        </Space>
      }
    >
      <Table
        dataSource={filteredPermissions}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={false}
        scroll={{ x: 260 + roles.length * 120 }}
        size="small"
      />
    </Card>
  )
}

// ── Roles Tab ──

const RolesTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const [roles, setRoles] = useState<Role[]>([])
  const [loading, setLoading] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)
  const [expandedPerms, setExpandedPerms] = useState<Record<number, any[]>>({})

  const loadRoles = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listAdminRoles()
      setRoles(Array.isArray(result) ? result : (result as any)?.data ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    loadRoles()
  }, [loadRoles])

  const handleCreate = useCallback(async () => {
    try {
      const values = await createForm.validateFields()
      setCreating(true)
      await tldwClient.createAdminRole(values.name, values.description)
      message.success("Role created")
      createForm.resetFields()
      await loadRoles()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, "Failed to create role"))
    } finally {
      setCreating(false)
    }
  }, [createForm, loadRoles])

  const handleDelete = useCallback(async (roleId: number) => {
    try {
      await tldwClient.deleteAdminRole(roleId)
      message.success("Role deleted")
      await loadRoles()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete role"))
    }
  }, [loadRoles])

  const loadRolePerms = useCallback(async (roleId: number) => {
    try {
      const result = await tldwClient.listRolePermissions(roleId)
      setExpandedPerms(prev => ({
        ...prev,
        [roleId]: Array.isArray(result) ? result : (result as any)?.data ?? []
      }))
    } catch {
      // silently fail
    }
  }, [])

  const columns = [
    { title: "Name", dataIndex: "name", key: "name" },
    { title: "Description", dataIndex: "description", key: "description" },
    {
      title: "System",
      dataIndex: "is_system",
      key: "is_system",
      width: 80,
      render: (v: boolean) => v ? <Tag color="blue">System</Tag> : null
    },
    {
      title: "Permissions",
      dataIndex: "permission_count",
      key: "permission_count",
      width: 110,
      render: (v: number) => v ?? "-"
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: any, record: Role) =>
        record.is_system ? (
          <Tag>Protected</Tag>
        ) : (
          <Popconfirm
            title="Delete this role?"
            onConfirm={() => handleDelete(record.id)}
          >
            <Button danger size="small" icon={<DeleteOutlined />} />
          </Popconfirm>
        )
    }
  ]

  return (
    <Card
      title="Roles"
      extra={
        <Button icon={<ReloadOutlined />} onClick={loadRoles} loading={loading}>
          Refresh
        </Button>
      }
    >
      <Card size="small" style={{ marginBottom: 16 }} title="Create Role">
        <Form form={createForm} layout="inline">
          <Form.Item name="name" rules={[{ required: true, message: "Name required" }]}>
            <Input placeholder="Role name" />
          </Form.Item>
          <Form.Item name="description">
            <Input placeholder="Description (optional)" style={{ width: 250 }} />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={handleCreate}
              loading={creating}
            >
              Create
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Table
        dataSource={roles}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={false}
        expandable={{
          expandedRowRender: (record: Role) => {
            const perms = expandedPerms[record.id]
            if (!perms) return <div style={{ padding: 8, color: "#999" }}>Loading...</div>
            if (perms.length === 0) return <div style={{ padding: 8, color: "#999" }}>No permissions assigned</div>
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

const UserPermissionsTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const [users, setUsers] = useState<any[]>([])
  const [usersLoading, setUsersLoading] = useState(false)
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)

  // User roles
  const [userRoles, setUserRoles] = useState<any[]>([])
  const [rolesLoading, setRolesLoading] = useState(false)

  // All roles for assignment
  const [allRoles, setAllRoles] = useState<Role[]>([])

  // Overrides
  const [overrides, setOverrides] = useState<UserOverride[]>([])
  const [overridesLoading, setOverridesLoading] = useState(false)

  // Effective permissions
  const [effectivePerms, setEffectivePerms] = useState<EffectivePerm[]>([])
  const [effectiveLoading, setEffectiveLoading] = useState(false)

  // All permissions for override modal
  const [allPermissions, setAllPermissions] = useState<Permission[]>([])

  // Modals
  const [addRoleModalOpen, setAddRoleModalOpen] = useState(false)
  const [addRoleId, setAddRoleId] = useState<number | null>(null)
  const [addingRole, setAddingRole] = useState(false)

  const [addOverrideModalOpen, setAddOverrideModalOpen] = useState(false)
  const [overrideForm] = Form.useForm()
  const [addingOverride, setAddingOverride] = useState(false)

  const searchUsers = useCallback(async (search: string) => {
    if (!search || search.length < 1) return
    setUsersLoading(true)
    try {
      const result = await tldwClient.listAdminUsers({ search, limit: 20 })
      const items = Array.isArray(result) ? result : (result as any)?.data ?? (result as any)?.users ?? []
      setUsers(items)
    } catch (err) {
      onGuardError(err)
    } finally {
      setUsersLoading(false)
    }
  }, [onGuardError])

  const loadUserData = useCallback(async (userId: number) => {
    setRolesLoading(true)
    setOverridesLoading(true)
    setEffectiveLoading(true)
    try {
      const [roles, overridesResult, effective] = await Promise.allSettled([
        tldwClient.listUserRoles(userId),
        tldwClient.listUserOverrides(userId),
        tldwClient.getUserEffectivePermissions(userId)
      ])
      if (roles.status === "fulfilled") {
        const r = roles.value
        setUserRoles(Array.isArray(r) ? r : (r as any)?.data ?? [])
      }
      if (overridesResult.status === "fulfilled") {
        const o = overridesResult.value
        setOverrides(Array.isArray(o) ? o : (o as any)?.data ?? [])
      }
      if (effective.status === "fulfilled") {
        const e = effective.value
        setEffectivePerms(Array.isArray(e) ? e : (e as any)?.data ?? [])
      }
    } catch (err) {
      onGuardError(err)
    } finally {
      setRolesLoading(false)
      setOverridesLoading(false)
      setEffectiveLoading(false)
    }
  }, [onGuardError])

  // Load all roles and permissions once
  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [rolesResult, permsResult] = await Promise.allSettled([
          tldwClient.listAdminRoles(),
          tldwClient.listPermissions()
        ])
        if (rolesResult.status === "fulfilled") {
          const r = rolesResult.value
          setAllRoles(Array.isArray(r) ? r : (r as any)?.data ?? [])
        }
        if (permsResult.status === "fulfilled") {
          const p = permsResult.value
          setAllPermissions(Array.isArray(p) ? p : (p as any)?.data ?? [])
        }
      } catch {
        // non-critical
      }
    }
    loadMeta()
  }, [])

  const handleUserSelect = useCallback((userId: number) => {
    setSelectedUserId(userId)
    loadUserData(userId)
  }, [loadUserData])

  const handleRemoveRole = useCallback(async (roleId: number) => {
    if (!selectedUserId) return
    try {
      await tldwClient.removeUserRole(selectedUserId, roleId)
      message.success("Role removed")
      await loadUserData(selectedUserId)
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to remove role"))
    }
  }, [selectedUserId, loadUserData])

  const handleAddRole = useCallback(async () => {
    if (!selectedUserId || !addRoleId) return
    setAddingRole(true)
    try {
      await tldwClient.assignUserRole(selectedUserId, addRoleId)
      message.success("Role assigned")
      setAddRoleModalOpen(false)
      setAddRoleId(null)
      await loadUserData(selectedUserId)
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to assign role"))
    } finally {
      setAddingRole(false)
    }
  }, [selectedUserId, addRoleId, loadUserData])

  const handleAddOverride = useCallback(async () => {
    if (!selectedUserId) return
    try {
      const values = await overrideForm.validateFields()
      setAddingOverride(true)
      await tldwClient.addUserOverride(selectedUserId, {
        permission_id: values.permission_id,
        effect: values.effect
      })
      message.success("Override added")
      setAddOverrideModalOpen(false)
      overrideForm.resetFields()
      await loadUserData(selectedUserId)
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to save user permission overrides")
      )
    } finally {
      setAddingOverride(false)
    }
  }, [selectedUserId, overrideForm, loadUserData])

  const handleRemoveOverride = useCallback(async (permissionId: number) => {
    if (!selectedUserId) return
    try {
      await tldwClient.deleteUserOverride(selectedUserId, permissionId)
      message.success("Override removed")
      await loadUserData(selectedUserId)
    } catch (err: any) {
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to revoke user permission override")
      )
    }
  }, [selectedUserId, loadUserData])

  const roleColumns = [
    {
      title: "Role",
      key: "name",
      render: (_: any, record: any) => record.name ?? record.role_name ?? `Role #${record.id ?? record.role_id}`
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: any, record: any) => (
        <Popconfirm
          title="Remove this role?"
          onConfirm={() => handleRemoveRole(record.id ?? record.role_id)}
        >
          <Button danger size="small" icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  const overrideColumns = [
    {
      title: "Permission",
      key: "permission_name",
      render: (_: any, record: UserOverride) =>
        record.permission_name ?? allPermissions.find(p => p.id === record.permission_id)?.name ?? `#${record.permission_id}`
    },
    {
      title: "Effect",
      dataIndex: "effect",
      key: "effect",
      width: 100,
      render: (v: string) => (
        <Tag color={v === "grant" ? "green" : "red"}>{v?.toUpperCase()}</Tag>
      )
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: any, record: UserOverride) => (
        <Popconfirm
          title="Remove this override?"
          onConfirm={() => handleRemoveOverride(record.permission_id)}
        >
          <Button danger size="small" icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  const effectiveColumns = [
    {
      title: "Permission",
      key: "permission_name",
      render: (_: any, record: EffectivePerm) =>
        record.permission_name ?? `#${record.permission_id}`
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      width: 120,
      render: (v: string) => v ? <Tag>{v}</Tag> : null
    },
    {
      title: "Effect",
      dataIndex: "effect",
      key: "effect",
      width: 100,
      render: (v: string) =>
        v ? <Tag color={v === "grant" ? "green" : v === "deny" ? "red" : "default"}>{v?.toUpperCase()}</Tag> : <Tag color="green">GRANT</Tag>
    }
  ]

  return (
    <Card title="User Permissions">
      <div style={{ marginBottom: 16 }}>
        <Select
          showSearch
          placeholder="Search for a user..."
          style={{ width: 400 }}
          loading={usersLoading}
          filterOption={false}
          onSearch={searchUsers}
          onChange={handleUserSelect}
          value={selectedUserId}
          options={users.map((u: any) => ({
            label: u.username ?? u.email ?? `User #${u.id}`,
            value: u.id
          }))}
        />
      </div>

      {selectedUserId && (
        <div>
          {/* Assigned Roles */}
          <Card
            size="small"
            title="Assigned Roles"
            style={{ marginBottom: 16 }}
            extra={
              <Button
                size="small"
                icon={<PlusOutlined />}
                onClick={() => setAddRoleModalOpen(true)}
              >
                Add Role
              </Button>
            }
          >
            <Table
              dataSource={userRoles}
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
            title="Permission Overrides"
            style={{ marginBottom: 16 }}
            extra={
              <Button
                size="small"
                icon={<PlusOutlined />}
                onClick={() => setAddOverrideModalOpen(true)}
              >
                Add Override
              </Button>
            }
          >
            <Table
              dataSource={overrides}
              columns={overrideColumns}
              rowKey="permission_id"
              loading={overridesLoading}
              pagination={false}
              size="small"
            />
          </Card>

          {/* Effective Permissions */}
          <Card size="small" title="Effective Permissions (read-only)">
            <Table
              dataSource={effectivePerms}
              columns={effectiveColumns}
              rowKey="permission_id"
              loading={effectiveLoading}
              pagination={false}
              size="small"
            />
          </Card>
        </div>
      )}

      {/* Add Role Modal */}
      <Modal
        title="Assign Role"
        open={addRoleModalOpen}
        onOk={handleAddRole}
        onCancel={() => { setAddRoleModalOpen(false); setAddRoleId(null) }}
        confirmLoading={addingRole}
        okButtonProps={{ disabled: !addRoleId }}
      >
        <Select
          placeholder="Select a role"
          style={{ width: "100%" }}
          value={addRoleId}
          onChange={setAddRoleId}
          options={allRoles.map(r => ({
            label: r.name,
            value: r.id,
            disabled: userRoles.some((ur: any) => (ur.id ?? ur.role_id) === r.id)
          }))}
        />
      </Modal>

      {/* Add Override Modal */}
      <Modal
        title="Add Permission Override"
        open={addOverrideModalOpen}
        onOk={handleAddOverride}
        onCancel={() => { setAddOverrideModalOpen(false); overrideForm.resetFields() }}
        confirmLoading={addingOverride}
      >
        <Form form={overrideForm} layout="vertical">
          <Form.Item
            name="permission_id"
            label="Permission"
            rules={[{ required: true, message: "Select a permission" }]}
          >
            <Select
              showSearch
              placeholder="Select permission"
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
            label="Effect"
            rules={[{ required: true, message: "Select an effect" }]}
          >
            <Select
              placeholder="Select effect"
              options={[
                { label: "Grant", value: "grant" },
                { label: "Deny", value: "deny" }
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
  const [guardError, setGuardError] = useState<string | null>(null)
  const guardRef = useRef(false)

  const handleGuardError = useCallback((err: any) => {
    const guard = deriveAdminGuardFromError(err)
    if (guard && !guardRef.current) {
      guardRef.current = true
      setGuardError(guard)
    }
  }, [])

  if (guardError) {
    return (
      <DesignSystemAlert variant="warning" title="Access Restricted" className="m-6">
        {guardError}
      </DesignSystemAlert>
    )
  }

  return (
    <div style={{ padding: 16 }}>
      <Tabs
        defaultActiveKey="matrix"
        items={[
          {
            key: "matrix",
            label: (
              <span>
                <SafetyOutlined /> Permission Matrix
              </span>
            ),
            children: <PermissionMatrixTab onGuardError={handleGuardError} />
          },
          {
            key: "roles",
            label: (
              <span>
                <LockOutlined /> Roles
              </span>
            ),
            children: <RolesTab onGuardError={handleGuardError} />
          },
          {
            key: "users",
            label: (
              <span>
                <LockOutlined /> User Permissions
              </span>
            ),
            children: <UserPermissionsTab onGuardError={handleGuardError} />
          }
        ]}
      />
    </div>
  )
}

export default RbacEditorPage
