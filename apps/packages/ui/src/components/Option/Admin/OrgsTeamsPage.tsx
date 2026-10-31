import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  Card,
  Table,
  Button,
  Input,
  Modal,
  Select,
  Space,
  Popconfirm,
  message,
  Form,
  InputNumber,
  Tag
} from "antd"
import {
  PlusOutlined,
  DeleteOutlined,
  ReloadOutlined,
  TeamOutlined
} from "@ant-design/icons"
import { Alert } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"

import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { useAdminRequestScope, useAdminMutation } from "./admin-request-scope"

type WorkspaceProps = { config: TldwConfig; onGuardError: (error: unknown) => void }

// ── Types ──

interface Org {
  id: number
  name: string
  slug?: string
  member_count?: number
  created_at?: string
}

interface OrgMember {
  user_id: number
  username?: string
  role: string
  joined_at?: string
}

interface Team {
  id: number
  name: string
  org_id: number
  member_count?: number
}

interface TeamMember {
  user_id: number
  username?: string
  role: string
}

// ── Server-side paging helpers ──

// The orgs/teams endpoints return either a plain array (today's backend for
// members/teams) or an {items, total} envelope. Normalize the rows...
const extractRows = (result: any, ...keys: string[]): any[] => {
  if (Array.isArray(result)) return result
  for (const key of keys) {
    if (Array.isArray(result?.[key])) return result[key]
  }
  return []
}

// ...and derive a usable antd total. When the server reports no total, keep a
// phantom next page while the current page is full (offset + rows + 1) so
// operators can keep paging through plain-array responses.
const pageTotal = (result: any, rows: any[], pageSize: number, offset: number): number => {
  if (typeof result?.total === "number") return result.total
  return offset + rows.length + (rows.length >= pageSize ? 1 : 0)
}

// ── Org Members Sub-Table ──

const OrgMembersTable: React.FC<{ orgId: number } & WorkspaceProps> = ({ orgId, config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [members, setMembers] = useState<OrgMember[]>([])
  const [loading, setLoading] = useState(false)
  const [addModalOpen, setAddModalOpen] = useState(false)
  const [addForm] = Form.useForm()
  // Server-side paging: the roster no longer needs to arrive in one response.
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [total, setTotal] = useState(0)


  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const [loadError, setLoadError] = useState<string | null>(null)
  const readRequest = useRef(0)

  const loadMembers = useCallback(async () => {
    const scope = captureScope()
    if (!scope) return
    const request = ++readRequest.current
    const current = () => !scope.abortSignal.aborted && request === readRequest.current
    setLoading(true)
    setLoadError(null)
    try {
      const offset = (page - 1) * pageSize
      const result = await tldwClient.listOrgMembers(orgId, { limit: pageSize, offset }, scope)
      if (!current()) return
      const rows = extractRows(result, "items", "data", "members")
      setMembers(rows)
      setTotal(pageTotal(result, rows, pageSize, offset))
    } catch (error) {
      if (!current()) return
      onGuardError(error)
      setMembers([])
      setLoadError(sanitizeAdminErrorMessage(error, "Organization members unavailable"))
    } finally {
      if (current()) setLoading(false)
    }
  }, [captureScope, orgId, onGuardError, page, pageSize])

  useEffect(() => { void loadMembers() }, [loadMembers])

  const handleAddMember = () => runMutation(
    t("settings:adminOrgs.addMemberFailed", "Failed to save membership"), async (scope) => {
      const values = await addForm.validateFields()
      if (scope.abortSignal.aborted) return
      await tldwClient.addOrgMember(orgId, { user_id: values.user_id, role: values.role }, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminOrgs.memberAdded", "Member added"))
      setAddModalOpen(false)
      addForm.resetFields()
      await loadMembers()
    }
  )

  const handleRemove = (userId: number) => runMutation(t("settings:adminOrgs.removeMemberFailed", "Failed to remove member"), async (scope) => {
    await tldwClient.removeOrgMember(orgId, userId, scope)
    if (scope.abortSignal.aborted) return
    message.success(t("settings:adminOrgs.memberRemoved", "Member removed"))
    await loadMembers()
  })

  const handleRoleChange = (userId: number, role: string) => runMutation(t("settings:adminOrgs.updateRoleFailed", "Failed to update role"), async (scope) => {
    await tldwClient.updateOrgMemberRole(orgId, userId, { role }, scope)
    if (scope.abortSignal.aborted) return
    message.success(t("settings:adminOrgs.roleUpdated", "Role updated"))
    await loadMembers()
  })

  const columns = [
    {
      title: t("settings:adminOrgs.colUserId", "User ID"),
      dataIndex: "user_id",
      key: "user_id",
      width: 100
    },
    {
      title: t("settings:adminOrgs.colUsername", "Username"),
      dataIndex: "username",
      key: "username",
      render: (v: string) => v || "\u2014"
    },
    {
      title: t("settings:adminOrgs.colRole", "Role"),
      dataIndex: "role",
      key: "role",
      width: 160,
      render: (role: string, record: OrgMember) => (
        <Select
          value={role}
          size="small"
          style={{ width: 130 }}
          disabled={mutating || loading || Boolean(loadError)}
          onChange={(val) => handleRoleChange(record.user_id, val)}
          options={[
            { value: "owner", label: t("settings:adminOrgs.roleOwner", "Owner") },
            { value: "admin", label: t("settings:adminOrgs.roleAdmin", "Admin") },
            { value: "member", label: t("settings:adminOrgs.roleMember", "Member") },
            { value: "viewer", label: t("settings:adminOrgs.roleViewer", "Viewer") }
          ]}
        />
      )
    },
    {
      title: t("settings:adminOrgs.colJoined", "Joined"),
      dataIndex: "joined_at",
      key: "joined_at",
      render: (v: string) => (v ? new Date(v).toLocaleDateString() : "\u2014")
    },
    {
      title: t("settings:adminOrgs.colActions", "Actions"),
      key: "actions",
      width: 80,
      render: (_: any, record: OrgMember) => (
        <Popconfirm
          title={t("settings:adminOrgs.removeMemberConfirm", "Remove this member?")}
          onConfirm={() => handleRemove(record.user_id)}
          okText={t("settings:adminOrgs.remove", "Remove")}
          okButtonProps={{ danger: true }}
        >
          <Button type="text" size="small" danger disabled={mutating || loading || Boolean(loadError)} icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <strong>{t("settings:adminOrgs.orgMembers", "Organization Members")}</strong>
        <Space>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadMembers()}>
            {t("common:refresh", "Refresh")}
          </Button>
          <Button size="small" type="primary" disabled={mutating || loading || Boolean(loadError)} icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>
            {t("settings:adminOrgs.addMember", "Add Member")}
          </Button>
        </Space>
      </div>
      {loadError && <Alert variant="error" title={t("settings:adminOrgs.OrgMembersTableUnavailable", "Organization members unavailable")}>
        <Space orientation="vertical"><span>{loadError}</span><Button onClick={() => void loadMembers()}>{t("common:retry", "Retry")}</Button></Space>
      </Alert>}
      <Table
        dataSource={members}
        columns={columns}
        rowKey="user_id"
        loading={loading}
        pagination={{
          current: page,
          pageSize,
          total,
          hideOnSinglePage: true,
          showTotal: (tot) => t("settings:adminOrgs.paginationTotal", "Total {{total}} items", { total: tot }),
          onChange: (p, ps) => {
            setPage(p)
            setPageSize(ps)
          }
        }}
        size="small"
      />
      <Modal
        title={t("settings:adminOrgs.addOrgMemberTitle", "Add Organization Member")}
        open={addModalOpen}
        onCancel={() => { if (!mutating) { setAddModalOpen(false); addForm.resetFields() } }}
        onOk={handleAddMember}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
      >
        <Form form={addForm} layout="vertical">
          <Form.Item name="user_id" label={t("settings:adminOrgs.colUserId", "User ID")} rules={[{ required: true, message: t("settings:adminOrgs.userIdRequired", "User ID is required") }]}>
            <InputNumber style={{ width: "100%" }} min={1} placeholder={t("settings:adminOrgs.userIdPlaceholder", "Enter user ID")} />
          </Form.Item>
          <Form.Item name="role" label={t("settings:adminOrgs.colRole", "Role")} initialValue="member">
            <Select
              options={[
                { value: "owner", label: t("settings:adminOrgs.roleOwner", "Owner") },
                { value: "admin", label: t("settings:adminOrgs.roleAdmin", "Admin") },
                { value: "member", label: t("settings:adminOrgs.roleMember", "Member") },
                { value: "viewer", label: t("settings:adminOrgs.roleViewer", "Viewer") }
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

// ── Team Members Sub-Table ──

const TeamMembersTable: React.FC<{ teamId: number } & WorkspaceProps> = ({ teamId, config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [members, setMembers] = useState<TeamMember[]>([])
  const [loading, setLoading] = useState(false)
  const [addModalOpen, setAddModalOpen] = useState(false)
  const [addForm] = Form.useForm()
  // Server-side paging (the backend honors these once the endpoint adopts
  // limit/offset like its siblings; phantom totals bridge plain arrays).
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [total, setTotal] = useState(0)


  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const [loadError, setLoadError] = useState<string | null>(null)
  const readRequest = useRef(0)

  const loadMembers = useCallback(async () => {
    const scope = captureScope()
    if (!scope) return
    const request = ++readRequest.current
    const current = () => !scope.abortSignal.aborted && request === readRequest.current
    setLoading(true)
    setLoadError(null)
    try {
      const offset = (page - 1) * pageSize
      const result = await tldwClient.listTeamMembers(teamId, { limit: pageSize, offset }, scope)
      if (!current()) return
      const rows = extractRows(result, "items", "data", "members")
      setMembers(rows)
      setTotal(pageTotal(result, rows, pageSize, offset))
    } catch (error) {
      if (!current()) return
      onGuardError(error)
      setMembers([])
      setLoadError(sanitizeAdminErrorMessage(error, "Team members unavailable"))
    } finally {
      if (current()) setLoading(false)
    }
  }, [captureScope, teamId, onGuardError, page, pageSize])

  useEffect(() => { void loadMembers() }, [loadMembers])

  const handleAddMember = () => runMutation(
    t("settings:adminOrgs.addMemberFailed", "Failed to save membership"), async (scope) => {
      const values = await addForm.validateFields()
      if (scope.abortSignal.aborted) return
      await tldwClient.addTeamMember(teamId, { user_id: values.user_id, role: values.role }, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminOrgs.memberAdded", "Team member added"))
      setAddModalOpen(false)
      addForm.resetFields()
      await loadMembers()
    }
  )

  const handleRemove = (userId: number) => runMutation(t("settings:adminOrgs.removeMemberFailed", "Failed to remove member"), async (scope) => {
    await tldwClient.removeTeamMember(teamId, userId, scope)
    if (scope.abortSignal.aborted) return
    message.success(t("settings:adminOrgs.memberRemoved", "Member removed"))
    await loadMembers()
  })

  const handleRoleChange = (userId: number, role: string) => runMutation(t("settings:adminOrgs.updateRoleFailed", "Failed to update role"), async (scope) => {
    await tldwClient.updateTeamMemberRole(teamId, userId, { role }, scope)
    if (scope.abortSignal.aborted) return
    message.success(t("settings:adminOrgs.roleUpdated", "Role updated"))
    await loadMembers()
  })

  const columns = [
    {
      title: t("settings:adminOrgs.colUserId", "User ID"),
      dataIndex: "user_id",
      key: "user_id",
      width: 100
    },
    {
      title: t("settings:adminOrgs.colUsername", "Username"),
      dataIndex: "username",
      key: "username",
      render: (v: string) => v || "\u2014"
    },
    {
      title: t("settings:adminOrgs.colRole", "Role"),
      dataIndex: "role",
      key: "role",
      width: 160,
      render: (role: string, record: TeamMember) => (
        <Select
          value={role}
          size="small"
          style={{ width: 130 }}
          disabled={mutating || loading || Boolean(loadError)}
          onChange={(val) => handleRoleChange(record.user_id, val)}
          options={[
            { value: "lead", label: t("settings:adminOrgs.roleLead", "Lead") },
            { value: "member", label: t("settings:adminOrgs.roleMember", "Member") },
            { value: "viewer", label: t("settings:adminOrgs.roleViewer", "Viewer") }
          ]}
        />
      )
    },
    {
      title: t("settings:adminOrgs.colActions", "Actions"),
      key: "actions",
      width: 80,
      render: (_: any, record: TeamMember) => (
        <Popconfirm
          title={t("settings:adminOrgs.removeTeamMemberConfirm", "Remove this team member?")}
          onConfirm={() => handleRemove(record.user_id)}
          okText={t("settings:adminOrgs.remove", "Remove")}
          okButtonProps={{ danger: true }}
        >
          <Button type="text" size="small" danger disabled={mutating || loading || Boolean(loadError)} icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <strong>{t("settings:adminOrgs.teamMembers", "Team Members")}</strong>
        <Space>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadMembers()}>
            {t("common:refresh", "Refresh")}
          </Button>
          <Button size="small" type="primary" disabled={mutating || loading || Boolean(loadError)} icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>
            {t("settings:adminOrgs.addMember", "Add Member")}
          </Button>
        </Space>
      </div>
      {loadError && <Alert variant="error" title={t("settings:adminOrgs.TeamMembersTableUnavailable", "Team members unavailable")}>
        <Space orientation="vertical"><span>{loadError}</span><Button onClick={() => void loadMembers()}>{t("common:retry", "Retry")}</Button></Space>
      </Alert>}
      <Table
        dataSource={members}
        columns={columns}
        rowKey="user_id"
        loading={loading}
        pagination={{
          current: page,
          pageSize,
          total,
          hideOnSinglePage: true,
          showTotal: (tot) => t("settings:adminOrgs.paginationTotal", "Total {{total}} items", { total: tot }),
          onChange: (p, ps) => {
            setPage(p)
            setPageSize(ps)
          }
        }}
        size="small"
      />
      <Modal
        title={t("settings:adminOrgs.addTeamMemberTitle", "Add Team Member")}
        open={addModalOpen}
        onCancel={() => { if (!mutating) { setAddModalOpen(false); addForm.resetFields() } }}
        onOk={handleAddMember}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
      >
        <Form form={addForm} layout="vertical">
          <Form.Item name="user_id" label={t("settings:adminOrgs.colUserId", "User ID")} rules={[{ required: true, message: t("settings:adminOrgs.userIdRequired", "User ID is required") }]}>
            <InputNumber style={{ width: "100%" }} min={1} placeholder={t("settings:adminOrgs.userIdPlaceholder", "Enter user ID")} />
          </Form.Item>
          <Form.Item name="role" label={t("settings:adminOrgs.colRole", "Role")} initialValue="member">
            <Select
              options={[
                { value: "lead", label: t("settings:adminOrgs.roleLead", "Lead") },
                { value: "member", label: t("settings:adminOrgs.roleMember", "Member") },
                { value: "viewer", label: t("settings:adminOrgs.roleViewer", "Viewer") }
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

// ── Teams Sub-Table (inside org expanded row) ──

const TeamsTable: React.FC<{ orgId: number } & WorkspaceProps> = ({ orgId, config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [teams, setTeams] = useState<Team[]>([])
  const [loading, setLoading] = useState(false)
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [createForm] = Form.useForm()
  // Server-side paging, mirroring OrgMembersTable.
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [total, setTotal] = useState(0)


  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const [loadError, setLoadError] = useState<string | null>(null)
  const readRequest = useRef(0)

  const loadTeams = useCallback(async () => {
    const scope = captureScope()
    if (!scope) return
    const request = ++readRequest.current
    const current = () => !scope.abortSignal.aborted && request === readRequest.current
    setLoading(true)
    setLoadError(null)
    try {
      const offset = (page - 1) * pageSize
      const result = await tldwClient.listTeams(orgId, { limit: pageSize, offset }, scope)
      if (!current()) return
      const rows = extractRows(result, "items", "data", "teams")
      setTeams(rows)
      setTotal(pageTotal(result, rows, pageSize, offset))
    } catch (error) {
      if (!current()) return
      onGuardError(error)
      setTeams([])
      setLoadError(sanitizeAdminErrorMessage(error, "Teams unavailable"))
    } finally {
      if (current()) setLoading(false)
    }
  }, [captureScope, orgId, onGuardError, page, pageSize])

  useEffect(() => { void loadTeams() }, [loadTeams])

  const handleCreateTeam = () => runMutation(
    t("settings:adminOrgs.createTeamFailed", "Failed to save membership"), async (scope) => {
      const values = await createForm.validateFields()
      if (scope.abortSignal.aborted) return
      await tldwClient.createTeam(orgId, { name: values.name }, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminOrgs.teamCreated", "Team created"))
      setCreateModalOpen(false)
      createForm.resetFields()
      await loadTeams()
    }
  )

  const teamColumns = [
    {
      title: t("settings:adminOrgs.colTeamName", "Team Name"),
      dataIndex: "name",
      key: "name"
    },
    {
      title: t("settings:adminOrgs.colMemberCount", "Member Count"),
      dataIndex: "member_count",
      key: "member_count",
      width: 120,
      render: (v: number) => v ?? "\u2014"
    }
  ]

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <strong><TeamOutlined /> {t("settings:adminOrgs.teams", "Teams")}</strong>
        <Space>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadTeams()}>
            {t("common:refresh", "Refresh")}
          </Button>
          <Button size="small" type="primary" disabled={mutating || loading || Boolean(loadError)} icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
            {t("settings:adminOrgs.createTeam", "Create Team")}
          </Button>
        </Space>
      </div>
      {loadError && <Alert variant="error" title={t("settings:adminOrgs.TeamsTableUnavailable", "Teams unavailable")}>
        <Space orientation="vertical"><span>{loadError}</span><Button onClick={() => void loadTeams()}>{t("common:retry", "Retry")}</Button></Space>
      </Alert>}
      <Table
        dataSource={teams}
        columns={teamColumns}
        rowKey="id"
        loading={loading}
        pagination={{
          current: page,
          pageSize,
          total,
          hideOnSinglePage: true,
          showTotal: (tot) => t("settings:adminOrgs.paginationTotal", "Total {{total}} items", { total: tot }),
          onChange: (p, ps) => {
            setPage(p)
            setPageSize(ps)
          }
        }}
        size="small"
        expandable={{
          expandedRowRender: (team: Team) => <TeamMembersTable key={team.id} teamId={team.id} config={config} onGuardError={onGuardError} />
        }}
      />
      <Modal
        title={t("settings:adminOrgs.createTeamTitle", "Create Team")}
        open={createModalOpen}
        onCancel={() => { if (!mutating) { setCreateModalOpen(false); createForm.resetFields() } }}
        onOk={handleCreateTeam}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item name="name" label={t("settings:adminOrgs.teamNameLabel", "Team Name")} rules={[{ required: true, message: t("settings:adminOrgs.teamNameRequired", "Team name is required") }]}>
            <Input placeholder={t("settings:adminOrgs.teamNamePlaceholder", "Enter team name")} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

// ── Main Page ──

const OrgsTeamsPage: React.FC = () => {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const { t } = useTranslation("settings")
  if (!config || authorityLoading) return <p role="status">{t("admin.resolvingConnection", "Resolving admin connection…")}</p>
  return <OrgsTeamsWorkspace key={deriveConnectionAuthorityId(config)} config={config} />
}

const OrgsTeamsWorkspace = ({ config }: { config: TldwConfig }) => {
  const { t } = useTranslation(["settings", "common"])
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  const [orgs, setOrgs] = useState<Org[]>([])
  const [orgsLoading, setOrgsLoading] = useState(false)
  const [searchText, setSearchText] = useState("")
  const [orgsTotal, setOrgsTotal] = useState<number | null>(null)

  const [createOrgModalOpen, setCreateOrgModalOpen] = useState(false)
  const [createOrgForm] = Form.useForm()


  const captureScope = useAdminRequestScope(config)
  const orgsRequest = useRef(0)
  const [orgsError, setOrgsError] = useState<string | null>(null)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  const { runMutation, mutating } = useAdminMutation(captureScope, markAdminGuardFromError)

  const loadOrgs = useCallback(async (search?: string) => {
    const scope = captureScope()
    if (!scope) return
    const request = ++orgsRequest.current
    const current = () => !scope.abortSignal.aborted && request === orgsRequest.current
    setOrgsLoading(true)
    setOrgsError(null)
    try {
      const params: { search?: string; limit?: number; offset?: number } = { limit: 100 }
      if (search) params.search = search
      const result = await tldwClient.listOrgs(params, scope)
      if (!current()) return
      setOrgs(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.organizations ?? [])
      setOrgsTotal(typeof result?.total === "number" ? result.total : null)
    } catch (error) {
      if (!current()) return
      markAdminGuardFromError(error)
      setOrgs([])
      setOrgsError(sanitizeAdminErrorMessage(error, "Organizations unavailable"))
    } finally {
      if (current()) setOrgsLoading(false)
    }
  }, [captureScope, markAdminGuardFromError])

  useEffect(() => { void loadOrgs() }, [loadOrgs])

  const handleSearch = () => { void loadOrgs(searchText || undefined) }

  const handleCreateOrg = () => runMutation(t("settings:adminOrgs.createOrgFailed", "Failed to create organization"), async (scope) => {
    const values = await createOrgForm.validateFields()
    if (scope.abortSignal.aborted) return
    const payload: { name: string; slug?: string } = { name: values.name }
    if (values.slug) payload.slug = values.slug
    await tldwClient.createOrg(payload, scope)
    if (scope.abortSignal.aborted) return
    message.success(t("settings:adminOrgs.orgCreated", "Organization created"))
    setCreateOrgModalOpen(false)
    createOrgForm.resetFields()
    await loadOrgs()
  })

  const orgColumns = [
    {
      title: t("settings:adminOrgs.colName", "Name"),
      dataIndex: "name",
      key: "name",
      render: (name: string) => <strong>{name}</strong>
    },
    {
      title: t("settings:adminOrgs.colSlug", "Slug"),
      dataIndex: "slug",
      key: "slug",
      render: (v: string) => v ? <Tag>{v}</Tag> : "\u2014"
    },
    {
      title: t("settings:adminOrgs.colMembers", "Members"),
      dataIndex: "member_count",
      key: "member_count",
      width: 100,
      render: (v: number) => v ?? "\u2014"
    },
    {
      title: t("settings:adminOrgs.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 140,
      render: (v: string) => (v ? new Date(v).toLocaleDateString() : "\u2014")
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminOrgs.forbiddenTitle", "Access Denied")}>
        {t("settings:adminOrgs.forbiddenBody", "You don't have permission to manage organizations.")}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminOrgs.notFoundTitle", "Not Available")}>
        {t("settings:adminOrgs.notFoundBody", "Organization management is not available on this server.")}
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminOrgs.title", "Organizations & Teams")}</h1>
      </div>

      {orgsError && <Alert variant="error" title={t("settings:adminOrgs.orgsUnavailable", "Organizations unavailable")}>
        <span>{orgsError}</span><Button onClick={() => void loadOrgs(searchText || undefined)}>{t("common:retry", "Retry")}</Button>
      </Alert>}
      <Card
        title={t("settings:adminOrgs.orgsCardTitle", "Organizations")}
        extra={
          <Space>
            <Input.Search
              placeholder={t("settings:adminOrgs.searchOrgs", "Search orgs...")}
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              onSearch={handleSearch}
              style={{ width: 220 }}
              size="small"
              allowClear
            />
            <Button size="small" icon={<ReloadOutlined />} onClick={() => loadOrgs(searchText || undefined)}>
              {t("common:refresh", "Refresh")}
            </Button>
            <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setCreateOrgModalOpen(true)}>
              {t("settings:adminOrgs.createOrg", "Create Org")}
            </Button>
          </Space>
        }
      >
        {/* Bounded snapshot disclosure (oversight-page pattern): the orgs list
            stays at limit:100 until full server pagination lands; say so when
            the server holds more instead of implying completeness. */}
        {(orgsTotal != null && orgsTotal > orgs.length) || orgs.length >= 100 ? (
          <span
            style={{
              display: "inline-block",
              marginBottom: 8,
              color: "var(--color-text-secondary, #888)",
              fontSize: "0.85rem"
            }}
          >
            {orgsTotal != null && orgsTotal > orgs.length
              ? t("settings:adminOrgs.showingOf", "Showing {{shown}} of {{total}}", {
                  shown: orgs.length,
                  total: orgsTotal
                })
              : t("settings:adminOrgs.showingFirst100", "Showing first 100")}
          </span>
        ) : null}
        <Table
          dataSource={orgs}
          columns={orgColumns}
          rowKey="id"
          loading={orgsLoading}
          pagination={orgs.length > 20 ? { pageSize: 20 } : false}
          size="small"
          expandable={{
            expandedRowRender: (org: Org) => (
              <div style={{ paddingLeft: 16 }}>
                <OrgMembersTable key={org.id} orgId={org.id} config={config} onGuardError={markAdminGuardFromError} />
                <TeamsTable key={org.id} orgId={org.id} config={config} onGuardError={markAdminGuardFromError} />
              </div>
            )
          }}
        />
      </Card>

      {/* Create Org Modal */}
      <Modal
        title={t("settings:adminOrgs.createOrgTitle", "Create Organization")}
        open={createOrgModalOpen}
        onCancel={() => { if (!mutating) { setCreateOrgModalOpen(false); createOrgForm.resetFields() } }}
        onOk={handleCreateOrg}
        confirmLoading={mutating}
        closable={!mutating}
        maskClosable={!mutating}
        cancelButtonProps={{ disabled: mutating }}
      >
        <Form form={createOrgForm} layout="vertical">
          <Form.Item name="name" label={t("settings:adminOrgs.orgNameLabel", "Organization Name")} rules={[{ required: true, message: t("settings:adminOrgs.orgNameRequired", "Name is required") }]}>
            <Input placeholder={t("settings:adminOrgs.orgNamePlaceholder", "Enter organization name")} />
          </Form.Item>
          <Form.Item name="slug" label={t("settings:adminOrgs.slugLabel", "Slug (optional)")}>
            <Input placeholder={t("settings:adminOrgs.slugPlaceholder", "e.g. my-org")} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default OrgsTeamsPage
