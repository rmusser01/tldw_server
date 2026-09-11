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
import { tldwClient } from "@/services/tldw/TldwApiClient"

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

// ── Org Members Sub-Table ──

const OrgMembersTable: React.FC<{ orgId: number }> = ({ orgId }) => {
  const { t } = useTranslation(["settings", "common"])
  const [members, setMembers] = useState<OrgMember[]>([])
  const [loading, setLoading] = useState(false)
  const [addModalOpen, setAddModalOpen] = useState(false)
  const [addForm] = Form.useForm()
  const [adding, setAdding] = useState(false)

  const loadMembers = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listOrgMembers(orgId)
      setMembers(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.members ?? [])
    } catch {
      // silently handled by parent guard
    } finally {
      setLoading(false)
    }
  }, [orgId])

  useEffect(() => {
    void loadMembers()
  }, [loadMembers])

  const handleAddMember = async () => {
    try {
      const values = await addForm.validateFields()
      setAdding(true)
      await tldwClient.addOrgMember(orgId, { user_id: values.user_id, role: values.role })
      message.success(t("settings:adminOrgs.memberAdded", "Member added"))
      setAddModalOpen(false)
      addForm.resetFields()
      void loadMembers()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.addMemberFailed", "Failed to add member")))
    } finally {
      setAdding(false)
    }
  }

  const handleRemove = async (userId: number) => {
    try {
      await tldwClient.removeOrgMember(orgId, userId)
      message.success(t("settings:adminOrgs.memberRemoved", "Member removed"))
      void loadMembers()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.removeMemberFailed", "Failed to remove member")))
    }
  }

  const handleRoleChange = async (userId: number, role: string) => {
    try {
      await tldwClient.updateOrgMemberRole(orgId, userId, { role })
      message.success(t("settings:adminOrgs.roleUpdated", "Role updated"))
      void loadMembers()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.updateRoleFailed", "Failed to update role")))
    }
  }

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
          <Button type="text" size="small" danger icon={<DeleteOutlined />} />
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
          <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>
            {t("settings:adminOrgs.addMember", "Add Member")}
          </Button>
        </Space>
      </div>
      <Table
        dataSource={members}
        columns={columns}
        rowKey="user_id"
        loading={loading}
        pagination={false}
        size="small"
      />
      <Modal
        title={t("settings:adminOrgs.addOrgMemberTitle", "Add Organization Member")}
        open={addModalOpen}
        onCancel={() => { setAddModalOpen(false); addForm.resetFields() }}
        onOk={handleAddMember}
        confirmLoading={adding}
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

const TeamMembersTable: React.FC<{ teamId: number }> = ({ teamId }) => {
  const { t } = useTranslation(["settings", "common"])
  const [members, setMembers] = useState<TeamMember[]>([])
  const [loading, setLoading] = useState(false)
  const [addModalOpen, setAddModalOpen] = useState(false)
  const [addForm] = Form.useForm()
  const [adding, setAdding] = useState(false)

  const loadMembers = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listTeamMembers(teamId)
      setMembers(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.members ?? [])
    } catch {
      // silently handled
    } finally {
      setLoading(false)
    }
  }, [teamId])

  useEffect(() => {
    void loadMembers()
  }, [loadMembers])

  const handleAddMember = async () => {
    try {
      const values = await addForm.validateFields()
      setAdding(true)
      await tldwClient.addTeamMember(teamId, { user_id: values.user_id, role: values.role })
      message.success(t("settings:adminOrgs.teamMemberAdded", "Team member added"))
      setAddModalOpen(false)
      addForm.resetFields()
      void loadMembers()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.addTeamMemberFailed", "Failed to add team member")))
    } finally {
      setAdding(false)
    }
  }

  const handleRemove = async (userId: number) => {
    try {
      await tldwClient.removeTeamMember(teamId, userId)
      message.success(t("settings:adminOrgs.teamMemberRemoved", "Team member removed"))
      void loadMembers()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.removeTeamMemberFailed", "Failed to remove team member")))
    }
  }

  const handleRoleChange = async (userId: number, role: string) => {
    try {
      await tldwClient.updateTeamMemberRole(teamId, userId, { role })
      message.success(t("settings:adminOrgs.roleUpdated", "Role updated"))
      void loadMembers()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.updateRoleFailed", "Failed to update role")))
    }
  }

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
          <Button type="text" size="small" danger icon={<DeleteOutlined />} />
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
          <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>
            {t("settings:adminOrgs.addMember", "Add Member")}
          </Button>
        </Space>
      </div>
      <Table
        dataSource={members}
        columns={columns}
        rowKey="user_id"
        loading={loading}
        pagination={false}
        size="small"
      />
      <Modal
        title={t("settings:adminOrgs.addTeamMemberTitle", "Add Team Member")}
        open={addModalOpen}
        onCancel={() => { setAddModalOpen(false); addForm.resetFields() }}
        onOk={handleAddMember}
        confirmLoading={adding}
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

const TeamsTable: React.FC<{ orgId: number }> = ({ orgId }) => {
  const { t } = useTranslation(["settings", "common"])
  const [teams, setTeams] = useState<Team[]>([])
  const [loading, setLoading] = useState(false)
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)

  const loadTeams = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listTeams(orgId)
      setTeams(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.teams ?? [])
    } catch {
      // silently handled
    } finally {
      setLoading(false)
    }
  }, [orgId])

  useEffect(() => {
    void loadTeams()
  }, [loadTeams])

  const handleCreateTeam = async () => {
    try {
      const values = await createForm.validateFields()
      setCreating(true)
      await tldwClient.createTeam(orgId, { name: values.name })
      message.success(t("settings:adminOrgs.teamCreated", "Team created"))
      setCreateModalOpen(false)
      createForm.resetFields()
      void loadTeams()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.createTeamFailed", "Failed to create team")))
    } finally {
      setCreating(false)
    }
  }

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
          <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
            {t("settings:adminOrgs.createTeam", "Create Team")}
          </Button>
        </Space>
      </div>
      <Table
        dataSource={teams}
        columns={teamColumns}
        rowKey="id"
        loading={loading}
        pagination={false}
        size="small"
        expandable={{
          expandedRowRender: (team: Team) => <TeamMembersTable teamId={team.id} />
        }}
      />
      <Modal
        title={t("settings:adminOrgs.createTeamTitle", "Create Team")}
        open={createModalOpen}
        onCancel={() => { setCreateModalOpen(false); createForm.resetFields() }}
        onOk={handleCreateTeam}
        confirmLoading={creating}
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
  const { t } = useTranslation(["settings", "common"])
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  const [orgs, setOrgs] = useState<Org[]>([])
  const [orgsLoading, setOrgsLoading] = useState(false)
  const [searchText, setSearchText] = useState("")

  const [createOrgModalOpen, setCreateOrgModalOpen] = useState(false)
  const [createOrgForm] = Form.useForm()
  const [creatingOrg, setCreatingOrg] = useState(false)

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  const loadOrgs = useCallback(async (search?: string) => {
    setOrgsLoading(true)
    try {
      const params: { search?: string; limit?: number; offset?: number } = { limit: 100 }
      if (search) params.search = search
      const result = await tldwClient.listOrgs(params)
      setOrgs(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.organizations ?? [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setOrgsLoading(false)
    }
  }, [markAdminGuardFromError])

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadOrgs()
  }, [loadOrgs])

  const handleSearch = () => {
    void loadOrgs(searchText || undefined)
  }

  const handleCreateOrg = async () => {
    try {
      const values = await createOrgForm.validateFields()
      setCreatingOrg(true)
      const payload: { name: string; slug?: string } = { name: values.name }
      if (values.slug) payload.slug = values.slug
      await tldwClient.createOrg(payload)
      message.success(t("settings:adminOrgs.orgCreated", "Organization created"))
      setCreateOrgModalOpen(false)
      createOrgForm.resetFields()
      void loadOrgs()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminOrgs.createOrgFailed", "Failed to create organization")))
    } finally {
      setCreatingOrg(false)
    }
  }

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
                <OrgMembersTable orgId={org.id} />
                <TeamsTable orgId={org.id} />
              </div>
            )
          }}
        />
      </Card>

      {/* Create Org Modal */}
      <Modal
        title={t("settings:adminOrgs.createOrgTitle", "Create Organization")}
        open={createOrgModalOpen}
        onCancel={() => { setCreateOrgModalOpen(false); createOrgForm.resetFields() }}
        onOk={handleCreateOrg}
        confirmLoading={creatingOrg}
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
