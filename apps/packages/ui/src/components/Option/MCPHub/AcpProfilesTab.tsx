import { useEffect, useMemo, useState } from "react"
import { Button, Card, Empty, List, Space, Tag, Typography } from "antd"

import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"
import { createAcpProfile, listAcpProfiles, type McpHubProfile } from "@/services/tldw/mcp-hub"

export const AcpProfilesTab = () => {
  const [profiles, setProfiles] = useState<McpHubProfile[]>([])
  const [loading, setLoading] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [name, setName] = useState("")
  const [saving, setSaving] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const canSave = useMemo(() => name.trim().length > 0 && !saving, [name, saving])

  const loadProfiles = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const rows = await listAcpProfiles()
      setProfiles(Array.isArray(rows) ? rows : [])
    } catch {
      setProfiles([])
      setErrorMessage("Failed to load ACP profiles.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadProfiles()
  }, [])

  const handleCreate = async () => {
    if (!canSave) return
    setSaving(true)
    setErrorMessage(null)
    try {
      await createAcpProfile({
        name: name.trim(),
        owner_scope_type: "global",
        profile: {}
      })
      setCreateOpen(false)
      setName("")
      await loadProfiles()
    } catch {
      setErrorMessage("Failed to create ACP profile.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        ACP profiles define reusable MCP execution defaults.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}

      <Space>
        <Button type="primary" onClick={() => setCreateOpen(true)}>
          Create Profile
        </Button>
      </Space>

      {createOpen ? (
        <Card title="Create ACP Profile">
          <Space orientation="vertical" style={{ width: "100%" }}>
            <label htmlFor="mcp-profile-name">Profile Name</label>
            <input
              id="mcp-profile-name"
              aria-label="Profile Name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="default-dev"
            />
            <Space>
              <Button type="primary" onClick={handleCreate} disabled={!canSave} loading={saving}>
                Save Profile
              </Button>
              <Button
                onClick={() => {
                  setCreateOpen(false)
                  setName("")
                }}
              >
                Cancel
              </Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      <List
        bordered
        loading={loading}
        dataSource={profiles}
        locale={{ emptyText: <Empty description="No ACP profiles yet" /> }}
        renderItem={(profile) => (
          <List.Item>
            <Space orientation="vertical" size={2}>
              <Typography.Text strong>{profile.name}</Typography.Text>
              <Space size={6}>
                <Tag>{profile.owner_scope_type}</Tag>
                {profile.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
              </Space>
            </Space>
          </List.Item>
        )}
      />
    </Space>
  )
}
