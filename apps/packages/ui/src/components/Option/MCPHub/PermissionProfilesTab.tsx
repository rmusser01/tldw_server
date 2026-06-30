import { useEffect, useMemo, useState } from "react"
import { Button, Card, Checkbox, Empty, Modal, Space, Tag, Typography } from "antd"
import { EmptyState } from "@/components/ui/feedback"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  createPermissionProfile,
  deletePermissionProfile,
  getToolRegistrySummary,
  listPathScopeObjects,
  listExternalServers,
  listProfileCredentialBindings,
  listPermissionProfiles,
  deleteProfileCredentialBinding,
  upsertProfileCredentialBinding,
  updatePermissionProfile,
  type McpHubCredentialBinding,
  type McpHubExternalServer,
  type McpHubPathScopeObject,
  type McpHubPermissionPolicyDocument,
  type McpHubPermissionProfile,
  type McpHubToolRegistryEntry,
  type McpHubToolRegistryModule
} from "@/services/tldw/mcp-hub"

import {
  getCredentialBindingKey,
  getManagedExternalServers,
  getManagedExternalServerSlots,
  getPathAllowlistSummary,
  getPathScopeLabel,
  MCP_HUB_PROFILE_MODE_OPTIONS,
  MCP_HUB_SCOPE_OPTIONS
} from "./policyHelpers"
import { PolicyDocumentEditor } from "./PolicyDocumentEditor"

export const PermissionProfilesTab = () => {
  const [profiles, setProfiles] = useState<McpHubPermissionProfile[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<"global" | "org" | "team" | "user">("global")
  const [mode, setMode] = useState<"custom" | "preset">("custom")
  const [pathScopeSource, setPathScopeSource] = useState<"inline" | "named">("inline")
  const [pathScopeObjectId, setPathScopeObjectId] = useState("")
  const [policyDocument, setPolicyDocument] = useState<McpHubPermissionPolicyDocument>({})
  const [isActive, setIsActive] = useState(true)
  const [registryEntries, setRegistryEntries] = useState<McpHubToolRegistryEntry[]>([])
  const [registryModules, setRegistryModules] = useState<McpHubToolRegistryModule[]>([])
  const [externalServers, setExternalServers] = useState<McpHubExternalServer[]>([])
  const [pathScopeObjects, setPathScopeObjects] = useState<McpHubPathScopeObject[]>([])
  const [profileBindings, setProfileBindings] = useState<McpHubCredentialBinding[]>([])
  const [bindingsLoading, setBindingsLoading] = useState(false)
  const [bindingServerId, setBindingServerId] = useState<string | null>(null)
  const managedExternalServers = useMemo(
    () => getManagedExternalServers(externalServers),
    [externalServers]
  )
  const grantedBindingKeys = useMemo(
    () =>
      new Set(
        profileBindings.map((binding) =>
          getCredentialBindingKey(binding.external_server_id, binding.slot_name)
        )
      ),
    [profileBindings]
  )

  const canSave = useMemo(() => name.trim().length > 0 && !saving, [name, saving])

  const loadProfiles = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const rows = await listPermissionProfiles()
      setProfiles(Array.isArray(rows) ? rows : [])
    } catch (err) {
      setProfiles([])
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to load permission profiles: ${msg}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadProfiles()
  }, [])

  useEffect(() => {
    let cancelled = false
    const loadRegistryAndServers = async () => {
      try {
        const [summary, serverRows, pathScopeRows] = await Promise.all([
          getToolRegistrySummary(),
          listExternalServers(),
          listPathScopeObjects()
        ])
        if (!cancelled) {
          setRegistryEntries(Array.isArray(summary?.entries) ? summary.entries : [])
          setRegistryModules(Array.isArray(summary?.modules) ? summary.modules : [])
          setExternalServers(Array.isArray(serverRows) ? serverRows : [])
          setPathScopeObjects(Array.isArray(pathScopeRows) ? pathScopeRows : [])
        }
      } catch {
        if (!cancelled) {
          setRegistryEntries([])
          setRegistryModules([])
          setExternalServers([])
          setPathScopeObjects([])
        }
      }
    }
    void loadRegistryAndServers()
    return () => {
      cancelled = true
    }
  }, [])

  const resetForm = () => {
    setCreateOpen(false)
    setEditingId(null)
    setName("")
    setDescription("")
    setOwnerScopeType("global")
    setMode("custom")
    setPathScopeSource("inline")
    setPathScopeObjectId("")
    setPolicyDocument({})
    setIsActive(true)
    setProfileBindings([])
    setBindingsLoading(false)
    setBindingServerId(null)
  }

  const loadProfileBindings = async (profileId: number) => {
    setBindingsLoading(true)
    try {
      const rows = await listProfileCredentialBindings(profileId)
      setProfileBindings(Array.isArray(rows) ? rows : [])
    } catch (err) {
      setProfileBindings([])
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to load external server bindings: ${msg}`)
    } finally {
      setBindingsLoading(false)
    }
  }

  const openForEdit = (profile: McpHubPermissionProfile) => {
    setCreateOpen(true)
    setEditingId(profile.id)
    setName(profile.name)
    setDescription(String(profile.description || ""))
    setOwnerScopeType(profile.owner_scope_type)
    setMode(profile.mode)
    setPathScopeSource(profile.path_scope_object_id ? "named" : "inline")
    setPathScopeObjectId(profile.path_scope_object_id ? String(profile.path_scope_object_id) : "")
    setPolicyDocument(profile.policy_document || {})
    setIsActive(profile.is_active)
    setProfileBindings([])
    void loadProfileBindings(profile.id)
  }

  const handleSave = async () => {
    if (!canSave) return
    setSaving(true)
    setErrorMessage(null)
    try {
      const payload = {
        name: name.trim(),
        description: description.trim() || null,
        owner_scope_type: ownerScopeType,
        mode,
        path_scope_object_id:
          pathScopeSource === "named" && pathScopeObjectId ? Number(pathScopeObjectId) : null,
        policy_document: policyDocument,
        is_active: isActive
      }
      if (editingId) {
        await updatePermissionProfile(editingId, payload)
      } else {
        await createPermissionProfile(payload)
      }
      resetForm()
      await loadProfiles()
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(
        editingId
          ? `Failed to update permission profile: ${msg}`
          : `Failed to create permission profile: ${msg}`
      )
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = (profile: McpHubPermissionProfile) => {
    Modal.confirm({
      title: "Delete Profile",
      content: `Are you sure you want to delete the profile "${profile.name}"? This cannot be undone.`,
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        setErrorMessage(null)
        try {
          await deletePermissionProfile(profile.id)
          await loadProfiles()
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error"
          setErrorMessage(`Failed to delete permission profile: ${msg}`)
        }
      }
    })
  }

  const handleToggleExternalServer = async (
    serverId: string,
    checked: boolean,
    slotName?: string | null
  ) => {
    if (!editingId) return
    setBindingServerId(getCredentialBindingKey(serverId, slotName))
    setErrorMessage(null)
    try {
      if (checked) {
        await upsertProfileCredentialBinding(editingId, serverId, slotName)
      } else {
        await deleteProfileCredentialBinding(editingId, serverId, slotName)
      }
      const [bindingRows, serverRows] = await Promise.all([
        listProfileCredentialBindings(editingId),
        listExternalServers()
      ])
      setProfileBindings(Array.isArray(bindingRows) ? bindingRows : [])
      setExternalServers(Array.isArray(serverRows) ? serverRows : [])
    } catch {
      setErrorMessage("Failed to update external server binding.")
    } finally {
      setBindingServerId(null)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Reusable tool-access profiles define capabilities, exact tool allowlists, and baseline restrictions.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        New Profile
      </Button>

      {createOpen ? (
        <Card title={editingId ? "Edit Permission Profile" : "Create Permission Profile"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-permission-profile-name">Profile Name</label>
              <input
                id="mcp-permission-profile-name"
                aria-label="Profile Name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Read Only"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-permission-profile-description">Description</label>
              <input
                id="mcp-permission-profile-description"
                aria-label="Description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Restricts this persona to low-risk read flows"
              />
            </Space>
            <Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-permission-profile-scope">Owner Scope</label>
                <select
                  id="mcp-permission-profile-scope"
                  aria-label="Owner Scope"
                  value={ownerScopeType}
                  onChange={(event) => setOwnerScopeType(event.target.value as typeof ownerScopeType)}
                >
                  {MCP_HUB_SCOPE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-permission-profile-mode">Profile Mode</label>
                <select
                  id="mcp-permission-profile-mode"
                  aria-label="Profile Mode"
                  value={mode}
                  onChange={(event) => setMode(event.target.value as typeof mode)}
                >
                  {MCP_HUB_PROFILE_MODE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Space>
            </Space>

            <Card size="small" title="Path Scope Source">
              <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                <Typography.Text type="secondary">
                  Named path scopes provide reusable relative file rules. Inline path fields below stay
                  preserved and can still replace object values.
                </Typography.Text>
                <Space wrap>
                  <label>
                    <input
                      type="radio"
                      name="mcp-profile-path-scope-source"
                      checked={pathScopeSource === "inline"}
                      onChange={() => setPathScopeSource("inline")}
                    />
                    <span style={{ marginLeft: 8 }}>Use inline rules</span>
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="mcp-profile-path-scope-source"
                      checked={pathScopeSource === "named"}
                      onChange={() => setPathScopeSource("named")}
                    />
                    <span style={{ marginLeft: 8 }}>Use named path scope</span>
                  </label>
                </Space>
                {pathScopeSource === "named" ? (
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    <label htmlFor="mcp-profile-path-scope-object">Named path scope</label>
                    <select
                      id="mcp-profile-path-scope-object"
                      aria-label="Named path scope"
                      value={pathScopeObjectId}
                      onChange={(event) => setPathScopeObjectId(event.target.value)}
                    >
                      <option value="">Select a path scope</option>
                      {pathScopeObjects.map((pathScopeObject) => (
                        <option key={pathScopeObject.id} value={pathScopeObject.id}>
                          {pathScopeObject.name}
                        </option>
                      ))}
                    </select>
                  </Space>
                ) : null}
              </Space>
            </Card>

            <PolicyDocumentEditor
              formId="mcp-permission-profile"
              policy={policyDocument}
              onChange={setPolicyDocument}
              registryEntries={registryEntries}
              registryModules={registryModules}
            />

            {editingId ? (
              <Card size="small" title="External Service Bindings">
                <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                  <Typography.Text type="secondary">
                    Grant reusable access to managed external MCP servers here. Legacy inventory is
                    visible in External Servers and cannot be selected until imported into MCP Hub.
                  </Typography.Text>
                  {bindingsLoading ? (
                    <Typography.Text type="secondary">Loading external service bindings...</Typography.Text>
                  ) : managedExternalServers.length > 0 ? (
                    <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                      {managedExternalServers.map((server) => {
                        const slots = getManagedExternalServerSlots(server)
                        return (
                          <Card key={server.id} size="small" title={server.name}>
                            <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                              <Space wrap size={4}>
                                {server.secret_configured ? (
                                  <Tag color="green">secret configured</Tag>
                                ) : (
                                  <Tag>no secret</Tag>
                                )}
                                {server.binding_count ? (
                                  <Tag>{`${server.binding_count} bindings`}</Tag>
                                ) : null}
                              </Space>
                              {slots.length > 0 ? (
                                <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                                  {slots.map((slot) => {
                                    const bindingKey = getCredentialBindingKey(server.id, slot.slot_name)
                                    return (
                                      <Checkbox
                                        key={bindingKey}
                                        checked={grantedBindingKeys.has(bindingKey)}
                                        disabled={bindingServerId === bindingKey}
                                        onChange={(event) =>
                                          void handleToggleExternalServer(
                                            server.id,
                                            event.target.checked,
                                            slot.slot_name
                                          )
                                        }
                                      >
                                        <Space wrap size={4}>
                                          <span>{slot.display_name}</span>
                                          <Tag>{slot.slot_name}</Tag>
                                          <Tag>{slot.privilege_class}</Tag>
                                          {slot.secret_configured ? (
                                            <Tag color="green">slot secret configured</Tag>
                                          ) : (
                                            <Tag>slot secret missing</Tag>
                                          )}
                                        </Space>
                                      </Checkbox>
                                    )
                                  })}
                                </Space>
                              ) : (
                                <Typography.Text type="secondary">
                                  Define credential slots in External Servers before binding this service.
                                </Typography.Text>
                              )}
                            </Space>
                          </Card>
                        )
                      })}
                    </Space>
                  ) : (
                    <EmptyState
                      variant="inline"
                      size="sm"
                      title="No managed external servers are available yet."
                    />
                  )}
                </Space>
              </Card>
            ) : null}

            <Checkbox checked={isActive} onChange={(event) => setIsActive(event.target.checked)}>
              Active
            </Checkbox>
            <Space>
              <Button type="primary" onClick={handleSave} disabled={!canSave} loading={saving}>
                {editingId ? "Update Profile" : "Save Profile"}
              </Button>
              <Button onClick={resetForm}>Cancel</Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      {loading ? (
        <Card loading size="small" />
      ) : profiles.length === 0 ? (
        <EmptyState
          variant="inline"
          title="No permission profiles yet"
          description={
            <Space orientation="vertical" size={4}>
              <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                Profiles control which tools users and personas can access.
                Before creating a profile, review the <Typography.Text strong>Tool Catalog</Typography.Text> to see available tools.
              </Typography.Text>
            </Space>
          }
          primaryAction={{
            label: "Create Profile",
            onClick: () => setCreateOpen(true),
          }}
        />
      ) : (
        <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
          {profiles.map((profile) => (
            <Card key={profile.id} size="small">
              <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                <Space wrap>
                  <Typography.Text strong>{profile.name}</Typography.Text>
                  <Tag>{profile.owner_scope_type}</Tag>
                  <Tag color="blue">{profile.mode}</Tag>
                  {profile.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
                  {profile.path_scope_object_id ? (
                    <Tag color="purple">
                      {`path scope ${
                        pathScopeObjects.find((row) => row.id === profile.path_scope_object_id)?.name ||
                        profile.path_scope_object_id
                      }`}
                    </Tag>
                  ) : null}
                  <Button size="small" onClick={() => openForEdit(profile)}>
                    Edit
                  </Button>
                  <Button size="small" danger onClick={() => handleDelete(profile)}>
                    Delete
                  </Button>
                </Space>
                {profile.description ? (
                  <Typography.Text type="secondary">{profile.description}</Typography.Text>
                ) : null}
                <Space wrap>
                  {(profile.policy_document.capabilities || []).map((capability) => (
                    <Tag key={capability}>{capability}</Tag>
                  ))}
                  {(profile.policy_document.allowed_tools || []).map((tool) => (
                    <Tag key={tool} color="green">
                      {tool}
                    </Tag>
                  ))}
                  {(profile.policy_document.denied_tools || []).map((tool) => (
                    <Tag key={tool} color="red">
                      {tool}
                    </Tag>
                  ))}
                  {getPathScopeLabel(profile.policy_document.path_scope_mode) ? (
                    <Tag color="cyan">{getPathScopeLabel(profile.policy_document.path_scope_mode)}</Tag>
                  ) : null}
                  {getPathAllowlistSummary(profile.policy_document.path_allowlist_prefixes) ? (
                    <Tag color="blue">
                      {`paths ${getPathAllowlistSummary(profile.policy_document.path_allowlist_prefixes)}`}
                    </Tag>
                  ) : null}
                </Space>
              </Space>
            </Card>
          ))}
        </Space>
      )}
    </Space>
  )
}
