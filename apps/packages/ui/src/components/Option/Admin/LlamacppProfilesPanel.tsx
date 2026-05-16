import React from "react"
import {
  Button,
  Card,
  Empty,
  Input,
  InputNumber,
  List,
  Modal,
  Select,
  Space,
  Switch,
  Tag,
  Typography
} from "antd"
import { Copy, Edit3, Plus, RefreshCw, Trash2 } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  LlamacppAsset,
  LlamacppAssetsResponse,
  LlamacppPortPolicy,
  LlamacppProfile,
  LlamacppProfileCreateRequest,
  LlamacppProfileMode,
  LlamacppProfileUpdateRequest
} from "@/types/llamacpp-admin"

const { Text } = Typography
const { TextArea } = Input

const PROFILE_MODES: LlamacppProfileMode[] = [
  "chat",
  "vision",
  "embedding",
  "rerank",
  "server_generic"
]

const PORT_POLICIES: LlamacppPortPolicy[] = ["explicit", "autoselect"]

type FormMode = "create" | "edit" | "duplicate"

interface LlamacppProfilesPanelProps {
  profiles: LlamacppProfile[]
  assets: LlamacppAssetsResponse | null
  loading?: boolean
  savingProfileId?: string | null
  error?: string | null
  onRefresh: () => void
  onCreate: (payload: LlamacppProfileCreateRequest) => Promise<boolean> | boolean
  onUpdate: (
    profileId: string,
    payload: LlamacppProfileUpdateRequest
  ) => Promise<boolean> | boolean
  onDelete: (profileId: string) => Promise<boolean> | boolean
}

interface ProfileFormState {
  name: string
  enabled: boolean
  mode: LlamacppProfileMode
  modelId: string
  modelPath: string
  mmprojModelId: string
  host: string
  port: number
  portPolicy: LlamacppPortPolicy
  autostart: boolean
  providerAlias: string
  tagsText: string
  serverArgsText: string
  restartPolicy: Record<string, unknown>
}

interface ActiveForm {
  mode: FormMode
  profile?: LlamacppProfile
}

const safeJsonStringify = (value: unknown) => {
  try {
    return JSON.stringify(value || {}, null, 2)
  } catch {
    return "{}"
  }
}

const firstAssetId = (assets: LlamacppAsset[], kind: LlamacppAsset["kind"]) =>
  assets.find((asset) => asset.kind === kind)?.asset_id || ""

const formFromProfile = (
  profile: LlamacppProfile | undefined,
  assets: LlamacppAsset[],
  mode: FormMode
): ProfileFormState => ({
  name:
    mode === "duplicate" && profile
      ? `${profile.name} copy`
      : profile?.name || "",
  enabled: profile?.enabled ?? true,
  mode: profile?.mode || "chat",
  modelId: profile?.model_id || firstAssetId(assets, "gguf"),
  modelPath: profile?.model_path || "",
  mmprojModelId: profile?.mmproj_model_id || "",
  host: profile?.host || "127.0.0.1",
  port: profile?.port || 8080,
  portPolicy: profile?.port_policy || "explicit",
  autostart: profile?.autostart ?? false,
  providerAlias: profile?.provider_alias || "",
  tagsText: (profile?.tags || []).join(", "),
  serverArgsText: safeJsonStringify(profile?.server_args),
  restartPolicy: profile?.restart_policy || {}
})

const parseTags = (value: string) =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)

const profileEndpoint = (profile: LlamacppProfile) => `${profile.host}:${profile.port}`

const assetOptions = (
  assets: LlamacppAsset[],
  kind: LlamacppAsset["kind"],
  selectedId?: string | null
) => {
  const options = assets
    .filter((asset) => asset.kind === kind)
    .map((asset) => ({
      label: `${asset.display_name} (${asset.asset_id})`,
      value: asset.asset_id
    }))

  if (selectedId && !options.some((option) => option.value === selectedId)) {
    options.push({ label: selectedId, value: selectedId })
  }

  return options
}

export const LlamacppProfilesPanel: React.FC<LlamacppProfilesPanelProps> = ({
  profiles,
  assets,
  loading = false,
  savingProfileId = null,
  error,
  onRefresh,
  onCreate,
  onUpdate,
  onDelete
}) => {
  const assetList = assets?.assets || []
  const [activeForm, setActiveForm] = React.useState<ActiveForm | null>(null)
  const [form, setForm] = React.useState<ProfileFormState>(() =>
    formFromProfile(undefined, assetList, "create")
  )
  const [formError, setFormError] = React.useState<string | null>(null)

  const openForm = (mode: FormMode, profile?: LlamacppProfile) => {
    setActiveForm({ mode, profile })
    setForm(formFromProfile(profile, assetList, mode))
    setFormError(null)
  }

  const closeForm = () => {
    setActiveForm(null)
    setFormError(null)
  }

  const updateForm = <K extends keyof ProfileFormState>(
    key: K,
    value: ProfileFormState[K]
  ) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const buildPayload = () => {
    const name = form.name.trim()
    if (!name) {
      setFormError("Profile name is required.")
      return null
    }

    let serverArgs: Record<string, unknown>
    try {
      const parsed = JSON.parse(form.serverArgsText || "{}")
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("Server args must be an object.")
      }
      serverArgs = parsed as Record<string, unknown>
    } catch {
      setFormError("Invalid server args JSON.")
      return null
    }

    return {
      name,
      enabled: form.enabled,
      mode: form.mode,
      model_id: form.modelId.trim() || null,
      model_path: form.modelPath.trim() || null,
      mmproj_model_id: form.mmprojModelId.trim() || null,
      host: form.host.trim() || "127.0.0.1",
      port: form.port,
      port_policy: form.portPolicy,
      server_args: serverArgs,
      autostart: form.autostart,
      restart_policy: form.restartPolicy,
      provider_alias: form.providerAlias.trim() || null,
      tags: parseTags(form.tagsText)
    }
  }

  const handleSave = async () => {
    if (!activeForm) return
    const payload = buildPayload()
    if (!payload) return

    const saved =
      activeForm.mode === "edit" && activeForm.profile
        ? await onUpdate(activeForm.profile.profile_id, payload)
        : await onCreate(payload)

    if (saved) {
      closeForm()
    }
  }

  const handleDelete = async (profile: LlamacppProfile) => {
    const confirmed = window.confirm(`Delete llama.cpp profile "${profile.name}"?`)
    if (!confirmed) return
    await onDelete(profile.profile_id)
  }

  return (
    <Card
      title="Profiles"
      loading={loading}
      extra={
        <Space>
          <Button size="small" icon={<RefreshCw size={14} />} onClick={onRefresh}>
            Refresh
          </Button>
          <Button
            size="small"
            type="primary"
            icon={<Plus size={14} />}
            onClick={() => openForm("create")}
          >
            New profile
          </Button>
        </Space>
      }
      aria-label="Profiles"
    >
      <Space orientation="vertical" size="middle" className="w-full">
        {error && <DesignSystemAlert variant="error" title={error} />}

        {profiles.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="No saved llama.cpp profiles are available."
          />
        ) : (
          <List
            size="small"
            bordered
            rowKey="profile_id"
            dataSource={profiles}
            renderItem={(profile) => (
              <List.Item
                actions={[
                  <Button
                    key="edit"
                    size="small"
                    icon={<Edit3 size={14} />}
                    onClick={() => openForm("edit", profile)}
                    loading={savingProfileId === profile.profile_id}
                    aria-label={`Edit ${profile.name}`}
                  >
                    Edit
                  </Button>,
                  <Button
                    key="duplicate"
                    size="small"
                    icon={<Copy size={14} />}
                    onClick={() => openForm("duplicate", profile)}
                    aria-label={`Duplicate ${profile.name}`}
                  >
                    Duplicate
                  </Button>,
                  <Button
                    key="delete"
                    size="small"
                    danger
                    icon={<Trash2 size={14} />}
                    onClick={() => {
                      void handleDelete(profile)
                    }}
                    loading={savingProfileId === profile.profile_id}
                    aria-label={`Delete ${profile.name}`}
                  >
                    Delete
                  </Button>
                ]}
              >
                <Space orientation="vertical" size={4} className="w-full">
                  <Space wrap size="small">
                    <Text strong>{profile.name}</Text>
                    <Tag>{profile.mode}</Tag>
                    <Tag color={profile.enabled ? "green" : "default"}>
                      {profile.enabled ? "enabled" : "disabled"}
                    </Tag>
                    {profile.autostart && <Tag color="blue">autostart</Tag>}
                    <Tag>{profile.port_policy}</Tag>
                  </Space>
                  <Space wrap size="small">
                    <Text code>{profile.profile_id}</Text>
                    <Text type="secondary">{profileEndpoint(profile)}</Text>
                    {profile.model_id && <Text type="secondary">{profile.model_id}</Text>}
                    {profile.mmproj_model_id && (
                      <Text type="secondary">{profile.mmproj_model_id}</Text>
                    )}
                  </Space>
                  {profile.tags.length > 0 && (
                    <Space wrap size="small">
                      {profile.tags.map((tag) => (
                        <Tag key={tag}>{tag}</Tag>
                      ))}
                    </Space>
                  )}
                </Space>
              </List.Item>
            )}
          />
        )}
      </Space>

      <Modal
        open={Boolean(activeForm)}
        title={
          activeForm?.mode === "edit"
            ? "Edit profile"
            : activeForm?.mode === "duplicate"
              ? "Duplicate profile"
              : "New profile"
        }
        okText="Save profile"
        onOk={() => {
          void handleSave()
        }}
        onCancel={closeForm}
        confirmLoading={
          savingProfileId === "__create__" ||
          Boolean(activeForm?.profile && savingProfileId === activeForm.profile.profile_id)
        }
        destroyOnHidden
      >
        <Space orientation="vertical" size="middle" className="w-full">
          {formError && <DesignSystemAlert variant="error" title={formError} />}

          <div>
            <Text>Name</Text>
            <Input
              aria-label="Profile name"
              value={form.name}
              onChange={(event) => updateForm("name", event.target.value)}
              placeholder="Vision runtime"
            />
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Text>Mode</Text>
              <Select
                aria-label="Profile mode"
                value={form.mode}
                onChange={(value) => updateForm("mode", value)}
                options={PROFILE_MODES.map((value) => ({ label: value, value }))}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div>
              <Text>Port policy</Text>
              <Select
                aria-label="Profile port policy"
                value={form.portPolicy}
                onChange={(value) => updateForm("portPolicy", value)}
                options={PORT_POLICIES.map((value) => ({ label: value, value }))}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div>
              <Text>Host</Text>
              <Input
                aria-label="Profile host"
                value={form.host}
                onChange={(event) => updateForm("host", event.target.value)}
              />
            </div>

            <div>
              <Text>Port</Text>
              <InputNumber
                aria-label="Profile port"
                value={form.port}
                min={1}
                max={65535}
                onChange={(value) => updateForm("port", value ?? 8080)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          </div>

          <div>
            <Text>Model asset</Text>
            <Select
              aria-label="Profile model"
              value={form.modelId || undefined}
              onChange={(value) => updateForm("modelId", value || "")}
              options={assetOptions(assetList, "gguf", form.modelId)}
              placeholder="Select a GGUF asset"
              allowClear
              showSearch
              optionFilterProp="label"
              style={{ width: "100%", marginTop: 4 }}
            />
          </div>

          <div>
            <Text>mmproj asset</Text>
            <Select
              aria-label="Profile mmproj"
              value={form.mmprojModelId || undefined}
              onChange={(value) => updateForm("mmprojModelId", value || "")}
              options={assetOptions(assetList, "mmproj", form.mmprojModelId)}
              placeholder="Optional projector asset"
              allowClear
              showSearch
              optionFilterProp="label"
              style={{ width: "100%", marginTop: 4 }}
            />
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="flex items-center justify-between">
              <Text>Enabled</Text>
              <Switch
                aria-label="Profile enabled"
                checked={form.enabled}
                onChange={(checked) => updateForm("enabled", checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Text>Autostart</Text>
              <Switch
                aria-label="Profile autostart"
                checked={form.autostart}
                onChange={(checked) => updateForm("autostart", checked)}
              />
            </div>
          </div>

          <div>
            <Text>Provider alias</Text>
            <Input
              aria-label="Profile provider alias"
              value={form.providerAlias}
              onChange={(event) => updateForm("providerAlias", event.target.value)}
              placeholder="optional local provider alias"
            />
          </div>

          <div>
            <Text>Tags</Text>
            <Input
              aria-label="Profile tags"
              value={form.tagsText}
              onChange={(event) => updateForm("tagsText", event.target.value)}
              placeholder="local, vision"
            />
          </div>

          <div>
            <Text>Server args JSON</Text>
            <TextArea
              aria-label="Profile server args JSON"
              value={form.serverArgsText}
              onChange={(event) => updateForm("serverArgsText", event.target.value)}
              rows={5}
            />
          </div>
        </Space>
      </Modal>
    </Card>
  )
}

export default LlamacppProfilesPanel
