import React from "react"
import { Alert, Button, Empty, Form, Input, Modal, Radio, Select, Space, Spin, Switch, Typography } from "antd"
import { ArrowUp, Check, FolderOpen } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import {
  useCreateIngestionSourceMutation,
  useIngestionSourceDirectoryBrowseQuery,
  useUpdateIngestionSourceMutation
} from "@/hooks/use-ingestion-sources"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import type {
  CreateIngestionSourceRequest,
  IngestionSourceDirectoryEntry,
  UpdateIngestionSourceRequest,
  IngestionSourceSummary,
  IngestionSourceType
} from "@/types/ingestion-sources"

type GitRepositoryMode = "local_repo" | "remote_github_repo"

type SourceFormValues = {
  source_type: IngestionSourceType
  sink_type: "media" | "notes"
  policy: "canonical" | "import_only"
  enabled: boolean
  schedule_enabled: boolean
  path?: string
  git_repository_mode?: GitRepositoryMode
  repo_path?: string
  repo_url?: string
  ref?: string
  root_subpath?: string
  account_id?: string
  respect_gitignore?: boolean
}

type SourceFormProps = {
  mode: "create" | "edit"
  source?: IngestionSourceSummary | null
  preset?: "notes-folder-sync"
}

const hasLockedSourceIdentity = (source?: IngestionSourceSummary | null): boolean => {
  if (!source) {
    return false
  }
  return Boolean(source.last_successful_snapshot_id)
}

const getSourceTypeLabel = (sourceType: IngestionSourceType): string => {
  if (sourceType === "archive_snapshot") {
    return "Archive snapshot"
  }
  if (sourceType === "git_repository") {
    return "Git repository"
  }
  return "Local directory"
}

const getSinkTypeLabel = (sinkType: IngestionSourceSummary["sink_type"] | "media" | "notes"): string =>
  sinkType === "media" ? "Media" : "Notes"

const getInitialGitRepositoryMode = (source?: IngestionSourceSummary | null): GitRepositoryMode =>
  source?.source_type === "git_repository" && source.config?.mode === "remote_github_repo"
    ? "remote_github_repo"
    : "local_repo"

export const SourceForm: React.FC<SourceFormProps> = ({ mode, source, preset }) => {
  const { t } = useTranslation(["sources", "common"])
  const navigate = useNavigate()
  const [form] = Form.useForm<SourceFormValues>()
  const presetDefaults =
    mode === "create" && preset === "notes-folder-sync"
      ? {
          source_type: "local_directory" as const,
          sink_type: "notes" as const,
          policy: "canonical" as const,
          enabled: true,
          schedule_enabled: false
        }
      : null
  const initialSourceType = source?.source_type ?? presetDefaults?.source_type ?? "local_directory"
  const initialSinkType = source?.sink_type ?? presetDefaults?.sink_type ?? "notes"
  const initialPolicy = source?.policy ?? presetDefaults?.policy ?? "canonical"
  const initialEnabled = source?.enabled ?? presetDefaults?.enabled ?? true
  const initialScheduleEnabled = source?.schedule_enabled ?? presetDefaults?.schedule_enabled ?? false
  const initialGitRepositoryMode = getInitialGitRepositoryMode(source)
  const identityLocked = mode === "edit" && hasLockedSourceIdentity(source)
  const [sourceType, setSourceType] = React.useState<IngestionSourceType>(initialSourceType)
  const [gitRepositoryMode, setGitRepositoryMode] =
    React.useState<GitRepositoryMode>(initialGitRepositoryMode)
  const [submitError, setSubmitError] = React.useState<string | null>(null)
  const [pathPickerOpen, setPathPickerOpen] = React.useState(false)
  const [browsePath, setBrowsePath] = React.useState<string | null>(null)
  const { capabilities, loading: capabilitiesLoading } = useServerCapabilities()

  const createMutation = useCreateIngestionSourceMutation()
  const updateMutation = useUpdateIngestionSourceMutation(source?.id ?? "")
  const activeMutation = mode === "edit" ? updateMutation : createMutation

  React.useEffect(() => {
    setSourceType(initialSourceType)
    setGitRepositoryMode(initialGitRepositoryMode)
    form.setFieldsValue({
      source_type: initialSourceType,
      sink_type: initialSinkType,
      policy: initialPolicy,
      enabled: initialEnabled,
      schedule_enabled: initialScheduleEnabled,
      path: typeof source?.config?.path === "string" ? source.config.path : "",
      git_repository_mode: initialGitRepositoryMode,
      repo_path:
        source?.source_type === "git_repository" && typeof source?.config?.path === "string"
          ? source.config.path
          : "",
      repo_url:
        source?.source_type === "git_repository" && typeof source?.config?.repo_url === "string"
          ? source.config.repo_url
          : "",
      ref:
        source?.source_type === "git_repository" && typeof source?.config?.ref === "string"
          ? source.config.ref
          : "",
      root_subpath:
        source?.source_type === "git_repository" && typeof source?.config?.root_subpath === "string"
          ? source.config.root_subpath
          : "",
      account_id:
        source?.source_type === "git_repository" && source?.config?.account_id != null
          ? String(source.config.account_id)
          : "",
      respect_gitignore:
        source?.source_type === "git_repository"
          ? source.config?.respect_gitignore !== false
          : true
    })
  }, [
    form,
    initialEnabled,
    initialGitRepositoryMode,
    initialPolicy,
    initialScheduleEnabled,
    initialSinkType,
    initialSourceType,
    source?.config?.account_id,
    source?.config?.path,
    source?.config?.ref,
    source?.config?.repo_url,
    source?.config?.respect_gitignore,
    source?.config?.root_subpath,
    source?.source_type
  ])

  const effectiveSourceType = identityLocked && source ? source.source_type : sourceType
  const localDirectoryCreateAllowed =
    capabilities?.canCreateLocalDirectoryIngestionSource === true
  const localDirectoryCreateBlocked =
    mode === "create" &&
    effectiveSourceType === "local_directory" &&
    (capabilitiesLoading || !localDirectoryCreateAllowed)
  const localDirectoryCapabilityMessage = capabilitiesLoading
    ? "Checking whether this server allows folder sync."
    : "The administrator must enable server folder sync before you can create a local directory source."
  const directoryBrowseQuery = useIngestionSourceDirectoryBrowseQuery(
    browsePath,
    undefined,
    {
      enabled:
        pathPickerOpen &&
        effectiveSourceType === "local_directory" &&
        !localDirectoryCreateBlocked
    }
  )
  const browseErrorMessage =
    directoryBrowseQuery.error instanceof Error
      ? directoryBrowseQuery.error.message
      : directoryBrowseQuery.data?.error || null
  const directoryEntries = directoryBrowseQuery.data?.entries ?? []

  const openPathPicker = React.useCallback(() => {
    const currentPath = String(form.getFieldValue("path") || "").trim()
    setBrowsePath(currentPath || null)
    setPathPickerOpen(true)
  }, [form])

  const handleUseDirectory = React.useCallback(
    (path: string) => {
      form.setFieldValue("path", path)
      setBrowsePath(path)
      setPathPickerOpen(false)
    },
    [form]
  )

  const renderDirectoryEntry = React.useCallback(
    (entry: IngestionSourceDirectoryEntry) => (
      <li className="flex items-center gap-3 px-3 py-2" key={entry.path}>
        <div className="min-w-0 flex-1">
          <Button
            aria-label={`Open ${entry.name}`}
            className="min-w-0 px-0"
            icon={<FolderOpen className="h-3.5 w-3.5 shrink-0" />}
            type="link"
            onClick={() => setBrowsePath(entry.path)}>
            <span className="truncate">{entry.name}</span>
          </Button>
          <div>
            <Typography.Text className="break-all text-xs" type="secondary">
              {entry.path}
            </Typography.Text>
          </div>
        </div>
        <Button
          aria-label={`Use ${entry.name} folder`}
          icon={<Check className="h-3.5 w-3.5" />}
          size="small"
          type="link"
          onClick={() => handleUseDirectory(entry.path)}>
          Use
        </Button>
      </li>
    ),
    [handleUseDirectory]
  )

  React.useEffect(() => {
    if (identityLocked) {
      return
    }
    if (sourceType === "git_repository" && form.getFieldValue("sink_type") !== "notes") {
      form.setFieldValue("sink_type", "notes")
    }
  }, [form, identityLocked, sourceType])

  const destinationOptions = React.useMemo(() => {
    if ((identityLocked && source ? source.source_type : sourceType) === "git_repository") {
      return [{ value: "notes", label: "Notes" }]
    }
    return [
      { value: "notes", label: "Notes" },
      { value: "media", label: "Media" }
    ]
  }, [identityLocked, source, sourceType])

  const handleFinish = async (values: SourceFormValues) => {
    setSubmitError(null)

    const effectiveSourceType = identityLocked && source ? source.source_type : sourceType
    const effectiveGitRepositoryMode =
      effectiveSourceType === "git_repository"
        ? identityLocked && source && source.config?.mode === "remote_github_repo"
          ? "remote_github_repo"
          : values.git_repository_mode ?? gitRepositoryMode
        : "local_repo"

    const sourceConfig = (() => {
      if (identityLocked && mode === "edit" && source) {
        return undefined
      }
      if (effectiveSourceType === "local_directory") {
        return {
          path:
            identityLocked && typeof source?.config?.path === "string"
              ? source.config.path
              : (values.path || "").trim()
        }
      }
      if (effectiveSourceType === "git_repository") {
        const ref = (values.ref || "").trim()
        const rootSubpath = (values.root_subpath || "").trim()
        const accountId = (values.account_id || "").trim()
        if (effectiveGitRepositoryMode === "remote_github_repo") {
          return {
            mode: "remote_github_repo",
            repo_url:
              identityLocked && typeof source?.config?.repo_url === "string"
                ? source.config.repo_url
                : (values.repo_url || "").trim(),
            ...(accountId ? { account_id: accountId } : {}),
            ...(ref ? { ref } : {}),
            ...(rootSubpath ? { root_subpath: rootSubpath } : {})
          }
        }
        return {
          mode: "local_repo",
          path:
            identityLocked && typeof source?.config?.path === "string"
              ? source.config.path
              : (values.repo_path || "").trim(),
          ...(ref ? { ref } : {}),
          ...(rootSubpath ? { root_subpath: rootSubpath } : {}),
          respect_gitignore: values.respect_gitignore !== false
        }
      }
      return {}
    })()
    const basePayload = {
      source_type: effectiveSourceType,
      sink_type: identityLocked && source ? source.sink_type : values.sink_type,
      policy: values.policy,
      enabled: values.enabled,
      schedule_enabled: values.schedule_enabled ?? false,
      schedule: {}
    }

    try {
      const result =
        mode === "edit" && source
          ? await updateMutation.mutateAsync({
              ...basePayload,
              ...(typeof sourceConfig === "undefined" ? {} : { config: sourceConfig })
            } satisfies UpdateIngestionSourceRequest)
          : await createMutation.mutateAsync({
              ...basePayload,
              config: sourceConfig ?? {}
            } satisfies CreateIngestionSourceRequest)

      if (mode === "create") {
        navigate(`/sources/${result.id}`)
      }
    } catch (error: any) {
      setSubmitError(error?.message || "Failed to save source")
    }
  }

  return (
    <div className="space-y-4">
      {submitError ? <Alert type="error" title={submitError} /> : null}

      <Modal
        destroyOnHidden
        footer={null}
        open={pathPickerOpen}
        title="Browse server folders"
        width={680}
        onCancel={() => setPathPickerOpen(false)}>
        <div className="space-y-4">
          <Typography.Text type="secondary">
            Only configured ingestion source roots and their folders are shown.
          </Typography.Text>

          {directoryBrowseQuery.data?.current_path ? (
            <div className="space-y-3">
              <Typography.Text className="block break-all" strong>
                {directoryBrowseQuery.data.current_path}
              </Typography.Text>
              <Space wrap>
                <Button
                  disabled={!directoryBrowseQuery.data.parent_path}
                  icon={<ArrowUp className="h-3.5 w-3.5" />}
                  onClick={() => setBrowsePath(directoryBrowseQuery.data?.parent_path ?? null)}>
                  Up
                </Button>
                <Button
                  icon={<Check className="h-3.5 w-3.5" />}
                  type="primary"
                  onClick={() =>
                    handleUseDirectory(String(directoryBrowseQuery.data?.current_path || ""))
                  }>
                  Use this folder
                </Button>
              </Space>
            </div>
          ) : null}

          {browseErrorMessage ? (
            <Alert
              showIcon
              type="warning"
              title={browseErrorMessage}
              action={
                <Button size="small" onClick={() => setBrowsePath(null)}>
                  Show roots
                </Button>
              }
            />
          ) : null}

          {directoryBrowseQuery.isLoading ? (
            <div className="flex min-h-32 items-center justify-center">
              <Spin />
            </div>
          ) : directoryEntries.length > 0 ? (
            <div className="space-y-2">
              {directoryBrowseQuery.isFetching ? (
                <div className="flex justify-end">
                  <Spin size="small" />
                </div>
              ) : null}
              <ul className="m-0 max-h-80 list-none divide-y divide-gray-200 overflow-auto rounded border border-solid border-gray-200 p-0 dark:divide-gray-700 dark:border-gray-700">
                {directoryEntries.map(renderDirectoryEntry)}
              </ul>
            </div>
          ) : (
            <div className="rounded border border-solid border-gray-200 p-4 dark:border-gray-700">
              <Empty
                description={
                  browseErrorMessage
                    ? "No folders available for this path."
                    : "No folders available."
                }
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            </div>
          )}
        </div>
      </Modal>

      <Form<SourceFormValues>
        form={form}
        layout="vertical"
        initialValues={{
          source_type: initialSourceType,
          sink_type: initialSinkType,
          policy: initialPolicy,
          enabled: initialEnabled,
          schedule_enabled: initialScheduleEnabled,
          path: typeof source?.config?.path === "string" ? source.config.path : "",
          git_repository_mode: initialGitRepositoryMode,
          repo_path:
            source?.source_type === "git_repository" && typeof source?.config?.path === "string"
              ? source.config.path
              : "",
          repo_url:
            source?.source_type === "git_repository" && typeof source?.config?.repo_url === "string"
              ? source.config.repo_url
              : "",
          ref:
            source?.source_type === "git_repository" && typeof source?.config?.ref === "string"
              ? source.config.ref
              : "",
          root_subpath:
            source?.source_type === "git_repository" && typeof source?.config?.root_subpath === "string"
              ? source.config.root_subpath
              : "",
          account_id:
            source?.source_type === "git_repository" && source?.config?.account_id != null
              ? String(source.config.account_id)
              : "",
          respect_gitignore:
            source?.source_type === "git_repository"
              ? source.config?.respect_gitignore !== false
              : true
        }}
        onFinish={(values) => {
          void handleFinish(values)
        }}>
        {identityLocked && source ? (
          <Alert
            type="info"
            title="Locked after first successful sync"
            description={
              <div className="space-y-2">
                <div>
                  <Typography.Text strong>
                    {t("sources:form.sourceType", "Source type")}
                  </Typography.Text>
                  <div>{getSourceTypeLabel(source.source_type)}</div>
                </div>
                <div>
                  <Typography.Text strong>Current destination</Typography.Text>
                  <div>{getSinkTypeLabel(source.sink_type)}</div>
                </div>
                {typeof source.config?.path === "string" && source.config.path.trim().length > 0 ? (
                  <div>
                    <Typography.Text strong>
                      {t("sources:form.path", "Server directory path")}
                    </Typography.Text>
                    <div>{source.config.path}</div>
                  </div>
                ) : null}
                {source.source_type === "git_repository" && typeof source.config?.repo_url === "string" ? (
                  <div>
                    <Typography.Text strong>
                      {t("sources:form.repoUrl", "GitHub repository URL")}
                    </Typography.Text>
                    <div>{source.config.repo_url}</div>
                  </div>
                ) : null}
              </div>
            }
          />
        ) : (
          <>
            <Form.Item
              name="source_type"
              label={t("sources:form.sourceType", "Source type")}>
              <Radio.Group
                onChange={(event) => setSourceType(event.target.value as IngestionSourceType)}>
                <Space orientation="vertical">
                  <Radio value="local_directory">
                    {t("sources:form.localDirectory", "Local directory")}
                  </Radio>
                  <Radio value="archive_snapshot">
                    {t("sources:form.archiveSnapshot", "Archive snapshot")}
                  </Radio>
                  <Radio value="git_repository">
                    {t("sources:form.gitRepository", "Git repository")}
                  </Radio>
                </Space>
              </Radio.Group>
            </Form.Item>

            <Form.Item name="sink_type" label="Destination">
              <Select options={destinationOptions} />
            </Form.Item>
          </>
        )}

        <Form.Item name="policy" label="Lifecycle policy">
          <Select
            options={[
              { value: "canonical", label: "Canonical" },
              { value: "import_only", label: "Import only" }
            ]}
          />
        </Form.Item>

        <Space align="start" size="large" wrap>
          <Form.Item name="enabled" label="Enabled" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item
            name="schedule_enabled"
            label="Scheduled rescans"
            valuePropName="checked"
            extra="When enabled, the server may rescan this source on its configured schedule. Cadence is managed by the server.">
            <Switch />
          </Form.Item>
        </Space>

        {effectiveSourceType === "local_directory" && !identityLocked ? (
          <>
            {mode === "create" && !capabilitiesLoading && !localDirectoryCreateAllowed ? (
              <Alert
                type="warning"
                showIcon
                title="Server folder sync is disabled"
                description={localDirectoryCapabilityMessage}
              />
            ) : mode === "create" && capabilitiesLoading ? (
              <Alert type="info" showIcon title={localDirectoryCapabilityMessage} />
            ) : null}
            <Form.Item
              label={t("sources:form.path", "Server directory path")}
              required>
              <Space.Compact block>
                <Form.Item
                  name="path"
                  noStyle
                  rules={[
                    {
                      required: true,
                      message: t("sources:form.path", "Server directory path")
                    }
                  ]}>
                  <Input aria-label={t("sources:form.path", "Server directory path")} />
                </Form.Item>
                <Button
                  aria-label="Browse server folders"
                  disabled={localDirectoryCreateBlocked}
                  htmlType="button"
                  icon={<FolderOpen className="h-3.5 w-3.5" />}
                  onClick={openPathPicker}>
                  Browse
                </Button>
              </Space.Compact>
            </Form.Item>
            <Typography.Text type="secondary">
              {t(
                "sources:form.pathHelp",
                "This is a path on the tldw server host, not a local browser or extension folder."
              )}
            </Typography.Text>
          </>
        ) : effectiveSourceType === "git_repository" && !identityLocked ? (
          <>
            <Form.Item
              name="git_repository_mode"
              label={t("sources:form.gitRepositoryMode", "Repository mode")}>
              <Radio.Group
                onChange={(event) => setGitRepositoryMode(event.target.value as GitRepositoryMode)}>
                <Space orientation="vertical">
                  <Radio value="local_repo">
                    {t("sources:form.localGitRepository", "Local checked-out repository")}
                  </Radio>
                  <Radio value="remote_github_repo">
                    {t("sources:form.remoteGitRepository", "Remote GitHub repository")}
                  </Radio>
                </Space>
              </Radio.Group>
            </Form.Item>

            {gitRepositoryMode === "local_repo" ? (
              <>
                <Form.Item
                  name="repo_path"
                  label={t("sources:form.repoPath", "Repository path")}
                  rules={[
                    {
                      required: true,
                      message: t("sources:form.repoPath", "Repository path")
                    }
                  ]}>
                  <Input />
                </Form.Item>
                <Typography.Text type="secondary">
                  {t(
                    "sources:form.repoPathHelp",
                    "Use a checked-out repository path on the tldw server host."
                  )}
                </Typography.Text>
                <Form.Item
                  name="respect_gitignore"
                  label={t("sources:form.respectGitignore", "Respect .gitignore")}
                  valuePropName="checked">
                  <Switch />
                </Form.Item>
              </>
            ) : (
              <>
                <Form.Item
                  name="repo_url"
                  label={t("sources:form.repoUrl", "GitHub repository URL")}
                  rules={[
                    {
                      required: true,
                      message: t("sources:form.repoUrl", "GitHub repository URL")
                    }
                  ]}>
                  <Input />
                </Form.Item>
                <Form.Item
                  name="account_id"
                  label={t("sources:form.accountId", "Linked account ID")}
                  rules={[
                    {
                      validator: async (_, value) => {
                        const text = String(value || "").trim()
                        if (!text) {
                          return
                        }
                        if (/^[1-9]\d*$/.test(text)) {
                          return
                        }
                        throw new Error("Linked account ID must be a positive integer")
                      }
                    }
                  ]}>
                  <Input />
                </Form.Item>
              </>
            )}

            <Form.Item
              name="ref"
              label={t("sources:form.ref", "Branch, tag, or ref")}>
              <Input />
            </Form.Item>
            <Form.Item
              name="root_subpath"
              label={t("sources:form.rootSubpath", "Root subpath")}>
              <Input />
            </Form.Item>
          </>
        ) : effectiveSourceType === "archive_snapshot" ? (
          <Alert
            type="info"
            title={t("sources:form.archiveHint", "Upload archive after creation")}
          />
        ) : (
          <Alert
            type="info"
            title={t("sources:form.gitRepositoryHint", "Git repository details are configured below.")}
          />
        )}

        <div className="pt-4">
          <Button
            type="primary"
            htmlType="submit"
            loading={Boolean((activeMutation as { isPending?: boolean }).isPending)}
            disabled={localDirectoryCreateBlocked}>
            {mode === "create" ? "Create source" : "Save changes"}
          </Button>
        </div>
      </Form>
    </div>
  )
}
