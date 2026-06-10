import React from "react"
import { Button, Card, Input, Select, Space, Typography } from "antd"

import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  ScheduledTaskAutomationFamily,
  ScheduledTaskDefinitionCreateLifecycle,
  ScheduledTaskDefinitionCreateRequest,
  ScheduledTaskDefinitionResponse,
  ScheduledTaskDefinitionUpdateRequest,
  ScheduledTaskPreviewCreateRequest,
  ScheduledTaskPreviewResponse
} from "@/services/scheduled-tasks-control-plane"
import { getAutomationDefinitionFamilyLabel } from "./scheduled-task-automation-status"

export interface ScheduledTaskAutomationDefinitionEditorValues {
  name?: string
  description?: string
  scheduleKind?: "manual" | "cron"
  cron?: string
  timezone?: string
  visibility?: "private" | "shared"
  question?: string
  successCriteria?: string
  scopeJson?: string
  agentRef?: string
  message?: string
  allowedToolClasses?: string
  deniedToolClasses?: string
  approvalMode?: "none" | "manual"
  initialLifecycle?: ScheduledTaskDefinitionCreateLifecycle
}

export interface ScheduledTaskAutomationDefinitionEditorProps {
  family: ScheduledTaskAutomationFamily
  mode: "create" | "update"
  definitionId?: string | null
  definitionVersion?: number | null
  initialValues?: ScheduledTaskAutomationDefinitionEditorValues
  onPreview: (
    payload: ScheduledTaskPreviewCreateRequest
  ) => Promise<ScheduledTaskPreviewResponse> | ScheduledTaskPreviewResponse
  onCreate?: (
    payload: ScheduledTaskDefinitionCreateRequest
  ) => Promise<ScheduledTaskDefinitionResponse | unknown> | ScheduledTaskDefinitionResponse | unknown
  onUpdate?: (
    payload: ScheduledTaskDefinitionUpdateRequest
  ) => Promise<ScheduledTaskDefinitionResponse | unknown> | ScheduledTaskDefinitionResponse | unknown
  onCancel: () => void
  onSaved?: (definition: ScheduledTaskDefinitionResponse | unknown) => void
}

const DEFAULT_SCOPE_JSON = "{}"
const DEFAULT_AGENT_REF_JSON = "{}"

const toStringList = (value: string): string[] =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)

const parseJsonObject = (value: string, fallback: Record<string, unknown>): Record<string, unknown> => {
  const trimmed = value.trim()
  if (!trimmed) return fallback

  try {
    const parsed = JSON.parse(trimmed)
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : fallback
  } catch {
    return fallback
  }
}

const readErrorMessage = (error: unknown, fallback: string): string => {
  if (error && typeof error === "object") {
    const detail = "detail" in error ? (error as { detail?: unknown }).detail : null
    if (detail && typeof detail === "object") {
      const code = "code" in detail ? String((detail as { code?: unknown }).code || "") : ""
      const message =
        "message" in detail ? String((detail as { message?: unknown }).message || "") : ""
      if (code && message) return `${code}: ${message}`
      if (message) return message
      if (code) return code
    }

    if ("message" in error && typeof (error as { message?: unknown }).message === "string") {
      return (error as { message: string }).message
    }
  }

  return fallback
}

const getValidationMessage = (entry: Record<string, unknown>): string => {
  const message =
    typeof entry.message === "string"
      ? entry.message
      : typeof entry.detail === "string"
        ? entry.detail
        : JSON.stringify(entry)

  return message
}

const isPreviewUsable = (
  preview: ScheduledTaskPreviewResponse | null
): preview is ScheduledTaskPreviewResponse => preview?.status === "valid"

const getFamilyLabel = (family: ScheduledTaskAutomationFamily): string =>
  getAutomationDefinitionFamilyLabel({ source_ref: { family } })

export const buildAutomationDefinitionPreviewPayload = ({
  family,
  mode,
  definitionId,
  definitionVersion,
  values
}: {
  family: ScheduledTaskAutomationFamily
  mode: "create" | "update"
  definitionId?: string | null
  definitionVersion?: number | null
  values: Required<ScheduledTaskAutomationDefinitionEditorValues>
}): ScheduledTaskPreviewCreateRequest => {
  const schedule =
    values.scheduleKind === "cron"
      ? {
          kind: "cron",
          cron: values.cron.trim(),
          timezone: values.timezone.trim()
        }
      : {
          kind: "manual",
          timezone: values.timezone.trim()
        }
  const visibility_policy = { visibility: values.visibility }
  const notification_policy = { channels: [] }

  if (family === "recurring_question") {
    return {
      mode,
      family,
      definition_id: mode === "update" ? definitionId ?? null : undefined,
      definition_version: mode === "update" ? definitionVersion ?? null : undefined,
      name: values.name.trim() || values.question.trim() || "Recurring question",
      description: values.description.trim() || null,
      input: {
        question: values.question.trim(),
        success_criteria: values.successCriteria.trim() || null,
        scope: parseJsonObject(values.scopeJson, {})
      },
      config: {},
      schedule,
      visibility_policy,
      notification_policy,
      approval_policy: { mode: "none" }
    }
  }

  return {
    mode,
    family,
    definition_id: mode === "update" ? definitionId ?? null : undefined,
    definition_version: mode === "update" ? definitionVersion ?? null : undefined,
    name: values.name.trim() || "Agent task",
    description: values.description.trim() || null,
    input: {
      agent_ref: parseJsonObject(values.agentRef, { raw: values.agentRef.trim() }),
      message: values.message.trim()
    },
    config: {
      allowed_tool_classes: toStringList(values.allowedToolClasses),
      denied_tool_classes: toStringList(values.deniedToolClasses)
    },
    schedule,
    visibility_policy,
    notification_policy,
    approval_policy: { mode: values.approvalMode }
  }
}

export const ScheduledTaskAutomationDefinitionEditor: React.FC<
  ScheduledTaskAutomationDefinitionEditorProps
> = ({
  family,
  mode,
  definitionId,
  definitionVersion,
  initialValues,
  onPreview,
  onCreate,
  onUpdate,
  onCancel,
  onSaved
}) => {
  const [values, setValues] = React.useState<Required<ScheduledTaskAutomationDefinitionEditorValues>>({
    name: initialValues?.name ?? "",
    description: initialValues?.description ?? "",
    scheduleKind: initialValues?.scheduleKind ?? "manual",
    cron: initialValues?.cron ?? "",
    timezone: initialValues?.timezone ?? "UTC",
    visibility: initialValues?.visibility ?? "private",
    question: initialValues?.question ?? "",
    successCriteria: initialValues?.successCriteria ?? "",
    scopeJson: initialValues?.scopeJson ?? DEFAULT_SCOPE_JSON,
    agentRef: initialValues?.agentRef ?? DEFAULT_AGENT_REF_JSON,
    message: initialValues?.message ?? "",
    allowedToolClasses: initialValues?.allowedToolClasses ?? "",
    deniedToolClasses: initialValues?.deniedToolClasses ?? "",
    approvalMode: initialValues?.approvalMode ?? "none",
    initialLifecycle: initialValues?.initialLifecycle ?? "configured"
  })
  const [preview, setPreview] = React.useState<ScheduledTaskPreviewResponse | null>(null)
  const [previewing, setPreviewing] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null)
  const [saved, setSaved] = React.useState(false)

  const updateValue = <K extends keyof ScheduledTaskAutomationDefinitionEditorValues>(
    key: K,
    value: Required<ScheduledTaskAutomationDefinitionEditorValues>[K]
  ) => {
    setValues((current) => ({ ...current, [key]: value }))
    setPreview(null)
    setSaved(false)
  }

  const handlePreview = async () => {
    setPreviewing(true)
    setErrorMessage(null)
    setSaved(false)
    try {
      const nextPreview = await onPreview(
        buildAutomationDefinitionPreviewPayload({
          family,
          mode,
          definitionId,
          definitionVersion,
          values
        })
      )
      setPreview(nextPreview)
    } catch (error) {
      setPreview(null)
      setErrorMessage(readErrorMessage(error, "Unable to preview definition"))
    } finally {
      setPreviewing(false)
    }
  }

  const handleSave = async () => {
    if (!isPreviewUsable(preview)) {
      setErrorMessage("Preview again before saving")
      return
    }

    setSaving(true)
    setErrorMessage(null)
    try {
      const result =
        mode === "create"
          ? await onCreate?.({
              preview_id: preview.id,
              initial_lifecycle: values.initialLifecycle
            })
          : await onUpdate?.({ preview_id: preview.id })
      setSaved(true)
      onSaved?.(result)
    } catch (error) {
      setErrorMessage(readErrorMessage(error, "Unable to save definition"))
    } finally {
      setSaving(false)
    }
  }

  const validationErrors = preview?.validation_errors ?? []
  const previewNeedsRefresh =
    preview && ["expired", "invalid", "consumed"].includes(preview.status)
  const redactedFields = Array.isArray(preview?.redaction_policy?.redacted_fields)
    ? (preview?.redaction_policy?.redacted_fields as unknown[]).filter(
        (field): field is string => typeof field === "string" && Boolean(field.trim())
      )
    : []
  const saveDisabled = !isPreviewUsable(preview) || saving || previewing

  return (
    <Card title={mode === "create" ? `Create ${getFamilyLabel(family)}` : `Update ${getFamilyLabel(family)}`} size="small">
      <Space orientation="vertical" size={12} style={{ width: "100%" }}>
        <Space wrap style={{ width: "100%" }}>
          <Input
            aria-label="Name"
            placeholder="Name"
            value={values.name}
            onChange={(event) => updateValue("name", event.target.value)}
            style={{ minWidth: 240 }}
          />
          <Select
            aria-label="Schedule kind"
            value={values.scheduleKind}
            onChange={(value) => updateValue("scheduleKind", value)}
            options={[
              { value: "manual", label: "Manual" },
              { value: "cron", label: "Cron" }
            ]}
            style={{ width: 140 }}
          />
          <Input
            aria-label="Timezone"
            value={values.timezone}
            onChange={(event) => updateValue("timezone", event.target.value)}
            style={{ width: 180 }}
          />
          <Select
            aria-label="Visibility"
            value={values.visibility}
            onChange={(value) => updateValue("visibility", value)}
            options={[
              { value: "private", label: "Private" },
              { value: "shared", label: "Shared" }
            ]}
            style={{ width: 140 }}
          />
        </Space>

        {values.scheduleKind === "cron" ? (
          <Input
            aria-label="Cron"
            placeholder="0 9 * * *"
            value={values.cron}
            onChange={(event) => updateValue("cron", event.target.value)}
          />
        ) : null}

        <Input.TextArea
          aria-label="Description"
          placeholder="Description"
          rows={2}
          value={values.description}
          onChange={(event) => updateValue("description", event.target.value)}
        />

        {family === "recurring_question" ? (
          <>
            <Input.TextArea
              aria-label="Question"
              placeholder="Question"
              rows={3}
              value={values.question}
              onChange={(event) => updateValue("question", event.target.value)}
            />
            <Input.TextArea
              aria-label="Success criteria"
              placeholder="Success criteria"
              rows={2}
              value={values.successCriteria}
              onChange={(event) => updateValue("successCriteria", event.target.value)}
            />
            <Input.TextArea
              aria-label="Scope JSON"
              value={values.scopeJson}
              rows={4}
              onChange={(event) => updateValue("scopeJson", event.target.value)}
            />
          </>
        ) : (
          <>
            <Input.TextArea
              aria-label="Agent ref"
              value={values.agentRef}
              rows={3}
              onChange={(event) => updateValue("agentRef", event.target.value)}
            />
            <Input.TextArea
              aria-label="Message"
              placeholder="Message"
              rows={3}
              value={values.message}
              onChange={(event) => updateValue("message", event.target.value)}
            />
            <Input
              aria-label="Allowed tool classes"
              placeholder="Allowed tool classes"
              value={values.allowedToolClasses}
              onChange={(event) => updateValue("allowedToolClasses", event.target.value)}
            />
            <Input
              aria-label="Denied tool classes"
              placeholder="Denied tool classes"
              value={values.deniedToolClasses}
              onChange={(event) => updateValue("deniedToolClasses", event.target.value)}
            />
            <Select
              aria-label="Approval mode"
              value={values.approvalMode}
              onChange={(value) => updateValue("approvalMode", value)}
              options={[
                { value: "none", label: "No approval" },
                { value: "manual", label: "Manual approval" }
              ]}
              style={{ width: 180 }}
            />
          </>
        )}

        {preview?.status === "valid" ? (
          <DesignSystemAlert variant="success" title="Preview ready" />
        ) : null}
        {previewNeedsRefresh ? (
          <DesignSystemAlert variant="warning" title="Preview again before saving" />
        ) : null}
        {validationErrors.length > 0 ? (
          <Space orientation="vertical" size={4}>
            {validationErrors.map((entry, index) => (
              <Typography.Text key={index} type="danger">
                {getValidationMessage(entry)}
              </Typography.Text>
            ))}
          </Space>
        ) : null}
        {redactedFields.length > 0 ? (
          <Typography.Text type="secondary">
            {`Redacted: ${redactedFields.join(", ")}`}
          </Typography.Text>
        ) : null}
        {errorMessage ? (
          <DesignSystemAlert variant="error" title={errorMessage} />
        ) : null}
        {(preview?.status === "valid" || saved) ? (
          <Typography.Text type="secondary">Execution is not available yet</Typography.Text>
        ) : null}

        <Space wrap>
          <Button onClick={onCancel}>Cancel</Button>
          <Button onClick={handlePreview} loading={previewing}>
            Preview
          </Button>
          <Button
            type="primary"
            disabled={saveDisabled}
            loading={saving}
            onClick={handleSave}
          >
            Save definition
          </Button>
        </Space>
      </Space>
    </Card>
  )
}

export default ScheduledTaskAutomationDefinitionEditor
