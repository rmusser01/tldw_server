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
  scheduleKind?: ScheduledTaskAutomationEditorScheduleKind
  schedule?: Record<string, unknown>
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
const DEFAULT_AGENT_REF = ""

export type ScheduledTaskAutomationEditorScheduleKind =
  | "one_time"
  | "interval"
  | "daily"
  | "weekly"
  | "cron"

const DEFAULT_SCHEDULE_KIND: ScheduledTaskAutomationEditorScheduleKind = "daily"
const SUPPORTED_SCHEDULE_KINDS = new Set<string>([
  "one_time",
  "interval",
  "daily",
  "weekly",
  "cron"
])

const toStringList = (value: string): string[] =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const isSupportedScheduleKind = (
  value: unknown
): value is ScheduledTaskAutomationEditorScheduleKind =>
  typeof value === "string" && SUPPORTED_SCHEDULE_KINDS.has(value)

const normalizeScheduleKind = (
  value: unknown
): ScheduledTaskAutomationEditorScheduleKind =>
  isSupportedScheduleKind(value) ? value : DEFAULT_SCHEDULE_KIND

const parseJsonObject = (
  value: string,
  fieldLabel: string
): { value: Record<string, unknown> } | { error: string } => {
  const trimmed = value.trim()
  if (!trimmed) return { value: {} }

  try {
    const parsed = JSON.parse(trimmed)
    return isRecord(parsed)
      ? { value: parsed }
      : { error: `${fieldLabel} must be a valid JSON object` }
  } catch {
    return { error: `${fieldLabel} must be a valid JSON object` }
  }
}

const readErrorDetail = (error: unknown): unknown => {
  if (!isRecord(error)) return null

  const details = error["details"]
  if (isRecord(details) && isRecord(details["detail"])) {
    return details["detail"]
  }

  const detail = error["detail"]
  return isRecord(detail) ? detail : null
}

const readErrorMessage = (error: unknown, fallback: string): string => {
  if (error && typeof error === "object") {
    const detail = readErrorDetail(error)
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
    typeof entry["message"] === "string"
      ? entry["message"]
      : typeof entry["detail"] === "string"
        ? entry["detail"]
        : JSON.stringify(entry)

  return message
}

const isPreviewUsable = (
  preview: ScheduledTaskPreviewResponse | null
): preview is ScheduledTaskPreviewResponse => preview?.status === "valid"

const getFamilyLabel = (family: ScheduledTaskAutomationFamily): string =>
  getAutomationDefinitionFamilyLabel({ source_ref: { family } })

const buildSchedule = (
  values: Required<ScheduledTaskAutomationDefinitionEditorValues>
): Record<string, unknown> => {
  const timezone = values.timezone.trim() || "UTC"
  const scheduleKind = normalizeScheduleKind(values.scheduleKind)
  const existingSchedule = isRecord(values.schedule) ? values.schedule : {}
  const baseSchedule =
    existingSchedule["kind"] === scheduleKind ? { ...existingSchedule } : {}

  if (scheduleKind === "cron") {
    return {
      ...baseSchedule,
      kind: "cron",
      cron: values.cron.trim(),
      timezone
    }
  }

  return {
    ...baseSchedule,
    kind: scheduleKind,
    timezone
  }
}

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
  const schedule = buildSchedule(values)
  const visibility_policy = { visibility: values.visibility }
  const notification_policy = { channels: [] }

  if (family === "recurring_question") {
    const parsedScope = parseJsonObject(values.scopeJson, "Scope JSON")
    if ("error" in parsedScope) {
      throw new Error(parsedScope.error)
    }

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
        scope: parsedScope.value
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
      agent_ref: values.agentRef.trim(),
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
    scheduleKind: normalizeScheduleKind(initialValues?.scheduleKind),
    schedule: initialValues?.schedule ?? {},
    cron: initialValues?.cron ?? "",
    timezone: initialValues?.timezone ?? "UTC",
    visibility: initialValues?.visibility ?? "private",
    question: initialValues?.question ?? "",
    successCriteria: initialValues?.successCriteria ?? "",
    scopeJson: initialValues?.scopeJson ?? DEFAULT_SCOPE_JSON,
    agentRef: initialValues?.agentRef ?? DEFAULT_AGENT_REF,
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
  const valuesSignatureRef = React.useRef(JSON.stringify(values))
  const latestPreviewRequestRef = React.useRef(0)

  const updateValue = <K extends keyof ScheduledTaskAutomationDefinitionEditorValues>(
    key: K,
    value: Required<ScheduledTaskAutomationDefinitionEditorValues>[K]
  ) => {
    setValues((current) => {
      const nextValues = { ...current, [key]: value }
      valuesSignatureRef.current = JSON.stringify(nextValues)
      return nextValues
    })
    setPreview(null)
    setSaved(false)
  }

  const handlePreview = async () => {
    const requestId = latestPreviewRequestRef.current + 1
    latestPreviewRequestRef.current = requestId
    const requestSignature = valuesSignatureRef.current
    setPreviewing(true)
    setErrorMessage(null)
    setSaved(false)
    try {
      const payload = buildAutomationDefinitionPreviewPayload({
        family,
        mode,
        definitionId,
        definitionVersion,
        values
      })
      const nextPreview = await onPreview(payload)
      if (
        latestPreviewRequestRef.current === requestId &&
        valuesSignatureRef.current === requestSignature
      ) {
        setPreview(nextPreview)
      }
    } catch (error) {
      if (
        latestPreviewRequestRef.current === requestId &&
        valuesSignatureRef.current === requestSignature
      ) {
        setPreview(null)
        setErrorMessage(readErrorMessage(error, "Unable to preview definition"))
      }
    } finally {
      if (latestPreviewRequestRef.current === requestId) {
        setPreviewing(false)
      }
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
      let result: ScheduledTaskDefinitionResponse
      if (mode === "create") {
        if (!onCreate) {
          throw new Error("Create handler is not configured")
        }
        result = await onCreate({
          preview_id: preview.id,
          initial_lifecycle: values.initialLifecycle
        })
      } else {
        if (!onUpdate) {
          throw new Error("Update handler is not configured")
        }
        result = await onUpdate({ preview_id: preview.id })
      }
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
  const redactionPolicy = isRecord(preview?.redaction_policy) ? preview.redaction_policy : {}
  const redactedFieldValues = redactionPolicy["redacted_fields"]
  const redactedFields = Array.isArray(redactedFieldValues)
    ? redactedFieldValues.filter(
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
              { value: "daily", label: "Daily" },
              { value: "weekly", label: "Weekly" },
              { value: "interval", label: "Interval" },
              { value: "one_time", label: "One time" },
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
