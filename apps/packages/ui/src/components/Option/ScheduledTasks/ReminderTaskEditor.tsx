import React, { useEffect } from "react"
import { Button, Card, Form, Input, Space, Switch } from "antd"
import type {
  CreateScheduledTaskReminderPayload,
  ScheduledTask,
  UpdateScheduledTaskReminderPayload
} from "@/services/scheduled-tasks-control-plane"
import { ReminderScheduleControls } from "./ReminderScheduleControls"
import {
  datetimeLocalToIsoString,
  getDefaultReminderTimezone,
  isoStringToDatetimeLocal,
  validateCronExpression
} from "./reminder-schedule-utils"

type ReminderTaskEditorValues = {
  title: string
  body?: string | null
  schedule_kind: "one_time" | "recurring"
  run_at?: string | null
  cron?: string | null
  timezone?: string | null
  enabled: boolean
}

type ReminderTaskEditorProps = {
  open: boolean
  task: ScheduledTask | null
  saving?: boolean
  onClose: () => void
  onSubmit: (payload: CreateScheduledTaskReminderPayload | UpdateScheduledTaskReminderPayload) => Promise<void> | void
}

const taskToValues = (task: ScheduledTask | null): ReminderTaskEditorValues => {
  const sourceRef = (task?.source_ref ?? {}) as Record<string, unknown>
  return {
    title: task?.title ?? "",
    body: typeof task?.description === "string" ? task.description : "",
    schedule_kind: sourceRef.schedule_kind === "recurring" ? "recurring" : "one_time",
    run_at: typeof sourceRef.run_at === "string" ? isoStringToDatetimeLocal(sourceRef.run_at) : "",
    cron: typeof sourceRef.cron === "string" ? sourceRef.cron : "",
    timezone:
      typeof sourceRef.timezone === "string"
        ? sourceRef.timezone
        : sourceRef.schedule_kind === "recurring"
          ? getDefaultReminderTimezone()
          : "",
    enabled: task ? Boolean(task.enabled) : true
  }
}

export const ReminderTaskEditor: React.FC<ReminderTaskEditorProps> = ({
  open,
  task,
  saving,
  onClose,
  onSubmit
}) => {
  const [form] = Form.useForm<ReminderTaskEditorValues>()

  useEffect(() => {
    if (open) {
      form.setFieldsValue(taskToValues(task))
    }
  }, [form, open, task])

  if (!open) {
    return null
  }

  const handleFinish = async () => {
    let values: ReminderTaskEditorValues
    try {
      values = await form.validateFields()
    } catch {
      return
    }

    const rawRunAt = values.run_at?.trim() || ""
    const runAt = datetimeLocalToIsoString(rawRunAt) || ""
    const cron = values.cron?.trim() || ""
    const timezone = values.timezone?.trim() || ""
    if (values.schedule_kind === "one_time" && !runAt) {
      form.setFields([
        {
          name: "run_at",
          errors: ["Run at is required for one-time reminders"]
        }
      ])
      return
    }
    const cronValidation = validateCronExpression(cron)
    if (values.schedule_kind === "recurring" && (!cron || !timezone || !cronValidation.valid)) {
      form.setFields([
        {
          name: "cron",
          errors: !cron
            ? ["Cron is required for recurring reminders"]
            : !cronValidation.valid
              ? [cronValidation.error]
              : []
        },
        {
          name: "timezone",
          errors: !timezone ? ["Timezone is required for recurring reminders"] : []
        }
      ])
      return
    }

    const payload =
      values.schedule_kind === "one_time"
        ? {
            title: values.title.trim(),
            body: values.body?.trim() || null,
            schedule_kind: "one_time" as const,
            run_at: runAt || null,
            timezone: timezone || null,
            enabled: Boolean(values.enabled)
          }
        : {
            title: values.title.trim(),
            body: values.body?.trim() || null,
            schedule_kind: "recurring" as const,
            cron: cron || null,
            timezone: timezone || null,
            enabled: Boolean(values.enabled)
          }

    await onSubmit(payload)
  }

  return (
    <Card title={task ? "Edit reminder" : "Create reminder"} style={{ marginTop: 16 }}>
      <Form form={form} layout="vertical">
        <Form.Item label="Title" name="title" rules={[{ required: true, message: "Title is required" }]}>
          <Input />
        </Form.Item>
        <Form.Item label="Body" name="body">
          <Input.TextArea rows={4} />
        </Form.Item>
        <ReminderScheduleControls />
        <Form.Item label="Task is active" name="enabled" valuePropName="checked">
          <Switch />
        </Form.Item>
        <Space>
          <Button type="primary" onClick={() => void handleFinish()} loading={saving}>
            Save reminder
          </Button>
          <Button onClick={onClose}>Cancel</Button>
        </Space>
      </Form>
    </Card>
  )
}

export default ReminderTaskEditor
