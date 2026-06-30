import React, { useEffect, useMemo, useRef, useState } from "react"
import { Form, Input, InputNumber, Radio, Select, Space, Typography } from "antd"

import {
  buildReminderCron,
  getDefaultReminderTimezone,
  getOneTimePreviewCopy,
  getRecurringPreviewCopy,
  parseReminderCron,
  type ReminderRecurrencePreset,
  type ReminderWeekdayToken
} from "./reminder-schedule-utils"

type ReminderScheduleFormValues = {
  schedule_kind?: "one_time" | "recurring"
  run_at?: string | null
  cron?: string | null
  timezone?: string | null
}

const weekdayOptions: Array<{ value: ReminderWeekdayToken; label: string }> = [
  { value: "MON", label: "Monday" },
  { value: "TUE", label: "Tuesday" },
  { value: "WED", label: "Wednesday" },
  { value: "THU", label: "Thursday" },
  { value: "FRI", label: "Friday" },
  { value: "SAT", label: "Saturday" },
  { value: "SUN", label: "Sunday" }
]

const presetOptions: Array<{ value: ReminderRecurrencePreset; label: string }> = [
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "custom", label: "Custom schedule" }
]

export const ReminderScheduleControls: React.FC = () => {
  const form = Form.useFormInstance<ReminderScheduleFormValues>()
  const scheduleKind = Form.useWatch("schedule_kind", form) || "one_time"
  const runAt = Form.useWatch("run_at", form)
  const cron = Form.useWatch("cron", form)
  const timezone = Form.useWatch("timezone", form)
  const localTimezone = useMemo(() => getDefaultReminderTimezone(), [])
  const previousScheduleKind = useRef(scheduleKind)

  const parsedCron = parseReminderCron(typeof cron === "string" ? cron : "")
  const [preset, setPreset] = useState<ReminderRecurrencePreset>(parsedCron?.preset || "daily")
  const [hour, setHour] = useState(parsedCron?.hour ?? 9)
  const [minute, setMinute] = useState(parsedCron?.minute ?? 0)
  const [weekday, setWeekday] = useState<ReminderWeekdayToken>(parsedCron?.weekday || "MON")

  const setPresetCron = (
    nextPreset: ReminderRecurrencePreset,
    nextWeekday: ReminderWeekdayToken,
    nextHour: number,
    nextMinute: number,
    customCron: string | null | undefined = normalizedCron
  ) => {
    form.setFieldValue("cron", buildReminderCron(nextPreset, nextWeekday, nextHour, nextMinute, customCron))
  }

  useEffect(() => {
    const enteredRecurring =
      previousScheduleKind.current !== "recurring" && scheduleKind === "recurring"
    previousScheduleKind.current = scheduleKind
    if (enteredRecurring && (!timezone || !timezone.trim())) {
      form.setFieldValue("timezone", localTimezone)
    }
    if (enteredRecurring && (!cron || !cron.trim())) {
      form.setFieldValue("cron", buildReminderCron(preset, weekday, hour, minute, cron))
    }
  }, [cron, form, hour, localTimezone, minute, preset, scheduleKind, timezone, weekday])

  useEffect(() => {
    if (scheduleKind !== "recurring") return
    const nextParsedCron = parseReminderCron(typeof cron === "string" ? cron : "")
    if (!nextParsedCron) {
      if (typeof cron === "string" && cron.trim()) {
        setPreset("custom")
      }
      return
    }

    setPreset(nextParsedCron.preset)
    setHour(nextParsedCron.hour)
    setMinute(nextParsedCron.minute)
    setWeekday(nextParsedCron.weekday)
  }, [cron, scheduleKind])

  const normalizedCron = typeof cron === "string" ? cron : ""
  const normalizedTimezone = typeof timezone === "string" ? timezone : localTimezone

  return (
    <>
      <Form.Item
        label="Schedule kind"
        name="schedule_kind"
        rules={[{ required: true, message: "Schedule kind is required" }]}
      >
        <Radio.Group
          optionType="button"
          buttonStyle="solid"
          options={[
            { value: "one_time", label: "Run once" },
            { value: "recurring", label: "Repeat" }
          ]}
        />
      </Form.Item>

      {scheduleKind === "one_time" ? (
        <>
          <Form.Item
            label="Run once at"
            name="run_at"
            extra={`Timezone: ${localTimezone}`}
          >
            <Input type="datetime-local" />
          </Form.Item>
          <Typography.Text type="secondary">{getOneTimePreviewCopy(runAt)}</Typography.Text>
          <Form.Item name="timezone" hidden>
            <Input />
          </Form.Item>
          <Form.Item name="cron" hidden>
            <Input />
          </Form.Item>
        </>
      ) : (
        <>
          <Space wrap align="start">
            <Form.Item label="Repeat preset">
              <Select
                aria-label="Repeat preset"
                style={{ minWidth: 180 }}
                value={preset}
                options={presetOptions}
                onChange={(value) => {
                  setPreset(value)
                  if (value !== "custom") {
                    setPresetCron(value, weekday, hour, minute, normalizedCron)
                  }
                }}
              />
            </Form.Item>
            {preset !== "custom" ? (
              <>
                <Form.Item label="Hour">
                  <InputNumber
                    min={0}
                    max={23}
                    value={hour}
                    onChange={(value) => {
                      const nextHour = Number(value ?? 0)
                      setHour(nextHour)
                      setPresetCron(preset, weekday, nextHour, minute, normalizedCron)
                    }}
                  />
                </Form.Item>
                <Form.Item label="Minute">
                  <InputNumber
                    min={0}
                    max={59}
                    value={minute}
                    onChange={(value) => {
                      const nextMinute = Number(value ?? 0)
                      setMinute(nextMinute)
                      setPresetCron(preset, weekday, hour, nextMinute, normalizedCron)
                    }}
                  />
                </Form.Item>
              </>
            ) : null}
            {preset === "weekly" ? (
              <Form.Item label="Weekday">
                <Select
                  aria-label="Weekday"
                  style={{ minWidth: 150 }}
                  value={weekday}
                  options={weekdayOptions}
                  onChange={(value) => {
                    setWeekday(value)
                    setPresetCron(preset, value, hour, minute, normalizedCron)
                  }}
                />
              </Form.Item>
            ) : null}
          </Space>
          {preset === "custom" ? (
            <Form.Item label="Custom cron" name="cron">
              <Input placeholder="0 9 * * MON" />
            </Form.Item>
          ) : (
            <Form.Item name="cron" hidden>
              <Input />
            </Form.Item>
          )}
          <Form.Item label="Timezone" name="timezone">
            <Input placeholder={localTimezone} />
          </Form.Item>
          <Typography.Text type="secondary">
            {getRecurringPreviewCopy(preset, normalizedCron, normalizedTimezone)}
          </Typography.Text>
          <Form.Item name="run_at" hidden>
            <Input />
          </Form.Item>
        </>
      )}
    </>
  )
}

export default ReminderScheduleControls
