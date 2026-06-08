import React, { useMemo, useState } from "react"
import { Alert, Button, Card, Empty, Input, Segmented, Space, Tag, Typography } from "antd"
import type { SegmentedValue } from "antd/es/segmented"

import type { CreateScheduledTaskReminderPayload } from "@/services/scheduled-tasks-control-plane"
import { ReminderTaskEditor } from "./ReminderTaskEditor"
import {
  SCHEDULED_TASK_TEMPLATE_FILTERS,
  filterScheduledTaskTemplates,
  findScheduledTaskTemplates,
  getScheduledTaskTemplate,
  getScheduledTaskTemplateStateLabel,
  toSafeHandoffSourceText,
  type ScheduledTaskTemplate,
  type ScheduledTaskTemplateFilterId,
  type ScheduledTaskTemplateId
} from "./scheduled-task-templates"

export interface ScheduledTaskCreatePanelProps {
  selectedTemplateId: ScheduledTaskTemplateId | null
  onSelectTemplate: (templateId: ScheduledTaskTemplateId | null) => void
  onCreateReminder: (payload: CreateScheduledTaskReminderPayload) => Promise<void> | void
  savingReminder?: boolean
}

const WATCHLISTS_HREF = "/watchlists"

const isTemplateFilterId = (value: SegmentedValue): value is ScheduledTaskTemplateFilterId =>
  SCHEDULED_TASK_TEMPLATE_FILTERS.some((filter) => filter.id === value)

const requiresWatchlistsHandoff = (template: ScheduledTaskTemplate): boolean =>
  template.id === "watch" || template.id === "ingest"

const TemplateCard: React.FC<{
  template: ScheduledTaskTemplate
  stateLabel: string
  onSelectTemplate: (templateId: ScheduledTaskTemplateId) => void
}> = ({ template, stateLabel, onSelectTemplate }) => (
  <Card
    size="small"
    style={{ height: "100%" }}
    title={
      <Space wrap size={8}>
        <Typography.Text strong>{template.title}</Typography.Text>
        <Tag>{stateLabel}</Tag>
      </Space>
    }
  >
    <Space orientation="vertical" size={8} style={{ width: "100%" }}>
      <Typography.Text type="secondary">{template.intent}</Typography.Text>
      <Typography.Paragraph style={{ marginBottom: 0 }}>
        {template.description}
      </Typography.Paragraph>
      <Button
        type={template.state === "available" ? "primary" : "default"}
        onClick={() => onSelectTemplate(template.id)}
      >
        {template.primaryActionLabel}
      </Button>
    </Space>
  </Card>
)

const HandoffPanel: React.FC<{ template: ScheduledTaskTemplate }> = ({ template }) => {
  const [sourceNote, setSourceNote] = useState("")
  const safeSourceText = toSafeHandoffSourceText(sourceNote)
  const hasUnsafeSource = Boolean(sourceNote.trim()) && !safeSourceText
  const watchlistsHandoff = requiresWatchlistsHandoff(template)
  const summaryLines = [
    `Template: ${template.title}`,
    `Intent: ${template.intent}`,
    safeSourceText ? `Source/setup note: ${safeSourceText}` : null
  ].filter((line): line is string => Boolean(line))

  return (
    <Card title={template.title} size="small">
      <Space orientation="vertical" size={12} style={{ width: "100%" }}>
        <Tag>{getScheduledTaskTemplateStateLabel(template.state)}</Tag>
        <Typography.Text>{template.intent}</Typography.Text>
        <Typography.Paragraph style={{ marginBottom: 0 }}>
          {watchlistsHandoff
            ? "Watchlists owns source setup, matching, scheduling, and outputs for this automation."
            : "Choose the workspace that owns the deeper automation setup."}
        </Typography.Paragraph>
        {watchlistsHandoff ? (
          <Typography.Text strong>Setup continues in Watchlists.</Typography.Text>
        ) : null}
        <Typography.Text>No scheduled task has been created yet.</Typography.Text>
        <Input.TextArea
          aria-label="Optional source or setup note"
          rows={3}
          value={sourceNote}
          onChange={(event) => setSourceNote(event.target.value)}
        />
        {hasUnsafeSource ? (
          <Alert
            type="warning"
            showIcon
            title="This source contains private-looking values. Remove secrets before copying or opening setup."
          />
        ) : null}
        <div aria-label="Setup summary">
          <Typography.Text copyable={summaryLines.length > 0 ? { text: summaryLines.join("\n") } : false}>
            {summaryLines.join(" · ")}
          </Typography.Text>
        </div>
        {watchlistsHandoff ? (
          <Typography.Link href={WATCHLISTS_HREF}>Open Watchlists setup</Typography.Link>
        ) : (
          <Space orientation="vertical" size={4}>
            <Typography.Text strong>Choose destination</Typography.Text>
            <Typography.Link href={WATCHLISTS_HREF}>Watchlists</Typography.Link>
            <Typography.Link href="/research">Research workspace</Typography.Link>
            <Typography.Link href="/agent-tasks">Agent tasks</Typography.Link>
          </Space>
        )}
      </Space>
    </Card>
  )
}

const PlannedPanel: React.FC<{ template: ScheduledTaskTemplate }> = ({ template }) => (
  <Card title={template.title} size="small">
    <Space orientation="vertical" size={8}>
      <Tag>{getScheduledTaskTemplateStateLabel(template.state)}</Tag>
      <Typography.Text>{template.intent}</Typography.Text>
      <Typography.Paragraph style={{ marginBottom: 0 }}>
        {template.description}
      </Typography.Paragraph>
    </Space>
  </Card>
)

export const ScheduledTaskCreatePanel: React.FC<ScheduledTaskCreatePanelProps> = ({
  selectedTemplateId,
  onSelectTemplate,
  onCreateReminder,
  savingReminder
}) => {
  const [finderText, setFinderText] = useState("")
  const [filterId, setFilterId] = useState<ScheduledTaskTemplateFilterId>("all")
  const selectedTemplate = getScheduledTaskTemplate(selectedTemplateId)
  const matches = useMemo(() => findScheduledTaskTemplates(finderText), [finderText])
  const bestMatch = matches[0] ?? null
  const templates = useMemo(() => filterScheduledTaskTemplates(filterId), [filterId])
  const templateStateLabels = useMemo(() => {
    const seenStateLabels = new Set<string>()

    return new Map(
      templates.map((template) => {
        const stateLabel = getScheduledTaskTemplateStateLabel(template.state)
        const displayLabel = seenStateLabels.has(stateLabel)
          ? `${stateLabel} (${template.title})`
          : stateLabel
        seenStateLabels.add(stateLabel)
        return [template.id, displayLabel]
      })
    )
  }, [templates])

  const renderSelectedTemplate = () => {
    if (!selectedTemplate) return null

    if (selectedTemplate.id === "reminder") {
      return (
        <ReminderTaskEditor
          open
          task={null}
          saving={savingReminder}
          onClose={() => onSelectTemplate(null)}
          onSubmit={(payload) => onCreateReminder(payload as CreateScheduledTaskReminderPayload)}
        />
      )
    }

    if (selectedTemplate.state === "planned") {
      return <PlannedPanel template={selectedTemplate} />
    }

    return <HandoffPanel template={selectedTemplate} />
  }

  return (
    <section>
      <Space orientation="vertical" size={16} style={{ width: "100%" }}>
        <div>
          <Typography.Title level={3} style={{ marginBottom: 4 }}>
            Choose what you want to automate
          </Typography.Title>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
            Pick a task shape first. Some automation setup stays in its owner workspace.
          </Typography.Paragraph>
        </div>

        <Input
          aria-label="Find a template"
          placeholder="Find a template"
          value={finderText}
          onChange={(event) => setFinderText(event.target.value)}
        />
        <div role="status" aria-live="polite">
          {bestMatch ? `Best match: ${bestMatch.title}` : null}
        </div>

        {selectedTemplate ? (
          <Space orientation="vertical" size={12} style={{ width: "100%" }}>
            <Button onClick={() => onSelectTemplate(null)}>Back to templates</Button>
            {renderSelectedTemplate()}
          </Space>
        ) : (
          <>
            <Segmented
              value={filterId}
              options={SCHEDULED_TASK_TEMPLATE_FILTERS.map((filter) => ({
                label: filter.label,
                value: filter.id
              }))}
              onChange={(value) => {
                if (isTemplateFilterId(value)) {
                  setFilterId(value)
                }
              }}
            />
            {templates.length > 0 ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                  gap: 12
                }}
              >
                {templates.map((template) => (
                  <TemplateCard
                    key={template.id}
                    template={template}
                    stateLabel={
                      templateStateLabels.get(template.id) ??
                      getScheduledTaskTemplateStateLabel(template.state)
                    }
                    onSelectTemplate={onSelectTemplate}
                  />
                ))}
              </div>
            ) : (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No templates match this filter." />
            )}
          </>
        )}
      </Space>
    </section>
  )
}

export default ScheduledTaskCreatePanel
