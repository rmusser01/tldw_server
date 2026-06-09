import React, { useMemo, useState } from "react"
import { Button, Card, Input, Segmented, Space, Typography } from "antd"
import type { SegmentedValue } from "antd/es/segmented"

import { EmptyState } from "@/components/ui/feedback"
import {
  Alert as DesignSystemAlert,
  Badge as DesignSystemBadge,
  type BadgeVariant
} from "@/components/ui/primitives"
import type { CreateScheduledTaskReminderPayload } from "@/services/scheduled-tasks-control-plane"
import { ReminderTaskEditor } from "./ReminderTaskEditor"
import {
  applyScheduledTaskTemplateCapabilities,
  buildNotificationPolicyCopy,
  buildResultDestinationCopy,
  buildSourceIntentCopy,
  getMissingAvailabilityGates,
  redactCapabilityPreviewText,
  type ScheduledTaskTemplateCapability,
  type ScheduledTaskTemplateCapabilityMap
} from "./scheduled-task-template-capabilities"
import {
  SCHEDULED_TASK_TEMPLATES,
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
  templateCapabilities?: ScheduledTaskTemplateCapabilityMap
}

const WATCHLISTS_HREF = "/watchlists"

const isTemplateFilterId = (value: SegmentedValue): value is ScheduledTaskTemplateFilterId =>
  SCHEDULED_TASK_TEMPLATE_FILTERS.some((filter) => filter.id === value)

const requiresWatchlistsHandoff = (template: ScheduledTaskTemplate): boolean =>
  template.id === "watch" || template.id === "ingest"

const templateStateToBadgeVariant = (
  state: ScheduledTaskTemplate["state"]
): BadgeVariant => {
  switch (state) {
    case "available":
      return "success"
    case "limited_availability":
    case "needs_setup":
      return "warning"
    case "unavailable":
      return "danger"
    case "planned":
      return "secondary"
    case "managed_in_watchlists":
    case "handoff_only":
      return "info"
  }

  const _exhaustive: never = state
  return _exhaustive
}

const PRIVATE_LOOKING_PROSE_PATTERN =
  /\b(api[_ -]?key|password|passphrase|secret|bearer\s+token|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret)\b|sk-[A-Za-z0-9_-]+/i

const containsPrivateLookingProse = (value: string): boolean =>
  PRIVATE_LOOKING_PROSE_PATTERN.test(value)

const formatAvailabilityGateLabel = (gate: string): string => gate.replace(/_/g, " ")

const WATCHLISTS_ADAPTER_UNAVAILABLE_COPY =
  "Creation from Scheduled Tasks is not available yet. Continue setup in Watchlists."

const OWNER_WORKSPACE_ADAPTER_UNAVAILABLE_COPY =
  "Creation from Scheduled Tasks is not available yet. Choose the owner workspace to continue setup."

const buildAvailabilityCopy = (
  template: ScheduledTaskTemplate,
  capability: ScheduledTaskTemplateCapability
): string[] => {
  const missingGateCopy = getMissingAvailabilityGates(template.id, capability).map(
    (gate) => `Missing: ${formatAvailabilityGateLabel(gate)}`
  )

  if (missingGateCopy.length > 0 || capability.creationAdapterSupported === true) {
    return missingGateCopy
  }

  const adapterCopy = requiresWatchlistsHandoff(template)
    ? WATCHLISTS_ADAPTER_UNAVAILABLE_COPY
    : OWNER_WORKSPACE_ADAPTER_UNAVAILABLE_COPY
  const redactedReason = capability.reason
    ? redactCapabilityPreviewText(capability.reason)
    : null

  return redactedReason && redactedReason !== adapterCopy
    ? [adapterCopy, redactedReason]
    : [adapterCopy]
}

const CapabilityCopyGroup: React.FC<{ title: string; lines: readonly string[] }> = ({
  title,
  lines
}) =>
  lines.length > 0 ? (
    <Space orientation="vertical" size={4}>
      <Typography.Text strong>{title}</Typography.Text>
      {lines.map((line) => (
        <Typography.Text key={line}>{line}</Typography.Text>
      ))}
    </Space>
  ) : null

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
        <DesignSystemBadge variant={templateStateToBadgeVariant(template.state)}>
          {stateLabel}
        </DesignSystemBadge>
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

const HandoffPanel: React.FC<{
  template: ScheduledTaskTemplate
  capability?: ScheduledTaskTemplateCapability | null
}> = ({ template, capability }) => {
  const [sourceNote, setSourceNote] = useState("")
  const safeSourceText = toSafeHandoffSourceText(sourceNote)
  const normalizedSourceNote = sourceNote.trim()
  const hasUnsafeSource =
    Boolean(normalizedSourceNote) &&
    (!safeSourceText || containsPrivateLookingProse(normalizedSourceNote))
  const hasSafeSourceNote = Boolean(normalizedSourceNote) && !hasUnsafeSource
  const watchlistsHandoff = requiresWatchlistsHandoff(template)
  const availabilityCopy = capability ? buildAvailabilityCopy(template, capability) : []
  const sourceIntentCopy = capability ? buildSourceIntentCopy(capability.sourceIntent) : []
  const shouldRenderDestinationFallback =
    watchlistsHandoff || Boolean(capability?.resultDestinations)
  const resultDestinationCopy = capability && shouldRenderDestinationFallback
    ? buildResultDestinationCopy(capability.resultDestinations)
    : []
  const notificationCopy = capability && shouldRenderDestinationFallback
    ? [buildNotificationPolicyCopy(capability.resultDestinations)]
    : []
  const summaryLines = [
    `Template: ${template.title}`,
    `Intent: ${template.intent}`,
    hasSafeSourceNote ? "Source/setup note provided" : null
  ].filter((line): line is string => Boolean(line))

  return (
    <Card title={template.title} size="small">
      <Space orientation="vertical" size={12} style={{ width: "100%" }}>
        <DesignSystemBadge variant={templateStateToBadgeVariant(template.state)}>
          {getScheduledTaskTemplateStateLabel(template.state)}
        </DesignSystemBadge>
        <Typography.Text>{template.intent}</Typography.Text>
        <Typography.Paragraph style={{ marginBottom: 0 }}>
          {watchlistsHandoff
            ? "Watchlists owns source setup, matching, scheduling, and outputs for this automation."
            : "Choose the workspace that owns the deeper automation setup."}
        </Typography.Paragraph>
        {watchlistsHandoff ? (
          <Typography.Text strong>Setup continues in Watchlists.</Typography.Text>
        ) : null}
        <CapabilityCopyGroup title="Availability" lines={availabilityCopy} />
        <CapabilityCopyGroup title="Source support" lines={sourceIntentCopy} />
        <CapabilityCopyGroup title="Result destinations" lines={resultDestinationCopy} />
        <CapabilityCopyGroup title="Notifications" lines={notificationCopy} />
        <Typography.Text>No scheduled task has been created yet.</Typography.Text>
        <Input.TextArea
          aria-label="Optional source or setup note"
          rows={3}
          value={sourceNote}
          onChange={(event) => setSourceNote(event.target.value)}
        />
        {hasUnsafeSource ? (
          <DesignSystemAlert
            variant="warning"
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
      <DesignSystemBadge variant={templateStateToBadgeVariant(template.state)}>
        {getScheduledTaskTemplateStateLabel(template.state)}
      </DesignSystemBadge>
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
  savingReminder,
  templateCapabilities
}) => {
  const [finderText, setFinderText] = useState("")
  const [filterId, setFilterId] = useState<ScheduledTaskTemplateFilterId>("all")
  const effectiveTemplates = useMemo(
    () => applyScheduledTaskTemplateCapabilities(SCHEDULED_TASK_TEMPLATES, templateCapabilities),
    [templateCapabilities]
  )
  const selectedTemplate = getScheduledTaskTemplate(selectedTemplateId, effectiveTemplates)
  const selectedCapability = selectedTemplate
    ? templateCapabilities?.[selectedTemplate.id] ?? null
    : null
  const matches = useMemo(() => findScheduledTaskTemplates(finderText), [finderText])
  const bestMatch = matches[0] ?? null
  const templates = useMemo(
    () => filterScheduledTaskTemplates(filterId, effectiveTemplates),
    [effectiveTemplates, filterId]
  )
  const templateStateLabels = useMemo(
    () =>
      new Map(
        templates.map((template) => [
          template.id,
          getScheduledTaskTemplateStateLabel(template.state)
        ])
      ),
    [templates]
  )

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

    return <HandoffPanel template={selectedTemplate} capability={selectedCapability} />
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
              <EmptyState
                variant="inline"
                size="sm"
                title="No templates match this filter."
              />
            )}
          </>
        )}
      </Space>
    </section>
  )
}

export default ScheduledTaskCreatePanel
