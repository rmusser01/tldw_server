import React from "react"
import { Button, Descriptions, Drawer, Space, Typography } from "antd"
import {
  Alert as DesignSystemAlert,
  Badge as DesignSystemBadge
} from "@/components/ui/primitives"

import { formatScheduledTaskTimestamp } from "./scheduled-task-status"
import {
  getScheduledTaskResultStatusLabel,
  scheduledTaskResultSeverityToBadgeVariant,
  type ScheduledTaskResultItem
} from "./scheduled-task-results"

export interface ScheduledTaskResultDetailDrawerProps {
  open: boolean
  result: ScheduledTaskResultItem | null
  onClose: () => void
  onReviewResult: (result: ScheduledTaskResultItem) => void
  onRetryRun: (result: ScheduledTaskResultItem) => void
}

const UNSUPPORTED_ACTION_COPY =
  "Review and retry actions appear when this server supports them for the selected result."

const optionalDescriptionItem = (
  label: string,
  value: string | number | null | undefined
): React.ReactNode => {
  if (value === null || value === undefined || String(value).trim() === "") {
    return null
  }

  return (
    <Descriptions.Item label={label} key={label}>
      {String(value)}
    </Descriptions.Item>
  )
}

const renderActionLinks = (result: ScheduledTaskResultItem): React.ReactNode => {
  const links: Array<{ href: string | null; label: string }> = [
    { href: result.resultHref, label: "Open result" },
    { href: result.runHref, label: "Open run" },
    { href: result.sourceHref, label: "Open owner workspace" }
  ]
  const visibleLinks = links.filter((link): link is { href: string; label: string } =>
    Boolean(link.href)
  )

  if (visibleLinks.length === 0) {
    return <Typography.Text type="secondary">No owner links are available yet.</Typography.Text>
  }

  return (
    <Space wrap>
      {visibleLinks.map((link) => (
        <Button key={link.label} href={link.href}>
          {link.label}
        </Button>
      ))}
    </Space>
  )
}

const renderStateBadge = (result: ScheduledTaskResultItem): React.ReactNode => {
  const badge = (
    <DesignSystemBadge variant={scheduledTaskResultSeverityToBadgeVariant(result.severity)}>
      {getScheduledTaskResultStatusLabel(result)}
    </DesignSystemBadge>
  )

  if (result.signalKind !== "running") {
    return badge
  }

  return <span aria-live="polite">{badge}</span>
}

const renderAnswerSection = (result: ScheduledTaskResultItem): React.ReactNode => {
  if (!result.answer) {
    return null
  }

  return (
    <div>
      <Typography.Title level={5}>Answer</Typography.Title>
      <Typography.Paragraph style={{ marginBottom: 0 }}>
        {result.answer}
      </Typography.Paragraph>
    </div>
  )
}

const renderEvidenceSection = (result: ScheduledTaskResultItem): React.ReactNode => {
  const sourceRefs = result.sourceRefs ?? []
  if (sourceRefs.length === 0) {
    return null
  }

  return (
    <div>
      <Typography.Title level={5}>Evidence</Typography.Title>
      <Space orientation="vertical" size={8} style={{ width: "100%" }}>
        {sourceRefs.map((sourceRef, index) => (
          <div key={`${sourceRef.sourceId ?? sourceRef.citationRef ?? "source"}:${index}`}>
            <Typography.Text strong>
              {sourceRef.title ?? sourceRef.sourceId ?? "Source"}
            </Typography.Text>
            {sourceRef.snippet ? (
              <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
                {sourceRef.snippet}
              </Typography.Paragraph>
            ) : null}
            {sourceRef.citationRef ? (
              <Typography.Text type="secondary">{sourceRef.citationRef}</Typography.Text>
            ) : null}
          </div>
        ))}
      </Space>
    </div>
  )
}

export const ScheduledTaskResultDetailDrawer: React.FC<
  ScheduledTaskResultDetailDrawerProps
> = ({ open, result, onClose, onReviewResult, onRetryRun }) => {
  const titleId = React.useId()
  const drawerTitle = result ? `${result.title} result` : "Scheduled task result"

  return (
    <Drawer
      aria-labelledby={titleId}
      title={<span id={titleId}>{drawerTitle}</span>}
      open={open}
      onClose={onClose}
      size={560}
    >
      {result ? (
        <Space orientation="vertical" size="large" style={{ width: "100%" }}>
          <div>
            <Typography.Title level={5}>Why this is here</Typography.Title>
            <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
              {result.summary}
            </Typography.Paragraph>
          </div>

          <Descriptions bordered size="small" column={1}>
            <Descriptions.Item label="State">
              {renderStateBadge(result)}
            </Descriptions.Item>
            <Descriptions.Item label="Owner">{result.ownerLabel}</Descriptions.Item>
            <Descriptions.Item label="Task type">{result.taskTypeLabel}</Descriptions.Item>
            {optionalDescriptionItem("Task id", result.taskId)}
            {optionalDescriptionItem("Run id", result.runId)}
            {optionalDescriptionItem("Result id", result.resultId)}
            {optionalDescriptionItem("Result count", result.resultCount)}
            {optionalDescriptionItem("Source", result.sourceLabel)}
            {optionalDescriptionItem("Matched rule", result.matchedRuleLabel)}
            {optionalDescriptionItem("Output", result.outputLabel)}
            <Descriptions.Item label="Last signal">
              {formatScheduledTaskTimestamp(result.occurredAt, "No run time")}
            </Descriptions.Item>
          </Descriptions>

          {renderAnswerSection(result)}
          {renderEvidenceSection(result)}

          <div>
            <Typography.Title level={5}>Continue in</Typography.Title>
            {renderActionLinks(result)}
          </div>

          <div>
            <Typography.Title level={5}>Actions</Typography.Title>
            {result.reviewAvailable || result.retryAvailable ? (
              <Space wrap>
                {result.reviewAvailable ? (
                  <Button type="primary" onClick={() => onReviewResult(result)}>
                    Mark reviewed
                  </Button>
                ) : null}
                {result.retryAvailable ? (
                  <Button onClick={() => onRetryRun(result)}>
                    Retry run
                  </Button>
                ) : null}
              </Space>
            ) : (
              <DesignSystemAlert variant="info" title={UNSUPPORTED_ACTION_COPY} />
            )}
          </div>
        </Space>
      ) : (
        <Typography.Text type="secondary">Select a result to inspect.</Typography.Text>
      )}
    </Drawer>
  )
}

export default ScheduledTaskResultDetailDrawer
