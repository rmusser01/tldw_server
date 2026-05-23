import React from "react"
import { Button, Empty, Space, Tag, Typography } from "antd"
import { Check, Copy, RefreshCw, X } from "lucide-react"
import { WritingRevisionDiff } from "./WritingRevisionDiff"
import type { WritingRevisionProposal, WritingRevisionStatus } from "./writing-revision-types"

const { Text, Paragraph } = Typography

export type WritingRevisionQueueProps = {
  proposals: WritingRevisionProposal[]
  onApply: (proposal: WritingRevisionProposal) => void
  onReject: (proposal: WritingRevisionProposal) => void
  onCopy: (proposal: WritingRevisionProposal) => void
  onRegenerate: (proposal: WritingRevisionProposal) => void
}

const STATUS_COLORS: Record<WritingRevisionStatus, string> = {
  pending: "blue",
  applied: "green",
  rejected: "default",
  conflict: "red",
  raw_suggestion: "gold",
  advisory: "purple"
}

const formatAction = (action: string): string =>
  action.replace(/_/g, " ").replace(/^\w/, (letter) => letter.toUpperCase())

const showApply = (proposal: WritingRevisionProposal): boolean =>
  proposal.status === "pending" && proposal.operation !== "advisory"

const showStandardActions = (proposal: WritingRevisionProposal): boolean =>
  proposal.status !== "raw_suggestion"

function ProposalBody({ proposal }: { proposal: WritingRevisionProposal }) {
  if (proposal.status === "raw_suggestion") {
    return (
      <pre className="m-0 whitespace-pre-wrap rounded border border-gray-200 bg-gray-50 p-2 text-xs">
        {proposal.rawText ?? proposal.replacementText ?? ""}
      </pre>
    )
  }

  if (proposal.operation === "advisory") {
    return (
      <Paragraph className="!mb-0 whitespace-pre-wrap text-sm">
        {proposal.rawText ?? proposal.replacementText ?? proposal.rationale}
      </Paragraph>
    )
  }

  return (
    <WritingRevisionDiff
      beforeText={proposal.target.beforeText}
      afterText={proposal.replacementText ?? ""}
    />
  )
}

export function WritingRevisionQueue({
  proposals,
  onApply,
  onReject,
  onCopy,
  onRegenerate
}: WritingRevisionQueueProps) {
  return (
    <section
      data-testid="writing-revision-queue"
      className="flex flex-col gap-2 rounded border border-gray-200 p-2"
    >
      <div className="flex items-center justify-between gap-2">
        <Text strong>Revision queue</Text>
        <Tag>{proposals.length}</Tag>
      </div>

      {proposals.length === 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No revision proposals" />
      ) : null}

      {proposals.map((proposal) => (
        <article
          key={proposal.id}
          data-testid="writing-revision-proposal"
          data-proposal-id={proposal.id}
          data-regenerated-from-id={proposal.regeneratedFromId ?? undefined}
          className="flex flex-col gap-2 rounded border border-gray-100 p-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <Text strong className="text-sm">
              {proposal.title ?? formatAction(proposal.action)}
            </Text>
            <Tag color={STATUS_COLORS[proposal.status]}>{proposal.status}</Tag>
            <Text type="secondary" className="text-xs">
              {proposal.target.label}
            </Text>
          </div>

          <ProposalBody proposal={proposal} />

          {proposal.rationale && proposal.operation !== "advisory" ? (
            <Text type="secondary" className="text-xs">
              {proposal.rationale}
            </Text>
          ) : null}

          {proposal.status === "conflict" ? (
            <Text type="danger" className="text-xs">
              Copy the suggestion and apply it manually.
            </Text>
          ) : null}

          {proposal.notes?.length ? (
            <div className="flex flex-col gap-1">
              {proposal.notes.map((note) => (
                <Text key={note} type="secondary" className="text-xs">
                  {note}
                </Text>
              ))}
            </div>
          ) : null}

          <Space wrap size={[6, 6]}>
            {showApply(proposal) ? (
              <Button
                size="small"
                type="primary"
                icon={<Check size={14} />}
                onClick={() => onApply(proposal)}
              >
                Apply
              </Button>
            ) : null}
            <Button size="small" icon={<Copy size={14} />} onClick={() => onCopy(proposal)}>
              Copy
            </Button>
            {showStandardActions(proposal) ? (
              <>
                <Button size="small" icon={<X size={14} />} onClick={() => onReject(proposal)}>
                  Reject
                </Button>
                <Button
                  size="small"
                  icon={<RefreshCw size={14} />}
                  onClick={() => onRegenerate(proposal)}
                >
                  Regenerate
                </Button>
              </>
            ) : null}
          </Space>
        </article>
      ))}
    </section>
  )
}
