import React from "react"
import { Button, Radio, Space } from "antd"

import { Alert } from "@/components/ui/primitives"
import type { TemplateComposerFlowCheckMode, TemplateComposerFlowIssue } from "@/services/watchlists"

interface FlowCheckDiffPanelProps {
  diff: string
  mode?: TemplateComposerFlowCheckMode
  issues?: TemplateComposerFlowIssue[]
  onModeChange?: (mode: TemplateComposerFlowCheckMode) => void
  onAcceptChunk: (chunkId: string) => void
  onRejectChunk: (chunkId: string) => void
  onRevertAll?: () => void
}

export const FlowCheckDiffPanel: React.FC<FlowCheckDiffPanelProps> = ({
  diff,
  mode = "suggest_only",
  issues = [],
  onModeChange,
  onAcceptChunk,
  onRejectChunk,
  onRevertAll
}) => {
  const trimmedDiff = String(diff || "").trim()

  return (
    <div className="space-y-3 rounded-lg border border-border p-3" data-testid="flow-check-diff-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-medium text-text-muted">Final flow-check</div>
        <Radio.Group
          value={mode}
          onChange={(event) => onModeChange?.(event.target.value as TemplateComposerFlowCheckMode)}
          size="small"
          optionType="button"
        >
          <Radio.Button value="suggest_only">Suggest only</Radio.Button>
          <Radio.Button value="auto_apply">Auto apply</Radio.Button>
        </Radio.Group>
      </div>

      {issues.length > 0 ? (
        <Alert variant="warning" title="Flow issues">
          {issues.map((issue) => issue.message).join("; ")}
        </Alert>
      ) : null}

      {trimmedDiff ? (
        <pre
          className="max-h-64 overflow-auto rounded bg-surface p-3 text-xs leading-5"
          role="region"
          aria-label="Template flow-check diff"
        >
          {trimmedDiff}
        </pre>
      ) : (
        <Alert variant="info" title="No diff available yet">
          Run flow-check to generate suggestions.
        </Alert>
      )}

      <Space size={8}>
        <Button type="primary" onClick={() => onAcceptChunk("all")}>
          Accept
        </Button>
        <Button onClick={() => onRejectChunk("all")}>
          Reject
        </Button>
        {onRevertAll ? (
          <Button onClick={onRevertAll}>
            Revert
          </Button>
        ) : null}
      </Space>
    </div>
  )
}

export default FlowCheckDiffPanel
