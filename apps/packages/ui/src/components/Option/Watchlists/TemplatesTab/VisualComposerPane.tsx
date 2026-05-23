import React from "react"
import { Button, Input, Select, Space } from "antd"

import { Alert } from "@/components/ui/primitives/Alert"
import type { TemplateComposerSectionResult } from "@/services/watchlists"

import {
  CORE_BLOCK_DEFINITIONS,
  createComposerNode,
  type ComposerAst,
  type ComposerNode,
  type ComposerNodeType,
  isPromptCapableBlock
} from "./composer-types"

interface VisualComposerPaneProps {
  ast: ComposerAst
  onChange: (nextAst: ComposerAst) => void
  runs?: Array<{ id: number; label: string }>
  selectedRunId?: number
  onSelectedRunIdChange?: (runId?: number) => void
  onGenerateSection?: (input: {
    run_id: number
    block_id: string
    prompt: string
    input_scope: "all_items" | "top_items" | "selected_items"
    style?: string
    length_target: "short" | "medium" | "long"
  }) => Promise<TemplateComposerSectionResult>
}

type GenerateState = {
  loading: boolean
  error?: string
  warnings?: string[]
}

const DEFAULT_PROMPT_BY_BLOCK: Record<ComposerNodeType, string> = {
  HeaderBlock: "",
  IntroSummaryBlock: "Write a concise intro in 2-3 sentences.",
  ItemLoopBlock: "Refine each item summary for readability and flow.",
  GroupSectionBlock: "Generate section transitions between grouped topics.",
  CtaFooterBlock: "Draft a short call-to-action closing paragraph.",
  FinalFlowCheckBlock: "Ensure the complete draft flows naturally end-to-end.",
  RawCodeBlock: ""
}

const readConfigString = (node: ComposerNode, key: string, fallback = ""): string => {
  const value = node.config?.[key]
  return typeof value === "string" ? value : fallback
}

const withUpdatedNode = (
  nodes: ComposerNode[],
  nodeId: string,
  updater: (node: ComposerNode) => ComposerNode
): ComposerNode[] => nodes.map((node) => (node.id === nodeId ? updater(node) : node))

export const VisualComposerPane: React.FC<VisualComposerPaneProps> = ({
  ast,
  onChange,
  runs = [],
  selectedRunId,
  onSelectedRunIdChange,
  onGenerateSection
}) => {
  const [generateState, setGenerateState] = React.useState<Record<string, GenerateState>>({})

  const nodes = Array.isArray(ast?.nodes) ? ast.nodes : []

  const pushNode = (type: ComposerNodeType) => {
    const existingIds = new Set(nodes.map((node) => node.id))
    onChange({
      ...ast,
      nodes: [...nodes, createComposerNode(type, undefined, existingIds)]
    })
  }

  const removeNode = (nodeId: string) => {
    onChange({
      ...ast,
      nodes: nodes.filter((node) => node.id !== nodeId)
    })
  }

  const updateNodeSource = (nodeId: string, source: string) => {
    onChange({
      ...ast,
      nodes: withUpdatedNode(nodes, nodeId, (node) => ({ ...node, source }))
    })
  }

  const updateNodeConfig = (nodeId: string, key: string, value: string) => {
    onChange({
      ...ast,
      nodes: withUpdatedNode(nodes, nodeId, (node) => ({
        ...node,
        config: { ...(node.config || {}), [key]: value }
      }))
    })
  }

  const runSectionGeneration = async (node: ComposerNode) => {
    if (!onGenerateSection || !selectedRunId) return

    const prompt =
      readConfigString(node, "prompt", DEFAULT_PROMPT_BY_BLOCK[node.type]).trim() ||
      DEFAULT_PROMPT_BY_BLOCK[node.type]
    if (!prompt.trim()) return

    const style = readConfigString(node, "style", "").trim()
    const lengthTarget = (readConfigString(node, "length_target", "medium") || "medium") as
      | "short"
      | "medium"
      | "long"

    setGenerateState((previous) => ({
      ...previous,
      [node.id]: { loading: true }
    }))

    try {
      const result = await onGenerateSection({
        run_id: selectedRunId,
        block_id: node.id,
        prompt,
        input_scope: "all_items",
        style: style || undefined,
        length_target: lengthTarget
      })

      onChange({
        ...ast,
        nodes: withUpdatedNode(nodes, node.id, (existing) => ({
          ...existing,
          source: result.content,
          config: {
            ...(existing.config || {}),
            prompt,
            style,
            length_target: lengthTarget
          }
        }))
      })

      setGenerateState((previous) => ({
        ...previous,
        [node.id]: {
          loading: false,
          warnings: Array.isArray(result.warnings) ? result.warnings : []
        }
      }))
    } catch (error: unknown) {
      setGenerateState((previous) => ({
        ...previous,
        [node.id]: {
          loading: false,
          error: error instanceof Error ? error.message : "Section generation failed"
        }
      }))
    }
  }

  return (
    <div className="space-y-3" data-testid="visual-composer-pane">
      <div className="rounded-lg border border-border p-3">
        <div className="mb-2 text-xs font-medium text-text-muted">Visual block composer</div>
        <Space size={8} wrap>
          {CORE_BLOCK_DEFINITIONS.map((definition) => (
            <Button
              key={definition.type}
              size="small"
              onClick={() => pushNode(definition.type)}
              data-testid={`visual-add-${definition.type}`}
            >
              {`Add ${definition.label}`}
            </Button>
          ))}
          <Button
            size="small"
            onClick={() => pushNode("RawCodeBlock")}
            data-testid="visual-add-raw-code"
          >
            Add Raw Code
          </Button>
        </Space>
      </div>

      {onGenerateSection ? (
        <div className="rounded-lg border border-border p-3">
          <div className="mb-2 text-xs font-medium text-text-muted">Manual section generation</div>
          <Select
            value={selectedRunId}
            onChange={(value) => onSelectedRunIdChange?.(value ? Number(value) : undefined)}
            placeholder="Select run for section generation"
            className="min-w-[220px]"
            options={runs.map((run) => ({ value: run.id, label: run.label }))}
            allowClear
          />
        </div>
      ) : null}

      {nodes.length === 0 ? (
        <Alert
          variant="info"
          aria-live="off"
          title="No visual blocks yet"
        >
          Add blocks above to start building this template.
        </Alert>
      ) : null}

      {nodes.map((node) => {
        const state = generateState[node.id]
        const prompt = readConfigString(node, "prompt", DEFAULT_PROMPT_BY_BLOCK[node.type])
        const style = readConfigString(node, "style", "")
        const lengthTarget =
          (readConfigString(node, "length_target", "medium") || "medium") as "short" | "medium" | "long"

        return (
          <div
            key={node.id}
            className="rounded-lg border border-border p-3 space-y-2"
            data-testid={`visual-block-${node.id}`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs font-medium text-text-muted">
                {node.type}
              </div>
              <Button
                danger
                size="small"
                onClick={() => removeNode(node.id)}
                aria-label={`Remove ${node.type} block`}
                data-testid={`visual-remove-${node.id}`}
              >
                Remove
              </Button>
            </div>

            {isPromptCapableBlock(node.type) ? (
              <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_180px_140px]">
                <Input
                  value={prompt}
                  onChange={(event) => updateNodeConfig(node.id, "prompt", event.currentTarget.value)}
                  placeholder="Inline prompt for this section"
                />
                <Input
                  value={style}
                  onChange={(event) => updateNodeConfig(node.id, "style", event.currentTarget.value)}
                  placeholder="Style hint"
                />
                <Select
                  value={lengthTarget}
                  onChange={(value) => updateNodeConfig(node.id, "length_target", String(value))}
                  options={[
                    { value: "short", label: "short" },
                    { value: "medium", label: "medium" },
                    { value: "long", label: "long" }
                  ]}
                />
              </div>
            ) : null}

            <Input.TextArea
              value={node.source}
              rows={6}
              onChange={(event) => updateNodeSource(node.id, event.currentTarget.value)}
              placeholder="Block content/Jinja source"
            />

            {onGenerateSection && isPromptCapableBlock(node.type) ? (
              <div className="space-y-1">
                <Button
                  size="small"
                  onClick={() => runSectionGeneration(node)}
                  loading={Boolean(state?.loading)}
                  disabled={!selectedRunId}
                  data-testid={`visual-generate-${node.id}`}
                >
                  Generate section
                </Button>
                {state?.error ? (
                  <Alert variant="error" title={state.error} />
                ) : null}
                {state?.warnings?.length ? (
                  <Alert variant="warning" title={state.warnings.join("; ")} />
                ) : null}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

export default VisualComposerPane
