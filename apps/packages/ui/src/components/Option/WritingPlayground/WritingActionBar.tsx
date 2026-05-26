import React, { useEffect, useMemo, useRef, useState } from "react"
import { Alert, Button, Checkbox, Input, Segmented, Space, Tag, Typography, type InputRef } from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"
import {
  ArrowRight,
  FileText,
  ListTree,
  Minimize2,
  PenLine,
  RefreshCw,
  SlidersHorizontal
} from "lucide-react"
import { WRITING_REVISION_PRESETS, getWritingRevisionPreset } from "./writing-revision-presets"
import type {
  WritingRevisionAction,
  WritingRevisionOperation,
  WritingRevisionPresetId,
  WritingRevisionTarget
} from "./writing-revision-types"

const { Text } = Typography

export type WritingActionBarRequest = {
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  presetId: WritingRevisionPresetId
  presetInstruction: string
  instruction: string
  target: WritingRevisionTarget
}

export type WritingActionBarProps = {
  generationAvailable: boolean
  target: WritingRevisionTarget
  selectedPresetId?: WritingRevisionPresetId
  isGenerating?: boolean
  onPresetChange?: (presetId: WritingRevisionPresetId) => void
  onRequest: (request: WritingActionBarRequest) => void
}

const TEXT_CHANGING_ACTIONS = new Set<WritingRevisionAction>([
  "continue",
  "rewrite",
  "expand",
  "tighten",
  "tone",
  "custom"
])

const ACTION_LABELS: Array<{
  action: WritingRevisionAction
  label: string
  icon: React.ReactNode
}> = [
  { action: "continue", label: "Continue", icon: <ArrowRight size={14} /> },
  { action: "rewrite", label: "Rewrite", icon: <RefreshCw size={14} /> },
  { action: "expand", label: "Expand", icon: <FileText size={14} /> },
  { action: "tighten", label: "Tighten", icon: <Minimize2 size={14} /> },
  { action: "tone", label: "Tone", icon: <SlidersHorizontal size={14} /> },
  { action: "outline", label: "Outline", icon: <ListTree size={14} /> },
  { action: "custom", label: "Custom", icon: <PenLine size={14} /> }
]

const DEFAULT_INSTRUCTIONS: Record<WritingRevisionAction, string> = {
  continue: "Continue from the current cursor position.",
  rewrite: "Rewrite the target while preserving intent.",
  expand: "Expand the target with useful detail.",
  tighten: "Tighten the target and remove redundancy.",
  tone: "Adjust the tone using the requested direction.",
  outline: "Provide an advisory outline for improving the current draft.",
  custom: "Follow the custom instruction for the selected target."
}

const getOperationForAction = (
  action: WritingRevisionAction
): WritingRevisionOperation => {
  if (action === "continue") return "insert"
  if (action === "outline") return "advisory"
  return "replace"
}

const getTargetIdentity = (target: WritingRevisionTarget): string =>
  [
    target.mode,
    target.start,
    target.end,
    target.beforeText,
    target.anchor.documentFingerprint
  ].join("|")

export function WritingActionBar({
  generationAvailable,
  target,
  selectedPresetId,
  isGenerating = false,
  onPresetChange,
  onRequest
}: WritingActionBarProps) {
  const [internalPresetId, setInternalPresetId] =
    useState<WritingRevisionPresetId>("polish_prose")
  const [activeInputAction, setActiveInputAction] =
    useState<WritingRevisionAction | null>(null)
  const [customInstruction, setCustomInstruction] = useState("")
  const [toneDirection, setToneDirection] = useState("")
  const [confirmed, setConfirmed] = useState(false)
  const [confirmationWarning, setConfirmationWarning] = useState(false)
  const customInputRef = useRef<TextAreaRef | null>(null)
  const toneInputRef = useRef<InputRef | null>(null)

  const activePresetId = selectedPresetId ?? internalPresetId
  const selectedPreset = useMemo(
    () => getWritingRevisionPreset(activePresetId) ?? WRITING_REVISION_PRESETS[0],
    [activePresetId]
  )
  const targetIdentity = useMemo(() => getTargetIdentity(target), [target])

  useEffect(() => {
    setConfirmed(false)
    setConfirmationWarning(false)
  }, [targetIdentity])

  const disabled = !generationAvailable || isGenerating

  const sendRequest = (action: WritingRevisionAction, explicitInstruction?: string) => {
    const operation = getOperationForAction(action)
    const requiresConfirmation =
      target.requiresConfirmation &&
      operation !== "insert" &&
      TEXT_CHANGING_ACTIONS.has(action)

    if (requiresConfirmation && !confirmed) {
      setConfirmationWarning(true)
      return
    }

    setConfirmationWarning(false)
    onRequest({
      action,
      operation,
      presetId: selectedPreset.id,
      presetInstruction: selectedPreset.instruction,
      instruction:
        explicitInstruction?.trim() ||
        (action === "outline"
          ? DEFAULT_INSTRUCTIONS.outline
          : DEFAULT_INSTRUCTIONS[action]),
      target
    })
  }

  const handleAction = (action: WritingRevisionAction) => {
    if (disabled) return
    if (action === "custom" || action === "tone") {
      setActiveInputAction(action)
      return
    }
    sendRequest(action)
  }

  const sendCustom = () =>
    sendRequest(
      "custom",
      customInputRef.current?.resizableTextArea?.textArea?.value ??
        customInstruction
    )
  const sendTone = () =>
    sendRequest("tone", toneInputRef.current?.input?.value ?? toneDirection)

  return (
    <section
      data-testid="writing-revision-action-bar"
      className="flex flex-col gap-2 rounded border border-gray-200 p-2"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Tag color={generationAvailable ? "green" : "default"}>
          {generationAvailable ? "Ready" : "Generation unavailable"}
        </Tag>
        <Text type="secondary" className="text-xs">
          Target:
        </Text>
        <Text type="secondary" className="text-xs">
          {target.label}
        </Text>
      </div>

      <Segmented
        size="small"
        value={activePresetId}
        onChange={(value) => {
          const nextPresetId = value as WritingRevisionPresetId
          setInternalPresetId(nextPresetId)
          onPresetChange?.(nextPresetId)
        }}
        options={WRITING_REVISION_PRESETS.map((preset) => ({
          label: preset.label,
          value: preset.id
        }))}
      />
      <Text type="secondary" className="text-xs">
        {selectedPreset.instruction}
      </Text>

      {target.requiresConfirmation ? (
        <div className="flex flex-col gap-1">
          <Checkbox
            checked={confirmed}
            onChange={(event) => setConfirmed(event.target.checked)}
            aria-label="Confirm whole-document text change"
          >
            Confirm whole-document text change
          </Checkbox>
          <Text type="secondary" className="text-xs">
            {target.confirmationReason ??
              "Confirm before sending a broad text-changing request."}
          </Text>
        </div>
      ) : null}

      {confirmationWarning ? (
        <Alert
          type="warning"
          title={
            target.confirmationReason ??
            "Confirm before sending a broad text-changing request."
          }
        />
      ) : null}

      <Space wrap size={[6, 6]}>
        {ACTION_LABELS.map(({ action, label, icon }) => (
          <Button
            key={action}
            size="small"
            icon={icon}
            disabled={disabled}
            loading={isGenerating}
            onClick={() => handleAction(action)}
          >
            {label}
          </Button>
        ))}
      </Space>

      {activeInputAction === "custom" ? (
        <div className="flex flex-col gap-2">
          <Input.TextArea
            ref={customInputRef}
            aria-label="Custom instruction"
            value={customInstruction}
            onChange={(event) => setCustomInstruction(event.target.value)}
            rows={2}
            placeholder="Describe the revision you want."
          />
          <Button
            size="small"
            type="primary"
            disabled={disabled}
            onClick={sendCustom}
          >
            Send Custom
          </Button>
        </div>
      ) : null}

      {activeInputAction === "tone" ? (
        <div className="flex flex-col gap-2">
          <Input
            ref={toneInputRef}
            aria-label="Tone direction"
            value={toneDirection}
            onChange={(event) => setToneDirection(event.target.value)}
            placeholder="warmer, sharper, more formal..."
          />
          <Button
            size="small"
            type="primary"
            disabled={disabled}
            onClick={sendTone}
          >
            Send Tone
          </Button>
        </div>
      ) : null}
    </section>
  )
}
