export type WritingRevisionAction =
  | "continue"
  | "rewrite"
  | "expand"
  | "tighten"
  | "tone"
  | "outline"
  | "custom"

export type WritingRevisionOperation = "insert" | "replace" | "advisory"

export type WritingRevisionPresetId =
  | "draft_freely"
  | "polish_prose"
  | "developmental_edit"
  | "preserve_voice"
  | "make_concise"
  | "expand_sensory_detail"

export type WritingRevisionStatus =
  | "pending"
  | "applied"
  | "rejected"
  | "conflict"
  | "raw_suggestion"
  | "advisory"

export type WritingRevisionAnchor = {
  documentFingerprint: string
  prefix: string
  suffix: string
}

export type WritingRevisionTarget = {
  mode: "selection" | "paragraph" | "cursor" | "document"
  start: number
  end: number
  beforeText: string
  anchor: WritingRevisionAnchor
  label: string
  requiresConfirmation: boolean
  confirmationReason?: string
}

export type WritingRevisionProposal = {
  id: string
  sessionId: string
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  presetId?: WritingRevisionPresetId | null
  presetInstruction?: string | null
  instruction: string
  target: WritingRevisionTarget
  replacementText?: string
  rawText?: string
  rationale?: string
  title?: string
  notes?: string[]
  regeneratedFromId?: string
  createdAt: string
  status: WritingRevisionStatus
}

export type WritingRevisionApplyPlan =
  | { type: "apply"; start: number; end: number; nextText: string }
  | { type: "retarget"; start: number; end: number; nextText: string }
  | { type: "conflict"; reason: string }
  | { type: "noop"; reason: string }

export type WritingRevisionPayload = {
  schemaVersion: 1
  items: WritingRevisionProposal[]
}
