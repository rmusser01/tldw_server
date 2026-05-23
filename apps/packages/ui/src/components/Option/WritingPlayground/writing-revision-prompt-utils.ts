import type {
  WritingRevisionAction,
  WritingRevisionOperation,
  WritingRevisionPresetId,
  WritingRevisionProposal,
  WritingRevisionTarget
} from "./writing-revision-types"

type WritingRevisionPromptContext = {
  selectedTemplateName?: string | null
  selectedThemeName?: string | null
  chatMode: boolean
  contextComposedPrompt?: string | null
  contextMessages?: Array<{ role: string; content: string }> | null
  memoryBlock?: unknown
  authorNote?: unknown
  worldInfoEntries?: unknown[]
  provider?: string | null
  model?: string | null
  generationSettingsSummary: Record<string, unknown>
}

type BuildRevisionUserPromptInput = {
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  instruction: string
  documentText: string
  target: WritingRevisionTarget
  presetInstruction?: string | null
  writingContext: WritingRevisionPromptContext
}

type ParseRevisionModelResponseInput = {
  responseText: string
  sessionId: string
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  instruction: string
  target: WritingRevisionTarget
  presetId?: WritingRevisionPresetId | null
  presetInstruction?: string | null
  createdAt?: string
  id?: string
}

type ParsedRevisionResponse = {
  title?: string
  replacement?: string
  rawText?: string
  rationale?: string
  notes?: string[]
}

const createRevisionProposalId = () =>
  globalThis.crypto?.randomUUID?.() ??
  `revision-${Date.now()}-${Math.random().toString(36).slice(2)}`

const getString = (value: unknown) =>
  typeof value === "string" ? value : undefined

const getNotes = (value: unknown) =>
  Array.isArray(value)
    ? value.filter((note): note is string => typeof note === "string")
    : undefined

const parseJsonObject = (responseText: string): Record<string, unknown> | null => {
  try {
    const parsed = JSON.parse(responseText.trim())
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

const normalizeParsedResponse = (
  response: Record<string, unknown>
): ParsedRevisionResponse => ({
  title: getString(response.title),
  replacement: getString(response.replacement),
  rawText: getString(response.rawText),
  rationale: getString(response.rationale),
  notes: getNotes(response.notes)
})

const hasAdvisoryContent = (response: ParsedRevisionResponse) =>
  response.title !== undefined ||
  response.rawText !== undefined ||
  response.rationale !== undefined ||
  response.notes !== undefined

const createBaseProposal = (
  input: ParseRevisionModelResponseInput
): WritingRevisionProposal => {
  const proposal: WritingRevisionProposal = {
    id: input.id ?? createRevisionProposalId(),
    sessionId: input.sessionId,
    action: input.action,
    operation: input.operation,
    instruction: input.instruction,
    target: input.target,
    createdAt: input.createdAt ?? new Date().toISOString(),
    status: "raw_suggestion"
  }

  if (input.presetId !== undefined) {
    proposal.presetId = input.presetId
  }
  if (input.presetInstruction !== undefined) {
    proposal.presetInstruction = input.presetInstruction
  }

  return proposal
}

const copyParsedDisplayFields = (
  proposal: WritingRevisionProposal,
  response: ParsedRevisionResponse
) => {
  if (response.title !== undefined) {
    proposal.title = response.title
  }
  if (response.rationale !== undefined) {
    proposal.rationale = response.rationale
  }
  if (response.notes !== undefined) {
    proposal.notes = response.notes
  }
  if (response.rawText !== undefined) {
    proposal.rawText = response.rawText
  }
}

export const buildRevisionUserPrompt = (
  input: BuildRevisionUserPromptInput
): string => {
  return [
    "You are helping revise a creative writing document.",
    "Return only valid JSON. Do not include markdown fences.",
    `Action: ${input.action}`,
    `Operation: ${input.operation}`,
    `Instruction: ${input.instruction}`,
    input.presetInstruction ? `Workflow preset: ${input.presetInstruction}` : null,
    `Template: ${input.writingContext.selectedTemplateName || "(none)"}`,
    `Theme: ${input.writingContext.selectedThemeName || "(none)"}`,
    `Chat mode: ${input.writingContext.chatMode ? "enabled" : "disabled"}`,
    `Provider: ${input.writingContext.provider || "(default)"}`,
    `Model: ${input.writingContext.model || "(unset)"}`,
    "Generation settings:",
    JSON.stringify(input.writingContext.generationSettingsSummary),
    "Composed writing context:",
    input.writingContext.contextComposedPrompt ||
      JSON.stringify(input.writingContext.contextMessages ?? []),
    "Memory / author note / world info:",
    JSON.stringify({
      memoryBlock: input.writingContext.memoryBlock,
      authorNote: input.writingContext.authorNote,
      worldInfoEntries: input.writingContext.worldInfoEntries ?? []
    }),
    "Target text:",
    input.target.beforeText || "(insertion point)",
    `Target summary: ${input.target.label}`,
    "Full document:",
    input.documentText,
    "JSON shape:",
    input.operation === "advisory"
      ? '{"title":"...","rawText":"...","rationale":"...","notes":["..."]}'
      : '{"title":"...","replacement":"...","rationale":"...","notes":["..."]}'
  ]
    .filter(Boolean)
    .join("\n\n")
}

export const parseRevisionModelResponse = (
  input: ParseRevisionModelResponseInput
): WritingRevisionProposal => {
  const proposal = createBaseProposal(input)
  const parsedJson = parseJsonObject(input.responseText)

  if (!parsedJson) {
    proposal.rawText = input.responseText
    return proposal
  }

  const parsed = normalizeParsedResponse(parsedJson)
  copyParsedDisplayFields(proposal, parsed)

  if (input.operation === "advisory") {
    if (hasAdvisoryContent(parsed)) {
      proposal.status = "advisory"
      return proposal
    }

    proposal.rawText = input.responseText
    return proposal
  }

  if (parsed.replacement !== undefined) {
    proposal.replacementText = parsed.replacement
    proposal.status = "pending"
    return proposal
  }

  proposal.rawText = parsed.rawText ?? input.responseText
  return proposal
}
