import { z } from "zod"

const revision = z.number().int().min(1).max(Number.MAX_SAFE_INTEGER)
const messageId = z.string().min(1).max(255)
const digest = z.string().regex(/^sha256:[0-9a-f]{64}$/)
const tail = z
  .object({
    message_id: messageId.nullable(),
    message_version: revision.nullable()
  })
  .strict()
  .refine(
    (value) => (value.message_id === null) === (value.message_version === null)
  )
const snapshot = z.object({
  status: z.enum(["valid", "missing", "invalid"]),
  schema_version: revision.nullable().optional(),
  digest: digest.nullable().optional()
})
const completionDisplay = z.object({
  provider: z.string().min(1).max(128),
  model: z.string().min(1).max(512)
})
const fence = z
  .object({
    version: z.literal(1),
    snapshot_digest: digest,
    settings_version: revision,
    history_version: revision,
    tail
  })
  .strict()
const textInput = z
  .string()
  .min(1)
  .max(2_000_000)
  .refine((value) => {
    let length = 0
    for (const character of value) {
      if (character && ++length > 1_000_000) return false
    }
    return true
  })
const appendInput = z
  .object({ message_id: messageId, content: textInput, expected: fence })
  .strict()
const appendReceipt = z
  .object({
    version: z.literal(1),
    conversation_id: messageId,
    message_id: messageId,
    message_version: revision,
    parent_message_id: messageId.nullable(),
    fence
  })
  .strict()
const completionInput = z
  .object({
    version: z.literal(1),
    input_message_id: messageId,
    input_message_version: revision,
    assistant_message_id: messageId,
    expected: fence
  })
  .strict()
const completionPayload = z
  .object({
    tldw_roleplay_resume_v1: completionInput,
    stream: z.literal(false).optional(),
    save_to_db: z.literal(true).optional(),
    include_character_context: z.literal(true).optional(),
    append_user_message: z.union([z.literal(false), z.null()]).optional()
  })
  .strict()
const completionResult = completionInput
  .extend({
    assistant_message_version: revision.nullable(),
    fence: fence.nullable(),
    unsaved_reason: z.string().min(1).max(100).nullable()
  })
  .strict()
const completionReceipt = z.object({
  chat_id: messageId,
  character_id: revision,
  provider: z.string().min(1).max(128),
  model: z.string().min(1).max(512).nullable(),
  saved: z.boolean(),
  user_message_id: messageId,
  assistant_message_id: messageId.nullable(),
  assistant_content: z.string().min(1),
  tldw_roleplay_resume_v1: completionResult
})
const intent = z
  .object({
    version: z.literal(1),
    conversation_id: messageId,
    input: appendInput,
    assistant_message_id: messageId
  })
  .strict()

export type RoleplayResumeFence = z.infer<typeof fence>
export type RoleplayResumeUserAppendRequest = z.infer<typeof appendInput>
export type RoleplayResumeUserAppendReceipt = z.infer<typeof appendReceipt>
export type RoleplayResumeCompletionRequest = z.infer<typeof completionPayload>
export type RoleplayResumeCompletionReceipt = z.infer<typeof completionReceipt>
export type RoleplayResumeIntent = z.infer<typeof intent>
export type RoleplayResumeRecovery = {
  intent: RoleplayResumeIntent
  append?: RoleplayResumeUserAppendReceipt
  completion?: RoleplayResumeCompletionReceipt
}
export type RoleplayResumeMetadata = {
  resume_completion?: z.infer<typeof completionDisplay> | null
  behavior_snapshot?: z.infer<typeof snapshot>
  resume_eligible?: boolean
  resume_ineligible_reason?: string | null
  settings_version?: number | null
  tail?: RoleplayResumeFence["tail"]
}

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
const invalid = (): never => {
  throw new Error("roleplay_resume_receipt_invalid")
}
export class RoleplayResumeResponseError extends Error {
  readonly generatedText: string
  constructor(value: unknown) {
    super("roleplay_resume_receipt_invalid")
    this.name = "RoleplayResumeResponseError"
    const text = record(value).assistant_content
    this.generatedText = typeof text === "string" ? text : ""
  }
}
export const sameRoleplayFence = (
  left: RoleplayResumeFence,
  right: RoleplayResumeFence
): boolean =>
  left.version === right.version &&
  left.snapshot_digest === right.snapshot_digest &&
  left.settings_version === right.settings_version &&
  left.history_version === right.history_version &&
  left.tail.message_id === right.tail.message_id &&
  left.tail.message_version === right.tail.message_version

export const readRoleplayResumeMetadata = (
  value: unknown
): RoleplayResumeMetadata => {
  const raw = record(value)
  const behavior = snapshot.safeParse(raw.behavior_snapshot)
  const currentTail = tail.safeParse(raw.tail)
  const settings = revision.nullable().safeParse(raw.settings_version)
  const completion = completionDisplay
    .nullable()
    .safeParse(raw.resume_completion)
  return {
    ...(completion.success ? { resume_completion: completion.data } : {}),
    ...(behavior.success ? { behavior_snapshot: behavior.data } : {}),
    ...(typeof raw.resume_eligible === "boolean"
      ? { resume_eligible: raw.resume_eligible }
      : {}),
    ...(raw.resume_ineligible_reason === null ||
    typeof raw.resume_ineligible_reason === "string"
      ? {
          resume_ineligible_reason:
            typeof raw.resume_ineligible_reason === "string"
              ? raw.resume_ineligible_reason
              : null
        }
      : {}),
    ...(settings.success ? { settings_version: settings.data } : {}),
    ...(currentTail.success ? { tail: currentTail.data } : {})
  }
}

/** Raw detail is checked before permissive legacy summary normalization. */
export const requireRoleplayResumeDetail = (
  conversationId: string,
  value: unknown
): RoleplayResumeFence => {
  const raw = record(value)
  const behavior = snapshot.safeParse(raw.behavior_snapshot)
  if (
    raw.id !== conversationId ||
    raw.scope_type !== "global" ||
    raw.workspace_id !== null ||
    raw.resume_eligible !== true ||
    raw.resume_ineligible_reason !== null ||
    !behavior.success ||
    behavior.data.status !== "valid" ||
    behavior.data.schema_version !== 1 ||
    !behavior.data.digest ||
    typeof raw.history_version !== "number" ||
    raw.history_version >= Number.MAX_SAFE_INTEGER - 1
  )
    throw new Error("roleplay_resume_unavailable")
  const parsed = fence.safeParse({
    version: 1,
    snapshot_digest: behavior.data.digest,
    settings_version: raw.settings_version,
    history_version: raw.history_version,
    tail: raw.tail
  })
  if (!parsed.success) throw new Error("roleplay_resume_unavailable")
  return parsed.data
}

export const parseRoleplayResumeIntent = (
  value: unknown
): RoleplayResumeIntent => {
  const parsed = intent.parse(value)
  if (
    parsed.input.message_id === parsed.assistant_message_id ||
    parsed.input.expected.tail.message_id === parsed.input.message_id ||
    parsed.input.expected.tail.message_id === parsed.assistant_message_id ||
    parsed.input.expected.history_version >= Number.MAX_SAFE_INTEGER - 1
  )
    invalid()
  return parsed
}
export const parseRoleplayResumeAppendRequest = (
  value: unknown
): RoleplayResumeUserAppendRequest => {
  const parsed = appendInput.parse(value)
  if (
    parsed.expected.history_version >= Number.MAX_SAFE_INTEGER - 1 ||
    parsed.message_id === parsed.expected.tail.message_id
  )
    invalid()
  return parsed
}
export const parseRoleplayResumeAppendReceipt = (
  conversationId: string,
  input: RoleplayResumeUserAppendRequest,
  value: unknown
): RoleplayResumeUserAppendReceipt => {
  const parsed = appendReceipt.safeParse(value)
  if (!parsed.success) invalid()
  const result = parsed.data
  if (
    result.conversation_id !== conversationId ||
    result.message_id !== input.message_id ||
    result.message_version !== 1 ||
    result.parent_message_id !== input.expected.tail.message_id ||
    !sameRoleplayFence(result.fence, {
      ...input.expected,
      history_version: input.expected.history_version + 1,
      tail: { message_id: input.message_id, message_version: 1 }
    })
  )
    invalid()
  return result
}
export const parseRoleplayResumeCompletionRequest = (
  value: unknown
): RoleplayResumeCompletionRequest => {
  const parsed = completionPayload.parse(value)
  const request = parsed.tldw_roleplay_resume_v1
  if (
    request.input_message_id === request.assistant_message_id ||
    request.input_message_version !== 1 ||
    request.expected.tail.message_id !== request.input_message_id ||
    request.expected.tail.message_version !== 1 ||
    request.expected.history_version >= Number.MAX_SAFE_INTEGER
  )
    invalid()
  return parsed
}
export const buildRoleplayResumeCompletion = (
  original: RoleplayResumeIntent,
  admitted: RoleplayResumeUserAppendReceipt
): RoleplayResumeCompletionRequest => {
  const accepted = parseRoleplayResumeAppendReceipt(
    original.conversation_id,
    original.input,
    admitted
  )
  return parseRoleplayResumeCompletionRequest({
    tldw_roleplay_resume_v1: {
      version: 1,
      input_message_id: accepted.message_id,
      input_message_version: accepted.message_version,
      assistant_message_id: original.assistant_message_id,
      expected: accepted.fence
    },
    stream: false,
    save_to_db: true,
    include_character_context: true,
    append_user_message: false
  })
}
export const parseRoleplayResumeCompletionReceipt = (
  conversationId: string,
  payload: RoleplayResumeCompletionRequest,
  value: unknown
): RoleplayResumeCompletionReceipt => {
  const parsed = completionReceipt.safeParse(value)
  if (!parsed.success) throw new RoleplayResumeResponseError(value)
  const result = parsed.data
  const expected = payload.tldw_roleplay_resume_v1
  const receipt = result.tldw_roleplay_resume_v1
  if (
    result.chat_id !== conversationId ||
    result.user_message_id !== expected.input_message_id ||
    receipt.input_message_id !== expected.input_message_id ||
    receipt.input_message_version !== 1 ||
    receipt.assistant_message_id !== expected.assistant_message_id ||
    !sameRoleplayFence(receipt.expected, expected.expected) ||
    (result.saved
      ? result.assistant_message_id !== expected.assistant_message_id ||
        receipt.assistant_message_version !== 1 ||
        receipt.unsaved_reason !== null ||
        !receipt.fence ||
        !sameRoleplayFence(receipt.fence, {
          ...expected.expected,
          history_version: expected.expected.history_version + 1,
          tail: {
            message_id: expected.assistant_message_id,
            message_version: 1
          }
        })
      : result.assistant_message_id !== null ||
        receipt.assistant_message_version !== null ||
        receipt.fence !== null ||
        !receipt.unsaved_reason)
  )
    throw new RoleplayResumeResponseError(value)
  return result
}

/** Only this structured pre-admission refusal proves that no input was added. */
export const isDefinitiveRoleplayResumeRefusal = (error: unknown): boolean => {
  const response = record(error)
  const detail = record(record(response.details).detail ?? response.detail)
  return (
    response.status === 409 &&
    detail.status === "resume_conflict" &&
    typeof detail.code === "string"
  )
}

export const parseRoleplayResumeRecovery = (
  value: unknown
): RoleplayResumeRecovery => {
  const raw = record(value)
  const original = parseRoleplayResumeIntent(raw.intent)
  const admitted =
    raw.append === undefined
      ? undefined
      : parseRoleplayResumeAppendReceipt(
          original.conversation_id,
          original.input,
          raw.append
        )
  if (raw.completion !== undefined && !admitted) invalid()
  const completed =
    raw.completion === undefined
      ? undefined
      : parseRoleplayResumeCompletionReceipt(
          original.conversation_id,
          buildRoleplayResumeCompletion(original, admitted!),
          raw.completion
        )
  return {
    intent: original,
    ...(admitted ? { append: admitted } : {}),
    ...(completed ? { completion: completed } : {})
  }
}

export type SavedRoleplayPresentation = {
  id: string
  name: string
  provider?: string
  model?: string
}
/** Display identity is read from the saved detail, never fetched from the current card. */
export const readSavedRoleplayPresentation = (
  value: unknown
): SavedRoleplayPresentation | null => {
  const raw = record(value)
  const behavior = snapshot.safeParse(raw.behavior_snapshot)
  const id = raw.assistant_id ?? raw.character_id
  const name = raw.assistant_name ?? raw.character_name
  if (
    !behavior.success ||
    behavior.data.status !== "valid" ||
    behavior.data.schema_version !== 1 ||
    !behavior.data.digest ||
    raw.assistant_kind !== "character" ||
    (typeof id !== "string" && typeof id !== "number") ||
    !String(id).length ||
    String(id).length > 255 ||
    typeof name !== "string" ||
    !name.trim() ||
    name.length > 512
  )
    return null
  const completion = completionDisplay.safeParse(raw.resume_completion)
  return {
    id: String(id),
    name,
    ...(completion.success ? completion.data : {})
  }
}
