import { z } from "zod"
import { canonicalHistoryJson, historyDigest } from "@/db/dexie/history-selection"
import { HistorySelectionError, selectionDigest } from "@/utils/history-selection"
import type { HistoryAdmissionReferenceV1, HistoryAdmissionV1, HistorySelectionV1, HistorySelectionCaptureV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import { bgRequest } from "./background-proxy"
import { requestScopeFields } from "./tldw/domains/service-prompts"
import { buildQuery } from "./tldw/client-utils"
import { toChatScopeParams } from "@/types/chat-scope"
import type {
  HistoryDurableEnvelopeV1, HistoryDurableRequestBodyV1, PreparedHistoryDurableTurnV1
} from "@/types/history-durable-turn"
import {
  finalizeHistorySelection, historyAdmissionReference, prepareHistoryContext,
  type NativeHistoryOwnerV1
} from "./chat-history-selection"
import { parseHistoryDurableResult, validateHistoryDurableResultReceipt } from "@/utils/history-durable-sources"
export {
  parseHistoryDurableResult, projectHistoryDurableSources, historyResultV1ToMessageSources,
  validateHistoryDurableResultReceipt
} from "@/utils/history-durable-sources"

const fail = (code = "invalid_history_durable_result"): never => {
  throw new HistorySelectionError(code)
}
const freeze = <T>(value: T): T => {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze)
    Object.freeze(value)
  }
  return value
}
const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) return fail()
  return value as Record<string, unknown>
}
const onlyKeys = (value: Record<string, unknown>, keys: readonly string[]) => {
  if (Object.keys(value).some(key => !keys.includes(key))) fail()
}
const bytes = (value: string) => new TextEncoder().encode(value).length
const scalarString = (value: string) => !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)
const text = (limit: number, allowEmpty = false) => z.string().refine(value =>
  scalarString(value) && bytes(value) <= limit && (allowEmpty || value.trim().length > 0))
const safeInt = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER)
const hex64 = z.string().regex(/^[0-9a-f]{64}$/)
const uuidSchema = z.guid()
const nonblank = z.string().min(1)
const revisionSchema = z.object({ id: nonblank, revision: nonblank }).strict()
const selectionSchema = z.object({
  version: z.literal(1), owner_key: nonblank, conversation_id: nonblank, purpose: z.literal("send"),
  interpretation: z.union([z.object({ kind: z.literal("parent_graph_v1") }).strict(),
    z.object({ kind: z.literal("legacy_linear_v1"), projection_id: nonblank }).strict()]),
  cursor: z.union([z.object({ kind: z.literal("empty") }).strict(),
    z.object({ kind: z.enum(["before_message", "after_message"]), message_id: nonblank }).strict()]),
  selection_revision: safeInt, messages: z.array(revisionSchema),
  fences: z.object({ conversation: z.string(), history: z.string(), settings: z.string() }).strict(),
  storage_context_digest: nonblank, request_context_digest: hex64, selection_digest: hex64
}).strict()
const parseSelection = (value: unknown): HistorySelectionV1 => {
  const parsed = selectionSchema.safeParse(value)
  if (!parsed.success) return fail("invalid_history_selection")
  const selection = parsed.data as HistorySelectionV1
  if (selectionDigest(selection) !== selection.selection_digest ||
      new Set(selection.messages.map(row => row.id)).size !== selection.messages.length) return fail("invalid_history_selection")
  return selection
}

/** H1 reference parsing plus the full manifest and originating-selection binding. */
export const validateHistoryDurableAdmission = (
  owner: Pick<HistoryAdmissionReferenceV1, "owner_key" | "conversation_id">,
  history: HistoryDurableEnvelopeV1, inputId: string, value: unknown
): HistoryAdmissionV1 => {
  const raw = record(value)
  onlyKeys(raw, ["version", "owner_key", "conversation_id", "input_message_id", "input_message_revision",
    "selection_digest", "messages", "originating_selection_revision"])
  const reference = historyAdmissionReference(raw as HistoryAdmissionReferenceV1)
  const messages = z.array(revisionSchema).safeParse(raw.messages)
  const originatingRevision = safeInt.safeParse(raw.originating_selection_revision)
  if (!messages.success || !originatingRevision.success || !uuidSchema.safeParse(inputId).success ||
      reference.input_message_id !== inputId || reference.input_message_revision !== "1" ||
      reference.owner_key !== owner.owner_key || reference.conversation_id !== owner.conversation_id ||
      !hex64.safeParse(reference.selection_digest).success) return fail("invalid_history_admission")
  const envelope = record(history)
  if (envelope.version !== 1) return fail("invalid_history_admission")
  if (history.kind === "selection") {
    onlyKeys(envelope, ["version", "kind", "selection"])
    const selection = parseSelection(history.selection)
    if (selection.owner_key !== owner.owner_key || selection.conversation_id !== owner.conversation_id ||
        reference.selection_digest !== selection.selection_digest ||
        originatingRevision.data !== selection.selection_revision ||
        canonicalHistoryJson(messages.data) !== canonicalHistoryJson(selection.messages)) return fail("invalid_history_admission")
  } else if (history.kind === "admission") {
    onlyKeys(envelope, ["version", "kind", "admission", "request_context_digest"])
    const expectedRaw = record(history.admission)
    onlyKeys(expectedRaw, ["version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest"])
    if (!hex64.safeParse(history.request_context_digest).success ||
        canonicalHistoryJson(historyAdmissionReference(history.admission)) !== canonicalHistoryJson(reference))
      return fail("invalid_history_admission")
  } else return fail("invalid_history_admission")
  return freeze({ ...reference, messages: messages.data, originating_selection_revision: originatingRevision.data })
}

/** Body-only projection. No scope/header synthesis, default insertion or stream rewrite. */
const requestProjection = (value: unknown): Record<string, unknown> => {
  const body = record(value)
  const turn = record(body.tldw_turn)
  const projectedTurn = { ...turn }
  delete projectedTurn.history_v1
  return { ...body, tldw_turn: projectedTurn }
}
export const historyDurableRequestDigest = (body: unknown): string => historyDigest(requestProjection(body))

export type PrepareHistoryDurableTurnOptions = {
  readonly owner: NativeHistoryOwnerV1
  readonly mode: "plain" | "rag"
  readonly validate_lease: () => boolean
  readonly history:
    | { readonly kind: "selection"; readonly capture: HistorySelectionCaptureV1; readonly current_view: HistoryViewSelectionV1 }
    | { readonly kind: "admission"; readonly admission: HistoryAdmissionReferenceV1 }
}

/** Finalize only the excluded H1 envelope; dispatch body, never the mutable input. */
export const prepareHistoryDurableTurn = (
  value: unknown, options: PrepareHistoryDurableTurnOptions
): PreparedHistoryDurableTurnV1 => {
  const body = record(value)
  const turn = record(body.tldw_turn)
  onlyKeys(turn, ["user_message_id", "result_v1"])
  const owner = options.owner
  if (owner.kind !== "native" || !owner.validate_lease() || !options.validate_lease())
    return fail("request_config_scope_changed")
  if (!uuidSchema.safeParse(turn.user_message_id).success || body.save_to_db !== true ||
      typeof body.stream !== "boolean" || !text(Number.MAX_SAFE_INTEGER).safeParse(body.model).success ||
      !text(Number.MAX_SAFE_INTEGER).safeParse(body.api_provider).success ||
      body.conversation_id !== owner.conversation_id || !owner.owner_key)
    return fail("invalid_history_durable_request")
  for (const key of ["tldw_history_selection_v1", "tldw_continuation", "tools", "functions", "tool_choice", "function_call", "rag_context"]) {
    if (body[key] !== undefined && body[key] !== null) return fail("invalid_history_durable_request")
  }
  const metadata = body.metadata === undefined || body.metadata === null ? {} : record(body.metadata)
  if (metadata.tldw_retry_failed_turn === true || metadata.tldw_regenerate_from_message_id != null)
    return fail("invalid_history_durable_request")
  if (body.extra_body !== undefined && body.extra_body !== null) {
    const extra = record(body.extra_body)
    if (Object.keys(extra).some(key => key.startsWith("tldw_") || ["stream", "model", "api_provider", "messages",
      "conversation_id", "save_to_db", "metadata", "tools", "functions", "tool_choice", "function_call", "rag_context"].includes(key)))
      return fail("invalid_history_durable_request")
  }
  const messageSchema = z.object({ role: z.enum(["system", "user"]), content: text(Number.MAX_SAFE_INTEGER) }).strict()
  const messages = z.array(messageSchema).safeParse(body.messages)
  if (!messages.success || messages.data.filter(message => message.role === "user").length !== 1)
    return fail("invalid_history_durable_request")
  const result = parseHistoryDurableResult(turn.result_v1, options.mode)
  const prepared = prepareHistoryContext({ ...body, tldw_turn: { ...turn, result_v1: result } }, options.validate_lease)
  let envelope: HistoryDurableEnvelopeV1
  if (options.history.kind === "selection") {
    if (options.history.capture.purpose !== "send") return fail("invalid_history_durable_request")
    const selection = finalizeHistorySelection(owner, options.history.capture, prepared, options.history.current_view)
    envelope = { version: 1, kind: "selection", selection }
  } else if (options.history.kind === "admission") {
    const raw = record(options.history.admission)
    onlyKeys(raw, ["version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest"])
    const admission = historyAdmissionReference(options.history.admission)
    if (admission.owner_key !== owner.owner_key || admission.conversation_id !== owner.conversation_id ||
        admission.input_message_id !== turn.user_message_id || !uuidSchema.safeParse(admission.input_message_id).success ||
        !hex64.safeParse(admission.selection_digest).success) return fail("invalid_history_admission")
    envelope = { version: 1, kind: "admission", admission, request_context_digest: prepared.request_context_digest }
  } else return fail("invalid_history_durable_request")
  if (!owner.validate_lease() || !prepared.validate_lease()) return fail("request_config_scope_changed")
  const projection = record(prepared.payload)
  const finalBody = freeze({ ...projection, tldw_turn: { ...record(projection.tldw_turn), history_v1: envelope } }) as HistoryDurableRequestBodyV1
  return Object.freeze({ body: finalBody, prepared, request_context_digest: prepared.request_context_digest })
}

/** Exact protected reads only. No enumeration, public metadata adoption or ledger writes. */
export const inspectHistoryDurableRecovery = async (
  owner: NativeHistoryOwnerV1, turn: HistoryTurnRecovery, signal?: AbortSignal
): Promise<HistoryTurnRecovery> => {
  const lease = () => {
    if (signal?.aborted || owner.kind !== "native" || !owner.owner_key || !owner.validate_lease())
      fail("request_config_scope_changed")
  }
  lease()
  if (turn.persistence !== "server" || turn.owner_key !== owner.owner_key || turn.conversation_id !== owner.conversation_id ||
      !uuidSchema.safeParse(turn.logical_user_message_id).success || !hex64.safeParse(turn.request_context_digest).success ||
      turn.input_images.length !== 0 || !turn.finalized_selection) return fail("invalid_history_recovery")
  const selection = parseSelection(turn.finalized_selection)
  if (selection.owner_key !== owner.owner_key || selection.conversation_id !== owner.conversation_id ||
      selection.selection_digest !== turn.selection_digest) return fail("invalid_history_recovery")
  const binding = { owner_key: owner.owner_key!, conversation_id: owner.conversation_id }
  const inputId = turn.logical_user_message_id!
  if (turn.input_id && turn.input_id !== inputId) return fail("invalid_history_recovery")
  const expectedScope = owner.scope?.type === "workspace"
    ? { scope_type: "workspace", workspace_id: owner.scope.workspaceId }
    : { scope_type: "global", workspace_id: null }
  const read = async (id: string) => {
    lease()
    const response = record(await bgRequest<unknown>({
      path: `/api/v1/messages/${id}${buildQuery({ ...toChatScopeParams(owner.scope), include_history_recovery_v1: true })}`,
      method: "GET", ...requestScopeFields(owner.request_scope), abortSignal: signal
    }))
    lease()
    if (response.id !== id || response.conversation_id !== owner.conversation_id || response.version !== 1)
      return fail("invalid_history_recovery")
    const proof = record(response.tldw_history_recovery_v1)
    if (proof.version !== 1 || canonicalHistoryJson(proof.scope) !== canonicalHistoryJson(expectedScope))
      return fail("unverified_history_recovery")
    return { response, proof }
  }
  let admission: HistoryAdmissionReferenceV1 | undefined = turn.admission
  if (admission) {
    onlyKeys(record(admission), ["version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest"])
    admission = historyAdmissionReference(admission)
    if (admission.owner_key !== owner.owner_key || admission.conversation_id !== owner.conversation_id ||
        admission.input_message_id !== inputId || admission.input_message_revision !== "1" ||
        admission.selection_digest !== selection.selection_digest || (turn.input_id && turn.input_id !== inputId))
      return fail("invalid_history_recovery")
  }
  if (!admission || !turn.observed_result) {
    const { response, proof } = await read(inputId)
    onlyKeys(proof, ["version", "status", "scope", "admission"])
    if (proof.status !== "input_verified" || response.sender !== "user" || response.content !== turn.input_text)
      return fail("unverified_history_recovery")
    const verified = validateHistoryDurableAdmission(binding, { version: 1, kind: "selection", selection }, inputId, proof.admission)
    const reference = historyAdmissionReference(verified)
    if (admission && canonicalHistoryJson(admission) !== canonicalHistoryJson(reference)) return fail("invalid_history_recovery")
    admission = reference
  }
  if (!turn.observed_result) return { ...turn, admission, input_id: inputId }
  const observed = validateHistoryDurableResultReceipt(binding, admission!, turn.request_context_digest,
    turn.observed_result.sources, turn.observed_result)
  if (turn.assistant_id && turn.assistant_id !== observed.result_message_id) return fail("invalid_history_recovery")
  const { response, proof } = await read(observed.result_message_id)
  onlyKeys(proof, ["version", "status", "scope", "result"])
  if (proof.status !== "result_verified" || response.sender !== "assistant" || response.parent_message_id !== inputId)
    return fail("unverified_history_recovery")
  const result = validateHistoryDurableResultReceipt(binding, admission!, turn.request_context_digest, observed.sources, proof.result)
  if (result.result_message_id !== observed.result_message_id) return fail("invalid_history_recovery")
  return { ...turn, admission, input_id: inputId, observed_result: result, assistant_id: result.result_message_id }
}
