/** Owner selection is separate from workspace routing, view identity and inference connection. */
import { z } from "zod"
import type { ChatScope } from "@/types/chat-scope"
import type {
  HistoryAdmissionReferenceV1,
  HistoryAdmissionV1,
  HistoryCaptureRequestV1,
  HistoryCaptureResultV1,
  HistorySelectionCaptureV1,
  HistorySelectionV1,
  HistoryViewSelectionV1,
  LegacyHistoryProjectionConfirmV1,
  LegacyHistoryProjectionV1,
  PreparedHistoryContextV1
} from "@/types/history-selection"
import type { HistoryBookmarkScope, Message } from "@/db/dexie/types"
import type { ServicePromptRequestScope } from "./tldw/domains/service-prompts"
import { tldwClient } from "./tldw/TldwApiClient"
import {
  acknowledgeHistoryConfirmation,
  appendLocalSelectedUser,
  canonicalHistoryJson,
  captureLocalHistorySnapshot,
  confirmLocalHistoryProjection,
  historyDigest,
  savePendingHistoryConfirmation,
  rejectPendingHistoryConfirmation,
  markPendingHistoryConfirmationDispatched,
  settleLocalAcceptedAssistant,
  type HistoryOperationOptions,
  type LocalHistoryOwnerV1
} from "@/db/dexie/history-selection"
import {
  bindSelectedHistoryContent,
  HistorySelectionError,
  resolveHistorySelection,
  selectionDigest
} from "@/utils/history-selection"

export type NativeHistoryOwnerV1 = {
  readonly kind: "native"
  readonly conversation_id: string
  readonly owner_key?: string
  readonly request_scope: ServicePromptRequestScope
  readonly scope?: ChatScope
  readonly validate_lease: () => boolean
}
export type HistoryOwnerV1 =
  | LocalHistoryOwnerV1
  | NativeHistoryOwnerV1
  | { readonly kind: "unavailable"; readonly code: string }
const fail = (code: string): never => {
  throw new HistorySelectionError(code)
}
const freeze = <T>(value: T): T => {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze)
    Object.freeze(value)
  }
  return value
}
const str = z.string().min(1)
const revision = z.object({ id: str, revision: str }).strict()
const fences = z
  .object({
    conversation: z.string(),
    history: z.string(),
    settings: z.string()
  })
  .strict()
const cursor = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("empty") }).strict(),
  z.object({ kind: z.literal("after_message"), message_id: str }).strict(),
  z.object({ kind: z.literal("before_message"), message_id: str }).strict()
])
const parent = z.object({ kind: z.literal("parent_graph_v1") }).strict()
const legacy = z
  .object({ kind: z.literal("legacy_linear_v1"), projection_id: str })
  .strict()
const interpretation = z.union([parent, legacy])
const viewSchema = z
  .object({
    view_session_id: str,
    owner_key: str,
    conversation_id: str,
    interpretation,
    cursor,
    selection_revision: z.number().int().nonnegative()
  })
  .strict()
const reference = z
  .object({ id: z.string(), revision: z.string(), kind: z.string() })
  .strict()
const node = revision
  .extend({
    preview: z.string().nullable().optional(),
    legacy_projection_id: z.string().nullable().optional(),
    parent_id: z.string().nullable(),
    role: str,
    settled: z.boolean(),
    conversation_id: z.string().nullable().optional(),
    metadata: z.array(reference).optional(),
    assets: z.array(reference).optional(),
    comparison: z
      .object({
        cluster_id: z.string(),
        model_id: z.string().nullable(),
        common: z.boolean()
      })
      .strict()
      .nullable()
      .optional()
  })
  .strict()
const snapshot = z
  .object({
    version: z.literal(1),
    owner_key: str,
    conversation_id: str,
    fences,
    nodes: z.array(node),
    source_digest: str,
    storage_context_digest: str,
    interpretation_status: z.union([
      parent,
      legacy.extend({ ordered_path_ids: z.array(str) }),
      z.object({ kind: z.literal("legacy_review_required") }).strict()
    ])
  })
  .strict()
const content = revision
  .extend({
    message: z.string(),
    images: z.array(z.string()),
    tool_calls: z
      .array(z.record(z.string(), z.unknown()))
      .nullable()
      .optional(),
    extra_metadata: z.record(z.string(), z.unknown()).nullable().optional()
  })
  .strict()
const capturedSchema = z
  .object({
    status: z.literal("captured"),
    snapshot,
    rows: z.array(node),
    selected_content: z.array(content),
    view: viewSchema,
    purpose: z.enum(["send", "fork"]),
    storage_context_digest: str
  })
  .strict()
const failureSchema = z
  .object({
    status: z.enum([
      "legacy_review_required",
      "stale_selection",
      "invalid_history",
      "unsupported_history_capability"
    ]),
    code: str,
    snapshot,
    view: viewSchema
  })
  .strict()
const captureSchema = z.union([capturedSchema, failureSchema])
const confirmationSchema = z
  .object({
    version: z.literal(1),
    projection_id: str,
    owner_key: str,
    conversation_id: str,
    source_digest: str,
    fences,
    source_members: z.array(revision),
    ordered_path_ids: z.array(str),
    cursor,
    selection_revision: z.number().int().nonnegative()
  })
  .strict()
const projectionSchema = confirmationSchema
  .extend({ projection_digest: str, created_at: str })
  .strict()
const admissionReferenceSchema = z
  .object({
    version: z.literal(1),
    owner_key: str,
    conversation_id: str,
    input_message_id: str,
    input_message_revision: str,
    selection_digest: str
  })
  .strict()
const admissionSchema = admissionReferenceSchema
  .extend({
    messages: z.array(revision),
    originating_selection_revision: z.number().int().nonnegative()
  })
  .strict()
const parse = <T>(schema: z.ZodType<T>, value: unknown): T => {
  const r = schema.safeParse(value)
  return r.success ? r.data : fail("unsupported_history_response")
}
// A live captured owner object is the capability handshake. No durable identity registry.
const nativeCapabilities = new WeakMap<
  NativeHistoryOwnerV1,
  { owner_key: string; captured: boolean }
>()
const assertOwnerLease = (
  owner: HistoryOwnerV1,
  opts?: HistoryOperationOptions
) => {
  if (owner.kind === "unavailable") fail(owner.code)
  if (
    opts?.signal?.aborted ||
    opts?.validate_lease?.() === false ||
    (owner.kind === "native" && !owner.validate_lease())
  )
    fail("request_config_scope_changed")
}
const nativeOptions = (owner: NativeHistoryOwnerV1, signal?: AbortSignal) => ({
  requestScope: owner.request_scope,
  scope: owner.scope,
  signal
})
const assertNativeBinding = (
  owner: NativeHistoryOwnerV1,
  binding: { owner_key: string; conversation_id: string },
  requireCapture = false
) => {
  const capability = nativeCapabilities.get(owner)
  if (!capability || (requireCapture && !capability.captured))
    fail("unsupported_history_capability")
  if (
    binding.owner_key !== capability.owner_key ||
    (owner.owner_key && owner.owner_key !== binding.owner_key) ||
    binding.conversation_id !== owner.conversation_id
  )
    fail("owner_conversation_mismatch")
}

export const captureHistorySnapshot = async (
  owner: HistoryOwnerV1,
  view: HistoryCaptureRequestV1["view"],
  purpose: "send" | "fork",
  signal?: AbortSignal
): Promise<HistoryCaptureResultV1> => {
  assertOwnerLease(owner, { signal })
  if (owner.kind === "unavailable") return fail(owner.code)
  if (owner.kind === "local") {
    if (!view.owner_key) fail("owner_conversation_mismatch")
    return freeze(
      await captureLocalHistorySnapshot(
        owner,
        view as HistoryViewSelectionV1,
        purpose,
        { signal }
      )
    )
  }
  if (
    view.conversation_id !== owner.conversation_id ||
    (owner.owner_key && view.owner_key !== owner.owner_key)
  )
    fail("owner_conversation_mismatch")
  const request = structuredClone({ view, purpose })
  let response: unknown
  try {
    response = await tldwClient.captureHistorySelection(
      owner.conversation_id,
      request,
      nativeOptions(owner, signal)
    )
  } catch (error) {
    assertOwnerLease(owner, { signal })
    if ([404, 405, 501].includes(Number((error as any)?.status)))
      fail("unsupported_history_capability")
    throw error
  }
  assertOwnerLease(owner, { signal })
  const result: HistoryCaptureResultV1 = parse(
    captureSchema,
    response
  ) as HistoryCaptureResultV1
  if (
    result.snapshot.owner_key !== result.view.owner_key ||
    result.snapshot.conversation_id !== owner.conversation_id ||
    (view.owner_key && view.owner_key !== result.view.owner_key) ||
    canonicalHistoryJson({
      ...request.view,
      owner_key: result.view.owner_key
    }) !== canonicalHistoryJson(result.view)
  )
    fail("owner_conversation_mismatch")
  if (
    new Set(result.snapshot.nodes.map((n) => n.id)).size !==
    result.snapshot.nodes.length
  )
    fail("duplicate_message_id")
  if (result.status === "captured") {
    if (
      result.purpose !== purpose ||
      result.storage_context_digest !== result.snapshot.storage_context_digest
    )
      fail("selected_content_mismatch")
    const resolved = resolveHistorySelection(
      result.snapshot,
      result.view,
      purpose,
      ""
    )
    if (
      resolved.status !== "ready" ||
      canonicalHistoryJson(resolved.rows) !== canonicalHistoryJson(result.rows)
    )
      fail("selected_content_mismatch")
    bindSelectedHistoryContent(result.rows, result.selected_content)
  }
  const previous = nativeCapabilities.get(owner)
  if (previous && previous.owner_key !== result.view.owner_key)
    fail("owner_conversation_mismatch")
  nativeCapabilities.set(owner, {
    owner_key: result.view.owner_key,
    captured: result.status === "captured" || previous?.captured === true
  })
  return freeze(result)
}

/** Consumers dispatch this exact detached payload; never dispatch the original mutable input. */
export const prepareHistoryContext = (
  payload: unknown,
  validate_lease: () => boolean
): PreparedHistoryContextV1 => {
  const detached = JSON.parse(canonicalHistoryJson(payload))
  return Object.freeze({
    payload: freeze(detached),
    request_context_digest: historyDigest(detached),
    validate_lease
  })
}
export const finalizeHistorySelection = (
  owner: HistoryOwnerV1,
  capture: HistorySelectionCaptureV1,
  prepared: PreparedHistoryContextV1,
  currentView: HistoryViewSelectionV1
): HistorySelectionV1 => {
  assertOwnerLease(owner)
  if (owner.kind === "unavailable") return fail(owner.code)
  if (!prepared.validate_lease()) fail("request_config_scope_changed")
  if (canonicalHistoryJson(currentView) !== canonicalHistoryJson(capture.view))
    fail("stale_selection")
  if (prepared.request_context_digest !== historyDigest(prepared.payload))
    fail("request_context_mismatch")
  if (owner.kind === "native") assertNativeBinding(owner, capture.view, true)
  else if (
    owner.owner_key !== capture.view.owner_key ||
    owner.conversation_id !== capture.view.conversation_id
  )
    fail("owner_conversation_mismatch")
  const resolved = resolveHistorySelection(
    capture.snapshot,
    capture.view,
    capture.purpose,
    prepared.request_context_digest
  )
  if (resolved.status !== "ready") return fail(resolved.code)
  return freeze(resolved.selection)
}

// These exact native HistorySelectionError responses precede projection commit.
// Transport/scope/abort errors and unrecognized responses cannot prove non-commit.
const isDefinitiveProjectionRejection = (error: unknown): boolean => {
  const response = error as {
    status?: number
    details?: { detail?: { status?: string; code?: string } }
  } | null
  const detail = response?.details?.detail
  return (
    response?.status === 409 &&
    detail?.status === "stale_selection" &&
    ["stale_source", "invalid_projection", "projection_id_conflict"].includes(
      detail.code ?? ""
    )
  )
}

/** Retry uncertain outcomes with the same persisted confirmation ID. */
export const confirmLegacyHistoryProjection = async (
  owner: HistoryOwnerV1,
  scope: HistoryBookmarkScope,
  confirmation: LegacyHistoryProjectionConfirmV1,
  view: HistoryViewSelectionV1,
  signal?: AbortSignal
): Promise<LegacyHistoryProjectionV1> => {
  assertOwnerLease(owner, { signal })
  if (owner.kind === "unavailable") return fail(owner.code)
  const intent = parse(
    confirmationSchema,
    confirmation
  ) as LegacyHistoryProjectionConfirmV1
  if (
    intent.owner_key !== view.owner_key ||
    intent.conversation_id !== view.conversation_id ||
    intent.selection_revision !== view.selection_revision
  )
    fail("stale_selection")
  if (owner.kind === "local")
    return confirmLocalHistoryProjection(owner, scope, intent, { signal, view })
  assertNativeBinding(owner, intent)
  const pendingView = structuredClone(view)
  const origin = await savePendingHistoryConfirmation(
    scope,
    pendingView,
    intent
  )
  const pendingOrigin = { ...pendingView, view_session_id: origin }
  let dispatched = false
  let response: unknown
  try {
    assertOwnerLease(owner, { signal })
    const options = nativeOptions(owner, signal)
    await markPendingHistoryConfirmationDispatched(scope, pendingOrigin, intent)
    assertOwnerLease(owner, { signal })
    dispatched = true
    response = await tldwClient.confirmHistoryProjection(
      owner.conversation_id,
      intent,
      options
    )
  } catch (error) {
    if (dispatched && isDefinitiveProjectionRejection(error)) {
      await rejectPendingHistoryConfirmation(scope, pendingOrigin, intent)
    } else if (!dispatched) {
      await rejectPendingHistoryConfirmation(scope, pendingOrigin, intent, {
        onlyIfUndispatched: true
      })
    }
    throw error
  }
  assertOwnerLease(owner, { signal })
  const projection = parse(
    projectionSchema,
    response
  ) as LegacyHistoryProjectionV1
  const { projection_digest: _digest, created_at: _date, ...echo } = projection
  if (canonicalHistoryJson(echo) !== canonicalHistoryJson(intent))
    fail("projection_id_conflict")
  await acknowledgeHistoryConfirmation(scope, projection)
  return freeze(projection)
}

/** MessageCreate supports text and a single inline image; reject other lost state. */
export const nativeHistoryMessagePayload = (
  message: Message
): Record<string, unknown> => {
  const images = (message.images ?? []).filter((image) => image !== "")
  if (!images.length && message.image) images.push(message.image)
  if (
    !message.id ||
    !["user", "assistant"].includes(message.role) ||
    images.length > 1 ||
    message.documents ||
    message.sources?.length ||
    message.search ||
    message.discoSkillComment ||
    message.generationInfo ||
    message.clusterId ||
    message.modelId ||
    // Empty display strings are absent defaults; measured reasoning (including zero) is substantive.
    message.messageType ||
    message.modelName ||
    message.modelImage ||
    message.reasoning_time_taken != null ||
    (message.metadataExtra && Object.keys(message.metadataExtra).length)
  )
    fail("unsupported_message_payload")
  const image = images[0]
  const parsed = image
    ? /^data:([^;,]+);base64,([A-Za-z0-9+/]*={0,2})$/.exec(image)
    : undefined
  if (image !== undefined && !parsed) fail("unsupported_message_payload")
  if (parsed) {
    let bytes: string
    try {
      bytes = atob(parsed[2])
    } catch {
      return fail("unsupported_message_payload")
    }
    // Match the native MessageCreate magic-byte contract before accepting an input.
    const detected =
      bytes.length < 12
        ? null
        : bytes.startsWith("\x89PNG\r\n\x1a\n")
          ? "image/png"
          : bytes.startsWith("\xff\xd8\xff")
            ? "image/jpeg"
            : bytes.startsWith("GIF87a") || bytes.startsWith("GIF89a")
              ? "image/gif"
              : bytes.startsWith("RIFF") && bytes.slice(8, 12) === "WEBP"
                ? "image/webp"
                : bytes.startsWith("BM")
                  ? "image/bmp"
                  : bytes.startsWith("\x00\x00\x01\x00")
                    ? "image/x-icon"
                    : null
    if (!detected || detected !== parsed[1]) fail("unsupported_message_payload")
  }
  if (!message.content.trim() && !parsed) fail("unsupported_message_payload")
  return {
    id: message.id,
    role: message.role,
    ...(message.content ? { content: message.content } : {}),
    ...(message.parent_message_id !== undefined
      ? { parent_message_id: message.parent_message_id }
      : {}),
    ...(parsed ? { image_base64: image } : {})
  }
}
export const historyAdmissionReference = (
  admission: HistoryAdmissionReferenceV1
): HistoryAdmissionReferenceV1 =>
  parse(admissionReferenceSchema, {
    version: admission.version,
    owner_key: admission.owner_key,
    conversation_id: admission.conversation_id,
    input_message_id: admission.input_message_id,
    input_message_revision: admission.input_message_revision,
    selection_digest: admission.selection_digest
  })

/** Validate native admission without inventing a client-selected server input ID. */
export const parseNativeHistoryAdmission = (
  owner: NativeHistoryOwnerV1, selection: HistorySelectionV1, value: unknown
): HistoryAdmissionV1 => {
  const admission = parse(admissionSchema, value) as HistoryAdmissionV1
  assertNativeBinding(owner, admission, true)
  if (admission.selection_digest !== selection.selection_digest ||
      admission.originating_selection_revision !== selection.selection_revision ||
      canonicalHistoryJson(admission.messages) !== canonicalHistoryJson(selection.messages))
    fail("invalid_history_admission")
  return freeze(admission)
}

export const appendSelectedUser = async (
  owner: HistoryOwnerV1,
  selection: HistorySelectionV1,
  input: Message,
  opts?: HistoryOperationOptions
): Promise<HistoryAdmissionV1> => {
  selection = structuredClone(selection)
  input = structuredClone(input)
  assertOwnerLease(owner, opts)
  if (owner.kind === "unavailable") return fail(owner.code)
  if (owner.kind === "local")
    return appendLocalSelectedUser(owner, selection, input, opts)
  assertNativeBinding(owner, selection, true)
  if (
    input.history_id !== owner.conversation_id ||
    input.role !== "user" ||
    selection.purpose !== "send" ||
    selectionDigest(selection) !== selection.selection_digest
  )
    fail("invalid_selection")
  const body = {
    ...nativeHistoryMessagePayload(input),
    tldw_history_selection_v1: structuredClone(selection)
  }
  const response = await tldwClient.addChatMessage(
    owner.conversation_id,
    body,
    nativeOptions(owner, opts?.signal)
  )
  // Admission is already committed remotely. Preserve the receipt even if navigation changed.
  const admission = parse(
    admissionSchema,
    response.tldw_history_admission_v1
  ) as HistoryAdmissionV1
  assertNativeBinding(owner, admission, true)
  if (
    response.id !== input.id ||
    admission.input_message_id !== input.id ||
    admission.selection_digest !== selection.selection_digest ||
    admission.originating_selection_revision !== selection.selection_revision ||
    canonicalHistoryJson(admission.messages) !==
      canonicalHistoryJson(selection.messages)
  )
    fail("invalid_history_admission")
  return freeze(admission)
}
export const settleAcceptedAssistant = async (
  owner: HistoryOwnerV1,
  admission: HistoryAdmissionReferenceV1,
  input: Message,
  opts?: HistoryOperationOptions
): Promise<Message | { id: string }> => {
  admission = structuredClone(admission)
  input = structuredClone(input)
  assertOwnerLease(owner, opts)
  if (owner.kind === "unavailable") return fail(owner.code)
  if (owner.kind === "local")
    return settleLocalAcceptedAssistant(owner, admission, input, opts)
  // Every versioned write requires the live endpoint handshake, including settlement.
  assertNativeBinding(owner, admission, true)
  if (
    admission.conversation_id !== owner.conversation_id ||
    (owner.owner_key && admission.owner_key !== owner.owner_key) ||
    input.history_id !== owner.conversation_id ||
    input.role !== "assistant"
  )
    fail("owner_conversation_mismatch")
  const response = await tldwClient.addChatMessage(
    owner.conversation_id,
    {
      ...nativeHistoryMessagePayload(input),
      tldw_history_admission_v1: historyAdmissionReference(admission)
    },
    nativeOptions(owner, opts?.signal)
  )
  if (response.id !== input.id) fail("invalid_history_settlement")
  return response
}
