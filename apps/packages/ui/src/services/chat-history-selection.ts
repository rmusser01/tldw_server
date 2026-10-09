/** Owner selection is separate from workspace routing, view identity and inference connection. */
import { z } from "zod"
import type { NativeHistoryEditResend, NativeHistoryRegenerate } from "@/types/chat-modes"
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
import { tldwClient, type ChatMessage } from "./tldw/TldwApiClient"
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
    native_fork_context: z
      .object({
        policy: z.literal("plain_v1"),
        storage_context_digest: str,
        supported: z.boolean()
      })
      .strict()
      .nullable()
      .optional(),
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
    (owner.kind !== "unavailable" && owner.validate_lease?.() === false)
  )
    fail("request_config_scope_changed")
}

/** A ready controller must still belong to the conversation being submitted. */
export const assertHistoryConversationTarget = async (
  owner: HistoryOwnerV1,
  target: { historyId?: string | null; serverChatId?: string | null }
): Promise<void> => {
  assertOwnerLease(owner)
  if (owner.kind === "unavailable") return fail(owner.code)
  if (target.serverChatId) {
    if (owner.kind !== "native" || owner.conversation_id !== target.serverChatId)
      fail("owner_conversation_mismatch")
    return
  }
  if (!target.historyId) fail("owner_conversation_mismatch")
  if (owner.kind === "local") {
    if (owner.conversation_id !== target.historyId) fail("owner_conversation_mismatch")
    return
  }
  const { PageAssistDatabase } = await import("@/db/dexie/chat")
  const { serverChatMirrorOwnerKey } = await import("@/db/dexie/server-chat-mirror")
  const details = await new PageAssistDatabase().getHistoryInfo(target.historyId)
  assertOwnerLease(owner)
  if (details?.server_chat_id !== owner.conversation_id ||
      details.server_scope_key !== serverChatMirrorOwnerKey({ requestScope: owner.request_scope }))
    fail("owner_conversation_mismatch")
}
const localOptions = (
  owner: LocalHistoryOwnerV1,
  opts?: HistoryOperationOptions
): HistoryOperationOptions => ({
  ...opts,
  validate_lease: () =>
    owner.validate_lease?.() !== false && opts?.validate_lease?.() !== false
})
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
        localOptions(owner, { signal })
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

/** Keep the existing read gate until the deployed operation can enforce P1 atomically. */
export const nativeHistoryNeedsBranchChoice = async (
  owner: NativeHistoryOwnerV1,
  capture: HistorySelectionCaptureV1,
  operation: "messages" | "completion",
  options?: HistoryOperationOptions
): Promise<boolean> => {
  assertOwnerLease(owner, options)
  assertNativeBinding(owner, capture.view, true)
  assertNativeBinding(owner, capture.snapshot, true)
  if (capture.purpose !== "send") fail("selected_content_mismatch")
  const spec = await tldwClient
    .getOpenAPISpec(nativeOptions(owner, options?.signal))
    .catch(() => null)
  assertOwnerLease(owner, options)
  const { detectNativeHistoryBranchAdmission } = await import(
    "./tldw/server-capabilities"
  )
  assertOwnerLease(owner, options)
  if (detectNativeHistoryBranchAdmission(spec, operation)) return false
  // ponytail: an older server can still change after this read; upgrade it for atomic P1 admission.
  const parents = new Map(
    capture.snapshot.nodes.map((node) => [node.id, node.parent_id])
  )
  const interpretation = capture.snapshot.interpretation_status
  if (
    capture.view.interpretation.kind === "legacy_linear_v1" &&
    interpretation.kind === "legacy_linear_v1"
  ) {
    interpretation.ordered_path_ids.forEach((id, index, path) =>
      parents.set(id, index ? path[index - 1] : null)
    )
  }
  const selectedParent = capture.rows.at(-1)?.id ?? null
  return capture.snapshot.nodes.some(
    (node) => parents.get(node.id) === selectedParent
  )
}

/** Read the deployed history fence without loading message content or changing the view. */
export const readNativeHistoryVersion = async (
  owner: NativeHistoryOwnerV1,
  options?: HistoryOperationOptions
): Promise<number> => {
  assertOwnerLease(owner, options)
  const capability = nativeCapabilities.get(owner)
  if (!capability?.captured) fail("unsupported_history_capability")
  const detail = await tldwClient.getChat(owner.conversation_id, {
    ...nativeOptions(owner, options?.signal),
    requireHistoryVersion: true
  })
  assertOwnerLease(owner, options)
  if (
    detail.id !== owner.conversation_id ||
    typeof detail.history_version !== "number" ||
    !Number.isSafeInteger(detail.history_version) ||
    detail.history_version < 0
  )
    fail("unsupported_history_version")
  return detail.history_version
}

export type NativeMessageEditReceipt = Readonly<{
  message_id: string
  message: string
  version: number
}>

const nativeMessageTarget = (
  capture: HistorySelectionCaptureV1,
  messageId: string
) => {
  const nodes = capture.rows.filter((row) => row.id === messageId)
  const members = capture.snapshot.nodes.filter((row) => row.id === messageId)
  const content = capture.selected_content.filter((row) => row.id === messageId)
  if (
    nodes.length !== 1 ||
    members.length !== 1 ||
    content.length !== 1 ||
    canonicalHistoryJson(nodes[0]) !== canonicalHistoryJson(members[0]) ||
    content[0].revision !== nodes[0].revision ||
    !nodes[0].settled ||
    !["user", "assistant"].includes(nodes[0].role)
  )
    fail("message_changed")
  return { node: nodes[0], content: content[0] }
}
const nativeEditTarget = (capture: HistorySelectionCaptureV1, messageId: string) => {
  const target = nativeMessageTarget(capture, messageId)
  if (target.content.images.length) fail("unsupported_message_payload")
  return target
}
const nativeEditPreservedState = (
  target: ReturnType<typeof nativeEditTarget>
) => ({
  parent: target.node.parent_id,
  role: target.node.role,
  settled: target.node.settled,
  legacy_projection_id: target.node.legacy_projection_id ?? null,
  metadata: target.node.metadata ?? [],
  assets: target.node.assets ?? [],
  comparison: target.node.comparison ?? null,
  extra_metadata: target.content.extra_metadata ?? null,
  tool_calls: target.content.tool_calls ?? null
})

/** A content-only edit must be canonically recaptured before replacing its mounted source. */
export const qualifyNativeMessageEditCapture = (
  original: HistorySelectionCaptureV1,
  refreshed: HistorySelectionCaptureV1,
  receipt: NativeMessageEditReceipt
): void => {
  const before = nativeEditTarget(original, receipt.message_id)
  const after = nativeEditTarget(refreshed, receipt.message_id)
  if (
    original.snapshot.owner_key !== refreshed.snapshot.owner_key ||
    original.snapshot.conversation_id !== refreshed.snapshot.conversation_id ||
    canonicalHistoryJson(original.view) !==
      canonicalHistoryJson(refreshed.view) ||
    !Number.isSafeInteger(receipt.version) ||
    receipt.version <= 1 ||
    after.node.revision === before.node.revision ||
    after.content.message !== receipt.message ||
    canonicalHistoryJson(nativeEditPreservedState(before)) !==
      canonicalHistoryJson(nativeEditPreservedState(after))
  )
    fail("message_changed")
}

/** Reuse the native target CAS; never turn an in-place Save into a local history mutation. */
export const saveNativeHistoryMessage = async (
  owner: NativeHistoryOwnerV1,
  capture: HistorySelectionCaptureV1,
  target: { id: string; message: string; isBot?: boolean },
  message: string,
  options?: HistoryOperationOptions & {
    acknowledgement_current?: () => boolean
  }
): Promise<NativeMessageEditReceipt> => {
  assertOwnerLease(owner, options)
  assertNativeBinding(owner, capture.snapshot, true)
  if (capture.view.interpretation.kind !== "parent_graph_v1")
    fail("unsupported_history_capability")
  const original = nativeEditTarget(capture, target.id)
  if (
    original.content.message !== target.message ||
    original.node.role !== (target.isBot ? "assistant" : "user")
  )
    fail("message_changed")
  if (!message.trim()) fail("unsupported_message_payload")
  const scoped = nativeOptions(owner, options?.signal)
  const row = await tldwClient.getMessage(target.id, scoped)
  assertOwnerLease(owner, options)
  if (
    !row ||
    row.id !== target.id ||
    row.conversation_id !== owner.conversation_id ||
    row.sender !== original.node.role ||
    row.parent_message_id !== original.node.parent_id ||
    row.has_image !== false ||
    row.content !== target.message ||
    !Number.isSafeInteger(row.version) ||
    row.version <= 0 ||
    row.version >= Number.MAX_SAFE_INTEGER
  )
    fail("message_changed")
  const fresh = await captureHistorySnapshot(
    owner,
    capture.view,
    "send",
    options?.signal
  )
  assertOwnerLease(owner, options)
  if (fresh.status !== "captured") return fail(fresh.code)
  if (
    canonicalHistoryJson(original) !==
    canonicalHistoryJson(nativeEditTarget(fresh, target.id))
  )
    fail("message_changed")
  const result = await tldwClient.editMessage(
    target.id,
    message,
    row.version,
    owner.conversation_id,
    scoped
  )
  assertOwnerLease(owner, {
    ...options,
    validate_lease: options?.acknowledgement_current ?? options?.validate_lease
  })
  // PUT may render roleplay placeholders; the subsequent raw capture certifies the edited text.
  if (
    !result ||
    result.id !== target.id ||
    result.conversation_id !== owner.conversation_id ||
    result.sender !== row.sender ||
    result.parent_message_id !== row.parent_message_id ||
    result.has_image !== false ||
    typeof result.content !== "string" ||
    !Number.isSafeInteger(result.version) ||
    result.version !== row.version + 1
  )
    fail("message_edit_unconfirmed")
  return Object.freeze({
    message_id: target.id,
    message,
    version: result.version
  })
}

/** Keep opaque snapshot revisions separate from the server message version CAS. */
export const verifyNativeMessageEditReceipt = async (
  owner: NativeHistoryOwnerV1,
  original: HistorySelectionCaptureV1,
  receipt: NativeMessageEditReceipt,
  options?: HistoryOperationOptions
): Promise<void> => {
  assertOwnerLease(owner, options)
  assertNativeBinding(owner, original.snapshot, true)
  const target = nativeEditTarget(original, receipt.message_id)
  const row = await tldwClient.getMessage(
    receipt.message_id,
    nativeOptions(owner, options?.signal)
  )
  assertOwnerLease(owner, options)
  if (
    !Number.isSafeInteger(receipt.version) ||
    receipt.version <= 1 ||
    !row ||
    row.id !== receipt.message_id ||
    row.conversation_id !== owner.conversation_id ||
    row.sender !== target.node.role ||
    row.parent_message_id !== target.node.parent_id ||
    row.has_image !== false ||
    row.content !== receipt.message ||
    row.version !== receipt.version
  )
    fail("message_edit_unconfirmed")
}

export type NativeMessageDeleteReceipt = Readonly<{
  message_id: string
  parent_message_id: string | null
}>

export const nativeMessageDeleteView = (
  original: HistorySelectionCaptureV1,
  receipt: NativeMessageDeleteReceipt
): HistoryViewSelectionV1 => {
  const target = nativeMessageTarget(original, receipt.message_id)
  if (target.node.parent_id !== receipt.parent_message_id)
    fail("message_changed")
  if (
    original.view.cursor.kind === "empty" ||
    original.view.cursor.message_id !== receipt.message_id
  )
    return original.view
  if (
    !Number.isSafeInteger(original.view.selection_revision) ||
    original.view.selection_revision >= Number.MAX_SAFE_INTEGER
  )
    fail("stale_selection")
  return {
    ...original.view,
    selection_revision: original.view.selection_revision + 1,
    cursor: receipt.parent_message_id
      ? { kind: "after_message", message_id: receipt.parent_message_id }
      : { kind: "empty" }
  }
}

/** Destructive deletion requires both physical CAS and exact saved content/assets. */
export const deleteNativeHistoryMessage = async (
  owner: NativeHistoryOwnerV1,
  capture: HistorySelectionCaptureV1,
  target: { id: string; message: string; isBot?: boolean; images?: string[] },
  options?: HistoryOperationOptions
): Promise<NativeMessageDeleteReceipt> => {
  assertOwnerLease(owner, options)
  assertNativeBinding(owner, capture.snapshot, true)
  if (capture.view.interpretation.kind !== "parent_graph_v1")
    fail("unsupported_history_capability")
  const original = nativeMessageTarget(capture, target.id)
  if (
    original.content.message !== target.message ||
    original.node.role !== (target.isBot ? "assistant" : "user") ||
    canonicalHistoryJson(original.content.images) !==
      canonicalHistoryJson(target.images ?? [])
  )
    fail("message_changed")
  if (capture.snapshot.nodes.some((node) => node.parent_id === target.id))
    fail("message_has_descendants")
  const receipt = Object.freeze({
    message_id: target.id,
    parent_message_id: original.node.parent_id
  })
  nativeMessageDeleteView(capture, receipt)
  const scoped = nativeOptions(owner, options?.signal)
  const row = await tldwClient.getMessage(target.id, scoped)
  assertOwnerLease(owner, options)
  if (
    !row ||
    row.id !== target.id ||
    row.conversation_id !== owner.conversation_id ||
    row.sender !== original.node.role ||
    row.parent_message_id !== original.node.parent_id ||
    row.content !== target.message ||
    row.has_image !== original.content.images.length > 0 ||
    !Number.isSafeInteger(row.version) ||
    row.version <= 0 ||
    row.version >= Number.MAX_SAFE_INTEGER
  )
    fail("message_changed")
  const fresh = await captureHistorySnapshot(
    owner,
    capture.view,
    "send",
    options?.signal
  )
  assertOwnerLease(owner, options)
  if (fresh.status !== "captured") return fail(fresh.code)
  if (
    canonicalHistoryJson(original) !==
    canonicalHistoryJson(nativeMessageTarget(fresh, target.id))
  )
    fail("message_changed")
  if (fresh.snapshot.nodes.some((node) => node.parent_id === target.id))
    fail("message_has_descendants")
  await tldwClient.deleteMessage(
    target.id,
    row.version,
    owner.conversation_id,
    {
      ...scoped,
      requireLeaf: true,
      expectedRevision: original.node.revision
    }
  )
  assertOwnerLease(owner, options)
  return receipt
}

/** Only the acknowledged leaf disappears; retained history is qualified before publication. */
export const qualifyNativeMessageDeleteCapture = (
  original: HistorySelectionCaptureV1,
  refreshed: HistorySelectionCaptureV1,
  receipt: NativeMessageDeleteReceipt
): void => {
  const expectedView = nativeMessageDeleteView(original, receipt)
  if (
    refreshed.snapshot.owner_key !== original.snapshot.owner_key ||
    refreshed.snapshot.conversation_id !== original.snapshot.conversation_id ||
    refreshed.snapshot.storage_context_digest !==
      original.snapshot.storage_context_digest ||
    canonicalHistoryJson(refreshed.view) !==
      canonicalHistoryJson(expectedView) ||
    refreshed.snapshot.nodes.some(
      (node) =>
        node.id === receipt.message_id || node.parent_id === receipt.message_id
    ) ||
    original.snapshot.nodes.some(
      (node) =>
        node.id !== receipt.message_id &&
        !refreshed.snapshot.nodes.some(
          (current) =>
            canonicalHistoryJson(current) === canonicalHistoryJson(node)
        )
    ) ||
    canonicalHistoryJson(refreshed.rows) !==
      canonicalHistoryJson(
        original.rows.filter((row) => row.id !== receipt.message_id)
      ) ||
    canonicalHistoryJson(refreshed.selected_content) !==
      canonicalHistoryJson(
        original.selected_content.filter((row) => row.id !== receipt.message_id)
      )
  )
    fail("message_delete_unconfirmed")
}

/** Capture an explicit branch without replacing the mounted source/editor. */
export const nativeEditResendView = (
  owner: HistoryOwnerV1,
  origin: HistoryViewSelectionV1,
  capture: HistorySelectionCaptureV1,
  intent: NativeHistoryEditResend,
  sourceCapture: HistorySelectionCaptureV1 = capture
): HistoryViewSelectionV1 => {
  assertOwnerLease(owner)
  if (owner.kind !== "native")
    return fail("native_history_requires_server_owner")
  if (
    owner.conversation_id !== intent.conversation_id ||
    origin.conversation_id !== intent.conversation_id ||
    origin.owner_key !== intent.owner_key ||
    capture.snapshot.conversation_id !== intent.conversation_id ||
    capture.snapshot.owner_key !== intent.owner_key ||
    (owner.owner_key && owner.owner_key !== intent.owner_key)
  )
    fail("owner_conversation_mismatch")
  const targets = capture.snapshot.nodes.filter(
    (node) => node.id === intent.message_id
  )
  if (
    targets.length !== 1 ||
    targets[0].role !== "user" ||
    !targets[0].settled ||
    targets[0].revision !== intent.message_revision
  )
    fail("message_changed")
  if (targets[0].assets?.length || targets[0].comparison)
    fail("unsupported_message_payload")
  if (targets[0].metadata?.length) {
    // Detached ancestry excludes the edited row; validate its metadata against the original capture.
    const sourceNodes = sourceCapture.snapshot.nodes.filter(
      (node) => node.id === intent.message_id
    )
    const sourceContent = sourceCapture.selected_content.filter(
      (row) => row.id === intent.message_id
    )
    if (
      sourceCapture.snapshot.owner_key !== intent.owner_key ||
      sourceCapture.snapshot.conversation_id !== intent.conversation_id ||
      sourceNodes.length !== 1 ||
      sourceNodes[0].revision !== targets[0].revision ||
      sourceContent.length !== 1 ||
      sourceContent[0].revision !== targets[0].revision ||
      targets[0].metadata.some(
        (reference) => reference.kind !== "message_metadata"
      ) ||
      canonicalHistoryJson(sourceNodes[0].metadata ?? []) !==
        canonicalHistoryJson(targets[0].metadata) ||
      sourceContent[0].tool_calls?.length ||
      sourceContent[0].extra_metadata?.sender_role !== "user"
    )
      fail("unsupported_message_payload")
    nativeHistoryMessagePayload({
      id: intent.message_id,
      history_id: intent.conversation_id,
      role: "user",
      name: "You",
      content: sourceContent[0].message,
      createdAt: 0,
      images: [...sourceContent[0].images],
      metadataExtra: sourceContent[0].extra_metadata
    })
  }
  if (!intent.isOriginCurrent() || !intent.isCurrent()) fail("stale_selection")
  const selection_revision = origin.selection_revision + 1
  if (!Number.isSafeInteger(selection_revision)) fail("invalid_selection")
  return {
    ...structuredClone(origin),
    selection_revision,
    cursor: { kind: "before_message", message_id: intent.message_id }
  }
}

const nativeRegenerationTarget = (capture: HistorySelectionCaptureV1) => {
  const user = capture.rows.at(-2)
  const reply = capture.rows.at(-1)
  if (!user || !reply || user.role !== "user" || reply.role !== "assistant" ||
      !user.settled || !reply.settled || reply.parent_id !== user.id ||
      user.comparison || reply.comparison)
    return fail("unsupported_history_regeneration")
  for (const row of [user, reply]) {
    const nodes = capture.snapshot.nodes.filter(node => node.id === row.id)
    if (nodes.length !== 1 || canonicalHistoryJson(nodes[0]) !== canonicalHistoryJson(row))
      fail("message_changed")
  }
  const content = capture.selected_content.filter(row => row.id === user.id)
  if (content.length !== 1 || content[0].revision !== user.revision)
    return fail("message_changed")
  return { user, reply, content: content[0] }
}

/** Regenerate the selected saved reply without changing the mounted selection. */
export const nativeRegenerateView = (
  owner: HistoryOwnerV1,
  origin: HistoryViewSelectionV1,
  capture: HistorySelectionCaptureV1,
  intent: NativeHistoryRegenerate
): HistoryViewSelectionV1 => {
  assertOwnerLease(owner)
  if (owner.kind !== "native") return fail("native_history_requires_server_owner")
  if (origin.owner_key !== intent.owner_key || origin.conversation_id !== intent.conversation_id ||
      capture.snapshot.owner_key !== intent.owner_key || capture.snapshot.conversation_id !== intent.conversation_id ||
      owner.conversation_id !== intent.conversation_id || (owner.owner_key && owner.owner_key !== intent.owner_key))
    fail("owner_conversation_mismatch")
  if (origin.interpretation.kind !== "parent_graph_v1" || capture.view.interpretation.kind !== "parent_graph_v1")
    fail("unsupported_history_regeneration")
  const { user, reply } = nativeRegenerationTarget(capture)
  if (reply.id !== intent.message_id || reply.revision !== intent.message_revision ||
      user.id !== intent.input_message_id || user.revision !== intent.input_message_revision)
    fail("message_changed")
  if (!intent.isOriginCurrent() || !intent.isCurrent()) fail("stale_selection")
  const selection_revision = origin.selection_revision + 1
  if (!Number.isSafeInteger(selection_revision)) fail("invalid_selection")
  return { ...structuredClone(origin), selection_revision,
    cursor: { kind: "after_message", message_id: intent.message_id } }
}

/** Saved completion input uses every raw attachment, rather than MessageCreate's one-image form. */
export const nativeRegenerationMessage = (
  capture: HistorySelectionCaptureV1,
  intent: NativeHistoryRegenerate,
  inputVersion: number
): Extract<ChatMessage, { role: "user" }> => {
  const { user, reply, content } = nativeRegenerationTarget(capture)
  if (user.id !== intent.input_message_id || user.revision !== intent.input_message_revision ||
      reply.id !== intent.message_id || reply.revision !== intent.message_revision)
    fail("message_changed")
  const extra = content.extra_metadata ?? {}
  const details = extra.image_details ?? content.images.map(() => "auto")
  if (!Number.isSafeInteger(inputVersion) || inputVersion <= 0 || inputVersion >= Number.MAX_SAFE_INTEGER ||
      (extra.sender_role != null && extra.sender_role !== "user") || content.tool_calls != null ||
      extra.function_call != null || extra.tool_call_id != null ||
      !Array.isArray(details) || details.length !== content.images.length ||
      details.some(detail => typeof detail !== "string" || !["auto", "high", "low"].includes(detail)) ||
      content.images.some(image => !/^data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/]*={0,2}$/.test(image)))
    fail("unsupported_history_regeneration")
  const text = inputVersion === 1 && extra.content_placeholder_reason === "image_attachment" &&
    content.images.length && content.message === `<Image attachment x${content.images.length}>`
    ? "" : content.message
  return { role: "user", content: content.images.length ? [
    ...(text ? [{ type: "text" as const, text }] : []),
    ...content.images.map((url, index) => ({ type: "image_url" as const,
      image_url: { url, detail: details[index] as "auto" | "high" | "low" } }))
  ] : text }
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

export { isDefinitiveNativeBranchRejection } from "@/utils/history-selection"

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
    return confirmLocalHistoryProjection(owner, scope, intent, {
      ...localOptions(owner, { signal }),
      view
    })
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
    (message.metadataExtra &&
      Object.entries(message.metadataExtra).some(
        ([key, value]) => key !== "sender_role" || value !== message.role
      ))
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
  owner: NativeHistoryOwnerV1,
  selection: HistorySelectionV1,
  value: unknown
): HistoryAdmissionV1 => {
  const admission = parse(admissionSchema, value) as HistoryAdmissionV1
  assertNativeBinding(owner, admission, true)
  if (
    admission.selection_digest !== selection.selection_digest ||
    admission.originating_selection_revision !== selection.selection_revision ||
    canonicalHistoryJson(admission.messages) !==
      canonicalHistoryJson(selection.messages)
  )
    fail("invalid_history_admission")
  return freeze(admission)
}

/** The original user admission retains its original selection provenance during regeneration. */
export const parseNativeRegenerationAdmission = (
  owner: NativeHistoryOwnerV1,
  capture: HistorySelectionCaptureV1,
  inputVersion: number,
  value: unknown
): HistoryAdmissionV1 => {
  assertOwnerLease(owner)
  assertNativeBinding(owner, capture.snapshot, true)
  const { user, reply } = nativeRegenerationTarget(capture)
  const admission = parse(admissionSchema, value) as HistoryAdmissionV1
  assertNativeBinding(owner, admission, true)
  const ids = admission.messages.map(row => row.id)
  if (!Number.isSafeInteger(inputVersion) || inputVersion <= 0 || inputVersion >= Number.MAX_SAFE_INTEGER ||
      ids.includes(user.id) || ids.includes(reply.id) || new Set(ids).size !== ids.length ||
      admission.input_message_id !== user.id || admission.input_message_revision !== String(inputVersion))
    fail("invalid_history_admission")
  return freeze(admission)
}

export const appendSelectedUser = async (
  owner: HistoryOwnerV1,
  selection: HistorySelectionV1,
  input: Message,
  opts?: HistoryOperationOptions & { historyBranch?: boolean }
): Promise<HistoryAdmissionV1> => {
  selection = structuredClone(selection)
  input = structuredClone(input)
  assertOwnerLease(owner, opts)
  if (owner.kind === "unavailable") return fail(owner.code)
  if (opts?.historyBranch !== undefined && owner.kind !== "native")
    fail("native_history_requires_server_owner")
  if (owner.kind === "local")
    return appendLocalSelectedUser(owner, selection, input, localOptions(owner, opts))
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
    tldw_history_selection_v1: structuredClone(selection),
    ...(opts?.historyBranch !== undefined
      ? { tldw_history_branch: opts.historyBranch }
      : {})
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
    return settleLocalAcceptedAssistant(owner, admission, input, localOptions(owner, opts))
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

const NATIVE_FORK_POLICY = { policy: "native-plain-fork-v1" }
type ForkRequest = import("@/types/history-selection").ForkRequestV1
type ForkResult = import("@/types/history-selection").ForkResultV1
type NativeForkPrepared = {
  readonly request: ForkRequest
  readonly rows: readonly Readonly<Record<string, unknown>>[]
}
const forkCaptures = new WeakMap<
  object,
  { owner: NativeHistoryOwnerV1; view: HistoryViewSelectionV1 }
>()
const nativeForkPreparations = new WeakMap<
  NativeForkPrepared,
  {
    owner: NativeHistoryOwnerV1
    options?: HistoryOperationOptions
    dispatchOptions: ReturnType<typeof nativeOptions>
  }
>()
const assertPlainForkContext = (capture: HistorySelectionCaptureV1) => {
  const proof = capture.snapshot.native_fork_context
  if (
    !proof?.supported ||
    proof.policy !== "plain_v1" ||
    proof.storage_context_digest !== capture.storage_context_digest
  )
    fail("native_fork_context_unsupported")
}
const nativeForkRows = (capture: HistorySelectionCaptureV1) => {
  assertPlainForkContext(capture)
  return capture.rows.map((row, index) => {
    const content = capture.selected_content[index]
    if (
      content.tool_calls?.length ||
      (content.extra_metadata && Object.keys(content.extra_metadata).length) ||
      row.comparison ||
      row.metadata?.length
    )
      fail("unsupported_message_payload")
    // Use the existing exact text/inline-image validator. The legacy endpoint supports system rows too.
    if (!["user", "assistant", "system"].includes(row.role))
      fail("unsupported_message_payload")
    const payload = nativeHistoryMessagePayload({
      id: row.id,
      role: row.role === "system" ? "user" : row.role,
      content: content.message,
      images: [...content.images]
    } as Message)
    const { id: _sourceId, ...detached } = payload
    return {
      ...detached,
      ...(typeof content.message === "string"
        ? { content: content.message }
        : {}),
      role: row.role
    }
  })
}
/** Capture a selected limited projection, with no live metadata/settings reads. */
export const captureNativeForkSelection = async (
  owner: NativeHistoryOwnerV1,
  view: HistoryViewSelectionV1,
  options?: HistoryOperationOptions
): Promise<ForkRequest["input"]> => {
  assertOwnerLease(owner, options)
  const capture = await captureHistorySnapshot(
    owner,
    view,
    "fork",
    options?.signal
  )
  if (capture.status !== "captured") return fail(capture.code)
  nativeForkRows(capture)
  const prepared = prepareHistoryContext(
    NATIVE_FORK_POLICY,
    () => owner.validate_lease() && options?.validate_lease?.() !== false
  )
  const input = freeze({
    kind: "normal" as const,
    selection: finalizeHistorySelection(owner, capture, prepared, view)
  })
  forkCaptures.set(input, { owner, view: freeze(structuredClone(view)) })
  return input
}
/** Recapture immediately before dispatch; once dispatched the accepted projection never changes. */
export const prepareNativeFork = async (
  owner: NativeHistoryOwnerV1,
  request: ForkRequest,
  options?: HistoryOperationOptions
): Promise<NativeForkPrepared> => {
  assertOwnerLease(owner, options)
  const source = forkCaptures.get(request.input)
  if (!source || source.owner !== owner || request.input.kind !== "normal")
    return fail("native_fork_projection_unavailable")
  if (
    request.owner_key !== request.destination_owner_key ||
    request.owner_key !== request.input.selection.owner_key ||
    request.request_digest !==
      historyDigest({
        operation_id: request.operation_id,
        owner_key: request.owner_key,
        destination_owner_key: request.destination_owner_key,
        input: request.input
      })
  )
    return fail("fork_request_mismatch")
  const capture = await captureHistorySnapshot(
    owner,
    source.view,
    "fork",
    options?.signal
  )
  if (capture.status !== "captured") return fail(capture.code)
  const rows = nativeForkRows(capture)
  const context = prepareHistoryContext(
    NATIVE_FORK_POLICY,
    () => owner.validate_lease() && options?.validate_lease?.() !== false
  )
  const selection = finalizeHistorySelection(
    owner,
    capture,
    context,
    source.view
  )
  if (
    canonicalHistoryJson(selection) !==
    canonicalHistoryJson(request.input.selection)
  )
    return fail("stale_selection")
  const prepared = freeze({ request: structuredClone(request), rows })
  nativeForkPreparations.set(prepared, {
    owner,
    options,
    dispatchOptions: {
      requestScope: freeze(structuredClone(owner.request_scope)),
      scope: freeze(structuredClone(owner.scope ?? { type: "global" })),
      signal: options?.signal
    }
  })
  return prepared
}
/** Multi-request legacy copy: acknowledged completion is not an atomic receipt. Never replay. */
export const commitNativeFork = async (
  prepared: NativeForkPrepared,
  onCandidate: (childId: string) => Promise<unknown> | unknown
): Promise<ForkResult> => {
  const capability = nativeForkPreparations.get(prepared)
  if (!capability) return fail("fork_preparation_consumed")
  nativeForkPreparations.delete(prepared)
  const { owner, options, dispatchOptions } = capability
  const binding = {
    owner_key: prepared.request.owner_key,
    operation_id: prepared.request.operation_id
  }
  let dispatched = false
  let childId: string | undefined
  try {
    assertOwnerLease(owner, options)
    dispatched = true
    const child = await tldwClient.createChat(
      {
        title: "Forked conversation",
        character_id: null,
        assistant_kind: null,
        assistant_id: null,
        persona_memory_mode: null
      },
      dispatchOptions
    )
    if (
      typeof child?.id !== "string" ||
      !child.id.trim() ||
      child.id === owner.conversation_id
    )
      fail("invalid_child_acknowledgement")
    childId = child.id
    // Save the acknowledged child before inspecting any later lease or row failure.
    await onCandidate(childId)
    assertOwnerLease(owner, { signal: options?.signal })
    let parentId: string | null = null
    const seen = new Set<string>()
    for (const row of prepared.rows) {
      assertOwnerLease(owner, { signal: options?.signal })
      const response = (await tldwClient.addChatMessage(
        childId,
        { ...row, parent_message_id: parentId },
        dispatchOptions
      )) as unknown as Record<string, unknown>
      const id = response?.id
      if (typeof id !== "string")
        throw new Error("invalid_message_acknowledgement")
      if (
        !id.trim() ||
        seen.has(id) ||
        response.conversation_id !== childId ||
        response.parent_message_id !== parentId ||
        response.sender !== row.role
      )
        fail("invalid_message_acknowledgement")
      seen.add(id)
      parentId = id
    }
    return { ...binding, state: "legacy_completed", child_id: childId }
  } catch (error) {
    const code = error instanceof Error ? error.message : "native_fork_failed"
    return childId
      ? { ...binding, state: "partial", candidate_child_id: childId, code }
      : { ...binding, state: dispatched ? "unknown" : "rejected", code }
  }
}

/** Known fork children use current scoped server settings, never the server-ID browser cache. */
export const readNativeForkSettings = async (
  owner: NativeHistoryOwnerV1,
  signal?: AbortSignal
) => {
  assertOwnerLease(owner, { signal })
  if (!nativeCapabilities.has(owner)) fail("unsupported_history_capability")
  const response = await tldwClient.getChatSettings(
    owner.conversation_id,
    nativeOptions(owner, signal)
  )
  assertOwnerLease(owner, { signal })
  if (response?.conversation_id !== owner.conversation_id)
    fail("fork_settings_owner_mismatch")
  const { normalizeChatSettingsRecord } =
    await import("@/services/chat-settings")
  assertOwnerLease(owner, { signal })
  return normalizeChatSettingsRecord(response.settings)
}
export const updateNativeForkSettings = async (
  owner: NativeHistoryOwnerV1,
  patch: Partial<import("@/types/chat-session-settings").ChatSettingsRecord>,
  signal?: AbortSignal
) => {
  assertOwnerLease(owner, { signal })
  if (!nativeCapabilities.has(owner)) fail("unsupported_history_capability")
  const detached = structuredClone(patch)
  const response = await tldwClient.updateChatSettings(
    owner.conversation_id,
    detached,
    nativeOptions(owner, signal)
  )
  assertOwnerLease(owner, { signal })
  if (response?.conversation_id !== owner.conversation_id)
    fail("fork_settings_owner_mismatch")
  const { normalizeChatSettingsRecord } =
    await import("@/services/chat-settings")
  assertOwnerLease(owner, { signal })
  return normalizeChatSettingsRecord(response.settings)
}
