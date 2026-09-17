/** Local H1 authority. All hashes run synchronously within the owning Dexie transaction. */
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import { db } from "./schema"
import type {
  HistoryInfo,
  Message,
  HistoryBookmark,
  HistoryBookmarkScope
} from "./types"
import type {
  HistoryAdmissionReferenceV1,
  HistoryAdmissionV1,
  HistoryCaptureResultV1,
  HistoryNodeV1,
  HistorySelectionSnapshotV1,
  HistorySelectionV1,
  HistoryViewSelectionV1,
  LegacyHistoryProjectionConfirmV1,
  LegacyHistoryProjectionV1
} from "@/types/history-selection"
import {
  bindSelectedHistoryContent,
  HistorySelectionError,
  resolveHistorySelection,
  resolveParentPath,
  selectionDigest
} from "@/utils/history-selection"

export type LocalHistoryOwnerV1 = {
  readonly kind: "local"
  readonly profile_id: string
  readonly owner_key: string
  readonly conversation_id: string
}
export type HistoryOperationOptions = {
  signal?: AbortSignal
  validate_lease?: () => boolean
}
const fail = (code: string): never => {
  throw new HistorySelectionError(code)
}

/** JSON-only canonical local context; unsupported opaque values cannot be silently dropped. */
export const canonicalHistoryJson = (value: unknown): string => {
  const normalize = (v: any): any => {
    if (v === null || typeof v === "string" || typeof v === "boolean") return v
    if (typeof v === "number" && Number.isFinite(v)) return v
    if (Array.isArray(v)) return v.map(normalize)
    if (
      v &&
      typeof v === "object" &&
      (Object.getPrototypeOf(v) === Object.prototype ||
        Object.getPrototypeOf(v) === null)
    ) {
      return Object.fromEntries(
        Object.keys(v)
          .sort()
          .filter((k) => v[k] !== undefined)
          .map((k) => [k, normalize(v[k])])
      )
    }
    return fail("unsupported_history_context")
  }
  return JSON.stringify(normalize(value))
}
export const historyDigest = (value: unknown): string =>
  bytesToHex(sha256(utf8ToBytes(canonicalHistoryJson(value))))
const protectedKey = (key: string) =>
  key.startsWith("tldw_history_") ||
  key === "history_admission" ||
  key === "history_provenance" ||
  key === "history_settlement"
/** Public writes/imports cannot install authority, even inside ordinary nested metadata. */
export const stripHistoryAuthority = <T>(value: T): T => {
  if (Array.isArray(value)) return value.map(stripHistoryAuthority) as T
  if (!value || typeof value !== "object") return value
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !protectedKey(key))
      .map(([key, v]) => [key, stripHistoryAuthority(v)])
  ) as T
}
export const sanitizeImportedHistory = (value: HistoryInfo): HistoryInfo => {
  const result = stripHistoryAuthority(value)
  delete result.local_owner_key
  delete result.server_scope_key
  return result
}
export const sanitizeImportedMessage = (value: Message): Message => {
  const result = stripHistoryAuthority(value)
  delete result.serverMessageId
  delete result.serverMessageVersion
  return result
}

export const ensureLocalProfileId = async (): Promise<string> =>
  db.transaction("rw", [db.userSettings], async () => {
    const settings = await db.userSettings.get("main")
    if (settings?.history_profile_id) return settings.history_profile_id
    const profile = settings?.user_id?.trim() || crypto.randomUUID()
    await db.userSettings.put({
      ...settings,
      id: "main",
      user_id: settings?.user_id || profile,
      history_profile_id: profile
    })
    return profile
  })
const localKey = (profile: string) => `local-history-v1:${profile}`
export const getLocalHistoryOwner = async (
  conversation_id: string
): Promise<LocalHistoryOwnerV1> => {
  const profile_id = await ensureLocalProfileId()
  const owner = {
    kind: "local" as const,
    profile_id,
    owner_key: localKey(profile_id),
    conversation_id
  }
  await db.transaction("rw", [db.chatHistories, db.userSettings], async () => {
    const record = await ownedHistory(owner)
    if (!record.local_owner_key)
      await db.chatHistories.update(record.id, {
        local_owner_key: owner.owner_key
      })
  })
  return owner
}
const ownedHistory = async (
  owner: LocalHistoryOwnerV1
): Promise<HistoryInfo> => {
  const profile = await db.userSettings.get("main")
  if (
    profile?.history_profile_id !== owner.profile_id ||
    owner.owner_key !== localKey(owner.profile_id)
  )
    fail("owner_mismatch")
  const record = await db.chatHistories.get(owner.conversation_id)
  if (!record) return fail("missing_conversation")
  if (record.server_chat_id || record.message_source === "server")
    fail(
      record.server_scope_key ? "server_owned_history" : "unbound_server_mirror"
    )
  if (record.local_owner_key && record.local_owner_key !== owner.owner_key)
    fail("owner_mismatch")
  return record
}
const tables = () => [
  db.chatHistories,
  db.messages,
  db.userSettings,
  db.sessionFiles,
  db.compareStates,
  db.historySelections,
  db.historyProjections
]
const assertLease = (opts?: HistoryOperationOptions) => {
  if (opts?.signal?.aborted || opts?.validate_lease?.() === false)
    fail("request_config_scope_changed")
}
const transaction = async <T>(
  mode: "r" | "rw",
  opts: HistoryOperationOptions | undefined,
  operation: () => Promise<T>
): Promise<T> => {
  assertLease(opts)
  let active: { abort(): void } | undefined
  const abort = () => {
    try {
      active?.abort()
    } catch {
      /* Transaction already closed. */
    }
  }
  opts?.signal?.addEventListener("abort", abort, { once: true })
  try {
    return await db.transaction(mode, tables(), async (tx) => {
      active = tx
      assertLease(opts)
      const result = await operation()
      assertLease(opts)
      return result
    })
  } finally {
    opts?.signal?.removeEventListener("abort", abort)
  }
}
const bookmarkKey = (
  scope: HistoryBookmarkScope,
  owner: { owner_key: string; conversation_id: string }
) => [
  scope.profile_id,
  scope.client_session_id,
  owner.owner_key,
  owner.conversation_id
]
const assertBookmarkProfile = async (scope: HistoryBookmarkScope) => {
  if (
    (await db.userSettings.get("main"))?.history_profile_id !==
      scope.profile_id ||
    !scope.client_session_id
  )
    fail("bookmark_owner_mismatch")
}
export const loadHistoryBookmark = async (
  scope: HistoryBookmarkScope,
  owner: { owner_key: string; conversation_id: string }
): Promise<HistoryBookmark | null> =>
  db.transaction("r", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    return (await db.historySelections.get(bookmarkKey(scope, owner))) ?? null
  })
export const saveHistoryBookmark = async (
  scope: HistoryBookmarkScope,
  view: HistoryViewSelectionV1
): Promise<void> =>
  db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    if (
      !view.owner_key ||
      !view.conversation_id ||
      !view.view_session_id ||
      !Number.isSafeInteger(view.selection_revision) ||
      view.selection_revision < 0
    )
      fail("invalid_bookmark")
    const previous = await db.historySelections.get(bookmarkKey(scope, view))
    if (
      previous?.view.view_session_id === view.view_session_id &&
      previous.view.selection_revision > view.selection_revision
    )
      fail("stale_selection")
    await db.historySelections.put({
      ...scope,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      view: structuredClone(view),
      ...(previous?.pending_confirmation
        ? {
            pending_confirmation: previous.pending_confirmation,
            pending_view_session_id: previous.pending_view_session_id
          }
        : {})
    })
  })
/** Durable pending identity is stored before remote dispatch; never replaces the accepted view. */
export const savePendingHistoryConfirmation = async (
  scope: HistoryBookmarkScope,
  view: HistoryViewSelectionV1,
  confirmation: LegacyHistoryProjectionConfirmV1
): Promise<void> =>
  db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    const prior = await db.historySelections.get(bookmarkKey(scope, view))
    if (
      prior?.pending_confirmation &&
      canonicalHistoryJson(prior.pending_confirmation) !==
        canonicalHistoryJson(confirmation)
    )
      fail("pending_confirmation_conflict")
    await db.historySelections.put({
      ...scope,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      view: prior?.view ?? structuredClone(view),
      pending_confirmation: structuredClone(confirmation),
      pending_view_session_id:
        prior?.pending_view_session_id ?? view.view_session_id
    })
  })
export const acknowledgeHistoryConfirmation = async (
  scope: HistoryBookmarkScope,
  projection: LegacyHistoryProjectionV1
): Promise<void> =>
  db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    const prior = await db.historySelections.get(bookmarkKey(scope, projection))
    if (
      !prior?.pending_confirmation ||
      prior.pending_confirmation.projection_id !== projection.projection_id
    )
      fail("missing_pending_confirmation")
    const {
      projection_digest: _digest,
      created_at: _date,
      ...confirmation
    } = projection
    if (
      canonicalHistoryJson(prior.pending_confirmation) !==
      canonicalHistoryJson(confirmation)
    )
      fail("projection_id_conflict")
    const unchanged =
      prior.view.selection_revision === confirmation.selection_revision &&
      prior.view.view_session_id === prior.pending_view_session_id
    await db.historySelections.put({
      ...scope,
      owner_key: projection.owner_key,
      conversation_id: projection.conversation_id,
      view: unchanged
        ? {
            ...prior.view,
            interpretation: {
              kind: "legacy_linear_v1",
              projection_id: projection.projection_id
            },
            cursor: projection.cursor
          }
        : prior.view
    })
  })

/** Substantive state excludes recursive admission/settlement, but binds protected ancestry. */
const messageState = (row: Message) => ({
  ...stripHistoryAuthority(row),
  history_provenance: row.history_provenance ?? null
})
const messageRevision = (row: Message) => historyDigest(messageState(row))
const localImages = (row: Message): string[] => {
  const images = (row.images ?? []).filter((image) => image !== "")
  return images.length ? images : row.image ? [row.image] : []
}
const nodeFor = (row: Message): HistoryNodeV1 => ({
  id: row.id,
  revision: messageRevision(row),
  parent_id: row.parent_message_id ?? null,
  role: row.role,
  settled: true,
  conversation_id: row.history_id,
  preview: Array.from(row.content || "")
    .slice(0, 200)
    .join(""),
  legacy_projection_id: row.history_provenance?.projection_id ?? null,
  assets: localImages(row).map((image, i) => ({
    id: `${row.id}:image:${i}`,
    revision: historyDigest(image),
    kind: "image"
  }))
})
const readSnapshot = async (
  owner: LocalHistoryOwnerV1,
  projectionId?: string
): Promise<{
  snapshot: HistorySelectionSnapshotV1
  records: Message[]
  unsupported?: string
  stale?: string
}> => {
  const record = await ownedHistory(owner)
  const records = (
    await db.messages
      .where("history_id")
      .equals(owner.conversation_id)
      .toArray()
  ).sort(
    (a, b) =>
      a.createdAt - b.createdAt || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
  )
  const nodes = records.map(nodeFor)
  const files = await db.sessionFiles.get(owner.conversation_id)
  const compare = await db.compareStates.get(owner.conversation_id)
  const storage_context_digest = historyDigest({
    prompt: record.last_used_prompt ?? null,
    character_id: record.character_id ?? null,
    doc_id: record.doc_id ?? null,
    is_rag: record.is_rag,
    files: files ?? null
  })
  const source_digest = historyDigest(nodes)
  let snapshot: HistorySelectionSnapshotV1 = {
    version: 1,
    owner_key: owner.owner_key,
    conversation_id: owner.conversation_id,
    nodes,
    source_digest,
    storage_context_digest,
    fences: {
      conversation: historyDigest([
        record.id,
        record.local_owner_key ?? owner.owner_key
      ]),
      history: source_digest,
      settings: storage_context_digest
    },
    interpretation_status: { kind: "legacy_review_required" }
  }
  let stale: string | undefined
  if (projectionId) {
    const projection = await db.historyProjections.get([
      owner.owner_key,
      owner.conversation_id,
      projectionId
    ])
    if (!projection) stale = "missing_projection"
    else {
      const revisions = new Map(
        projection.source_members.map((n) => [n.id, n.revision])
      )
      const live = new Map(nodes.map((n) => [n.id, n.revision]))
      if (
        projection.ordered_path_ids.some(
          (id) => live.get(id) !== revisions.get(id)
        )
      )
        stale = "stale_projection"
      snapshot = {
        ...snapshot,
        interpretation_status: {
          kind: "legacy_linear_v1",
          projection_id: projectionId,
          ordered_path_ids: projection.ordered_path_ids
        }
      }
    }
  } else {
    try {
      resolveParentPath(nodes, { kind: "empty" })
      // A unique old chain remains usable when protected new edges branch from it.
      // Ambiguity is multiple unversioned children (including roots), not new admitted edges.
      const unversionedParents = new Set<string | null>()
      const ambiguous = records.some((row) => {
        if (row.history_provenance?.projection_id) return true
        if (row.history_provenance?.owner_key === owner.owner_key) return false
        const parent = row.parent_message_id ?? null
        if (unversionedParents.has(parent)) return true
        unversionedParents.add(parent)
        return false
      })
      if (!ambiguous)
        snapshot = {
          ...snapshot,
          interpretation_status: { kind: "parent_graph_v1" }
        }
    } catch {
      /* Ambiguous legacy stays readable for review. */
    }
  }
  const unsupported =
    record.last_used_prompt?.prompt_id &&
    record.last_used_prompt.prompt_content === undefined
      ? "unsupported_local_prompt_reference"
      : record.character_id
        ? "unsupported_local_character_context"
        : record.doc_id || record.is_rag
          ? "unsupported_local_external_context"
          : compare?.compareMode
            ? "unsupported_comparison_history"
            : undefined
  return { snapshot, records, unsupported, stale }
}
const capture = async (
  owner: LocalHistoryOwnerV1,
  view: HistoryViewSelectionV1,
  purpose: "send" | "fork"
): Promise<HistoryCaptureResultV1> => {
  const { snapshot, records, unsupported, stale } = await readSnapshot(
    owner,
    view.interpretation.kind === "legacy_linear_v1"
      ? view.interpretation.projection_id
      : undefined
  )
  if (stale)
    return {
      status: "stale_selection",
      code: stale,
      snapshot,
      view: structuredClone(view)
    }
  if (unsupported)
    return {
      status: "unsupported_history_capability",
      code: unsupported,
      snapshot,
      view: structuredClone(view)
    }
  const resolved = resolveHistorySelection(snapshot, view, purpose, "")
  if (resolved.status !== "ready")
    return { ...resolved, snapshot, view: structuredClone(view) }
  const byId = new Map(records.map((r) => [r.id, r]))
  if (
    resolved.rows.some((node) =>
      localImages(byId.get(node.id)!).some(
        (image) => !/^data:[^,]+,/.test(image)
      )
    )
  ) {
    return {
      status: "unsupported_history_capability",
      code: "unsupported_local_message_assets",
      snapshot,
      view: structuredClone(view)
    }
  }
  const selected_content = bindSelectedHistoryContent(
    resolved.rows,
    resolved.rows.map((n) => {
      const r = byId.get(n.id)!
      return {
        id: n.id,
        revision: n.revision,
        message: r.content,
        images: localImages(r),
        extra_metadata: {
          ...r.metadataExtra,
          sender_name: r.name,
          sender_role: r.role,
          local_history: {
            messageType: r.messageType,
            sources: r.sources,
            search: r.search,
            documents: r.documents,
            discoSkillComment: r.discoSkillComment,
            generationInfo: r.generationInfo,
            reasoning_time_taken: r.reasoning_time_taken,
            clusterId: r.clusterId,
            modelId: r.modelId,
            modelName: r.modelName,
            modelImage: r.modelImage
          }
        },
        ...(Array.isArray(r.metadataExtra?.tool_calls)
          ? {
              tool_calls: r.metadataExtra.tool_calls as Record<
                string,
                unknown
              >[]
            }
          : {})
      }
    })
  )
  return {
    status: "captured",
    snapshot,
    rows: resolved.rows,
    selected_content,
    view: structuredClone(view),
    purpose,
    storage_context_digest: snapshot.storage_context_digest
  }
}
export const captureLocalHistorySnapshot = (
  owner: LocalHistoryOwnerV1,
  view: HistoryViewSelectionV1,
  purpose: "send" | "fork",
  opts?: HistoryOperationOptions
): Promise<HistoryCaptureResultV1> => {
  const intent = structuredClone(view)
  return transaction("r", opts, () => capture(owner, intent, purpose))
}

export const confirmLocalHistoryProjection = (
  owner: LocalHistoryOwnerV1,
  scope: HistoryBookmarkScope,
  confirmation: LegacyHistoryProjectionConfirmV1,
  opts?: HistoryOperationOptions & { view?: HistoryViewSelectionV1 }
): Promise<LegacyHistoryProjectionV1> => {
  confirmation = structuredClone(confirmation)
  return transaction("rw", opts, async () => {
    await ownedHistory(owner)
    await assertBookmarkProfile(scope)
    if (
      confirmation.version !== 1 ||
      confirmation.owner_key !== owner.owner_key ||
      confirmation.conversation_id !== owner.conversation_id ||
      !confirmation.projection_id
    )
      fail("invalid_projection")
    const existing = await db.historyProjections.get([
      owner.owner_key,
      owner.conversation_id,
      confirmation.projection_id
    ])
    const digest = historyDigest(confirmation)
    let projection = existing
    if (existing) {
      if (existing.projection_digest !== digest) fail("projection_id_conflict")
    } else {
      const { snapshot, unsupported } = await readSnapshot(owner)
      if (unsupported) fail(unsupported)
      if (
        confirmation.source_digest !== snapshot.source_digest ||
        canonicalHistoryJson(confirmation.fences) !==
          canonicalHistoryJson(snapshot.fences) ||
        canonicalHistoryJson(confirmation.source_members) !==
          canonicalHistoryJson(
            snapshot.nodes.map(({ id, revision }) => ({ id, revision }))
          )
      )
        fail("stale_source")
      const ids = new Set(snapshot.nodes.map((n) => n.id))
      if (
        new Set(confirmation.ordered_path_ids).size !==
          confirmation.ordered_path_ids.length ||
        confirmation.ordered_path_ids.some((id) => !ids.has(id)) ||
        (confirmation.cursor.kind !== "empty" &&
          !confirmation.ordered_path_ids.includes(
            confirmation.cursor.message_id
          ))
      )
        fail("invalid_projection")
      projection = {
        ...structuredClone(confirmation),
        projection_digest: digest,
        created_at: new Date().toISOString()
      }
      await db.historyProjections.add(projection)
    }
    const prior = await db.historySelections.get(bookmarkKey(scope, owner))
    if (
      !prior ||
      (prior.view.selection_revision === confirmation.selection_revision &&
        (!opts?.view ||
          prior.view.view_session_id === opts.view.view_session_id))
    )
      await db.historySelections.put({
        ...scope,
        owner_key: owner.owner_key,
        conversation_id: owner.conversation_id,
        view: {
          view_session_id:
            prior?.view.view_session_id ??
            opts?.view?.view_session_id ??
            scope.client_session_id,
          owner_key: owner.owner_key,
          conversation_id: owner.conversation_id,
          interpretation: {
            kind: "legacy_linear_v1",
            projection_id: confirmation.projection_id
          },
          cursor: confirmation.cursor,
          selection_revision: confirmation.selection_revision
        }
      })
    return projection!
  })
}
const normalizeMessage = (
  owner: LocalHistoryOwnerV1,
  input: Message,
  role: "user" | "assistant",
  parent: string | null,
  projection_id?: string
): Message => {
  if (
    !input.id ||
    input.role !== role ||
    input.history_id !== owner.conversation_id
  )
    fail("invalid_message")
  if (
    input.parent_message_id !== undefined &&
    input.parent_message_id !== parent
  )
    fail("parent_mismatch")
  if (
    input.documents?.length ||
    localImages(input).some((image) => !/^data:[^,]+,/.test(image))
  )
    fail("unsupported_local_message_assets")
  return {
    ...sanitizeImportedMessage(input),
    images: localImages(input),
    parent_message_id: parent,
    history_provenance: {
      version: 1,
      owner_key: owner.owner_key,
      projection_id: projection_id ?? null
    }
  }
}
export const appendLocalSelectedUser = (
  owner: LocalHistoryOwnerV1,
  selection: HistorySelectionV1,
  input: Message,
  opts?: HistoryOperationOptions
): Promise<HistoryAdmissionV1> => {
  selection = structuredClone(selection)
  input = structuredClone(input)
  return transaction("rw", opts, async () => {
    if (
      selection.version !== 1 ||
      selection.owner_key !== owner.owner_key ||
      selection.conversation_id !== owner.conversation_id ||
      selection.purpose !== "send" ||
      selectionDigest(selection) !== selection.selection_digest
    )
      fail("invalid_selection")
    await ownedHistory(owner)
    const parent = selection.messages.at(-1)?.id ?? null
    const message = normalizeMessage(
      owner,
      input,
      "user",
      parent,
      selection.interpretation.kind === "legacy_linear_v1"
        ? selection.interpretation.projection_id
        : undefined
    )
    const admission: HistoryAdmissionV1 = {
      version: 1,
      owner_key: owner.owner_key,
      conversation_id: owner.conversation_id,
      input_message_id: message.id,
      input_message_revision: messageRevision(message),
      selection_digest: selection.selection_digest,
      messages: structuredClone(selection.messages),
      originating_selection_revision: selection.selection_revision
    }
    const existing = await db.messages.get(message.id)
    if (existing) {
      if (
        canonicalHistoryJson(existing.history_admission ?? null) !==
          canonicalHistoryJson(admission) ||
        messageRevision(existing) !== admission.input_message_revision
      )
        fail("message_id_conflict")
      return existing.history_admission!
    }
    const current = await capture(
      owner,
      { view_session_id: "admission", ...selection },
      "send"
    )
    if (current.status !== "captured") fail(current.code)
    const resolved = resolveHistorySelection(
      current.snapshot,
      current.view,
      "send",
      selection.request_context_digest
    )
    if (
      resolved.status !== "ready" ||
      resolved.selection.selection_digest !== selection.selection_digest
    )
      fail("stale_selection")
    await db.messages.add({ ...message, history_admission: admission })
    return admission
  })
}
export const settleLocalAcceptedAssistant = (
  owner: LocalHistoryOwnerV1,
  admission: HistoryAdmissionReferenceV1,
  input: Message,
  opts?: HistoryOperationOptions
): Promise<Message> => {
  admission = structuredClone(admission)
  input = structuredClone(input)
  return transaction("rw", opts, async () => {
    await ownedHistory(owner)
    const parent = await db.messages.get(admission.input_message_id)
    const stored = parent?.history_admission
    if (
      !stored ||
      parent?.history_id !== owner.conversation_id ||
      admission.version !== 1 ||
      admission.owner_key !== owner.owner_key ||
      admission.conversation_id !== owner.conversation_id ||
      (
        [
          "input_message_id",
          "input_message_revision",
          "selection_digest"
        ] as const
      ).some((k) => admission[k] !== stored[k]) ||
      messageRevision(parent) !== stored.input_message_revision
    )
      fail("stale_parent")
    const message = normalizeMessage(
      owner,
      input,
      "assistant",
      parent.id,
      parent.history_provenance?.projection_id ?? undefined
    )
    const existing = await db.messages.get(message.id)
    const settlement = {
      input_message_id: parent.id,
      selection_digest: admission.selection_digest,
      result_revision: messageRevision(message)
    }
    if (existing) {
      if (
        canonicalHistoryJson(existing.history_settlement ?? null) !==
          canonicalHistoryJson(settlement) ||
        messageRevision(existing) !== settlement.result_revision
      )
        fail("message_id_conflict")
      return existing
    }
    const result = { ...message, history_settlement: settlement }
    await db.messages.add(result)
    return result
  })
}
