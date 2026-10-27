import type { RoleplayResumeRecovery } from "@/services/tldw/roleplay-resume"
/** Local H1 authority. All hashes run synchronously within the owning Dexie transaction. */
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import { db } from "./schema"
import type {
  HistoryInfo,
  Message,
  HistoryBookmark,
  HistoryTurnRecovery,
  HistoryBookmarkScope,
  UploadedFile
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
  LOCAL_HISTORY_OWNER_KEY_PREFIX,
  resolveHistorySelection,
  resolveParentPath,
  selectionDigest
} from "@/utils/history-selection"

/** A captured reader lease, not permission to adopt or mutate a history. */
export type OnDeviceHistoryScope = {
  profileId: string | null;
  serverScopeKey: string | null;
  isCurrent: () => boolean;
};

export function isOnDeviceHistory(history: HistoryInfo, scope: OnDeviceHistoryScope): boolean {
  if (history.id === "temp" || history.server_chat_id || history.message_source === "server") return false;
  const profileKey = scope.profileId ? `local-history-v1:${scope.profileId}` : null;
  if (history.local_owner_key && history.local_owner_key !== profileKey) return false;
  if (history.server_scope_key) return !!scope.serverScopeKey && history.server_scope_key === scope.serverScopeKey;
  return !!profileKey && history.local_owner_key === profileKey;
}

/** A view receipt, valid only while the same successful Server page is visible. */
export type VisibleServerChats = {
  serverScopeKey: string;
  query: string;
  ids: ReadonlySet<string>;
  isCurrent: () => boolean;
};
export type OnDeviceHistoryReadOptions = {
  includeCachedServerChats?: boolean;
  visibleServerChats?: VisibleServerChats | null;
};

export function isOwnedCachedServerHistory(history: HistoryInfo, scope: OnDeviceHistoryScope): boolean {
  if (history.id === "temp" || !history.server_chat_id?.trim() || !scope.serverScopeKey ||
      history.server_scope_key !== scope.serverScopeKey) return false;
  return !history.local_owner_key ||
    (!!scope.profileId && history.local_owner_key === `local-history-v1:${scope.profileId}`);
}

/** Qualification and exact visible-page deduplication precede counts and paging. */
export function isReadableOnDeviceHistory(history: HistoryInfo, scope: OnDeviceHistoryScope,
  options?: OnDeviceHistoryReadOptions, query = ""): boolean {
  if (isOnDeviceHistory(history, scope)) return true;
  if (!options?.includeCachedServerChats || !isOwnedCachedServerHistory(history, scope)) return false;
  const visible = options.visibleServerChats;
  return !(history.server_scope_type === "global" && !history.server_workspace_id &&
    visible?.serverScopeKey === scope.serverScopeKey &&
    visible.query.trim().toLowerCase() === query.trim().toLowerCase() &&
    visible.isCurrent() && visible.ids.has(history.server_chat_id!));
}

export type LocalHistoryOwnerV1 = {
  readonly kind: "local"
  readonly profile_id: string
  readonly owner_key: string
  readonly conversation_id: string
  /** Present only when this local history is bound to a live account. */
  readonly validate_lease?: () => boolean
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
  key === "history_settlement" ||
  key === "history_existing_settlement"
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
  delete result.local_settings_guard
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

export const ensureLocalProfileId = async (
  expectedProfileId?: string | null
): Promise<string> => {
  // A qualified reader can verify its existing profile but cannot create a
  // replacement while its storage observer is still catching up.
  if (expectedProfileId !== undefined)
    return db.transaction("r", [db.userSettings], async () => {
      const settings = await db.userSettings.get("main")
      if (!expectedProfileId || settings?.history_profile_id !== expectedProfileId)
        fail("owner_mismatch")
      return expectedProfileId
    })
  return db.transaction("rw", [db.userSettings], async () => {
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
}
const localKey = (profile: string) => `${LOCAL_HISTORY_OWNER_KEY_PREFIX}${profile}`
export const getLocalHistoryOwner = async (
  conversation_id: string,
  readerScope?: OnDeviceHistoryScope
): Promise<LocalHistoryOwnerV1> => {
  // A qualified click already captured a profile. Do not initialize/adopt a
  // replacement if that profile disappeared while the observer was pending.
  const profile_id = readerScope ? readerScope.profileId : await ensureLocalProfileId()
  if (!profile_id || (readerScope && !readerScope.isCurrent())) fail("owner_mismatch")
  const owner = {
    kind: "local" as const,
    profile_id,
    owner_key: localKey(profile_id),
    conversation_id
  }
  await db.transaction("rw", [db.chatHistories, db.userSettings], async () => {
    const record = await ownedHistory(owner)
    if (readerScope && (!readerScope.isCurrent() || !isOnDeviceHistory(record, readerScope))) fail("owner_mismatch")
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
/** History replacement/deletion must not erase the fence for an uncertain external write. */
export const assertNoPendingLocalSettingsWrite = (
  history: HistoryInfo | undefined
) => {
  const guard = history?.local_settings_guard
  if (guard && (!Array.isArray(guard.pending) || guard.pending.length))
    fail("history_settings_write_pending")
}

/** ID-only legacy callers use the current profile; H1 actions must supply their captured owner. */
export const withLocalMessageMutation = async <T,>(
  conversation_id: string,
  capturedOwner: LocalHistoryOwnerV1 | undefined,
  mutate: () => Promise<T>,
  options?: HistoryOperationOptions
): Promise<T> =>
  db.transaction(
    "rw",
    [db.messages, db.chatHistories, db.userSettings],
    async () => {
      assertLease(capturedOwner)
      assertLease(options)
      const profile = await db.userSettings.get("main")
      const owner = capturedOwner ?? {
        kind: "local" as const,
        profile_id: profile?.history_profile_id ?? "",
        owner_key: localKey(profile?.history_profile_id ?? ""),
        conversation_id
      }
      if (!owner.profile_id || owner.conversation_id !== conversation_id)
        fail("owner_mismatch")
      await ownedHistory(owner)
      assertLease(owner)
      assertLease(options)
      const result = await mutate()
      assertLease(owner)
      assertLease(options)
      return result
    }
  )

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
/** Explicit consent binds a plain local import; imported trust fields never do. */
export const bindUnboundLocalHistory = async (
  conversation_id: string,
  opts: HistoryOperationOptions & { signal: AbortSignal; validate_lease: () => boolean }
): Promise<LocalHistoryOwnerV1> => {
  assertLease(opts)
  const profile_id = await ensureLocalProfileId()
  const owner: LocalHistoryOwnerV1 = { kind: "local", profile_id, owner_key: localKey(profile_id), conversation_id }
  return transaction("rw", opts, async () => {
    const record = await ownedHistory(owner)
    if (record.local_owner_key != null || record.server_scope_key != null || record.server_chat_id != null || record.message_source === "server")
      fail("not_unbound_local_history")
    assertNoPendingLocalSettingsWrite(record)
    await db.chatHistories.update(record.id, { local_owner_key: owner.owner_key })
    return owner
  })
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
      pending_turns: previous?.pending_turns,
      history_turn_outcomes: previous?.history_turn_outcomes,
      ...scope,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      view: structuredClone(view),
      ...(previous?.pending_confirmation
        ? {
            pending_confirmation: previous.pending_confirmation,
            pending_view_session_id: previous.pending_view_session_id,
            pending_dispatch_started: previous.pending_dispatch_started
          }
        : {})
    })
  })
/** Durable pending identity is stored before remote dispatch; never replaces the accepted view. */
export const savePendingHistoryConfirmation = async (
  scope: HistoryBookmarkScope,
  view: HistoryViewSelectionV1,
  confirmation: LegacyHistoryProjectionConfirmV1
): Promise<string> =>
  db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    const prior = await db.historySelections.get(bookmarkKey(scope, view))
    if (
      prior?.pending_confirmation &&
      canonicalHistoryJson(prior.pending_confirmation) !==
        canonicalHistoryJson(confirmation)
    )
      fail("pending_confirmation_conflict")
    const origin = prior?.pending_view_session_id ?? view.view_session_id
    await db.historySelections.put({
      pending_turns: prior?.pending_turns,
      history_turn_outcomes: prior?.history_turn_outcomes,
      ...scope,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      view: prior?.view ?? structuredClone(view),
      pending_confirmation: structuredClone(confirmation),
      pending_view_session_id: origin,
      pending_dispatch_started: prior?.pending_confirmation
        ? prior.pending_dispatch_started !== false
        : false
    })
    return origin
  })
const matchesPendingConfirmation = (
  prior: HistoryBookmark | undefined,
  view: HistoryViewSelectionV1,
  confirmation: LegacyHistoryProjectionConfirmV1
): boolean =>
  confirmation.owner_key === view.owner_key &&
  confirmation.conversation_id === view.conversation_id &&
  confirmation.selection_revision === view.selection_revision &&
  prior?.pending_view_session_id === view.view_session_id &&
  !!prior?.pending_confirmation &&
  canonicalHistoryJson(prior.pending_confirmation) ===
    canonicalHistoryJson(confirmation)

/** Persist before dispatch so another matching attempt cannot mistake uncertainty for non-dispatch. */
export const markPendingHistoryConfirmationDispatched = async (
  scope: HistoryBookmarkScope,
  view: HistoryViewSelectionV1,
  confirmation: LegacyHistoryProjectionConfirmV1
): Promise<void> =>
  db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    const prior = await db.historySelections.get(bookmarkKey(scope, view))
    if (!prior || !matchesPendingConfirmation(prior, view, confirmation))
      fail("missing_pending_confirmation")
    await db.historySelections.put({ ...prior, pending_dispatch_started: true })
  })

/** Resolve a proven non-commit rejection, or an intent proven never dispatched by any attempt. */
export const rejectPendingHistoryConfirmation = async (
  scope: HistoryBookmarkScope,
  view: HistoryViewSelectionV1,
  confirmation: LegacyHistoryProjectionConfirmV1,
  options?: { onlyIfUndispatched?: boolean }
): Promise<boolean> =>
  db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    const prior = await db.historySelections.get(bookmarkKey(scope, view))
    if (
      !prior ||
      !matchesPendingConfirmation(prior, view, confirmation) ||
      (options?.onlyIfUndispatched && prior.pending_dispatch_started !== false)
    )
      return false
    const {
      pending_confirmation: _intent,
      pending_view_session_id: _session,
      pending_dispatch_started: _dispatched,
      ...accepted
    } = prior
    await db.historySelections.put(accepted)
    return true
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
      pending_turns: prior?.pending_turns,
      history_turn_outcomes: prior?.history_turn_outcomes,
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
/** Forks share the owner/profile fence and transaction, with a separately retained context policy. */
export const withLocalForkSource = <T>(
  owner: LocalHistoryOwnerV1,
  projectionId: string | undefined,
  mode: "r" | "rw",
  operation: (source: {
    snapshot: HistorySelectionSnapshotV1; records: Message[]; history: HistoryInfo;
    files: import("./types").SessionFiles | undefined; comparison: boolean
  }) => Promise<T>,
  opts?: HistoryOperationOptions
): Promise<T> => transaction(mode, opts, async () => {
  const result = await readSnapshot(owner, projectionId)
  if (result.stale) fail(result.stale)
  if (result.unsupported && result.unsupported !== "unsupported_comparison_history") fail(result.unsupported)
  return operation({snapshot: result.snapshot, records: result.records,
    history: await ownedHistory(owner), files: await db.sessionFiles.get(owner.conversation_id),
    comparison: result.unsupported === "unsupported_comparison_history"})
})

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
        pending_turns: prior?.pending_turns,
        history_turn_outcomes: prior?.history_turn_outcomes,
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
/** Captured document references are display metadata, never mutable context authority. */
const validDocumentPresentation = (documents: unknown): boolean => {
  if (!Array.isArray(documents)) return false
  const fields = new Set(["type", "title", "url", "tabId", "favIconUrl", "filename", "fileSize", "processed"])
  return Array.from(documents).every(document => {
    if (!document || typeof document !== "object" || Array.isArray(document) ||
        (document.type !== "file" && document.type !== "tab") ||
        Object.keys(document).some(key => !fields.has(key))) return false
    return Object.entries(document).every(([key, value]) => value === undefined ||
      (key === "fileSize" ? typeof value === "number" && Number.isFinite(value) && value >= 0 :
       key === "tabId" ? typeof value === "number" && Number.isSafeInteger(value) && value >= 0 :
       key === "processed" ? typeof value === "boolean" : typeof value === "string"))
  })
}

const normalizeMessage = (
  owner: LocalHistoryOwnerV1,
  input: Message,
  role: "user" | "assistant",
  parent: string | null,
  projection_id?: string,
  documentPresentation = false
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
    (input.documents !== undefined &&
      (!Array.isArray(input.documents) ||
        (input.documents.length > 0 && (!documentPresentation || !validDocumentPresentation(input.documents))))) ||
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
  opts?: HistoryOperationOptions & { documentPresentation?: boolean }
): Promise<HistoryAdmissionV1> => {
  selection = structuredClone(selection)
  input = structuredClone(input)
  const documentPresentation = opts?.documentPresentation === true
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
        : undefined,
      documentPresentation
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
/**
 * The admission a local question was accepted with, while the question is
 * unchanged. A retried reply settles against it again (CS-04), so the
 * question is not admitted a second time. Null when there is none.
 */
export const loadLocalHistoryAdmission = (
  owner: LocalHistoryOwnerV1,
  messageId: string,
  opts?: HistoryOperationOptions
): Promise<HistoryAdmissionReferenceV1 | null> =>
  transaction("r", opts, async () => {
    await ownedHistory(owner)
    const message = await db.messages.get(messageId)
    const admission = message?.history_admission
    if (
      !message ||
      !admission ||
      message.history_id !== owner.conversation_id ||
      admission.version !== 1 ||
      admission.owner_key !== owner.owner_key ||
      admission.conversation_id !== owner.conversation_id ||
      admission.input_message_id !== message.id ||
      admission.input_message_revision !== messageRevision(message)
    )
      return null
    return {
      version: admission.version,
      owner_key: admission.owner_key,
      conversation_id: admission.conversation_id,
      input_message_id: admission.input_message_id,
      input_message_revision: admission.input_message_revision,
      selection_digest: admission.selection_digest
    }
  })
export const settleLocalAcceptedAssistant = (
  owner: LocalHistoryOwnerV1,
  admission: HistoryAdmissionReferenceV1,
  input: Message,
  opts?: HistoryOperationOptions,
  sessionFilesToAdd: UploadedFile[] = []
): Promise<Message> => {
  admission = structuredClone(admission)
  input = structuredClone(input)
  sessionFilesToAdd = structuredClone(sessionFilesToAdd)
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
    if (sessionFilesToAdd.length) {
      const session = await db.sessionFiles.get(owner.conversation_id)
      if (session && (session.sessionId !== owner.conversation_id || !Array.isArray(session.files)))
        fail("unsupported_local_message_assets")
      const files = [...(session?.files ?? [])]
      if (files.some(file => !file || typeof file !== "object" || Array.isArray(file) ||
          typeof file.id !== "string" || !file.id)) fail("unsupported_local_message_assets")
      for (const file of sessionFilesToAdd) {
        if (!file || typeof file !== "object" || Array.isArray(file) ||
            typeof file.id !== "string" || !file.id || typeof file.filename !== "string" ||
            typeof file.type !== "string" || typeof file.content !== "string" ||
            typeof file.processed !== "boolean" || !Number.isFinite(file.size) || file.size < 0 ||
            !Number.isFinite(file.uploadedAt)) fail("unsupported_local_message_assets")
        const previous = files.find(candidate => candidate.id === file.id)
        if (previous && canonicalHistoryJson(previous) !== canonicalHistoryJson(file))
          fail("local_file_id_conflict")
        if (!previous) files.push(file)
      }
      await db.sessionFiles.put({
        ...session, sessionId: owner.conversation_id, files,
        retrievalEnabled: session?.retrievalEnabled ?? false,
        createdAt: session?.createdAt ?? Date.now()
      })
    }
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

export const settleLocalSelectedExistingInput = (
  owner: LocalHistoryOwnerV1,
  selection: HistorySelectionV1,
  input: Message,
  opts?: HistoryOperationOptions
): Promise<Message> => {
  selection = structuredClone(selection)
  input = structuredClone(input)
  return transaction("rw", opts, async () => {
    await ownedHistory(owner)
    const questionId = selection.messages.at(-1)?.id
    if (
      selection.version !== 1 || selection.owner_key !== owner.owner_key ||
      selection.conversation_id !== owner.conversation_id || selection.purpose !== "send" ||
      selection.cursor.kind !== "after_message" || selection.cursor.message_id !== questionId ||
      selectionDigest(selection) !== selection.selection_digest || !questionId
    ) fail("invalid_selection")
    const question = await db.messages.get(questionId)
    if (!question || question.history_id !== owner.conversation_id ||
        question.role !== "user" || messageRevision(question) !== selection.messages.at(-1)?.revision)
      fail("stale_parent")
    const message = normalizeMessage(
      owner, input, "assistant", questionId,
      selection.interpretation.kind === "legacy_linear_v1"
        ? selection.interpretation.projection_id : undefined
    )
    const intent = historyDigest(messageState(message))
    const existing = await db.messages.get(message.id)
    if (existing) {
      const proof = existing.history_existing_settlement
      if (!proof || existing.history_id !== owner.conversation_id ||
          canonicalHistoryJson(proof.selection) !== canonicalHistoryJson(selection) ||
          proof.intent_digest !== intent || messageRevision(existing) !== proof.result_revision)
        fail("message_id_conflict")
      return existing
    }
    const current = await capture(owner, { view_session_id: "existing-input", ...selection }, "send")
    if (current.status !== "captured") fail(current.code)
    const resolved = resolveHistorySelection(
      current.snapshot, current.view, "send", selection.request_context_digest
    )
    if (resolved.status !== "ready" || resolved.selection.selection_digest !== selection.selection_digest)
      fail("stale_selection")
    const result = {
      ...message,
      history_existing_settlement: {
        selection,
        intent_digest: intent,
        result_revision: messageRevision(message)
      }
    }
    await db.messages.add(result)
    return result
  })
}


const HISTORY_TURN_OUTCOMES = new Set<string>(["complete", "interrupted", "stopped"])

const projectHistoryTurnRecovery = async (
  turn: HistoryTurnRecovery
): Promise<HistoryTurnRecovery> => {
  if (
    turn.action !== undefined &&
    turn.action !== "edit_resend" &&
    turn.action !== "regenerate"
  )
    fail("unsupported_history_action")
  if (
    turn.action === "regenerate" &&
    (turn.persistence !== "server" ||
      turn.origin_view?.interpretation.kind !== "parent_graph_v1" ||
      turn.origin_view.cursor.kind !== "after_message")
  )
    fail("unsupported_history_action")
  // Whitelist the recovery projection; runtime extras cannot persist credentials/capabilities.
  const detached: HistoryTurnRecovery = structuredClone({
    operation_id: turn.operation_id,
    origin_view: turn.origin_view,
    ...(turn.action ? { action: turn.action } : {}),
    selection_digest: turn.selection_digest,
    request_context_digest: turn.request_context_digest,
    owner_key: turn.owner_key,
    conversation_id: turn.conversation_id,
    persistence: turn.persistence ?? "client",
    input_id: turn.input_id,
    assistant_id: turn.assistant_id,
    created_at: turn.created_at,
    input_text: turn.input_text,
    input_images: turn.input_images,
    result_text: turn.result_text,
    state: turn.state,
    ...(turn.outcome && HISTORY_TURN_OUTCOMES.has(turn.outcome)
      ? { outcome: turn.outcome }
      : {}),
    ...(turn.interruption_reason
      ? { interruption_reason: turn.interruption_reason }
      : {}),
    ...(turn.settled_message_id
      ? { settled_message_id: turn.settled_message_id }
      : {}),
    ...(turn.model_name ? { model_name: turn.model_name } : {}),
    ...(turn.model_id ? { model_id: turn.model_id } : {}),
    ...(typeof turn.failure_reason === "string" && turn.failure_reason ? { failure_reason: turn.failure_reason } : {}),
    ...(turn.existing_selection ? {existing_selection: structuredClone(turn.existing_selection)} : {}),
    ...(turn.settlement_payload ? {settlement_payload: structuredClone({
      id: turn.settlement_payload.id, history_id: turn.settlement_payload.history_id,
      role: turn.settlement_payload.role, name: turn.settlement_payload.name,
      content: turn.settlement_payload.content, createdAt: turn.settlement_payload.createdAt,
      images: turn.settlement_payload.images, parent_message_id: turn.settlement_payload.parent_message_id,
      sources: turn.settlement_payload.sources, generationInfo: turn.settlement_payload.generationInfo,
      metadataExtra: turn.settlement_payload.metadataExtra,
      reasoning_time_taken: turn.settlement_payload.reasoning_time_taken,
      modelId: turn.settlement_payload.modelId, messageType: turn.settlement_payload.messageType
    })} : {}),
    ...(turn.roleplay_resume !== undefined
      ? { roleplay_resume: turn.roleplay_resume }
      : {}),
    ...(turn.retained_result_texts !== undefined
      ? { retained_result_texts: turn.retained_result_texts }
      : {}),
    ...(turn.admission
      ? {
          admission: {
            version: turn.admission.version,
            owner_key: turn.admission.owner_key,
            conversation_id: turn.admission.conversation_id,
            input_message_id: turn.admission.input_message_id,
            input_message_revision: turn.admission.input_message_revision,
            selection_digest: turn.admission.selection_digest
          }
        }
      : {})
  }) as HistoryTurnRecovery
  turn = detached
  let roleplay: RoleplayResumeRecovery | undefined
  if (turn.roleplay_resume !== undefined) {
    try {
      const { parseRoleplayResumeRecovery } = await import(
        "@/services/tldw/roleplay-resume"
      )
      roleplay = parseRoleplayResumeRecovery(turn.roleplay_resume)
    } catch {
      fail("history_operation_conflict")
    }
    const completed = roleplay!.completion
    if (
      turn.persistence !== "server" ||
      turn.action ||
      turn.admission ||
      turn.input_images.length ||
      roleplay!.intent.conversation_id !== turn.conversation_id ||
      roleplay!.intent.input.content !== turn.input_text ||
      turn.input_id !== roleplay!.append?.message_id ||
      turn.assistant_id !==
        (completed?.saved ? completed.assistant_message_id : undefined) ||
      (completed && turn.result_text !== completed.assistant_content) ||
      (turn.settled_message_id &&
        (!completed?.saved ||
          turn.settled_message_id !== completed.assistant_message_id)) ||
      (turn.outcome === "complete" && !completed?.saved)
    )
      fail("history_operation_conflict")
  }
  if (
    turn.retained_result_texts !== undefined &&
    (!roleplay ||
      !Array.isArray(turn.retained_result_texts) ||
      turn.retained_result_texts.some((value) => typeof value !== "string"))
  )
    fail("history_operation_conflict")
  if (roleplay) detached.roleplay_resume = roleplay
  return detached
}

// Storage failure keeps original-owner text in this window until it can be copied or saved.
const memoryTurnRecoveries = new Map<
  string,
  { scope: HistoryBookmarkScope; turn: HistoryTurnRecovery }
>()
const recoveryKey = (
  scope: HistoryBookmarkScope,
  turn: Pick<
    HistoryTurnRecovery,
    "owner_key" | "conversation_id" | "operation_id"
  >
) =>
  canonicalHistoryJson([
    scope.profile_id,
    scope.client_session_id,
    turn.owner_key,
    turn.conversation_id,
    turn.operation_id
  ])
const retainedTexts = (
  prior: HistoryTurnRecovery,
  later: HistoryTurnRecovery
): string[] =>
  [
    ...new Set([
      ...(prior.retained_result_texts ?? []),
      ...(later.retained_result_texts ?? []),
      ...(prior.result_text && prior.result_text !== later.result_text
        ? [prior.result_text]
        : [])
    ])
  ].filter((value) => value !== later.result_text)

/** A detached result grants no ancestry and is visible only to its original verified owner. */
export const retainHistoryTurnRecoveryInMemory = async (
  scope: HistoryBookmarkScope,
  view: HistoryViewSelectionV1,
  turn: HistoryTurnRecovery
): Promise<void> => {
  const detached = await projectHistoryTurnRecovery(turn);
  const ordinary =
    detached.persistence === 'server' && !detached.roleplay_resume && detached.action === undefined;
  if (
    !scope.profile_id ||
    !scope.client_session_id ||
    (!detached.roleplay_resume && !ordinary) ||
    !detached.result_text ||
    !detached.operation_id ||
    detached.owner_key !== view.owner_key ||
    detached.conversation_id !== view.conversation_id ||
    detached.origin_view.owner_key !== view.owner_key ||
    detached.origin_view.conversation_id !== view.conversation_id
  )
    fail('history_operation_conflict');
  if (
    ordinary &&
    (!detached.selection_digest ||
      !detached.request_context_digest ||
      (detached.admission &&
        (detached.admission.owner_key !== detached.owner_key ||
          detached.admission.conversation_id !== detached.conversation_id ||
          detached.admission.input_message_id !== detached.input_id ||
          detached.admission.selection_digest !== detached.selection_digest)) ||
      (!detached.admission && (detached.input_id || detached.assistant_id)) ||
      (detached.settled_message_id &&
        (!detached.admission || detached.settled_message_id !== detached.assistant_id)))
  )
    fail('history_operation_conflict');
  const key = recoveryKey(scope, detached),
    prior = memoryTurnRecoveries.get(key)
  if (prior) {
    if (ordinary) {
      if (prior.turn.roleplay_resume) fail('history_operation_conflict');
      for (const field of [
        'persistence',
        'action',
        'origin_view',
        'selection_digest',
        'request_context_digest',
        'created_at',
        'input_text',
        'input_images',
      ] as const)
        if (
          canonicalHistoryJson(prior.turn[field] ?? null) !==
          canonicalHistoryJson(detached[field] ?? null)
        )
          fail('history_operation_conflict');
      if (
        (prior.turn.admission &&
          detached.admission &&
          canonicalHistoryJson(prior.turn.admission) !==
            canonicalHistoryJson(detached.admission)) ||
        (prior.turn.input_id && detached.input_id && prior.turn.input_id !== detached.input_id) ||
        (prior.turn.assistant_id &&
          detached.assistant_id &&
          prior.turn.assistant_id !== detached.assistant_id)
      )
        fail('history_operation_conflict');
      if (
        (prior.turn.admission && !detached.admission) ||
        (prior.turn.settled_message_id && !detached.settled_message_id)
      )
        return;
    } else if (
      canonicalHistoryJson(prior.turn.roleplay_resume?.intent) !==
      canonicalHistoryJson(detached.roleplay_resume?.intent)
    )
      fail('history_operation_conflict');
    if (!ordinary) detached.retained_result_texts = retainedTexts(prior.turn, detached);
  }
  memoryTurnRecoveries.set(key, {
    scope: structuredClone(scope),
    turn: { ...detached, recovery_storage: "memory" }
  })
}

/** Store a credential-free operation intent before dispatch, separate from all ancestry. */
export const saveHistoryTurnRecovery = async (
  scope: HistoryBookmarkScope, view: HistoryViewSelectionV1, turn: HistoryTurnRecovery
): Promise<void> => {
  const detached = await projectHistoryTurnRecovery(turn)
  const key = recoveryKey(scope, detached),
    volatile = memoryTurnRecoveries.get(key)
  if (volatile && detached.roleplay_resume)
    detached.retained_result_texts = retainedTexts(volatile.turn, detached)
  let written = false
  await db.transaction("rw", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    if (!detached.operation_id || detached.owner_key !== view.owner_key || detached.conversation_id !== view.conversation_id)
      fail("owner_conversation_mismatch")
    const previous = await db.historySelections.get(bookmarkKey(scope, view))
    if (previous?.history_turn_outcomes?.[detached.operation_id]) return
    if (!detached.origin_view || detached.origin_view.owner_key !== view.owner_key || detached.origin_view.conversation_id !== view.conversation_id || !detached.selection_digest || !detached.request_context_digest || (detached.persistence !== "server" && (!detached.input_id || !detached.assistant_id)))
      fail("invalid_history_operation")
    if (detached.persistence === "server" && !detached.roleplay_resume && !detached.existing_selection && ((detached.input_id && !detached.admission) || (detached.assistant_id && !detached.admission)))
      fail("history_operation_conflict")
    if (detached.admission && (detached.admission.owner_key !== detached.owner_key || detached.admission.conversation_id !== detached.conversation_id || detached.admission.input_message_id !== detached.input_id || (detached.action !== "regenerate" && detached.admission.selection_digest !== detached.selection_digest)))
      fail("history_operation_conflict")
    if (
      detached.state === "refused" &&
      ((detached.persistence === "server" && (detached.input_id || detached.assistant_id)) ||
        detached.admission ||
        detached.result_text ||
        detached.outcome ||
        detached.settled_message_id ||
        (detached.persistence === "server" && detached.action !== "regenerate" && detached.input_images?.length))
    )
      fail("history_operation_conflict")

    if (detached.persistence === "server" && detached.existing_selection &&
        (detached.admission || detached.existing_selection.purpose !== "send" ||
         detached.existing_selection.cursor.kind !== "after_message" ||
         detached.existing_selection.cursor.message_id !== detached.input_id)) fail("history_operation_conflict")
    if (detached.existing_selection && (detached.existing_selection.owner_key !== detached.owner_key ||
      detached.existing_selection.conversation_id !== detached.conversation_id ||
      detached.existing_selection.messages.at(-1)?.id !== detached.input_id ||
      detached.existing_selection.selection_digest !== detached.selection_digest ||
      detached.existing_selection.request_context_digest !== detached.request_context_digest ||
      selectionDigest(detached.existing_selection) !== detached.selection_digest)) fail("history_operation_conflict")
    if (detached.settlement_payload && (!detached.existing_selection ||
      detached.settlement_payload.id !== detached.assistant_id ||
      detached.settlement_payload.history_id !== detached.conversation_id ||
      detached.settlement_payload.role !== "assistant" ||
      detached.settlement_payload.parent_message_id !== detached.input_id ||
      detached.settlement_payload.content !== detached.result_text ||
      detached.settlement_payload.createdAt !== detached.created_at)) fail("history_operation_conflict")
    const prior = previous?.pending_turns?.[detached.operation_id]
    if (prior) {
      const immutable = (value: HistoryTurnRecovery) => ({
        origin_view: value.origin_view, selection_digest: value.selection_digest, request_context_digest: value.request_context_digest,
        operation_id: value.operation_id, action: value.action, owner_key: value.owner_key, conversation_id: value.conversation_id,
        persistence: value.persistence ?? "client",
        ...(value.persistence === "server" ? {} : {input_id: value.input_id, assistant_id: value.assistant_id}), created_at: value.created_at,
        input_text: value.input_text, input_images: value.input_images, roleplay_intent: value.roleplay_resume?.intent
      })
      if (canonicalHistoryJson(immutable(prior)) !== canonicalHistoryJson(immutable(detached))) fail("history_operation_conflict")
      if (prior.state === "refused") {
        if (
          detached.admission ||
          (detached.persistence === "server" && detached.assistant_id) ||
          detached.result_text ||
          detached.outcome ||
          detached.settled_message_id
        )
          fail("history_operation_conflict")
        return
      }

      if (prior.roleplay_resume) {
        if (
          prior.settled_message_id &&
          detached.settled_message_id &&
          prior.settled_message_id !== detached.settled_message_id
        )
          fail("history_operation_conflict")
        if (
          (prior.outcome === "complete" && detached.outcome !== "complete") ||
          (prior.settled_message_id && !detached.settled_message_id)
        )
          return
        if (prior.state === "unknown" && detached.state === "refused")
          fail("history_operation_conflict")
        for (const field of ["append", "completion"] as const) {
          const earlier = prior.roleplay_resume[field]
          const later = detached.roleplay_resume?.[field]
          if (
            earlier &&
            later &&
            (field === "append" || prior.roleplay_resume.completion?.saved) &&
            canonicalHistoryJson(earlier) !== canonicalHistoryJson(later)
          )
            fail("history_operation_conflict")
          if (earlier && !later) return
        }
        detached.retained_result_texts = retainedTexts(prior, detached)
      }
      if (prior.persistence === "server") {
        if (Boolean(prior.existing_selection) !== Boolean(detached.existing_selection)) fail("history_operation_conflict")
        if ((prior.input_id && detached.input_id && prior.input_id !== detached.input_id) || (prior.assistant_id && detached.assistant_id && prior.assistant_id !== detached.assistant_id)) fail("history_operation_conflict")
        if (prior.assistant_id && !detached.assistant_id) return
      }
      if (prior.admission && detached.admission && canonicalHistoryJson(prior.admission) !== canonicalHistoryJson(detached.admission)) fail("history_operation_conflict")
      if (prior.existing_selection && canonicalHistoryJson(prior.existing_selection) !== canonicalHistoryJson(detached.existing_selection)) fail("history_operation_conflict")
      if (prior.settlement_payload && canonicalHistoryJson(prior.settlement_payload) !== canonicalHistoryJson(detached.settlement_payload)) fail("history_operation_conflict")
      if (prior.admission && !detached.admission) return
      if (prior.result_text && !detached.result_text) return
    }
    await db.historySelections.put({
      ...(previous ?? {...scope, owner_key: view.owner_key, conversation_id: view.conversation_id, view: structuredClone(view)}),
      pending_turns: {...previous?.pending_turns, [detached.operation_id]: detached}
    })
    written = true
  })
  if (written) memoryTurnRecoveries.delete(key)
}

/** Reopened views discover only the same profile and verified owner/conversation. */
export const loadHistoryTurnRecoveries = async (
  scope: HistoryBookmarkScope, owner: {owner_key: string; conversation_id: string}
): Promise<Array<{scope: HistoryBookmarkScope; turn: HistoryTurnRecovery}>> => {
  const {entries:durable,dismissedKeys} = await db.transaction("r", [db.userSettings, db.historySelections], async () => {
    await assertBookmarkProfile(scope)
    const bookmarks = await db.historySelections.where("owner_key").equals(owner.owner_key).toArray()
    const owned = bookmarks.filter(
      (record) =>
        record.profile_id === scope.profile_id &&
        record.conversation_id === owner.conversation_id
    )
    const dismissedKeys = owned.flatMap((record) =>
      Object.keys(record.history_turn_outcomes ?? {}).map((operation_id) =>
        recoveryKey(
          {
            profile_id: record.profile_id,
            client_session_id: record.client_session_id
          },
          { ...owner, operation_id }
        )
      )
    )
    const entries = owned.flatMap(record => Object.values(record.pending_turns ?? {}).map(turn => {
        const { action, ...ordinary } = turn
        if (action !== undefined && action !== "edit_resend" && action !== "regenerate") fail("unsupported_history_action")
        return { scope: { profile_id: record.profile_id, client_session_id: record.client_session_id },
          turn: { ...ordinary, ...(action ? { action } : {}) } }
      }))
    return { entries, dismissedKeys }
  })
  for (const key of dismissedKeys) memoryTurnRecoveries.delete(key)
  const results = new Map(
    durable.map((entry) => [recoveryKey(entry.scope, entry.turn), entry])
  )
  for (const [key, entry] of memoryTurnRecoveries) {
    if (
      entry.scope.profile_id !== scope.profile_id ||
      entry.turn.owner_key !== owner.owner_key ||
      entry.turn.conversation_id !== owner.conversation_id
    )
      continue
    const existing = results.get(key)
    const active = existing?.turn.roleplay_resume?.completion?.saved
      ? existing
      : entry
    results.set(key, {
      scope: active.scope,
      turn: {
        ...active.turn,
        recovery_storage: 'memory',
        ...(active.turn.roleplay_resume
          ? {
              retained_result_texts: existing
                ? retainedTexts(active === entry ? existing.turn : entry.turn, active.turn)
                : entry.turn.retained_result_texts,
            }
          : {}),
      },
    });
  }
  return [...results.values()].map((entry) => structuredClone(entry))
}

export const dismissHistoryTurnRecovery = async (
  scope: HistoryBookmarkScope,
  owner: { owner_key: string; conversation_id: string },
  operationId: string,
  outcome: "completed" | "dismissed" = "dismissed",
  options?: HistoryOperationOptions
): Promise<boolean> => {
  let tombstoned = false
  const dismissed = await transaction("rw", options, async () => {
    assertLease(options)
    await assertBookmarkProfile(scope)
    assertLease(options)
    const prior = await db.historySelections.get(bookmarkKey(scope, owner))
    assertLease(options)
    if (!prior?.pending_turns?.[operationId]) {
      tombstoned = Boolean(prior?.history_turn_outcomes?.[operationId])
      return false
    }
    const turns = { ...prior.pending_turns }
    delete turns[operationId]
    await db.historySelections.put({
      ...prior,
      pending_turns: turns,
      history_turn_outcomes: {
        ...prior.history_turn_outcomes,
        [operationId]: outcome
      }
    })
    // Retirement after the write must abort the transaction, not commit a false receipt.
    assertLease(options)
    return true
  })
  if (dismissed || tombstoned)
    memoryTurnRecoveries.delete(
      recoveryKey(scope, { ...owner, operation_id: operationId })
    )
  return dismissed
}
