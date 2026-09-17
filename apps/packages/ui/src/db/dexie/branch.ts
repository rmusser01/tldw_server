/** Independent local copies. Only this projector writes fork children. */
import { db } from "./schema"
import type { HistoryInfo, Message, SessionFiles, UploadedFile } from "./types"
import {
  getLocalHistoryOwner,
  historyDigest,
  withLocalForkSource,
  type LocalHistoryOwnerV1,
  type HistoryOperationOptions
} from "./history-selection"
import type {
  CompareHistorySelectionV1,
  ForkRequestV1,
  ForkResultV1,
  HistoryViewSelectionV1,
  HistoryCursorV1
} from "@/types/history-selection"
import {
  comparisonDigest,
  resolveComparisonProjection,
  resolveParentPath,
  resolveHistorySelection,
  selectionDigest,
  HistorySelectionError
} from "@/utils/history-selection"

const fail = (code: string): never => {
  throw new HistorySelectionError(code)
}
export const LOCAL_FORK_POLICY = "local-fork-allowlist-v1"
export const forkRequestDigest = (request: ForkRequestV1): string =>
  historyDigest({
    operation_id: request.operation_id,
    owner_key: request.owner_key,
    destination_owner_key: request.destination_owner_key,
    input: request.input
  })
type Source = Parameters<Parameters<typeof withLocalForkSource>[3]>[0]
export type ComparisonForkBoundary = {
  model_id: string
  cluster_id: string | null
  cursor: HistoryCursorV1
}
/** Only self-contained completed files have independent local ownership. */
const retainedFile = (file: UploadedFile): UploadedFile => {
  if (
    !file.id ||
    typeof file.content !== "string" ||
    !file.processed ||
    (file.processingStatus && file.processingStatus !== "ready") ||
    typeof file.filename !== "string" ||
    typeof file.type !== "string" ||
    !Number.isFinite(file.size) ||
    !Number.isFinite(file.uploadedAt)
  )
    fail("unsupported_local_file")
  // Unknown asset carriers cannot be assumed to be independently owned bytes.
  const allowed = new Set([
    "id",
    "filename",
    "type",
    "content",
    "size",
    "uploadedAt",
    "embedding",
    "processed",
    "processingMode",
    "processingStatus",
    "processingCapabilities",
    "processingSummary",
    "processingError",
    "processingBlockedReason",
    "processingRecoveryActions",
    "processingResultRef",
    "processingPageEstimate",
    "processingTokenEstimate",
    "documentDraftId",
    "ingestJobId",
    "ingestBatchId",
    "ingestIdempotencyKey"
  ])
  if (Object.keys(file).some((key) => !allowed.has(key)))
    fail("unsupported_local_file")
  return {
    id: file.id,
    filename: file.filename,
    type: file.type,
    content: file.content,
    size: file.size,
    uploadedAt: file.uploadedAt,
    processed: true,
    ...(file.embedding ? { embedding: [...file.embedding] } : {}),
    ...(file.processingSummary
      ? { processingSummary: file.processingSummary }
      : {}),
    ...(file.processingPageEstimate != null
      ? { processingPageEstimate: file.processingPageEstimate }
      : {}),
    ...(file.processingTokenEstimate != null
      ? { processingTokenEstimate: file.processingTokenEstimate }
      : {})
  }
}
const retainedContext = (source: Source) => {
  const files = source.files?.files ?? []
  if (new Set(files.map((file) => file.id)).size !== files.length)
    fail("duplicate_file_id")
  return {
    title: source.history.title,
    prompt: source.history.last_used_prompt?.prompt_content ?? null,
    model: source.history.model_id ?? null,
    files: source.files
      ? {
          retrievalEnabled: source.files.retrievalEnabled,
          files: source.files.files.map(retainedFile)
        }
      : null
  }
}
const forkSnapshot = (source: Source) => {
  const digest = historyDigest(retainedContext(source))
  return {
    ...source.snapshot,
    storage_context_digest: digest,
    fences: { ...source.snapshot.fences, settings: digest }
  }
}
/** Common round ancestry and each model's unique thread define semantic order, never timestamps. */
const comparisonNodes = (source: Source, modelId: string) => {
  resolveParentPath(source.snapshot.nodes, { kind: "empty" })
  const byId = new Map(source.records.map((row) => [row.id, row]))
  const common = new Map<string, Message>()
  const modelKey = (row: Message) => row.modelId || row.modelName || row.name
  for (const row of source.records) {
    if (
      row.history_id !== source.history.id ||
      !row.clusterId ||
      !row.messageType?.startsWith("compare:")
    )
      fail("unsupported_comparison_shape")
    if (row.messageType === "compare:user") {
      if (row.role !== "user" || common.has(row.clusterId))
        fail("ambiguous_comparison_round")
      common.set(row.clusterId, row)
    }
  }
  const nextRound = new Map<string | null, Message>()
  for (const root of common.values()) {
    let parent = root.parent_message_id
      ? byId.get(root.parent_message_id)
      : undefined
    while (parent && parent.messageType !== "compare:user")
      parent = parent.parent_message_id
        ? byId.get(parent.parent_message_id)
        : undefined
    const previous = parent?.id ?? null
    if (nextRound.has(previous)) fail("ambiguous_comparison_round")
    nextRound.set(previous, root)
  }
  const ordered: Message[] = []
  let root = nextRound.get(null)
  let rounds = 0
  while (root) {
    rounds++
    ordered.push(root)
    const rows = source.records.filter(
      (row) =>
        row.clusterId === root!.clusterId &&
        row.messageType !== "compare:user" &&
        modelKey(row) === modelId
    )
    const next = new Map<string, Message>()
    for (const row of rows) {
      if (!row.parent_message_id || next.has(row.parent_message_id))
        fail("ambiguous_comparison_thread")
      next.set(row.parent_message_id, row)
    }
    let parent = root.id
    let visited = 0
    while (next.has(parent)) {
      const row = next.get(parent)!
      ordered.push(row)
      visited++
      parent = row.id
    }
    if (visited !== rows.length) fail("unsupported_comparison_thread")
    root = nextRound.get(root.id)
  }
  if (rounds !== common.size || !rounds) fail("ambiguous_comparison_round")
  return ordered.map((row) => ({
    ...source.snapshot.nodes.find((node) => node.id === row.id)!,
    comparison: {
      cluster_id: row.clusterId!,
      model_id: row.messageType === "compare:user" ? null : modelId,
      common: row.messageType === "compare:user"
    }
  }))
}

const select = (
  source: Source,
  boundary: HistoryViewSelectionV1 | ComparisonForkBoundary
): ForkRequestV1["input"] => {
  const snapshot = forkSnapshot(source)
  if ("model_id" in boundary) {
    if (
      !source.comparison ||
      !boundary.model_id ||
      boundary.cursor.kind === "empty"
    )
      return fail("invalid_comparison_boundary")
    const nodes = comparisonNodes(source, boundary.model_id)
    let rows = resolveComparisonProjection(
      nodes,
      boundary.model_id,
      boundary.cursor.message_id
    )
    if (boundary.cursor.kind === "before_message") rows = rows.slice(0, -1)
    const boundaryId = boundary.cursor.message_id
    const endpoint = nodes.find((node) => node.id === boundaryId)
    if (
      boundary.cluster_id !== null &&
      endpoint?.comparison.cluster_id !== boundary.cluster_id
    )
      return fail("invalid_comparison_boundary")
    const selection: CompareHistorySelectionV1 = {
      version: 1,
      owner_key: snapshot.owner_key,
      conversation_id: snapshot.conversation_id,
      model_id: boundary.model_id,
      cluster_id: boundary.cluster_id,
      cursor: boundary.cursor,
      messages: rows.map(({ id, revision }) => ({ id, revision })),
      fences: snapshot.fences,
      storage_context_digest: snapshot.storage_context_digest,
      request_context_digest: LOCAL_FORK_POLICY,
      selection_digest: ""
    }
    return {
      kind: "comparison",
      selection: { ...selection, selection_digest: comparisonDigest(selection) }
    }
  }
  if (source.comparison) return fail("unsupported_comparison_history")
  const resolved = resolveHistorySelection(
    snapshot,
    boundary,
    "fork",
    LOCAL_FORK_POLICY
  )
  if (resolved.status !== "ready") return fail(resolved.code)
  return { kind: "normal", selection: resolved.selection }
}
const projectionId = (input: ForkRequestV1["input"]) =>
  input.kind === "normal" &&
  input.selection.interpretation.kind === "legacy_linear_v1"
    ? input.selection.interpretation.projection_id
    : undefined
export const captureLocalForkSelection = (
  owner: LocalHistoryOwnerV1,
  boundary: HistoryViewSelectionV1 | ComparisonForkBoundary,
  opts?: HistoryOperationOptions
): Promise<ForkRequestV1["input"]> => {
  boundary = structuredClone(boundary)
  const projection =
    "interpretation" in boundary &&
    boundary.interpretation.kind === "legacy_linear_v1"
      ? boundary.interpretation.projection_id
      : undefined
  return withLocalForkSource(
    owner,
    projection,
    "r",
    async (source) => select(source, boundary),
    opts
  )
}
const validate = (source: Source, request: ForkRequestV1) => {
  const selection = request.input.selection
  if (
    !request.operation_id ||
    request.owner_key !== source.snapshot.owner_key ||
    request.destination_owner_key !== request.owner_key ||
    request.request_digest !== forkRequestDigest(request) ||
    selection.owner_key !== request.owner_key ||
    selection.conversation_id !== source.history.id ||
    selection.version !== 1 ||
    selection.request_context_digest !== LOCAL_FORK_POLICY
  )
    fail("invalid_fork_request")
  const digest =
    request.input.kind === "normal"
      ? selectionDigest(request.input.selection)
      : comparisonDigest(request.input.selection)
  if (
    selection.selection_digest !== digest ||
    new Set(selection.messages.map((row) => row.id)).size !==
      selection.messages.length
  )
    fail("invalid_selection")
  const current = select(
    source,
    request.input.kind === "normal"
      ? { view_session_id: "fork", ...request.input.selection }
      : request.input.selection
  )
  if (current.selection.selection_digest !== selection.selection_digest)
    fail("stale_selection")
}
export type PreparedLocalFork = {
  request: ForkRequestV1
  owner: LocalHistoryOwnerV1
  history: HistoryInfo
  messages: Message[]
  files?: SessionFiles
  message_map: Record<string, string>
}
const project = (
  source: Source,
  request: ForkRequestV1
): Omit<PreparedLocalFork, "owner"> => {
  validate(source, request)
  const childId = crypto.randomUUID()
  const message_map = Object.fromEntries(
    request.input.selection.messages.map(({ id }) => [id, crypto.randomUUID()])
  )
  const fileMap = Object.fromEntries(
    (source.files?.files ?? []).map((file) => [file.id, crypto.randomUUID()])
  )
  const history: HistoryInfo = {
    id: childId,
    title: `${source.history.title} · Branch`,
    createdAt: Date.now(),
    is_rag: false,
    is_pinned: false,
    message_source: "branch",
    local_owner_key: request.destination_owner_key,
    ...(source.history.model_id ? { model_id: source.history.model_id } : {}),
    ...(source.history.last_used_prompt?.prompt_content !== undefined
      ? {
          last_used_prompt: {
            prompt_content: source.history.last_used_prompt.prompt_content
          }
        }
      : {})
  }
  const linear =
    request.input.kind === "comparison" ||
    request.input.selection.interpretation.kind === "legacy_linear_v1"
  const depths = new Map<string, number>()
  const messages = request.input.selection.messages.map(
    ({ id }, index): Message => {
      const row = source.records.find((row) => row.id === id)
      if (!row) return fail("missing_message")
      const images = row.images?.filter(Boolean).length
        ? row.images.filter(Boolean)
        : row.image
          ? [row.image]
          : []
      if (
        (row.documents &&
          (!Array.isArray(row.documents) || row.documents.length > 0)) ||
        row.sources?.some(
          (ref) =>
            typeof ref !== "string" ||
            (!fileMap[ref] && !/^data:[^,]+,/.test(ref))
        ) ||
        images.some((image) => !/^data:[^,]+,/.test(image))
      )
        fail("unsupported_local_message_assets")
      if (
        row.generationInfo?.image_generation ||
        row.generationInfo?.file_id ||
        row.metadataExtra?.tool_calls ||
        row.metadataExtra?.tool_call_id ||
        row.role === "tool"
      )
        fail("unsupported_local_rich_state")
      const parent = linear
        ? (request.input.selection.messages[index - 1]?.id ?? null)
        : (row.parent_message_id ?? null)
      if (parent && !message_map[parent]) fail("missing_parent")
      const depth = parent ? (depths.get(parent) ?? -1) + 1 : 0
      depths.set(id, depth)
      return {
        id: message_map[id],
        history_id: childId,
        role: row.role,
        name: row.name,
        content: row.content,
        createdAt: row.createdAt,
        parent_message_id: parent ? message_map[parent] : null,
        depth,
        history_provenance: {
          version: 1,
          owner_key: request.destination_owner_key,
          projection_id: null
        },
        images: [...images],
        ...(row.sources
          ? { sources: row.sources.map((ref) => fileMap[ref] ?? ref) }
          : {}),
        ...(row.search
          ? {
              search: {
                search_engine: row.search.search_engine,
                search_url: row.search.search_url,
                search_query: row.search.search_query,
                search_results: row.search.search_results.map((result) => ({
                  title: result.title,
                  link: result.link
                }))
              }
            }
          : {}),
        ...(row.messageType && !row.messageType.startsWith("compare:")
          ? { messageType: row.messageType }
          : {}),
        ...(row.modelId ? { modelId: row.modelId } : {}),
        ...(row.modelName ? { modelName: row.modelName } : {}),
        ...(row.reasoning_time_taken !== undefined
          ? { reasoning_time_taken: row.reasoning_time_taken }
          : {})
      }
    }
  )
  const files = source.files
    ? {
        sessionId: childId,
        createdAt: Date.now(),
        retrievalEnabled: source.files.retrievalEnabled,
        files: source.files.files.map((file) => ({
          ...retainedFile(file),
          id: fileMap[file.id]
        }))
      }
    : undefined
  return {
    request: structuredClone(request),
    history,
    messages,
    files,
    message_map
  }
}
const preparations = new WeakMap<PreparedLocalFork, PreparedLocalFork>()

export const prepareLocalFork = async (
  request: ForkRequestV1,
  opts?: HistoryOperationOptions
): Promise<PreparedLocalFork> => {
  request = structuredClone(request)
  const owner = await getLocalHistoryOwner(
    request.input.selection.conversation_id
  )
  const prepared = await withLocalForkSource(
    owner,
    projectionId(request.input),
    "r",
    async (source) => ({ ...project(source, request), owner }),
    opts
  )
  preparations.set(prepared, structuredClone(prepared))
  return prepared
}
/** Resolution means the Dexie transaction committed; failures never invoke another writer. */
export const commitLocalFork = async (
  prepared: PreparedLocalFork,
  opts?: HistoryOperationOptions
): Promise<ForkResultV1> => {
  const trusted = preparations.get(prepared)
  if (!trusted)
    return {
      state: "rejected",
      operation_id: prepared.request.operation_id,
      owner_key: prepared.request.owner_key,
      code: "invalid_fork_preparation"
    }
  preparations.delete(prepared)
  const { request, owner } = trusted
  let committed = false
  try {
    await withLocalForkSource(
      owner,
      projectionId(request.input),
      "rw",
      async (source) => {
        validate(source, request)
        await db.chatHistories.add(trusted.history)
        await db.messages.bulkAdd(trusted.messages)
        if (trusted.files) await db.sessionFiles.add(trusted.files)
      },
      opts
    )
    committed = true
    Object.assign(prepared, structuredClone(trusted))
    return {
      state: "committed",
      operation_id: request.operation_id,
      owner_key: request.owner_key,
      child_id: trusted.history.id,
      message_map: { ...trusted.message_map }
    }
  } catch (error) {
    if (committed)
      return {
        state: "unknown",
        operation_id: request.operation_id,
        owner_key: request.owner_key,
        candidate_child_id: trusted.history.id,
        code: "local_commit_observation_failed"
      }
    return {
      state: "rejected",
      operation_id: request.operation_id,
      owner_key: request.owner_key,
      code:
        error instanceof HistorySelectionError
          ? error.code
          : error instanceof Error
            ? error.message
            : "local_fork_failed"
    }
  }
}
