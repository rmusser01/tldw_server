import { z } from "zod"
import { DEFAULT_WORKSPACE_NOTE } from "../types/workspace"
import type { OwnedWorkspaceBundle } from "./workspace-api"

export type OwnedWorkspaceScope = {
  serverBase: string
  principalId: string
  organizationId: string | null
}

export type OwnedWorkspaceAttempt = {
  scope: OwnedWorkspaceScope
  generation: number
  workspaceId: string
}

export const ownedWorkspaceRenameDraftSchema = z
  .object({
    name: z.string(),
    baseVersion: z.number().int().positive().max(Number.MAX_SAFE_INTEGER)
  })
  .strict()
export type OwnedWorkspaceRenameDraft = z.infer<
  typeof ownedWorkspaceRenameDraftSchema
>

export const ownedWorkspaceAssistantDraftSchema = z
  .object({
    assistantId: z.string(),
    personaMemoryMode: z.enum(["read_only", "read_write"]),
    baseVersion: z.number().int().positive().max(Number.MAX_SAFE_INTEGER)
  })
  .strict()
export type OwnedWorkspaceAssistantDraft = z.infer<
  typeof ownedWorkspaceAssistantDraftSchema
>

export const ownedWorkspaceBannerDraftSchema = z
  .object({
    title: z.string(),
    subtitle: z.string(),
    baseVersion: z.number().int().positive().max(Number.MAX_SAFE_INTEGER)
  })
  .strict()
export type OwnedWorkspaceBannerDraft = z.infer<
  typeof ownedWorkspaceBannerDraftSchema
>

const idSchema = z
  .string()
  .min(1)
  .refine((value) => value === value.trim())
const dateSchema = z
  .string()
  .refine((value) => Number.isFinite(Date.parse(value)))
const scopeSchema = z
  .object({
    serverBase: z.string(),
    principalId: idSchema,
    organizationId: idSchema.nullable()
  })
  .strict()
const draftSchema = z
  .object({
    schemaVersion: z.literal(1),
    scope: scopeSchema,
    workspaceId: idSchema,
    notes: z.string(),
    composer: z.string(),
    renameDraft: ownedWorkspaceRenameDraftSchema.nullable().optional(),
    assistantDraft: ownedWorkspaceAssistantDraftSchema.nullable().optional(),
    bannerDraft: ownedWorkspaceBannerDraftSchema.nullable().optional(),
    currentNote: z
      .object({
        id: z.number().int().positive().optional(),
        title: z.string(),
        content: z.string(),
        keywords: z.array(z.string()),
        version: z.number().int().positive().optional(),
        isDirty: z.boolean(),
        createUncertain: z.boolean().optional()
      })
      .strict(),
    sourceFolders: z.array(
      z
        .object({
          id: idSchema,
          workspaceId: idSchema,
          name: z.string(),
          parentFolderId: idSchema.nullable(),
          createdAt: dateSchema,
          updatedAt: dateSchema
        })
        .strict()
    ),
    sourceFolderMemberships: z.array(
      z.object({ folderId: idSchema, sourceId: idSchema }).strict()
    ),
    selectedSourceFolderIds: z.array(idSchema),
    activeFolderId: idSchema.nullable(),
    leftPaneCollapsed: z.boolean(),
    rightPaneCollapsed: z.boolean(),
    // Quarantined canonical edits require explicit reconciliation, never auto-application.
    pendingChanges: z.record(z.string(), z.json())
  })
  .strict()

/** Wire-format draft: dates are ISO strings, and canonical edits are quarantined. */
export type OwnedWorkspaceDraft = Omit<z.infer<typeof draftSchema>, "scope"> & {
  scope: OwnedWorkspaceScope
}

const DRAFT_NAMESPACE = "tldw:research-workspace:owned-drafts:v1"

/** Canonical scope keys contain identity only, never auth tokens or URL credentials. */
export function ownedWorkspaceDraftKey(
  scope: OwnedWorkspaceScope,
  workspaceId: string
): string {
  scopeSchema.parse(scope)
  idSchema.parse(workspaceId)
  const base = new URL(scope.serverBase)
  if (
    !["http:", "https:"].includes(base.protocol) ||
    base.username ||
    base.password ||
    base.search ||
    base.hash
  ) {
    throw new Error("Invalid workspace server base")
  }
  const serverBase = `${base.origin}${base.pathname.replace(/\/+$/, "")}`
  return `${DRAFT_NAMESPACE}:${encodeURIComponent(
    JSON.stringify([
      serverBase,
      scope.principalId,
      scope.organizationId,
      workspaceId
    ])
  )}`
}

function parseDraft(value: unknown): OwnedWorkspaceDraft {
  const draft = draftSchema.parse(value)
  const scope: OwnedWorkspaceScope = {
    ...draft.scope,
    organizationId: draft.scope.organizationId ?? null
  }
  ownedWorkspaceDraftKey(scope, draft.workspaceId)
  const folders = new Map(
    draft.sourceFolders.map((folder) => [folder.id, folder])
  )
  if (folders.size !== draft.sourceFolders.length)
    throw new Error("Duplicate draft folders")
  for (const folder of folders.values()) {
    if (folder.workspaceId !== draft.workspaceId)
      throw new Error("Foreign draft folder")
    const visited = new Set<string>([folder.id])
    let parent = folder.parentFolderId
    while (parent != null) {
      if (visited.has(parent) || !folders.has(parent))
        throw new Error("Invalid draft folder ancestry")
      visited.add(parent)
      parent = folders.get(parent)!.parentFolderId
    }
  }
  return { ...draft, scope }
}

type DraftStorage = Pick<
  Storage,
  "getItem" | "setItem" | "removeItem" | "key" | "length"
>
type DraftWriteResult =
  | { status: "saved" | "unavailable" | "invalid" }
  | { status: "conflict"; variants: OwnedWorkspaceDraftVariant[] }
const revisionSchema = z
  .object({
    schemaVersion: z.literal(1),
    writerId: z.uuid(),
    sequence: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
    parents: z.record(
      z.uuid(),
      z.number().int().positive().max(Number.MAX_SAFE_INTEGER)
    ),
    draft: draftSchema.nullable(),
    legacyBaseline: z.string().nullable().optional()
  })
  .strict()
type DraftRevision = Omit<z.infer<typeof revisionSchema>, "draft"> & {
  draft: OwnedWorkspaceDraft | null
}
export type OwnedWorkspaceDraftVariant = {
  revisionId: string
  draft: OwnedWorkspaceDraft | null
  durable: boolean
  /** Exact legacy bytes remain recoverable even when they cannot be parsed. */
  raw?: string
}
export type OwnedWorkspaceDraftRead =
  | { status: "ready"; draft: OwnedWorkspaceDraft; durable: boolean }
  | { status: "conflict"; variants: OwnedWorkspaceDraftVariant[] }
  | { status: "missing"; deleted?: true }
  | { status: "unavailable" | "invalid" }

/** Each writer owns one replaceable slot per target; no cross-window CAS is needed. */
export function createOwnedWorkspaceDraftStore(getStorage: () => DraftStorage) {
  // getRandomValues also works on self-hosted HTTP origins, unlike randomUUID.
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("")
  const writerId = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
  const unsaved = new Map<string, DraftRevision>()
  const observed = new Map<string, DraftRevision["parents"]>()
  const legacyBaselines = new Map<string, string | null>()
  const sequences = new Map<string, number>()
  const savedDrafts = new Map<
    string,
    { draft: string; slot: string; raw: string }
  >()
  const prefix = (key: string) => `${key}:revisions:`
  const variant = (
    revision: DraftRevision,
    durable: boolean
  ): OwnedWorkspaceDraftVariant => ({
    revisionId: `${revision.writerId}:${revision.sequence}`,
    draft: structuredClone(revision.draft),
    durable
  })
  const write = (
    key: string,
    draft: OwnedWorkspaceDraft | null
  ): DraftWriteResult => {
    const sequence = (sequences.get(key) ?? 0) + 1
    if (!Number.isSafeInteger(sequence)) return { status: "invalid" }
    const revision: DraftRevision = {
      schemaVersion: 1,
      writerId,
      sequence,
      parents: { ...observed.get(key) },
      draft: structuredClone(draft),
      legacyBaseline: legacyBaselines.get(key) ?? null
    }
    sequences.set(key, sequence)
    if (draft !== null) {
      observed.set(key, { ...revision.parents, [writerId]: sequence })
      unsaved.set(key, revision)
    }
    try {
      const storage = getStorage()
      // Removal is a causal tombstone, never deletion of another writer's slot.
      const slot = `${prefix(key)}${writerId}`
      const raw = JSON.stringify(revision)
      storage.setItem(slot, raw)
      // Never mutate the legacy key: an old tab can write it at any point.
      unsaved.delete(key)
      observed.set(key, { ...revision.parents, [writerId]: sequence })
      if (draft !== null)
        savedDrafts.set(key, { draft: JSON.stringify(draft), slot, raw })
      else savedDrafts.delete(key)
      return writeResult(key)
    } catch {
      return { status: "unavailable" }
    }
  }
  const read = (key: string, adopt: boolean): OwnedWorkspaceDraftRead => {
    const pending = unsaved.get(key)
    let raw: string | null = null
    const entries: Array<{ key: string; raw: string }> = []
    try {
      const storage = getStorage()
      for (let index = 0; index < storage.length; index++) {
        const slot = storage.key(index)
        if (!slot?.startsWith(prefix(key))) continue
        const value = storage.getItem(slot)
        if (value !== null) entries.push({ key: slot, raw: value })
      }
      raw = storage.getItem(key)
    } catch {
      if (pending?.draft)
        return {
          status: "ready",
          draft: structuredClone(pending.draft),
          durable: false
        }
      return { status: "unavailable" }
    }
    try {
      const revisions = entries
        .map((entry) => {
          const parsed = revisionSchema.parse(JSON.parse(entry.raw))
          const revision: DraftRevision = {
            ...parsed,
            draft: parsed.draft ? parseDraft(parsed.draft) : null
          }
          if (
            entry.key !== `${prefix(key)}${revision.writerId}` ||
            (revision.parents[revision.writerId] ?? 0) >= revision.sequence
          )
            throw new Error("Invalid draft revision")
          if (revision.draft) {
            if (
              ownedWorkspaceDraftKey(
                revision.draft.scope,
                revision.draft.workspaceId
              ) !== key
            )
              throw new Error("Foreign draft revision")
          }
          if (revision.legacyBaseline != null) {
            const baseline = parseDraft(JSON.parse(revision.legacyBaseline))
            if (
              ownedWorkspaceDraftKey(baseline.scope, baseline.workspaceId) !==
              key
            )
              throw new Error("Foreign legacy baseline")
          }
          return revision
        })
        .filter((revision) => !pending || revision.writerId !== writerId)
      if (pending) revisions.push(pending)
      if (revisions.length) {
        const heads = revisions.filter(
          (revision) =>
            !revisions.some(
              (other) =>
                other !== revision &&
                (other.parents[revision.writerId] ?? 0) >= revision.sequence
            )
        )
        // A pending retry must keep its original ancestry even if another writer
        // continued a revision whose storage write did not acknowledge success.
        if (pending && !heads.includes(pending)) heads.push(pending)
        if (!heads.length) return { status: "invalid" }
        const variants = heads.map((head) => variant(head, head !== pending))
        if (
          raw !== null &&
          !heads.every((head) => head.legacyBaseline === raw)
        ) {
          let legacy: OwnedWorkspaceDraft | null = null
          try {
            const parsed = parseDraft(JSON.parse(raw))
            if (
              ownedWorkspaceDraftKey(parsed.scope, parsed.workspaceId) === key
            )
              legacy = parsed
          } catch {
            // Retain unparseable legacy bytes without adopting them as a baseline.
          }
          variants.push({
            revisionId: "legacy",
            draft: legacy,
            raw,
            durable: true
          })
        }
        if (variants.length > 1)
          return {
            status: "conflict",
            variants
          }
        const head = heads[0]
        // Only an unambiguous loaded revision becomes the next edit's parent.
        // Reading a conflict must never merge its independent histories.
        if (adopt) {
          observed.set(key, { ...head.parents, [head.writerId]: head.sequence })
          legacyBaselines.set(key, head.legacyBaseline ?? null)
          if (head.draft && head !== pending)
            savedDrafts.set(key, {
              draft: JSON.stringify(head.draft),
              slot: `${prefix(key)}${head.writerId}`,
              raw: entries.find(
                (entry) => entry.key === `${prefix(key)}${head.writerId}`
              )!.raw
            })
          else savedDrafts.delete(key)
        }
        return head.draft
          ? {
              status: "ready",
              draft: structuredClone(head.draft),
              durable: head !== pending
            }
          : { status: "missing", deleted: true }
      }
      if (adopt) {
        observed.delete(key)
        savedDrafts.delete(key)
        legacyBaselines.delete(key)
      }
      if (raw === null) return { status: "missing" }
      const draft = parseDraft(JSON.parse(raw))
      if (ownedWorkspaceDraftKey(draft.scope, draft.workspaceId) !== key) {
        return { status: "invalid" }
      }
      if (adopt) {
        legacyBaselines.set(key, raw)
        savedDrafts.set(key, {
          draft: JSON.stringify(draft),
          slot: key,
          raw: raw
        })
      }
      return { status: "ready", draft, durable: true }
    } catch {
      return { status: "invalid" }
    }
  }
  const writeResult = (key: string): DraftWriteResult => {
    // Inspect without adopting another writer's ancestry or foreign legacy content.
    const loaded = read(key, false)
    return loaded.status === "ready" || loaded.status === "missing"
      ? { status: "saved" }
      : loaded
  }
  const store = {
    hasPendingWrites(): boolean {
      return unsaved.size > 0
    },
    hasPendingWrite(scope: OwnedWorkspaceScope, workspaceId: string): boolean {
      return unsaved.has(ownedWorkspaceDraftKey(scope, workspaceId))
    },
    save(value: OwnedWorkspaceDraft, forceWrite = false): DraftWriteResult {
      let draft: OwnedWorkspaceDraft
      let key: string
      try {
        draft = parseDraft(value)
        key = ownedWorkspaceDraftKey(draft.scope, draft.workspaceId)
      } catch {
        return { status: "invalid" }
      }
      const saved = savedDrafts.get(key)
      if (
        !forceWrite &&
        !unsaved.has(key) &&
        saved?.draft === JSON.stringify(draft)
      ) {
        try {
          if (getStorage().getItem(saved.slot) === saved.raw)
            return writeResult(key)
        } catch {
          // Reattempt the writer's own slot and retain memory if storage is blocked.
        }
      }
      return write(key, draft)
    },
    load(
      scope: OwnedWorkspaceScope,
      workspaceId: string
    ): OwnedWorkspaceDraftRead {
      return read(ownedWorkspaceDraftKey(scope, workspaceId), true)
    },
    remove(scope: OwnedWorkspaceScope, workspaceId: string): DraftWriteResult {
      const key = ownedWorkspaceDraftKey(scope, workspaceId)
      const loaded = store.load(scope, workspaceId)
      if (loaded.status === "conflict" || loaded.status === "invalid")
        return loaded
      if (loaded.status === "unavailable") return { status: "unavailable" }
      return write(key, null)
    }
  }
  return store
}

export type OwnedWorkspaceActivationPreparation =
  | { status: "stale" }
  | {
      status: "draft-conflict"
      reason: "server-edits" | "note-changed" | "note-missing"
      draft: OwnedWorkspaceDraft
    }
  | {
      status: "activated"
      bundle: OwnedWorkspaceBundle
      draft: OwnedWorkspaceDraft | null
    }

export type OwnedWorkspaceActivationResult =
  OwnedWorkspaceActivationPreparation["status"]

/** Prepare one store commit from a validated load and the latest same-scope draft. */
export function prepareOwnedWorkspaceActivation(
  attempt: OwnedWorkspaceAttempt,
  expected: OwnedWorkspaceAttempt | null,
  bundle: OwnedWorkspaceBundle,
  recoveredDraft?: OwnedWorkspaceDraft
): OwnedWorkspaceActivationPreparation {
  const key = ownedWorkspaceDraftKey(attempt.scope, attempt.workspaceId)
  if (
    !expected ||
    attempt.generation !== expected.generation ||
    !Number.isSafeInteger(attempt.generation) ||
    attempt.generation < 1 ||
    key !== ownedWorkspaceDraftKey(expected.scope, expected.workspaceId) ||
    bundle.workspace.id !== attempt.workspaceId ||
    bundle.workspace.archived ||
    bundle.workspace.deleted
  ) {
    return { status: "stale" }
  }
  const draft = recoveredDraft ? parseDraft(recoveredDraft) : null
  if (draft && ownedWorkspaceDraftKey(draft.scope, draft.workspaceId) !== key) {
    return { status: "stale" }
  }
  if (draft && Object.keys(draft.pendingChanges).length > 0) {
    return { status: "draft-conflict", reason: "server-edits", draft }
  }
  if (draft?.currentNote.id != null) {
    const note = bundle.notes.find((item) => item.id === draft.currentNote.id)
    if (draft.currentNote.isDirty && !note) {
      return { status: "draft-conflict", reason: "note-missing", draft }
    }
    if (
      draft.currentNote.isDirty &&
      note?.version !== draft.currentNote.version
    ) {
      return { status: "draft-conflict", reason: "note-changed", draft }
    }
    if (!draft.currentNote.isDirty) {
      const keywords = note ? (JSON.parse(note.keywords_json) as unknown) : []
      if (
        !Array.isArray(keywords) ||
        keywords.some((value) => typeof value !== "string")
      ) {
        throw new Error("Invalid workspace note keywords")
      }
      draft.currentNote = note
        ? {
            id: note.id,
            title: note.title,
            content: note.content,
            keywords,
            version: note.version,
            isDirty: false
          }
        : { ...DEFAULT_WORKSPACE_NOTE, keywords: [] }
    }
  }
  if (draft) {
    const sourceIds = new Set(bundle.sources.map((source) => source.id))
    const folderIds = new Set(draft.sourceFolders.map((folder) => folder.id))
    draft.sourceFolderMemberships = draft.sourceFolderMemberships.filter(
      (item) => sourceIds.has(item.sourceId) && folderIds.has(item.folderId)
    )
    draft.selectedSourceFolderIds = draft.selectedSourceFolderIds.filter((id) =>
      folderIds.has(id)
    )
    if (draft.activeFolderId && !folderIds.has(draft.activeFolderId))
      draft.activeFolderId = null
  }
  return { status: "activated", bundle: structuredClone(bundle), draft }
}
