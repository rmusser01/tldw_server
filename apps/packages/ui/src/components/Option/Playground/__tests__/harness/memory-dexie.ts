/**
 * In-memory stand-in for the PageAssist Dexie database.
 *
 * jsdom has no IndexedDB and the repo does not ship fake-indexeddb, so the
 * Playground integration harness swaps `@/db/dexie/schema`'s `db` for this
 * object. It implements the subset of the Dexie Table / Collection API that
 * the chat, history-selection, fork and mirror modules use, with IndexedDB
 * semantics that matter to them: structured-clone isolation, primary and
 * compound keys, key ordering, and all-or-nothing `transaction()` rollback.
 */
import { AsyncLocalStorage } from "node:async_hooks"

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- IndexedDB rows are schemaless
type Row = Record<string, any>
type Key = string | number | Date | Key[]
type Predicate = (row: Row) => boolean

/** Primary keys and indexes, mirroring `PageAssistDexieDB`'s latest version. */
const PAGE_ASSIST_SCHEMA: Record<string, string> = {
  forkOperations: "[owner_key+operation_id], source_key, candidate_key, &active_intent",
  historySelections:
    "[profile_id+client_session_id+owner_key+conversation_id], owner_key, conversation_id",
  historyProjections: "[owner_key+conversation_id+projection_id], owner_key, conversation_id",
  chatHistories:
    "id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id",
  messages:
    "id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id",
  prompts:
    "id, title, content, is_system, createdBy, createdAt, deletedAt, serverId, studioProjectId, syncStatus, sourceSystem, usageCount, lastUsedAt",
  webshares: "id, title, url, api_url, share_id, createdAt",
  sessionFiles: "sessionId, retrievalEnabled, createdAt",
  userSettings: "id, user_id",
  customModels: "id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type",
  modelNickname: "id, model_id, model_name, model_avatar",
  processedMedia: "id, url, createdAt",
  folders: "id, name, parent_id, deleted",
  keywords: "id, keyword, deleted",
  folderKeywordLinks: "[folder_id+keyword_id], folder_id, keyword_id",
  conversationKeywordLinks: "[conversation_id+keyword_id], conversation_id, keyword_id",
  compareStates: "history_id",
  contentDrafts: "id, batchId, status, mediaType, createdAt, updatedAt, expiresAt",
  draftBatches: "id, createdAt, updatedAt",
  draftAssets: "id, draftId, createdAt",
  audiobookProjects: "id, title, status, createdAt, updatedAt, lastOpenedAt",
  audiobookChapterAssets: "id, projectId, chapterId, createdAt",
  ttsClips: "id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider",
  sttRecordings: "id, createdAt",
  mediaReadAlongAudioCache:
    "id, createdAt, lastUsedAt, [lastUsedAt+sizeBytes+id], mediaId, mediaKind, segmentId, settingsSignature, textHash"
}

const clone = <T,>(value: T): T => (value === undefined ? value : structuredClone(value))

const parseIndex = (spec: string): string | string[] => {
  const name = spec.trim().replace(/^(\+\+|&|\*)/, "")
  return name.startsWith("[") ? name.slice(1, -1).split("+") : name
}

const readPath = (row: Row, path: string | string[]): Key | undefined => {
  if (Array.isArray(path)) {
    const parts = path.map((part) => readPath(row, part))
    return parts.some((part) => part === undefined) ? undefined : (parts as Key[])
  }
  let value: unknown = row
  for (const part of path.split(".")) value = (value as Row | undefined)?.[part]
  return value as Key | undefined
}

const typeRank = (value: Key): number =>
  typeof value === "number" ? 0 : value instanceof Date ? 1 : typeof value === "string" ? 2 : 3

/** IndexedDB key ordering: number < Date < string < array. */
export const compareKeys = (left: Key, right: Key): number => {
  const rank = typeRank(left) - typeRank(right)
  if (rank !== 0) return rank
  if (Array.isArray(left) && Array.isArray(right)) {
    for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
      const result = compareKeys(left[index], right[index])
      if (result !== 0) return result
    }
    return left.length - right.length
  }
  const a = left instanceof Date ? left.getTime() : left
  const b = right instanceof Date ? right.getTime() : right
  return a < b ? -1 : a > b ? 1 : 0
}

const isValidKey = (value: unknown): value is Key =>
  typeof value === "number" ||
  typeof value === "string" ||
  value instanceof Date ||
  (Array.isArray(value) && value.every(isValidKey))

const applyChanges = (row: Row, changes: Row) => {
  for (const [path, value] of Object.entries(changes)) {
    const parts = path.split(".")
    let target = row
    for (const part of parts.slice(0, -1)) {
      target[part] = target[part] && typeof target[part] === "object" ? target[part] : {}
      target = target[part]
    }
    if (value === undefined) delete target[parts.at(-1)!]
    else target[parts.at(-1)!] = clone(value)
  }
}

type CollectionState = {
  index: string | string[] | null
  predicates: Predicate[]
  reverse: boolean
  offset: number
  limit: number | null
  union?: MemoryCollection[]
}

export class MemoryCollection {
  constructor(
    private readonly table: MemoryTable,
    private readonly state: CollectionState
  ) {}

  private next(patch: Partial<CollectionState>) {
    return new MemoryCollection(this.table, { ...this.state, ...patch })
  }

  private matches(): Row[] {
    if (this.state.union) {
      const seen = new Map<string, Row>()
      for (const collection of this.state.union)
        for (const row of collection.matches()) seen.set(this.table.encode(this.table.keyOf(row)), row)
      return [...seen.values()].filter((row) => this.state.predicates.every((p) => p(row)))
    }
    const rows = [...this.table.rows.values()].filter((row) =>
      this.state.predicates.every((predicate) => predicate(row))
    )
    const index = this.state.index
    rows.sort((left, right) => {
      if (index) {
        const a = readPath(left, index)
        const b = readPath(right, index)
        if (a !== undefined && b !== undefined) {
          const result = compareKeys(a, b)
          if (result !== 0) return result
        }
      }
      return compareKeys(this.table.keyOf(left), this.table.keyOf(right))
    })
    return rows
  }

  private window(): Row[] {
    let rows = this.matches()
    if (this.state.reverse) rows = rows.reverse()
    rows = rows.slice(this.state.offset)
    if (this.state.limit !== null) rows = rows.slice(0, this.state.limit)
    return rows
  }

  filter(predicate: Predicate) {
    return this.next({ predicates: [...this.state.predicates, (row) => Boolean(predicate(clone(row)))] })
  }
  and(predicate: Predicate) {
    return this.filter(predicate)
  }
  or(index: string) {
    return new MemoryWhereClause(this.table, index, (other) =>
      new MemoryCollection(this.table, {
        index: null,
        predicates: [],
        reverse: false,
        offset: 0,
        limit: null,
        union: [this, other]
      })
    )
  }
  reverse() {
    return this.next({ reverse: !this.state.reverse })
  }
  offset(count: number) {
    return this.next({ offset: this.state.offset + count })
  }
  limit(count: number) {
    return this.next({ limit: count })
  }
  async toArray() {
    return this.window().map(clone)
  }
  async first() {
    return clone(this.window()[0])
  }
  async last() {
    return clone(this.window().at(-1))
  }
  async count() {
    return this.window().length
  }
  async each(callback: (row: Row) => void) {
    for (const row of this.window()) callback(clone(row))
  }
  async sortBy(path: string) {
    const rows = this.window().map(clone)
    rows.sort((left, right) => {
      const a = readPath(left, path)
      const b = readPath(right, path)
      if (a === undefined || b === undefined) return a === b ? 0 : a === undefined ? 1 : -1
      return compareKeys(a, b)
    })
    return this.state.reverse ? rows.reverse() : rows
  }
  async primaryKeys() {
    return this.window().map((row) => clone(this.table.keyOf(row)))
  }
  async keys() {
    const index = this.state.index
    return this.window().map((row) => clone(index ? readPath(row, index) : this.table.keyOf(row)))
  }
  async uniqueKeys() {
    const keys = await this.keys()
    const seen = new Map<string, unknown>()
    for (const key of keys) seen.set(JSON.stringify(key), key)
    return [...seen.values()]
  }
  async modify(changes: Row | ((row: Row) => void | boolean)) {
    const rows = this.window()
    for (const row of rows) {
      const next = clone(row)
      if (typeof changes === "function") {
        if (changes(next) === false) continue
      } else applyChanges(next, changes)
      this.table.write(next, this.table.keyOf(row))
    }
    return rows.length
  }
  async delete() {
    const rows = this.window()
    for (const row of rows) this.table.rows.delete(this.table.encode(this.table.keyOf(row)))
    return rows.length
  }
}

export class MemoryWhereClause {
  constructor(
    private readonly table: MemoryTable,
    private readonly index: string,
    private readonly wrap: (collection: MemoryCollection) => MemoryCollection = (c) => c
  ) {}

  private path() {
    return parseIndex(this.index)
  }

  private collection(test: (value: Key) => boolean) {
    const path = this.path()
    return this.wrap(
      new MemoryCollection(this.table, {
        index: path,
        predicates: [
          (row) => {
            const value = readPath(row, path)
            return value !== undefined && isValidKey(value) && test(value)
          }
        ],
        reverse: false,
        offset: 0,
        limit: null
      })
    )
  }

  equals(value: Key) {
    return this.collection((candidate) => compareKeys(candidate, value) === 0)
  }
  notEqual(value: Key) {
    return this.collection((candidate) => compareKeys(candidate, value) !== 0)
  }
  equalsIgnoreCase(value: string) {
    return this.collection(
      (candidate) => typeof candidate === "string" && candidate.toLowerCase() === value.toLowerCase()
    )
  }
  anyOf(...values: Array<Key | Key[]>) {
    const single = values.length === 1 && Array.isArray(values[0]) ? values[0] : null
    const list = (single &&
    (!Array.isArray(this.path()) || single.every((value) => Array.isArray(value)))
      ? single
      : values) as Key[]
    return this.collection((candidate) => list.some((value) => compareKeys(candidate, value) === 0))
  }
  noneOf(values: Key[]) {
    return this.collection((candidate) => values.every((value) => compareKeys(candidate, value) !== 0))
  }
  startsWith(prefix: string) {
    return this.collection((candidate) => typeof candidate === "string" && candidate.startsWith(prefix))
  }
  above(value: Key) {
    return this.collection((candidate) => compareKeys(candidate, value) > 0)
  }
  aboveOrEqual(value: Key) {
    return this.collection((candidate) => compareKeys(candidate, value) >= 0)
  }
  below(value: Key) {
    return this.collection((candidate) => compareKeys(candidate, value) < 0)
  }
  belowOrEqual(value: Key) {
    return this.collection((candidate) => compareKeys(candidate, value) <= 0)
  }
  between(lower: Key, upper: Key, includeLower = true, includeUpper = false) {
    return this.collection((candidate) => {
      const low = compareKeys(candidate, lower)
      const high = compareKeys(candidate, upper)
      return (includeLower ? low >= 0 : low > 0) && (includeUpper ? high <= 0 : high < 0)
    })
  }
}

export class MemoryTable {
  readonly rows = new Map<string, Row>()
  readonly primaryKey: string | string[]
  private readonly autoIncrement: boolean
  private readonly uniqueIndexes: Array<string | string[]>
  private sequence = 0

  constructor(
    readonly name: string,
    spec: string
  ) {
    const [primary, ...indexes] = spec.split(",")
    this.primaryKey = parseIndex(primary)
    this.autoIncrement = primary.trim().startsWith("++")
    this.uniqueIndexes = indexes.filter((index) => index.trim().startsWith("&")).map(parseIndex)
  }

  encode(key: Key) {
    return JSON.stringify(key instanceof Date ? { date: key.getTime() } : key)
  }

  keyOf(row: Row): Key {
    return readPath(row, this.primaryKey) as Key
  }

  /** Store a row under its key; unique indexes behave like IndexedDB constraints. */
  write(row: Row, explicitKey?: Key) {
    let key = explicitKey ?? this.keyOf(row)
    if (key === undefined && this.autoIncrement && typeof this.primaryKey === "string") {
      this.sequence += 1
      key = this.sequence
      row[this.primaryKey] = key
    }
    if (!isValidKey(key)) throw new Error(`DataError: invalid key for ${this.name}`)
    const encoded = this.encode(key)
    for (const index of this.uniqueIndexes) {
      const value = readPath(row, index)
      if (value === undefined) continue
      for (const [otherKey, other] of this.rows) {
        if (otherKey === encoded) continue
        const otherValue = readPath(other, index)
        if (otherValue !== undefined && compareKeys(otherValue, value) === 0) {
          const error = new Error(`ConstraintError: ${this.name} unique index`)
          error.name = "ConstraintError"
          throw error
        }
      }
    }
    this.rows.set(encoded, clone(row))
    return key
  }

  private whereCriteria(criteria: Row) {
    return new MemoryCollection(this, {
      index: null,
      predicates: [
        (row) =>
          Object.entries(criteria).every(([path, value]) => {
            const candidate = readPath(row, path)
            return candidate !== undefined && compareKeys(candidate, value) === 0
          })
      ],
      reverse: false,
      offset: 0,
      limit: null
    })
  }

  toCollection() {
    return new MemoryCollection(this, { index: null, predicates: [], reverse: false, offset: 0, limit: null })
  }

  async get(keyOrCriteria: Key | Row) {
    if (keyOrCriteria === undefined || keyOrCriteria === null) return undefined
    if (!isValidKey(keyOrCriteria)) return this.whereCriteria(keyOrCriteria as Row).first()
    return clone(this.rows.get(this.encode(keyOrCriteria)))
  }
  async bulkGet(keys: Key[]) {
    return keys.map((key) => clone(this.rows.get(this.encode(key))))
  }
  async put(row: Row, key?: Key) {
    return clone(this.write(clone(row), key))
  }
  async add(row: Row, key?: Key) {
    const candidate = key ?? this.keyOf(row)
    if (candidate !== undefined && this.rows.has(this.encode(candidate))) {
      const error = new Error(`ConstraintError: ${this.name} key already exists`)
      error.name = "ConstraintError"
      throw error
    }
    return clone(this.write(clone(row), key))
  }
  async bulkPut(rows: Row[]) {
    let last: Key | undefined
    for (const row of rows) last = this.write(clone(row))
    return last
  }
  async bulkAdd(rows: Row[]) {
    let last: Key | undefined
    for (const row of rows) last = await this.add(row)
    return last
  }
  async update(keyOrRow: Key | Row, changes: Row | ((row: Row) => void)) {
    const key = isValidKey(keyOrRow) ? keyOrRow : this.keyOf(keyOrRow as Row)
    const existing = this.rows.get(this.encode(key))
    if (!existing) return 0
    const next = clone(existing)
    if (typeof changes === "function") changes(next)
    else applyChanges(next, changes)
    this.write(next, key)
    return 1
  }
  async delete(key: Key) {
    this.rows.delete(this.encode(key))
  }
  async bulkDelete(keys: Key[]) {
    for (const key of keys) this.rows.delete(this.encode(key))
  }
  async clear() {
    this.rows.clear()
  }
  async count() {
    return this.rows.size
  }
  async toArray() {
    return this.toCollection().toArray()
  }
  async each(callback: (row: Row) => void) {
    return this.toCollection().each(callback)
  }
  where(index: string): MemoryWhereClause
  where(criteria: Row): MemoryCollection
  where(index: string | Row): MemoryWhereClause | MemoryCollection {
    if (typeof index !== "string") return this.whereCriteria(index)
    return new MemoryWhereClause(this, index)
  }
  filter(predicate: Predicate) {
    return this.toCollection().filter(predicate)
  }
  orderBy(index: string) {
    const path = parseIndex(index)
    return new MemoryCollection(this, {
      index: path,
      predicates: [(row) => readPath(row, path) !== undefined],
      reverse: false,
      offset: 0,
      limit: null
    })
  }
  reverse() {
    return this.toCollection().reverse()
  }
  limit(count: number) {
    return this.toCollection().limit(count)
  }
  offset(count: number) {
    return this.toCollection().offset(count)
  }
}

export type MemoryDexie = Record<string, MemoryTable> & {
  transaction: (...args: unknown[]) => Promise<unknown>
  open: () => Promise<void>
  delete: () => Promise<void>
  close: () => void
  isOpen: () => boolean
  tables: MemoryTable[]
  table: (name: string) => MemoryTable
  /** Test-only: drop every row in every table. */
  resetAll: () => void
}

/**
 * Builds a fresh in-memory database. Transactions run one at a time (like
 * overlapping IndexedDB read-write transactions), nested calls join the
 * outer transaction, and a throwing transaction restores every table.
 */
export const createMemoryDexie = (): MemoryDexie => {
  const tables = Object.fromEntries(
    Object.entries(PAGE_ASSIST_SCHEMA).map(([name, spec]) => [name, new MemoryTable(name, spec)])
  )
  const active = new AsyncLocalStorage<{ aborted: boolean }>()
  let tail: Promise<unknown> = Promise.resolve()
  const transaction = (...args: unknown[]) => {
    const operation = args.at(-1) as (tx: unknown) => Promise<unknown>
    const outer = active.getStore()
    if (outer) return Promise.resolve().then(() => operation({ abort: () => { throw new Error("AbortError") } }))
    const run = async () => {
      const snapshots = Object.values(tables).map((table) => new Map(table.rows))
      const context = { aborted: false }
      try {
        return await active.run(context, () =>
          operation({
            abort() {
              context.aborted = true
              const error = new Error("Transaction aborted")
              error.name = "AbortError"
              throw error
            }
          })
        )
      } catch (error) {
        Object.values(tables).forEach((table, index) => {
          table.rows.clear()
          snapshots[index].forEach((row, key) => table.rows.set(key, row))
        })
        throw error
      }
    }
    const result = tail.then(run, run)
    tail = result.catch(() => undefined)
    return result
  }
  const db = {
    ...tables,
    transaction,
    open: async () => undefined,
    delete: async () => {
      Object.values(tables).forEach((table) => table.rows.clear())
    },
    close: () => undefined,
    isOpen: () => true,
    tables: Object.values(tables),
    table: (name: string) => tables[name],
    resetAll: () => {
      Object.values(tables).forEach((table) => table.rows.clear())
      tail = Promise.resolve()
    }
  }
  return db as unknown as MemoryDexie
}
