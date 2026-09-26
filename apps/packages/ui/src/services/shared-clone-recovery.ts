import { z } from "zod"
import {
  cloneIdSchema,
  cloneKeySchema,
  cloneNameSchema
} from "@/types/shared-workspace-clone"

export const CLONE_RECOVERY_KEY = "tldw:sharing:clone-operations:v1"
export const CLONE_RECOVERY_TTL = 7 * 24 * 60 * 60 * 1000
export const CLONE_RECOVERY_LIMIT = 32
const MAX_BYTES = 32 * 1024
const LOCK_WAIT_MS = 5000

const entrySchema = z
  .strictObject({
    share_id: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
    expires_at: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
    name: cloneNameSchema.optional(),
    idempotency_key: cloneKeySchema.optional(),
    operation_id: cloneIdSchema.optional()
  })
  .refine((entry) => Boolean(entry.idempotency_key || entry.operation_id))
  .refine((entry) => !entry.name || Boolean(entry.idempotency_key))

const envelopeSchema = z
  .strictObject({
    version: z.literal(1),
    scope: z.string().min(1).max(2048),
    records: z.array(entrySchema).max(CLONE_RECOVERY_LIMIT)
  })
  .refine(
    (value) =>
      new Set(value.records.map((entry) => entry.share_id)).size ===
      value.records.length
  )

export type CloneRecoveryEntry = z.infer<typeof entrySchema>
export type CloneStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">
export class CloneRecoveryFullError extends Error {}
export class CloneRecoveryScopeMismatchError extends Error {
  constructor() {
    super("Clone recovery belongs to another account or server")
    this.name = "CloneRecoveryScopeMismatchError"
  }
}

let generation = 0

/** All map access, including read cleanup, shares one origin-wide lock. */
async function withRecoveryLock<T>(
  action: () => T,
  signal?: AbortSignal
): Promise<T> {
  const expectedGeneration = generation
  if (typeof navigator === "undefined" || !navigator.locks?.request)
    throw new Error("Durable clone recovery requires Web Locks")
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason)
  const cancelled = new Promise<never>((_resolve, reject) => {
    controller.signal.addEventListener(
      "abort",
      () => reject(controller.signal.reason),
      { once: true }
    )
  })
  signal?.addEventListener("abort", abort, { once: true })
  if (signal?.aborted) abort()
  const timer = setTimeout(
    () =>
      controller.abort(
        new DOMException("Clone recovery lock timed out", "TimeoutError")
      ),
    LOCK_WAIT_MS
  )
  try {
    return await Promise.race([
      cancelled,
      navigator.locks.request(
        CLONE_RECOVERY_KEY,
        { signal: controller.signal },
        () => {
          // The callback is synchronous; cancelled waiters must never mutate later.
          if (controller.signal.aborted) throw controller.signal.reason
          if (generation !== expectedGeneration)
            throw new DOMException("Clone recovery session ended", "AbortError")
          return action()
        }
      )
    ])
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener("abort", abort)
  }
}

export async function clearCloneRecovery(
  expectedScope?: string
): Promise<void> {
  generation++
  if (typeof window === "undefined") return
  const clear = () => {
    if (expectedScope !== undefined) {
      const raw = window.localStorage.getItem(CLONE_RECOVERY_KEY)
      if (!raw || raw.length > MAX_BYTES) return
      const value = envelopeSchema.safeParse(JSON.parse(raw))
      if (!value.success || value.data.scope !== expectedScope) return
    }
    window.localStorage.removeItem(CLONE_RECOVERY_KEY)
  }
  try {
    if (typeof navigator === "undefined" || !navigator.locks?.request) {
      // No durable writers exist in unsupported browsers; logout still clears data.
      clear()
      return
    }
    await withRecoveryLock(clear)
  } catch {
    // Authentication cleanup must still succeed when storage is unavailable.
  }
}

function readUnlocked(
  storage: CloneStorage,
  scope: string,
  now = Date.now()
): CloneRecoveryEntry[] {
  const raw = storage.getItem(CLONE_RECOVERY_KEY)
  if (!raw) return []
  let value: z.infer<typeof envelopeSchema>
  try {
    if (
      raw.length > MAX_BYTES ||
      new TextEncoder().encode(raw).byteLength > MAX_BYTES
    )
      throw new Error("Oversized recovery map")
    const parsed = JSON.parse(raw)
    // Ownership precedes validation/expiry cleanup and every writer's entry CAS.
    if (typeof parsed?.scope === "string" && parsed.scope !== scope) {
      throw new CloneRecoveryScopeMismatchError()
    }
    value = envelopeSchema.parse(parsed)
    if (
      value.records.some((entry) => entry.expires_at > now + CLONE_RECOVERY_TTL)
    ) {
      throw new Error("Invalid recovery expiry")
    }
  } catch (error) {
    if (error instanceof CloneRecoveryScopeMismatchError) throw error
    storage.removeItem(CLONE_RECOVERY_KEY)
    return []
  }
  const records = value.records.filter((entry) => entry.expires_at > now)
  if (records.length !== value.records.length) {
    storage.setItem(CLONE_RECOVERY_KEY, JSON.stringify({ ...value, records }))
  }
  return records
}

export async function readCloneRecovery(
  storage: CloneStorage,
  scope: string,
  now?: number,
  signal?: AbortSignal
): Promise<CloneRecoveryEntry[]> {
  return withRecoveryLock(() => readUnlocked(storage, scope, now), signal)
}

export function sameCloneRecovery(
  left: CloneRecoveryEntry | null | undefined,
  right: CloneRecoveryEntry | null | undefined
): boolean {
  return (
    left?.share_id === right?.share_id &&
    left?.expires_at === right?.expires_at &&
    left?.idempotency_key === right?.idempotency_key &&
    left?.operation_id === right?.operation_id &&
    left?.name === right?.name
  )
}

export async function writeCloneRecovery(
  storage: CloneStorage,
  scope: string,
  entry: CloneRecoveryEntry | null,
  now?: number,
  removeShareId?: number,
  options: { expected?: CloneRecoveryEntry | null; signal?: AbortSignal } = {}
): Promise<CloneRecoveryEntry | null> {
  return withRecoveryLock(() => {
    const records = readUnlocked(storage, scope, now)
    const shareId = entry?.share_id ?? removeShareId
    const latest =
      records.find((existing) => existing.share_id === shareId) ?? null
    // Compare under the lock: even equal-TTL retries must not overwrite each other.
    if (
      options.expected !== undefined &&
      !sameCloneRecovery(latest, options.expected)
    )
      return latest
    const updated = records.filter((existing) => existing.share_id !== shareId)
    if (entry) updated.push(entrySchema.parse(entry))
    updated.sort((left, right) => left.share_id - right.share_id)
    if (updated.length > CLONE_RECOVERY_LIMIT)
      throw new CloneRecoveryFullError("Recovery map is full")
    const value = envelopeSchema.parse({ version: 1, scope, records: updated })
    const raw = JSON.stringify(value)
    // Reserve the fixed UUID field before admitting commands, not after POST succeeds.
    const withPointers = JSON.stringify({
      ...value,
      records: value.records.map((record) => ({
        ...record,
        operation_id:
          record.operation_id ?? "00000000-0000-4000-8000-000000000000"
      }))
    })
    if (new TextEncoder().encode(withPointers).byteLength > MAX_BYTES)
      throw new CloneRecoveryFullError("Recovery map is full")
    if (storage.getItem(CLONE_RECOVERY_KEY) !== raw)
      storage.setItem(CLONE_RECOVERY_KEY, raw)
    return entry
  }, options.signal)
}
