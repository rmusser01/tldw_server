import {
  cloneIdSchema,
  type SharedCloneOperation
} from "@/types/shared-workspace-clone"
import { getStructuredApiErrorDetail, TldwApiError } from "./tldw/api-error"
import type { SharedWorkspaceCloneApi } from "./tldw/domains/shared-workspaces"
import {
  CLONE_RECOVERY_LIMIT,
  CLONE_RECOVERY_TTL,
  CloneRecoveryFullError,
  CloneRecoveryScopeMismatchError,
  readCloneRecovery,
  sameCloneRecovery,
  writeCloneRecovery,
  type CloneRecoveryEntry,
  type CloneStorage
} from "./shared-clone-recovery"

export type CloneRowState = {
  entry: CloneRecoveryEntry
  operation?: SharedCloneOperation
  issue?: "uncertain" | "rejected" | "unavailable" | "recovery_full"
  pending: boolean
  recoveryAvailable: boolean
}
type ScheduledRow = CloneRowState & {
  admissionUncertain: boolean
  nextAt: number
  cooldownUntil: number
  controller?: AbortController
}
const REQUEST_DEADLINE_MS = 30_000

const terminal = (operation?: SharedCloneOperation) =>
  operation?.status === "succeeded" || operation?.status === "failed"

/** One page-owned scheduler: at most four requests, no polling while hidden. */
export class SharedCloneManager {
  private states = new Map<number, ScheduledRow>()
  private timer?: ReturnType<typeof setTimeout>
  private disposed = false
  private visible = true
  private inFlight = 0
  private recoveryAvailable = true
  private activity = new AbortController()
  private starting = new Set<number>()

  constructor(
    private scope: string,
    private storage: CloneStorage,
    private api: SharedWorkspaceCloneApi,
    private changed: () => void,
    private scopeChanged?: (
      reason: "auth_required" | "recovery_conflict"
    ) => void
  ) {}

  rows(): CloneRowState[] {
    return [...this.states.values()].map(
      ({ entry, operation, issue, pending, recoveryAvailable }) => ({
        entry,
        operation,
        issue,
        pending,
        recoveryAvailable
      })
    )
  }

  private async persist(
    entry: CloneRecoveryEntry | null,
    expected: CloneRecoveryEntry | null,
    signal: AbortSignal
  ): Promise<CloneRecoveryEntry | null | false> {
    if (signal.aborted) return false
    try {
      if (!this.recoveryAvailable) {
        // Volatile receipts must not overwrite older durable rows, but a
        // temporary storage failure must not disable future scope checks.
        await readCloneRecovery(this.storage, this.scope, undefined, signal)
        return entry
      }
      return await writeCloneRecovery(
        this.storage,
        this.scope,
        entry,
        undefined,
        expected?.share_id,
        { expected, signal }
      )
    } catch (error) {
      if (
        signal.aborted ||
        (error instanceof DOMException && error.name === "AbortError")
      )
        return false
      if (error instanceof CloneRecoveryFullError) return false
      if (error instanceof CloneRecoveryScopeMismatchError) {
        this.suspend()
        this.changed()
        this.scopeChanged?.("recovery_conflict")
        return false
      }
      this.recoveryAvailable = false
      for (const row of this.states.values()) row.recoveryAvailable = false
      return entry
    }
  }

  async resume(api?: SharedWorkspaceCloneApi): Promise<void> {
    if (this.disposed) return
    if (api && api !== this.api) {
      // Replacement is only supplied after same-scope verification. Fence every
      // old request before future requests can use the rotated credentials.
      const visible = this.visible
      this.suspend()
      this.api = api
      this.visible = visible
    }
    if (this.activity.signal.aborted) this.activity = new AbortController()
    await this.sync()
  }

  private adopt(entry: CloneRecoveryEntry | null, shareId: number): void {
    const current = this.states.get(shareId)
    if (entry && sameCloneRecovery(current?.entry, entry)) return
    current?.controller?.abort()
    if (!entry) {
      this.states.delete(shareId)
      return
    }
    this.states.set(shareId, {
      entry,
      // A recovered command may already have reached the server in another tab.
      admissionUncertain: true,
      pending: false,
      recoveryAvailable: this.recoveryAvailable,
      nextAt: Date.now(),
      cooldownUntil: 0
    })
  }

  async sync(): Promise<void> {
    const signal = this.activity.signal
    if (this.disposed || signal.aborted) return
    try {
      const records = await readCloneRecovery(
        this.storage,
        this.scope,
        undefined,
        signal
      )
      if (this.disposed || signal.aborted) return
      if (this.recoveryAvailable)
        for (const entry of records) this.adopt(entry, entry.share_id)
    } catch (error) {
      if (this.disposed || signal.aborted) return
      if (error instanceof CloneRecoveryScopeMismatchError) {
        this.suspend()
        this.changed()
        this.scopeChanged?.("recovery_conflict")
        return
      }
      this.recoveryAvailable = false
      for (const row of this.states.values()) row.recoveryAvailable = false
    }
    this.changed()
    this.pump()
  }

  async begin(shareId: number, workspaceName: string): Promise<void> {
    const signal = this.activity.signal
    if (
      this.disposed ||
      signal.aborted ||
      this.starting.has(shareId) ||
      !Number.isSafeInteger(shareId) ||
      shareId < 1
    )
      return
    this.starting.add(shareId)
    try {
      await this.sync()
      if (this.disposed || signal.aborted) return
      const previous = this.states.get(shareId)
      if (
        previous &&
        previous.issue !== "rejected" &&
        previous.issue !== "recovery_full" &&
        !(
          previous.operation?.status === "failed" &&
          previous.operation.retryable
        )
      )
        return
      const entry: CloneRecoveryEntry = {
        share_id: shareId,
        expires_at: Date.now() + CLONE_RECOVERY_TTL,
        name: `${Array.from(
          workspaceName.trim().replace(/\s+/g, " ") || "Workspace"
        )
          .slice(0, 248)
          .join("")} (Copy)`,
        idempotency_key: crypto.randomUUID()
      }
      const room =
        this.states.has(shareId) || this.states.size < CLONE_RECOVERY_LIMIT
      const expected =
        previous?.issue === "rejected" || previous?.issue === "recovery_full"
          ? null
          : (previous?.entry ?? null)
      const saved = room ? await this.persist(entry, expected, signal) : false
      if (this.disposed || signal.aborted) return
      if (saved !== false && !sameCloneRecovery(saved, entry)) {
        this.adopt(saved, shareId)
        this.changed()
        this.pump()
        return
      }
      this.states.set(shareId, {
        entry,
        admissionUncertain: false,
        pending: false,
        recoveryAvailable: this.recoveryAvailable,
        issue: saved !== false ? undefined : "recovery_full",
        nextAt: saved !== false ? Date.now() : Infinity,
        cooldownUntil: 0
      })
      this.changed()
      this.pump()
    } finally {
      if (this.activity.signal === signal) this.starting.delete(shareId)
    }
  }

  refresh(shareId?: number): void {
    for (const [id, row] of this.states) {
      if (
        (shareId === undefined || shareId === id) &&
        (shareId !== undefined || !terminal(row.operation)) &&
        row.issue !== "rejected" &&
        row.issue !== "recovery_full"
      ) {
        row.nextAt = Math.max(Date.now(), row.cooldownUntil)
      }
    }
    this.pump()
  }

  setVisible(visible: boolean): void {
    this.visible = visible
    if (visible) this.refresh()
    else this.pump()
  }

  suspend(): void {
    this.visible = false
    this.activity.abort()
    this.starting.clear()
    if (this.timer) clearTimeout(this.timer)
    for (const row of this.states.values()) {
      row.controller?.abort()
      row.controller = undefined
      row.pending = false
    }
  }

  private pump(): void {
    if (this.timer) clearTimeout(this.timer)
    const signal = this.activity.signal
    if (this.disposed || signal.aborted) return
    const now = Date.now()
    for (const [id, row] of this.states) {
      if (row.entry.expires_at <= now) {
        row.controller?.abort()
        this.states.delete(id)
        void this.persist(null, row.entry, signal).then((latest) => {
          if (this.disposed || signal.aborted || !latest) return
          this.adopt(latest, id)
          this.changed()
          this.pump()
        })
        this.changed()
      } else if (
        this.visible &&
        !row.pending &&
        row.nextAt <= now &&
        this.inFlight < 4
      ) {
        void this.request(row)
      }
    }
    const next = Math.min(
      ...[...this.states.values()].flatMap((row) => [
        row.entry.expires_at,
        this.visible && !row.pending && this.inFlight < 4
          ? row.nextAt
          : Infinity
      ])
    )
    if (Number.isFinite(next))
      this.timer = setTimeout(() => this.pump(), Math.max(1, next - Date.now()))
  }

  private async request(row: ScheduledRow): Promise<void> {
    const entry = row.entry
    const admissionUncertain = row.admissionUncertain
    // Fence even aborted requests: losing a response is not proof of rejection.
    if (!entry.operation_id) row.admissionUncertain = true
    const controller = new AbortController()
    row.controller = controller
    row.pending = true
    this.inFlight++
    this.changed()
    const current = () =>
      !this.disposed &&
      !this.activity.signal.aborted &&
      row.controller === controller &&
      this.states.get(entry.share_id) === row
    let deadline: ReturnType<typeof setTimeout> | undefined
    let aborted!: () => void
    const interrupted = new Promise<never>((_resolve, reject) => {
      aborted = () =>
        reject(new DOMException("Clone request interrupted", "AbortError"))
      controller.signal.addEventListener("abort", aborted, { once: true })
      deadline = setTimeout(() => controller.abort(), REQUEST_DEADLINE_MS)
    })
    try {
      const operation = await Promise.race([
        interrupted,
        entry.operation_id
          ? this.api.cloneStatus(
              entry.share_id,
              entry.operation_id,
              controller.signal
            )
          : this.api.clone(
              entry.share_id,
              { name: entry.name },
              entry.idempotency_key!,
              controller.signal
            )
      ])
      clearTimeout(deadline)
      if (!current()) return
      const updated = terminal(operation)
        ? {
            share_id: entry.share_id,
            expires_at: entry.expires_at,
            operation_id: operation.operation_id
          }
        : { ...entry, operation_id: operation.operation_id }
      if (
        !(await this.updateRecovery(row, updated, controller.signal)) ||
        !current()
      )
        return
      row.operation = operation
      row.issue = undefined
      row.cooldownUntil = 0
      row.entry = updated
      row.nextAt = terminal(operation) ? Infinity : Date.now() + 2000
      row.recoveryAvailable = this.recoveryAvailable
    } catch (error) {
      if (!current()) return
      const detail = getStructuredApiErrorDetail(error)
      const status = error instanceof TldwApiError ? error.status : 0
      if (
        status === 401 ||
        status === 403 ||
        (status === 412 && detail?.code === "request_config_scope_changed")
      ) {
        // The original admission may already exist. Keep its recovery key,
        // but stop all dispatch until the owner verifies a new context.
        row.issue = "unavailable"
        row.nextAt = Infinity
        this.suspend()
        this.changed()
        this.scopeChanged?.("auth_required")
        return
      }
      const conflictId = cloneIdSchema.safeParse(detail?.operation_id)
      if (
        !entry.operation_id &&
        status === 409 &&
        detail?.code === "clone_already_in_progress" &&
        conflictId.success
      ) {
        const updated = { ...entry, operation_id: conflictId.data }
        if (
          !(await this.updateRecovery(row, updated, controller.signal)) ||
          !current()
        )
          return
        row.entry = updated
        row.nextAt = Date.now()
      } else if (
        status >= 400 &&
        status < 500 &&
        status !== 408 &&
        status !== 429 &&
        !(status === 409 && detail?.code === "clone_already_in_progress")
      ) {
        if (
          !entry.operation_id &&
          !admissionUncertain &&
          (!(await this.updateRecovery(row, null, controller.signal)) ||
            !current())
        )
          return
        // A later permission/validation failure says nothing about an earlier
        // ambiguous POST. Keep its key; only a definite first rejection clears it.
        row.issue =
          entry.operation_id || admissionUncertain ? "unavailable" : "rejected"
        row.nextAt = Infinity
      } else {
        row.issue = "uncertain"
        const delay = Math.max(
          5000,
          Math.min(detail?.retry_after_ms ?? 0, 1_800_000)
        )
        row.cooldownUntil = Date.now() + delay
        row.nextAt = row.cooldownUntil
      }
    } finally {
      clearTimeout(deadline)
      controller.signal.removeEventListener("abort", aborted)
      this.inFlight--
      if (current()) {
        row.pending = false
        row.controller = undefined
        this.changed()
      }
      this.pump()
    }
  }

  private async updateRecovery(
    row: ScheduledRow,
    entry: CloneRecoveryEntry | null,
    signal: AbortSignal
  ): Promise<boolean> {
    const saved = await this.persist(entry, row.entry, signal)
    if (
      this.disposed ||
      signal.aborted ||
      this.states.get(row.entry.share_id) !== row
    )
      return false
    if (saved === false) {
      // A received operation remains authoritative even if a legacy map is full.
      this.recoveryAvailable = false
      for (const current of this.states.values())
        current.recoveryAvailable = false
      return true
    }
    if (!sameCloneRecovery(saved, entry)) {
      this.adopt(saved, row.entry.share_id)
      this.changed()
      return false
    }
    return true
  }

  dispose(): void {
    this.disposed = true
    this.activity.abort()
    if (this.timer) clearTimeout(this.timer)
    for (const row of this.states.values()) row.controller?.abort()
  }
}
