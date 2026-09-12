export type RecipeUncertaintyState = "clear" | "scoped" | "unknown_owner"

export type RecipeDeliveryReceipt = Readonly<{
  id: string
  ownerId: string
  operationId: string
}>

/** Process-local authority state; unknown outcomes quarantine an exact local ID. */
export class RecipePersistenceRegistry {
  private readonly scoped = new Map<string, Set<string>>()
  private readonly unknown = new Set<string>()
  private readonly provisional = new Map<string, RecipeDeliveryReceipt>()
  private readonly exclusive = new Map<string, string>()

  private hasScoped(id: string): boolean {
    for (const ids of this.scoped.values()) {
      if (ids.has(id)) return true
    }
    return false
  }

  read(id: string, ownerId: string | null): RecipeUncertaintyState {
    if (
      this.unknown.has(id) ||
      this.provisional.has(id) ||
      this.exclusive.has(id)
    )
      return "unknown_owner"
    return ownerId && this.scoped.get(ownerId)?.has(id) ? "scoped" : "clear"
  }

  markScoped(id: string, ownerId: string): void {
    let ids = this.scoped.get(ownerId)
    if (!ids) this.scoped.set(ownerId, (ids = new Set()))
    ids.add(id)
  }

  /** Check and install synchronously within the one dispatch authority. */
  reserve(id: string, ownerId: string, operationId?: string): void {
    if (this.read(id, ownerId) !== "clear")
      throw new Error("Recipe has an unresolved operation")
    this.markScoped(id, ownerId)
    if (operationId) this.provisional.set(id, { id, ownerId, operationId })
  }

  /** Receipt proof removes only its provisional handoff, never scoped/unknown evidence. */
  acknowledge(id: string, ownerId: string, operationId: string): boolean {
    const pending = this.provisional.get(id)
    if (pending?.ownerId !== ownerId || pending.operationId !== operationId)
      return false
    this.provisional.delete(id)
    return true
  }

  markUnknown(id: string): void {
    this.unknown.add(id)
  }

  clearScoped(id: string, ownerId: string): void {
    const ids = this.scoped.get(ownerId)
    ids?.delete(id)
    if (ids?.size === 0) this.scoped.delete(ownerId)
  }

  /** Clear only when every exact-ID scoped marker belongs to the reported owner. */
  reconcileExact(id: string, ownerId: string): boolean {
    if (
      this.unknown.has(id) ||
      this.provisional.has(id) ||
      this.exclusive.has(id)
    )
      return false
    for (const [candidateOwner, ids] of this.scoped) {
      if (candidateOwner !== ownerId && ids.has(id)) return false
    }
    this.clearScoped(id, ownerId)
    return true
  }

  /** Hold a process-wide exact-ID guard while unlink commits locally. */
  beginExclusive(id: string, operationId: string): boolean {
    if (
      this.unknown.has(id) ||
      this.provisional.has(id) ||
      this.exclusive.has(id) ||
      this.hasScoped(id)
    )
      return false
    this.exclusive.set(id, operationId)
    return true
  }

  endExclusive(id: string, operationId: string): boolean {
    if (this.exclusive.get(id) !== operationId) return false
    this.exclusive.delete(id)
    return true
  }

  forgetUnknown(id: string): void {
    this.unknown.delete(id)
    this.provisional.delete(id)
  }
}
