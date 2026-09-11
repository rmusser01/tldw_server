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

  read(id: string, ownerId: string | null): RecipeUncertaintyState {
    if (this.unknown.has(id) || this.provisional.has(id)) return "unknown_owner"
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

  forgetUnknown(id: string): void {
    this.unknown.delete(id)
    this.provisional.delete(id)
  }
}
