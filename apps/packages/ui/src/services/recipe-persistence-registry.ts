export type RecipeUncertaintyState = "clear" | "scoped" | "unknown_owner"

/** Process-local authority state; unknown outcomes quarantine an exact local ID. */
export class RecipePersistenceRegistry {
  private readonly scoped = new Map<string, Set<string>>()
  private readonly unknown = new Set<string>()

  read(id: string, ownerId: string | null): RecipeUncertaintyState {
    if (this.unknown.has(id)) return "unknown_owner"
    return ownerId && this.scoped.get(ownerId)?.has(id) ? "scoped" : "clear"
  }

  markScoped(id: string, ownerId: string): void {
    let ids = this.scoped.get(ownerId)
    if (!ids) this.scoped.set(ownerId, (ids = new Set()))
    ids.add(id)
  }

  /** Check and install synchronously within the one dispatch authority. */
  reserve(id: string, ownerId: string): void {
    if (this.read(id, ownerId) !== "clear")
      throw new Error("Recipe has an unresolved operation")
    this.markScoped(id, ownerId)
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
  }
}
