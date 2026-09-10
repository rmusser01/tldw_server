const uncertainRecipeIds = new Set<string>()

export const markRecipePersistenceUncertain = (id: string) => {
  uncertainRecipeIds.add(id)
}

export const clearRecipePersistenceUncertainty = (id: string) => {
  uncertainRecipeIds.delete(id)
}

export const isRecipePersistenceUncertain = (id: string) =>
  uncertainRecipeIds.has(id)
