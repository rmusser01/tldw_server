let generation = 0
const listeners = new Set<() => void>()

export const getChatModelsCacheGeneration = () => generation
export const subscribeChatModelsInvalidation = (listener: () => void) => {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}
export const invalidateChatModels = () => {
  generation++
  listeners.forEach(listener => listener())
}
