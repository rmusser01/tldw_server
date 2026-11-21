import { watchChatAccountChanges } from "@/services/chat-account-boundary"

let generation = 0
const listeners = new Set<() => void>()

export const getWatchlistsAccountGeneration = () => generation

export const watchWatchlistsAccountChanges = (listener: () => void) => {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

const advanceGeneration = () => {
  generation += 1
  for (const listener of listeners) listener()
}

const stopWatching = watchChatAccountChanges((invalidated) => {
  if (invalidated) advanceGeneration()
})
const hot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
hot?.dispose(stopWatching)
