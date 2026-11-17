import { useStoreChatModelSettings } from "@/store/model"

export type ModelSettingsOperation = {
  isCurrent: () => boolean
  write: (publish: () => void) => boolean
}

export const captureModelSettingsOperation = (): ModelSettingsOperation => {
  let expected = useStoreChatModelSettings.getState()
  let retired = false
  const isCurrent = () =>
    !retired && useStoreChatModelSettings.getState() === expected
  return {
    isCurrent,
    write: (publish) => {
      if (!isCurrent()) return false
      const original = expected
      let seen = false
      const stop = useStoreChatModelSettings.subscribe((next, previous) => {
        // Zustand exposes live state to outer listeners after a nested publication.
        if (retired) return
        if (seen || previous !== original) {
          retired = true
          return
        }
        seen = true
        expected = next
      })
      try {
        publish()
      } catch (error) {
        retired = true
        throw error
      } finally {
        stop()
      }
      if (!isCurrent()) retired = true
      return !retired
    }
  }
}
