import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { resolveStartupSelectedModel } from "@/utils/model-startup-selection"

type StartupModel = Parameters<typeof resolveStartupSelectedModel>[0]["models"]

type UseSidepanelStartupModelParams = {
  /** True once the server connection is ready. */
  enabled: boolean
  /** The verified chat model catalog; undefined while it is still loading. */
  models: StartupModel
}

/**
 * Choose a model for a side panel that has none (XS-02, #3111), the same way
 * the full chat page does: a favourite, then the server default, then a
 * provider that has answered, then the first available model. A model the
 * person already chose is never replaced.
 */
export function useSidepanelStartupModel({
  enabled,
  models
}: UseSidepanelStartupModelParams) {
  const { selectedModel, setSelectedModel, selectedModelIsLoading } = useSelectedModel()
  const [favoriteModels, , favoriteModelsMeta] = useStorage<string[]>(
    "favoriteChatModels",
    []
  )
  const favoritesLoading = Boolean(favoriteModelsMeta?.isLoading)

  React.useEffect(() => {
    if (!enabled || !Array.isArray(models)) return
    const nextModel = resolveStartupSelectedModel({
      currentModel: selectedModel,
      models,
      preferredModelIds: favoriteModels,
      isCurrentModelHydrating: selectedModelIsLoading,
      arePreferencesHydrating: favoritesLoading
    })
    if (nextModel) {
      void Promise.resolve(setSelectedModel(nextModel)).catch(() => undefined)
    }
  }, [
    enabled,
    favoriteModels,
    favoritesLoading,
    models,
    selectedModel,
    selectedModelIsLoading,
    setSelectedModel
  ])
}
