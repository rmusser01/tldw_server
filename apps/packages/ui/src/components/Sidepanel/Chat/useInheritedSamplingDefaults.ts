import React from "react"

import {
  resolveInheritedSampling,
  type InheritedSampling
} from "@/models/chat-sampling-defaults"
import { getAllDefaultModelSettings } from "@/services/model-settings"
import { useStoreChatModelSettings } from "@/store/model"

/**
 * The value each sampling parameter takes when the current chat leaves it
 * unset (XS-16): the user's saved default model settings, else the value
 * ChatTldw sends. Starts from the ChatTldw values and updates once the saved
 * defaults have loaded.
 */
export const useInheritedSamplingDefaults = (): InheritedSampling => {
  const [savedDefaults, setSavedDefaults] = React.useState<
    Parameters<typeof resolveInheritedSampling>[0]
  >(null)

  React.useEffect(() => {
    let cancelled = false
    getAllDefaultModelSettings()
      .then((settings) => {
        if (!cancelled) setSavedDefaults(settings)
      })
      .catch((error: unknown) => {
        console.debug("[ModelParamsPanel] saved model defaults unavailable:", error)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return React.useMemo(
    () => resolveInheritedSampling(savedDefaults),
    [savedDefaults]
  )
}

type SamplingKey =
  | "temperature"
  | "topP"
  | "frequencyPenalty"
  | "presencePenalty"
  | "minP"

/**
 * Whether a parameter is set where the panel writes it: the active model's
 * settings, or the all-models settings when no model is active. A value that
 * only comes from the all-models settings cannot be reset from a model's
 * panel, because a reset only clears the model's own value.
 */
export const useIsSamplingSetHere = (): ((key: SamplingKey) => boolean) => {
  const ownSettings = useStoreChatModelSettings((state) =>
    state.activeSettingsScope
      ? state.scopedSettingsByModelKey[state.activeSettingsScope]
      : state.globalSettings
  )
  return React.useCallback(
    (key: SamplingKey) => {
      const value = ownSettings?.[key]
      return typeof value === "number" && Number.isFinite(value)
    },
    [ownSettings]
  )
}
