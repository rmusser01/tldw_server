import React from "react"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"

import { chatModelCatalogQueryKey } from "@/hooks/useChatModelCatalog"
import {
  getChatModelsCacheGeneration,
  subscribeChatModelsInvalidation
} from "@/services/chat-models-invalidation"
import { fetchChatModels } from "@/services/tldw-server"
import { resolveSamplingSupport } from "@/utils/chat-sampling-support"

export type SamplingAvailability = {
  /** Why Top K can't be used with this model; undefined when it can. */
  topKUnavailableReason?: string
  /** Why Min P can't be used with this model; undefined when it can. */
  minPUnavailableReason?: string
}

/**
 * Whether the selected model's provider accepts Top K and Min P (#3283). It
 * reads the chat model catalog the composer has already loaded and never
 * fetches it. A setting the provider would reject is not sent, so its
 * control says why.
 */
export const useSelectedModelSamplingSupport = (
  selectedModel: string | null | undefined
): SamplingAvailability => {
  const { t } = useTranslation("common")
  const generation = React.useSyncExternalStore(
    subscribeChatModelsInvalidation,
    getChatModelsCacheGeneration,
    getChatModelsCacheGeneration
  )
  const { data } = useQuery({
    queryKey: chatModelCatalogQueryKey(generation),
    queryFn: () => fetchChatModels({ requireFresh: true }),
    enabled: false
  })
  return React.useMemo(() => {
    const support = resolveSamplingSupport(data, selectedModel)
    const reason = (supported: boolean | null) =>
      supported === true
        ? undefined
        : supported === false
          ? (t("modelSettings.samplingNotSupported", "Not supported by {{provider}}", {
              provider:
                support.providerName ||
                t("modelSettings.samplingThisProvider", "this provider")
            }) as string)
          : (t(
              "modelSettings.samplingSupportUnknown",
              "Not sent: the server doesn't list whether this model accepts it"
            ) as string)
    return {
      topKUnavailableReason: reason(support.topK),
      minPUnavailableReason: reason(support.minP)
    }
  }, [data, selectedModel, t])
}
