import React from "react"
import {
  getChatModelSendOutcomesVersion,
  resolveChatModelReadinessState,
  subscribeChatModelSendOutcomes,
  type ChatModelReadinessState
} from "@/utils/chat-model-readiness"

type UseChatModelReadinessStateParams = {
  /** The selected model's catalog entry, when it is known. */
  model?: Record<string, unknown> | null
  provider: unknown
  modelId: unknown
}

/**
 * The selected model's readiness ("not_checked", "working", "unreachable"),
 * kept current as sends in this session succeed or fail (CC-01, XS-02).
 * Returns null when no model is selected.
 */
export function useChatModelReadinessState({
  model,
  provider,
  modelId
}: UseChatModelReadinessStateParams): ChatModelReadinessState | null {
  const outcomesVersion = React.useSyncExternalStore(
    subscribeChatModelSendOutcomes,
    getChatModelSendOutcomesVersion,
    getChatModelSendOutcomesVersion
  )
  return React.useMemo(() => {
    if (typeof modelId !== "string" || !modelId.trim()) return null
    return resolveChatModelReadinessState({ model, provider, modelId })
    // The outcome version re-reads the outcomes recorded by sends.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, modelId, outcomesVersion, provider])
}
