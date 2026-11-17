import React from "react"
import { useTranslation } from "react-i18next"
import {
  getCanonicalModelKey,
  getModelId,
  getModelProvider,
  isConfiguredUsableModel,
  type ModelSelectorDescriptor
} from "@/hooks/playground/modelSelectorUtils"
import {
  chatModelReadinessKey,
  getChatModelSendOutcome,
  getChatModelSendOutcomesVersion,
  resolveChatModelReadinessState,
  subscribeChatModelSendOutcomes,
  type ChatModelReadinessState
} from "@/utils/chat-model-readiness"
import { formatChatModelLabel } from "@/utils/chat-model-label"
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider"

export type ComposerModelSwitchTarget = {
  /** Selection key, as the model picker stores it. */
  key: string
  /** "Switch to <model (provider)>" */
  label: string
  /** The target model's label alone, for announcements. */
  modelLabel: string
}

export type ComposerModelReadiness = {
  state: ChatModelReadinessState
  label: string
  title: string
  switchTarget: ComposerModelSwitchTarget | null
  switchToTarget: () => void
}

type UseComposerModelReadinessParams = {
  composerModels: ModelSelectorDescriptor[] | null | undefined
  selectedModel: string | null | undefined
  selectedModelMeta: ModelSelectorDescriptor | null | undefined
  providerKey: string | null | undefined
  onSelectModel: (modelKey: string) => unknown
}

const readinessOf = (model: ModelSelectorDescriptor) =>
  resolveChatModelReadinessState({
    model,
    provider: getModelProvider(model),
    modelId: getModelId(model)
  })

/**
 * The selected model's readiness for the composer chip (CC-01, #3111): "Not
 * checked" until there is evidence, "Working" after it answers, "Unreachable"
 * after its provider fails. When it is unreachable, offer a one-click switch
 * to the server's default model, or to another model that is known to work.
 */
export function useComposerModelReadiness({
  composerModels,
  selectedModel,
  selectedModelMeta,
  providerKey,
  onSelectModel
}: UseComposerModelReadinessParams): ComposerModelReadiness | null {
  const { t } = useTranslation(["playground"])
  const outcomesVersion = React.useSyncExternalStore(
    subscribeChatModelSendOutcomes,
    getChatModelSendOutcomesVersion,
    getChatModelSendOutcomesVersion
  )

  const readiness = React.useMemo(() => {
    const selected = selectedModel?.trim()
    if (!selected) return null
    const selection = parseProviderQualifiedModelSelection(selected)
    const modelId = selectedModelMeta ? getModelId(selectedModelMeta) : selection.modelId
    const provider = selectedModelMeta
      ? getModelProvider(selectedModelMeta)
      : selection.provider || providerKey
    const state = resolveChatModelReadinessState({
      model: selectedModelMeta,
      provider,
      modelId
    })
    // Without a send from this session the state comes from the server's
    // report on the whole provider, so the copy must not claim this model
    // answered or failed.
    const fromProviderReport = getChatModelSendOutcome(provider, modelId) === null

    let switchTarget: ComposerModelSwitchTarget | null = null
    if (state === "unreachable") {
      const currentKey = chatModelReadinessKey(provider, modelId)
      const candidates = (composerModels ?? []).filter(
        (model) =>
          isConfiguredUsableModel(model) &&
          chatModelReadinessKey(getModelProvider(model), getModelId(model)) !== currentKey
      )
      const target =
        candidates.find(
          (model) => model?.is_server_default === true && readinessOf(model) !== "unreachable"
        ) ?? candidates.find((model) => readinessOf(model) === "working")
      if (target) {
        const modelLabel = formatChatModelLabel({ model: target })
        switchTarget = {
          key: getCanonicalModelKey(target),
          modelLabel,
          label: String(
            t("playground:composer.modelReadiness.switchTo", {
              defaultValue: "Switch to {{model}}",
              model: modelLabel
            } as never)
          )
        }
      }
    }

    const copy: Record<ChatModelReadinessState, { label: string; title: string }> = {
      not_checked: {
        label: String(t("playground:composer.modelReadiness.notChecked", "Not checked")),
        title: String(
          t(
            "playground:composer.modelReadiness.notCheckedTitle",
            "This model has not answered yet. Send a message to check it."
          )
        )
      },
      working: {
        label: String(t("playground:composer.modelReadiness.working", "Working")),
        title: fromProviderReport
          ? String(
              t(
                "playground:composer.modelReadiness.providerWorkingTitle",
                "Its provider answered recently."
              )
            )
          : String(
              t("playground:composer.modelReadiness.workingTitle", "This model answered recently.")
            )
      },
      unreachable: {
        label: String(t("playground:composer.modelReadiness.unreachable", "Unreachable")),
        title: fromProviderReport
          ? String(
              t(
                "playground:composer.modelReadiness.providerUnreachableTitle",
                "Its provider has been failing. Choose another model."
              )
            )
          : String(
              t(
                "playground:composer.modelReadiness.unreachableTitle",
                "The last request to this model failed. Its provider may not be running. Choose another model."
              )
            )
      }
    }
    return { state, ...copy[state], switchTarget }
    // The outcome version re-reads the session outcomes recorded by sends.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [composerModels, outcomesVersion, providerKey, selectedModel, selectedModelMeta, t])

  const targetKey = readiness?.switchTarget?.key ?? null
  const switchToTarget = React.useCallback(() => {
    if (!targetKey) return
    void Promise.resolve(onSelectModel(targetKey)).catch(() => undefined)
  }, [onSelectModel, targetKey])

  return React.useMemo(
    () => (readiness ? { ...readiness, switchToTarget } : null),
    [readiness, switchToTarget]
  )
}
