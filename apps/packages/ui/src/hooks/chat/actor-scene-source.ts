import { useStoreMessageOption } from "@/store/option"
import {
  captureActorSettingsTarget,
  type ActorSettingsTarget
} from "@/utils/actor-settings-target"
import type { ActorSceneRead } from "@/services/actor-settings"
import { useActorStore } from "@/store/actor"
import { buildActorPrompt, estimateActorTokens } from "@/utils/actor"
import { chatDraftAddressForReceipt } from "./chat-draft-transition"
import type {
  HistorySelectionController,
  HistoryLoadReceipt
} from "./useHistorySelection"

/** Destination ownership is independent of an editor's panel, model and cursor. */
export const captureCurrentActorSceneTarget = (
  controller: HistorySelectionController | null,
  commandIsCurrent: () => boolean = () => true
) => {
  const source = useStoreMessageOption.getState()
  const address = source.chatDraftAddress
  if (
    !address ||
    (address.kind === "fresh" &&
      (source.serverChatId ||
        (source.historyId && source.historyId !== "temp"))) ||
    (address.kind === "native" && source.serverChatId !== address.id) ||
    (address.kind === "local" &&
      (source.serverChatId || source.historyId !== address.id))
  )
    return null
  const selected = controller?.getCurrent()
  const owner = selected?.owner
  const view = selected?.view
  const account = controller?.draftAccount
  const accountIsCurrent =
    address.kind === "native" && owner?.kind === "native"
      ? owner.validate_lease
      : account?.isCurrent
  const requestScope =
    address.kind === "native" && owner?.kind === "native"
      ? owner.request_scope
      : account?.requestScope
  const isCurrent = () => {
    const current = useStoreMessageOption.getState()
    if (
      !commandIsCurrent() ||
      current.chatDraftAddress !== address ||
      current.chatDraftRevision !== source.chatDraftRevision ||
      current.historyId !== source.historyId ||
      current.serverChatId !== source.serverChatId
    )
      return false
    if (address.kind === "fresh") return true
    const latest = controller?.getCurrent()
    return (
      latest?.owner === owner &&
      latest?.view?.owner_key === view?.owner_key &&
      latest?.view?.conversation_id === view?.conversation_id
    )
  }
  return captureActorSettingsTarget({
    address,
    owner,
    view,
    requestScope,
    accountIsCurrent,
    isCurrent
  })
}

/** Read once before command preparation; defaults remain staged until an explicit scene Save. */
export const readActorSceneForSend = async (
  controller: HistorySelectionController | null,
  commandIsCurrent: () => boolean = () => true
): Promise<ActorSceneRead> => {
  const target = captureCurrentActorSceneTarget(controller, commandIsCurrent)
  if (!target) throw new Error("actor_settings_unavailable")
  const { getActorSceneForChat } = await import("@/services/actor-settings")
  return getActorSceneForChat({ target })
}

/** The caller retains this receipt only after its bind was actually accepted. */
export const captureBoundActorSceneTarget = (
  source: ActorSettingsTarget,
  receipt: HistoryLoadReceipt,
  bindingIsCurrent: () => boolean
): ActorSettingsTarget | null => {
  const fresh = source.address
  const account = source.account
  if (
    fresh?.kind !== "fresh" ||
    fresh.temporary ||
    !account?.isCurrent() ||
    !bindingIsCurrent()
  )
    return null
  const address = chatDraftAddressForReceipt(receipt)
  if (
    !address ||
    address.kind === "fresh" ||
    (address.kind === "native" &&
      (address.scope.type !== fresh.scope.type ||
        (address.scope.type === "workspace" &&
          fresh.scope.type === "workspace" &&
          address.scope.workspaceId !== fresh.scope.workspaceId)))
  )
    return null
  return captureActorSettingsTarget({
    address,
    ...receipt,
    requestScope: account.requestScope,
    accountIsCurrent: account.isCurrent,
    isCurrent: bindingIsCurrent
  })
}

export const bindActorSceneAfterLoad = async (
  source: ActorSceneRead,
  receipt: HistoryLoadReceipt,
  bindingIsCurrent: () => boolean,
  preparationIsCurrent: () => boolean = () => true
): Promise<boolean> => {
  const destination = captureBoundActorSceneTarget(
    source.target,
    receipt,
    bindingIsCurrent
  )
  if (!destination || !preparationIsCurrent()) return false
  if (source.originalRecord === null)
    return destination.isCurrent() && preparationIsCurrent()
  const { copyActorSceneToBoundChat } = await import(
    "@/services/actor-settings"
  )
  if (
    !preparationIsCurrent() ||
    !(await copyActorSceneToBoundChat(
      source,
      destination,
      preparationIsCurrent
    )) ||
    !destination.isCurrent() ||
    !preparationIsCurrent()
  )
    return false
  const preview = buildActorPrompt(source.settings)
  const applied = useActorStore
    .getState()
    .applyScene({
      target: destination,
      settings: source.settings,
      preview,
      tokenCount: estimateActorTokens(preview)
    })
  return (
    applied &&
    preparationIsCurrent() &&
    source.target.account?.isCurrent() === true
  )
}
