import { useStoreMessageOption } from "@/store/option"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import type { HistorySelectionController } from "./useHistorySelection"

/** Destination ownership is independent of an editor's panel, model and cursor. */
export const captureCurrentActorSceneTarget = (
  controller: HistorySelectionController | null
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
