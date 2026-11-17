import type { HistoryOwnerV1 } from "@/services/chat-history-selection"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { HistoryViewSelectionV1 } from "@/types/history-selection"
import { chatDraftOwnerKey } from "./chat-draft-owner"
import { isChatDraftAddress, type ChatDraftAddress } from "./chat-draft-address"

export type ActorSettingsTarget = Readonly<{
  key: string
  accountKey: string | null
  localProfileId?: string
  account?: Readonly<{
    requestScope: ServicePromptRequestScope
    isCurrent: () => boolean
  }>
  address?: ChatDraftAddress
  isCurrent: () => boolean
  profileIsCurrent: () => boolean
}>
export type ActorProfileTarget = Readonly<{
  key: string
  isCurrent: () => boolean
}>

export const captureActorProfileTarget = (
  accountKey: string | null | undefined,
  characterId: string | number | null | undefined,
  isCurrent: () => boolean
): ActorProfileTarget | null => {
  if (
    !accountKey ||
    !/^[a-f0-9]{64}$/.test(accountKey) ||
    characterId == null ||
    !String(characterId).length ||
    !isCurrent()
  )
    return null
  return Object.freeze({
    key: JSON.stringify([accountKey, String(characterId)]),
    isCurrent
  })
}

/** A private locator retains the original owner; it grants no history capability. */
export const captureActorSettingsTarget = (params: {
  address: ChatDraftAddress | null
  owner?: HistoryOwnerV1 | null
  view?: HistoryViewSelectionV1 | null
  requestScope?: ServicePromptRequestScope | null
  accountIsCurrent?: () => boolean
  isCurrent: () => boolean
}): ActorSettingsTarget | null => {
  const { address, owner, view, requestScope, accountIsCurrent, isCurrent } =
    params
  if (!isChatDraftAddress(address) || !isCurrent()) return null
  const capturedAddress = Object.freeze(
    address.kind === "local"
      ? { ...address }
      : { ...address, scope: Object.freeze(structuredClone(address.scope)) }
  )
  const accountCurrent = () => accountIsCurrent?.() === true
  const accountKey =
    requestScope && accountCurrent() ? chatDraftOwnerKey(requestScope) : null
  const account =
    accountKey && requestScope
      ? Object.freeze({
          requestScope: structuredClone(requestScope),
          isCurrent: accountCurrent
        })
      : undefined
  if (address.kind === "fresh") {
    if (!accountKey) return null
    const current = () => isCurrent() && accountCurrent()
    return Object.freeze({
      key: JSON.stringify([
        "fresh",
        accountKey,
        address.scope.type,
        address.scope.type === "workspace" ? address.scope.workspaceId : null,
        address.id
      ]),
      accountKey,
      account,
      address: capturedAddress,
      isCurrent: current,
      profileIsCurrent: current
    })
  }
  if (
    !owner ||
    owner.kind === "unavailable" ||
    !view ||
    owner.conversation_id !== address.id ||
    view.conversation_id !== address.id ||
    !view.owner_key ||
    (owner.owner_key && view.owner_key !== owner.owner_key) ||
    owner.validate_lease?.() === false
  )
    return null
  const sourceCurrent = () => isCurrent() && owner.validate_lease?.() !== false
  if (address.kind === "local") {
    if (owner.kind !== "local" || address.profileId !== owner.profile_id)
      return null
    return Object.freeze({
      key: JSON.stringify(["local", owner.profile_id, owner.conversation_id]),
      localProfileId: owner.profile_id,
      address: capturedAddress,
      accountKey,
      account,
      isCurrent: sourceCurrent,
      profileIsCurrent: () =>
        Boolean(accountKey) && sourceCurrent() && accountCurrent()
    })
  }
  if (
    owner.kind !== "native" ||
    address.ownerKey !== view.owner_key ||
    !accountCurrent()
  )
    return null
  const nativeAccountKey = chatDraftOwnerKey(owner.request_scope)
  if (requestScope && chatDraftOwnerKey(requestScope) !== nativeAccountKey)
    return null
  const scope = owner.scope ?? { type: "global" }
  if (
    address.scope.type !== scope.type ||
    (address.scope.type === "workspace" &&
      scope.type === "workspace" &&
      address.scope.workspaceId !== scope.workspaceId)
  )
    return null
  const current = () => sourceCurrent() && accountCurrent()
  return Object.freeze({
    key: JSON.stringify([
      "native",
      nativeAccountKey,
      scope.type,
      scope.type === "workspace" ? scope.workspaceId : null,
      address.id
    ]),
    accountKey: nativeAccountKey,
    address: capturedAddress,
    account: Object.freeze({
      requestScope: structuredClone(owner.request_scope),
      isCurrent: accountCurrent
    }),
    isCurrent: current,
    profileIsCurrent: current
  })
}
