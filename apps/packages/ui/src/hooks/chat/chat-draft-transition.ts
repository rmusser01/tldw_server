import i18n from "i18next";
import { useStoreMessageOption } from "@/store/option";
import { usePlaygroundSessionStore } from "@/store/playground-session";
import type { HistoryLoadReceipt } from "./useHistorySelection";
import {
  isChatDraftAddress,
  type ChatDraftAddress,
  type ChatDraftTransition,
} from "@/utils/chat-draft-address";

export const chatDraftAddressForReceipt = ({
  owner,
  view,
}: HistoryLoadReceipt): ChatDraftAddress | null => {
  if (
    owner.kind === "unavailable" ||
    owner.validate_lease?.() === false ||
    view.conversation_id !== owner.conversation_id ||
    !view.owner_key ||
    (owner.owner_key && view.owner_key !== owner.owner_key)
  )
    return null;
  const address: ChatDraftAddress =
    owner.kind === "local"
      ? {
          kind: "local",
          id: owner.conversation_id,
          profileId: owner.profile_id,
        }
      : {
          kind: "native",
          id: owner.conversation_id,
          ownerKey: view.owner_key,
          scope: owner.scope ?? { type: "global" },
        };
  return isChatDraftAddress(address) ? address : null;
};

/** Capture the source before loading; a qualified H1 result locates the private draft. */
export const captureChatDraftLoad = (
  kind: ChatDraftTransition["kind"],
  isCurrent: () => boolean = () => true,
) => {
  const source = useStoreMessageOption.getState();
  const expectedRevision = source.chatDraftRevision;
  const sourceAddress = source.chatDraftAddress;
  let activeRevision = expectedRevision;
  let activeAddress = sourceAddress;
  let committed = false;
  const current = () => {
    const state = useStoreMessageOption.getState();
    return (
      isCurrent() &&
      state.chatDraftRevision === activeRevision &&
      state.chatDraftAddress === activeAddress
    );
  };
  const accept = (receipt: HistoryLoadReceipt): boolean => {
    if (committed || !current()) return false;
    const address = chatDraftAddressForReceipt(receipt);
    if (!address) return false;
    if (kind === "bind" && !sourceAddress) return false;
    committed = useStoreMessageOption
      .getState()
      .setChatDraftAddress(address, { kind, expectedRevision, sourceAddress });
    if (committed) {
      const state = useStoreMessageOption.getState();
      activeRevision = state.chatDraftRevision;
      activeAddress = state.chatDraftAddress;
    }
    return committed;
  };
  return Object.assign(accept, { isCurrent: current });
};

/** Retain the original conversation and command intent through asynchronous preparation. */
export const captureChatDraftSource = (
  isCurrent: () => boolean = () => true,
  isHistoryIdCurrent: (
    current: string | null,
    original: string | null,
  ) => boolean = (current, original) => current === original,
) => {
  const source = useStoreMessageOption.getState();
  const session = usePlaygroundSessionStore.getState();
  const current = () => {
    const state = useStoreMessageOption.getState();
    return (
      isCurrent() &&
      usePlaygroundSessionStore.getState().restoreRevision ===
        session.restoreRevision &&
      usePlaygroundSessionStore.getState().sourceSelectionRevision ===
        session.sourceSelectionRevision &&
      state.chatDraftRevision === source.chatDraftRevision &&
      state.chatDraftAddress === source.chatDraftAddress &&
      isHistoryIdCurrent(state.historyId, source.historyId) &&
      state.serverChatId === source.serverChatId
    );
  };
  return {
    isCurrent: current,
    isQuickPromptCurrent: () =>
      current() &&
      useStoreMessageOption.getState().quickPromptRevision ===
        source.quickPromptRevision,
  };
};

/** Only explicit interactive replacement asks; retained tab switches keep their source. */
export const hasTemporaryChatContent = (chat: {
  temporaryChat: boolean;
  historyId?: string | null;
  serverChatId?: string | null;
  history: readonly unknown[];
  messages: readonly unknown[];
}) =>
  Boolean(
    (chat.historyId === "temp" || (chat.temporaryChat && !chat.serverChatId)) &&
    (chat.history.length || chat.messages.length)
  );

export const confirmTemporaryChatDiscard = (
  chat: Parameters<
    typeof hasTemporaryChatContent
  >[0] = useStoreMessageOption.getState()
): boolean =>
  !hasTemporaryChatContent(chat) ||
  (typeof window !== "undefined" &&
    window.confirm(
      i18n.t("common:temporaryChatDiscardConfirmation", {
        defaultValue:
          "Discard this Temporary conversation?\n\nCancel keeps it on this device so you can copy messages before continuing. OK discards its messages."
      })
    ));
