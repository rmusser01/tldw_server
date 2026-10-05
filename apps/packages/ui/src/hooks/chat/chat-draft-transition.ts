import { useStoreMessageOption } from '@/store/option';
import type { HistoryLoadReceipt } from './useHistorySelection';
import {
  isChatDraftAddress,
  type ChatDraftAddress,
  type ChatDraftTransition,
} from '@/utils/chat-draft-address';

export const chatDraftAddressForReceipt = ({
  owner,
  view,
}: HistoryLoadReceipt): ChatDraftAddress | null => {
  if (
    owner.kind === 'unavailable' ||
    owner.validate_lease?.() === false ||
    view.conversation_id !== owner.conversation_id ||
    !view.owner_key ||
    (owner.owner_key && view.owner_key !== owner.owner_key)
  )
    return null;
  const address: ChatDraftAddress =
    owner.kind === 'local'
      ? { kind: 'local', id: owner.conversation_id, profileId: owner.profile_id }
      : {
          kind: 'native',
          id: owner.conversation_id,
          ownerKey: view.owner_key,
          scope: owner.scope ?? { type: 'global' },
        };
  return isChatDraftAddress(address) ? address : null;
};

/** Capture the source before loading; a qualified H1 result locates the private draft. */
export const captureChatDraftLoad = (
  kind: ChatDraftTransition['kind'],
  isCurrent: () => boolean = () => true
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
    if (kind === 'bind' && !sourceAddress) return false;
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
