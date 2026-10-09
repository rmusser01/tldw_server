import { beforeEach, describe, expect, it, vi } from 'vitest';

const account = vi.hoisted(() => ({ invalidators: [] as Array<(changed: boolean) => void> }));
vi.mock('@/services/chat-account-boundary', () => ({
  watchChatAccountChanges: (callback: (changed: boolean) => void) => {
    account.invalidators.push(callback);
    return () => {};
  },
}));
import { useStoreMessageOption } from '../option';
import {
  captureChatDraftLoad,
  captureChatDraftSource,
  chatDraftAddressForReceipt
} from "@/hooks/chat/chat-draft-transition"

import { usePlaygroundSessionStore } from '../playground-session';
import type { HistoryLoadReceipt } from '@/hooks/chat/useHistorySelection';

const native = (id = 'a', valid = true): HistoryLoadReceipt => ({
  owner: {
    kind: 'native',
    conversation_id: id,
    request_scope: {} as never,
    validate_lease: () => valid,
  },
  view: {
    owner_key: 'server-owner',
    conversation_id: id,
    view_session_id: 'view',
    selection_revision: 1,
    cursor: { kind: 'empty' },
    interpretation: { kind: 'parent_graph_v1' },
  },
});

describe('explicit conversation draft address transitions', () => {
  beforeEach(() => {
    useStoreMessageOption.setState({
      chatDraftAddress: null,
      chatDraftTransition: null,
      chatDraftRevision: 0,
      temporaryChat: false,
    });
    useStoreMessageOption.getState().startNewChatDraft();
  });

  it('binds a ready first-ID receipt to its current source', () => {
    const source = useStoreMessageOption.getState().chatDraftAddress;
    const accept = captureChatDraftLoad('bind');
    expect(accept(native())).toBe(true);
    expect(useStoreMessageOption.getState().chatDraftTransition).toMatchObject({
      kind: 'bind',
      sourceAddress: source,
    });
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({
      kind: 'native',
      id: 'a',
      ownerKey: 'server-owner',
    });
  });

  it.each(
    (["bind", "switch"] as const).flatMap((kind) =>
      (["native", "local"] as const).flatMap((ownerKind) =>
        (["New", "destination ABA"] as const).map((boundary) => ({
          kind,
          ownerKind,
          boundary
        }))
      )
    )
  )(
    "rejects $kind into $ownerKind after $boundary during synchronous address publication",
    ({ kind, ownerKind, boundary }) => {
      const receipt: HistoryLoadReceipt =
        ownerKind === "native"
          ? native()
          : {
              ...native(),
              owner: {
                kind: "local",
                conversation_id: "a",
                profile_id: "profile-one",
                owner_key: "local-owner",
                validate_lease: () => true
              },
              view: { ...native().view, owner_key: "local-owner" }
            }
      const accept = captureChatDraftLoad(kind)
      let replaced = false
      const stop = useStoreMessageOption.subscribe((state) => {
        const destination = state.chatDraftAddress
        if (replaced || destination?.kind !== ownerKind || destination.id !== "a")
          return
        replaced = true
        state.startNewChatDraft()
        if (boundary === "destination ABA")
          useStoreMessageOption.getState().setChatDraftAddress(destination)
      })
      try {
        expect(accept(receipt)).toBe(false)
        expect(replaced).toBe(true)
        expect(accept.isCurrent()).toBe(false)
        expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe(
          boundary === "New" ? "fresh" : ownerKind
        )
      } finally {
        stop()
      }
    }
  )

  it.each(["bind", "switch"] as const)(
    "keeps a current same-key %s without inventing another address revision",
    (kind) => {
      const receipt = native()
      const address = chatDraftAddressForReceipt(receipt)!
      expect(useStoreMessageOption.getState().setChatDraftAddress(address)).toBe(
        true
      )
      const revision = useStoreMessageOption.getState().chatDraftRevision
      const accept = captureChatDraftLoad(kind)
      expect(accept(receipt)).toBe(true)
      expect(accept.isCurrent()).toBe(true)
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(address)
      expect(useStoreMessageOption.getState().chatDraftRevision).toBe(revision)
    }
  )

  it.each(["bind", "switch"] as const)(
    "keeps a current %s after an unrelated synchronous store publication",
    (kind) => {
      const accept = captureChatDraftLoad(kind)
      let published = false
      const stop = useStoreMessageOption.subscribe((state) => {
        if (published || state.chatDraftAddress?.kind !== "native") return
        published = true
        state.setServerChatTitle("Unrelated current title")
      })
      try {
        expect(accept(native())).toBe(true)
        expect(accept.isCurrent()).toBe(true)
        expect(useStoreMessageOption.getState().serverChatTitle).toBe(
          "Unrelated current title"
        )
      } finally {
        stop()
      }
    }
  )

  it('rejects an obsolete receipt after New even before any ID exists', () => {
    const accept = captureChatDraftLoad('switch');
    useStoreMessageOption.getState().startNewChatDraft();
    const target = useStoreMessageOption.getState().chatDraftAddress;
    expect(accept(native())).toBe(false);
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(target);
  });

  it('rejects a revoked native lease', () => {
    const source = useStoreMessageOption.getState().chatDraftAddress;
    expect(captureChatDraftLoad('switch')(native('a', false))).toBe(false);
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source);
  });

  it('invalidates a pending receipt synchronously on account change', () => {
    const accept = captureChatDraftLoad('switch');
    account.invalidators.forEach((callback) => callback(true));
    expect(accept(native())).toBe(false);
    expect(useStoreMessageOption.getState().chatDraftAddress).toBeNull();
  });

  it('does not treat raw ID or mirror publication as draft selection', () => {
    const source = useStoreMessageOption.getState().chatDraftAddress;
    useStoreMessageOption.getState().setServerChatId('a');
    useStoreMessageOption.getState().setHistoryId('mirror', { preserveServerChatId: true });
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source);
  });
  it.each(['restoreRevision', 'sourceSelectionRevision'] as const)('retires source preparation when the session %s changes under the same chat IDs', revision => {
    const before = usePlaygroundSessionStore.getState();
    const source = captureChatDraftSource();
    expect(source.isCurrent()).toBe(true);
    usePlaygroundSessionStore.setState({ [revision]: before[revision] + 1 });
    expect(source.isCurrent()).toBe(false);
  });

});
