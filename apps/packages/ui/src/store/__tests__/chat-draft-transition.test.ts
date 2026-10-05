import { beforeEach, describe, expect, it, vi } from 'vitest';

const account = vi.hoisted(() => ({ invalidators: [] as Array<(changed: boolean) => void> }));
vi.mock('@/services/chat-account-boundary', () => ({
  watchChatAccountChanges: (callback: (changed: boolean) => void) => {
    account.invalidators.push(callback);
    return () => {};
  },
}));
import { useStoreMessageOption } from '../option';
import { captureChatDraftLoad } from '@/hooks/chat/chat-draft-transition';
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
});
