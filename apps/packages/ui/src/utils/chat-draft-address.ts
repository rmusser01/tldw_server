import type { ChatScope } from '@/types/chat-scope';

export type ChatDraftAddress =
  | { kind: 'fresh'; id: string; scope: ChatScope; temporary: boolean }
  | { kind: 'local'; id: string; profileId: string }
  | { kind: 'native'; id: string; ownerKey: string; scope: ChatScope };

export type ChatDraftTransition = {
  kind: 'switch' | 'bind';
  sourceAddress: ChatDraftAddress | null;
  destinationKey: string;
  revision: number;
};

/** An address locates private draft text; it never grants history authority. */
export const chatDraftAddressKey = (address: ChatDraftAddress): string =>
  JSON.stringify(
    address.kind === 'local'
      ? [address.kind, address.profileId, address.id]
      : [
          address.kind,
          address.scope.type,
          address.scope.type === 'workspace' ? address.scope.workspaceId : null,
          address.kind === 'native' ? address.ownerKey : null,
          address.id,
        ]
  );

export const conversationDraftKey = (surfaceKey: string, address: ChatDraftAddress): string =>
  `${surfaceKey}:conversation:${encodeURIComponent(chatDraftAddressKey(address))}`;

export const freshChatDraftAddress = (temporary: boolean): ChatDraftAddress => ({
  kind: 'fresh',
  id: crypto.randomUUID(),
  scope: { type: 'global' },
  temporary,
});

export const isChatDraftAddress = (value: unknown): value is ChatDraftAddress => {
  if (!value || typeof value !== 'object') return false;
  const address = value as Record<string, unknown>;
  const text = (v: unknown) => typeof v === 'string' && v.length > 0 && v.length <= 1024;
  if (!text(address.id)) return false;
  if (address.kind === 'local') return text(address.profileId);
  const scope = address.scope as ChatScope | undefined;
  if (
    !scope ||
    (scope.type !== 'global' && (scope.type !== 'workspace' || !text(scope.workspaceId)))
  )
    return false;
  return address.kind === 'fresh'
    ? typeof address.temporary === 'boolean'
    : address.kind === 'native' && text(address.ownerKey);
};
