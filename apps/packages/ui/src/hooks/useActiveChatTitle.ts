import { useEffect, useReducer, useRef } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { getTitleById, updateHistory } from '@/db/dexie/helpers';
import { runChatPersistenceTransaction } from '@/db/dexie/chat-persistence-transaction';
import { useStoreMessageOption } from '@/store/option';
import { useCanonicalConnectionConfig } from '@/hooks/useCanonicalConnectionConfig';
import { connectionAuthoritiesMatch } from '@/services/chat-surface-scope';
import { loadServicePromptSnapshot } from '@/services/service-prompts';
import { tldwClient, type TldwConfig } from '@/services/tldw/TldwApiClient';

/** Both title surfaces follow the active conversation, never the legacy catalog. */
export function useActiveChatTitle() {
  const { historyId, serverChatId, serverChatTitle, serverChatMetaLoaded, temporaryChat } = useStoreMessageOption();
  const { config, authorityLoading } = useCanonicalConnectionConfig();
  const [, refresh] = useReducer((value: number) => value + 1, 0);
  const active = useRef<{
    historyId: string | null;
    serverChatId: string | null;
    temporaryChat: boolean;
    config: TldwConfig | null;
    controller: AbortController;
    saving: boolean;
    authorityRefreshPending: boolean;
    sawUnloadedServerMetadata: boolean;
  } | null>(null);
  const serverChatMetaLoadedRef = useRef(serverChatMetaLoaded);
  serverChatMetaLoadedRef.current = serverChatMetaLoaded;
  let owner = active.current;
  const authorityChanged = !authorityLoading && !!config && !!owner && (
    owner.controller.signal.aborted ||
    (!!owner.config && !connectionAuthoritiesMatch(config, owner.config))
  );
  if (
    !owner ||
    owner.historyId !== historyId ||
    owner.serverChatId !== serverChatId ||
    owner.temporaryChat !== temporaryChat ||
    authorityChanged
  ) {
    const previousOwner = owner;
    const selectionChanged = !!previousOwner && (
      previousOwner.historyId !== historyId ||
      previousOwner.serverChatId !== serverChatId ||
      previousOwner.temporaryChat !== temporaryChat
    );
    const authorityRefreshPending = !selectionChanged && (
      authorityChanged || previousOwner?.authorityRefreshPending === true
    );
    owner?.controller.abort();
    owner = {
      historyId,
      serverChatId,
      temporaryChat,
      config: authorityLoading ? null : config,
      controller: new AbortController(),
      saving: false,
      authorityRefreshPending,
      sawUnloadedServerMetadata: authorityRefreshPending && serverChatMetaLoaded === false,
    };
    active.current = owner;
  } else if (!authorityLoading && config) {
    if (!owner.config) owner.config = config;
  }
  if (owner.authorityRefreshPending && owner.serverChatId) {
    if (serverChatMetaLoaded === false) owner.sawUnloadedServerMetadata = true;
    else if (owner.sawUnloadedServerMetadata) owner.authorityRefreshPending = false;
  }
  const ready = !authorityLoading && !!config && !owner.controller.signal.aborted &&
    !owner.authorityRefreshPending && !temporaryChat;
  const serverChatTitleReady = serverChatMetaLoaded !== false;
  const readyRef = useRef(ready);
  readyRef.current = ready;
  const isCurrent = () => {
    const current = useStoreMessageOption.getState();
    return (
      active.current === owner &&
      readyRef.current &&
      !owner.controller.signal.aborted &&
      current.historyId === historyId &&
      current.serverChatId === serverChatId &&
      !current.temporaryChat
    );
  };

  useEffect(() => {
    // React Strict Mode remounts effects before rendering the surviving instance.
    if (!active.current) refresh();
    const invalidate = () => {
      const current = active.current;
      if (current) {
        current.authorityRefreshPending = true;
        current.sawUnloadedServerMetadata = serverChatMetaLoadedRef.current === false;
        current.controller.abort();
      }
      refresh();
    };
    const configChanged = (event: Event) => {
      if ((event as CustomEvent).detail?.authorityChanged === true) invalidate();
    };
    window.addEventListener('tldw:auth-principal-changed', invalidate);
    window.addEventListener('tldw:config-updated', configChanged);
    return () => {
      active.current?.controller.abort();
      active.current = null;
      window.removeEventListener('tldw:auth-principal-changed', invalidate);
      window.removeEventListener('tldw:config-updated', configChanged);
    };
  }, []);

  const local = useLiveQuery(async () => {
    if (!ready || serverChatId || !historyId || historyId === 'temp') return null;
    const title = await getTitleById(historyId).catch(() => '');
    return isCurrent() ? { owner, title: title || '' } : null;
  }, [owner, ready]);
  const title = !ready
    ? ''
    : serverChatId
      ? serverChatTitleReady ? serverChatTitle || '' : ''
      : local?.owner === owner
        ? local.title
        : '';

  const renameTitle = async (value: string) => {
    if (!isCurrent() || owner.saving || !historyId || historyId === 'temp' || (serverChatId && !serverChatTitleReady)) return;
    owner.saving = true;
    refresh();
    const nextTitle = value.trim() || 'Untitled';
    let snapshot: Awaited<ReturnType<typeof loadServicePromptSnapshot>> | undefined;
    try {
      if (serverChatId) {
        snapshot = await loadServicePromptSnapshot([], { signal: owner.controller.signal });
        // Scope resolution may await authentication. Do not move old content to
        // whichever account became active while that work was pending.
        const currentConfig = await tldwClient.getConfig();
        if (
          !isCurrent() ||
          !connectionAuthoritiesMatch(currentConfig, owner.config) ||
          snapshot.scopeSignal.aborted
        )
          return;
        const saved = await tldwClient.updateChat(
          serverChatId,
          { title: nextTitle },
          {
            expectedVersion: useStoreMessageOption.getState().serverChatVersion ?? undefined,
            requestScope: snapshot.requestScope,
            signal: snapshot.scopeSignal,
          }
        );
        if (!isCurrent() || snapshot.scopeSignal.aborted) return;
        const state = useStoreMessageOption.getState();
        state.setServerChatTitle(saved.title || nextTitle);
        if (typeof saved.version === 'number') state.setServerChatVersion(saved.version);
      }
      if (!isCurrent()) return;
      await runChatPersistenceTransaction(snapshot?.scopeSignal ?? owner.controller.signal, () =>
        updateHistory(historyId, nextTitle)
      );
    } finally {
      snapshot?.release();
      owner.saving = false;
      if (active.current === owner) refresh();
    }
  };
  return { title, renameTitle, owner, ready };
}
