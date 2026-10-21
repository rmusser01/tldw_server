import React from 'react';
import { useChatDraftOwner } from '@/hooks/useChatDraftOwner';
import { resolveServicePromptScope } from '@/services/service-prompts';
import type { AudioStudioRequestOptions } from '@/services/audio-studio';
import { useAudioStudioStore } from '@/store/audio-studio';

import { restoreAudioStudioIntents } from './audioStudioAdmission';

type Session = { ownerKey: string; options: AudioStudioRequestOptions; readOnlyReason?: string };
const AudioStudioSessionContext = React.createContext<Session | null>(null);
export const useAudioStudioSession = () => React.useContext(AudioStudioSessionContext);

export const AudioStudioSession: React.FC<React.PropsWithChildren> = ({ children }) => {
  const controller = React.useRef<AbortController | null>(null);
  const clear = React.useCallback(() => {
    controller.current?.abort();
    useAudioStudioStore.getState().resetAudioStudio();
  }, []);
  const { ownerKey, isCurrent } = useChatDraftOwner(clear);
  const recovery = React.useRef<Session | null>(null);
  const [error, setError] = React.useState(false);
  const [session, setSession] = React.useState<Session | null>(null);
  React.useEffect(() => {
    if (!ownerKey) return;
    setError(false);
    recovery.current = null;
    useAudioStudioStore.getState().resetAudioStudio();
    const lease = new AbortController();
    controller.current = lease;
    const assertCurrent = () => {
      if (lease.signal.aborted || !isCurrent())
        throw new Error('Audio Studio account changed. Reopen the project.');
    };
    void resolveServicePromptScope({ signal: lease.signal })
      .then((requestScope) => {
        assertCurrent();
        const verified: Session = {
          ownerKey,
          options: { requestScope, signal: lease.signal, assertCurrent },
        };
        try {
          restoreAudioStudioIntents(ownerKey);
        } catch {
          const reason =
            'Pending job identity could not be restored. New work is disabled; review the server jobs and reload after browser storage is available.';
          recovery.current = {
            ...verified,
            readOnlyReason: reason,
            options: {
              ...verified.options,
              assertCanWrite: () => {
                throw new Error(reason);
              },
            },
          };
          throw new Error(reason);
        }
        setSession(verified);
      })
      .catch(() => {
        if (!lease.signal.aborted) {
          setSession(null);
          setError(true);
        }
      });
    return () => {
      lease.abort();
    };
  }, [ownerKey, isCurrent]);
  if (error && !session)
    return (
      <div role="alert">
        <p>
          Audio Studio account or pending job identity could not be restored. No new work was sent.
        </p>
        <button type="button" onClick={() => window.location.reload()}>
          Reload Audio Studio
        </button>
        {recovery.current ? (
          <button
            type="button"
            onClick={() => {
              recovery.current?.options.assertCurrent?.();
              setSession(recovery.current);
            }}
          >
            Review server jobs without new work
          </button>
        ) : null}
      </div>
    );
  if (!session || session.ownerKey !== ownerKey || !isCurrent())
    return <p role="status">Checking Audio Studio account…</p>;
  return (
    <AudioStudioSessionContext.Provider value={session}>
      {session.readOnlyReason ? <p role="alert">{session.readOnlyReason}</p> : null}
      {children}
    </AudioStudioSessionContext.Provider>
  );
};
