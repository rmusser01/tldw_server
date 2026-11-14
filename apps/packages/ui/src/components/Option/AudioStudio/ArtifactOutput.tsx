import React from 'react';
import { Button } from 'antd';
import {
  mintAudioStudioArtifactMediaTicket,
  type AudioStudioArtifact,
} from '@/services/audio-studio';
import { useAudioStudioSession } from './AudioStudioSession';

export const ArtifactOutput: React.FC<{ projectId: string; artifact: AudioStudioArtifact }> = ({
  projectId,
  artifact,
}) => {
  const session = useAudioStudioSession();
  const resourceKey = `${projectId}:${artifact.artifact_id}`;
  const resource = React.useRef(resourceKey);
  resource.current = resourceKey;
  const previewKey = React.useRef<string | null>(null);
  const current = React.useRef(0);
  const [url, setUrl] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  React.useEffect(
    () => () => {
      current.current++;
    },
    [artifact.artifact_id, projectId],
  );
  const open = async (download: boolean) => {
    const lease = ++current.current;
    const key = resourceKey;
    setBusy(true);
    setError(null);
    try {
      const ticket = await mintAudioStudioArtifactMediaTicket(
        projectId,
        artifact.artifact_id,
        download ? 'download' : 'playback',
        session?.options,
      );
      session?.options.assertCurrent?.();
      if (current.current !== lease || resource.current !== key) return;
      if (download) {
        const anchor = document.createElement('a');
        anchor.href = ticket.ticket_url;
        anchor.download = '';
        anchor.rel = 'noreferrer';
        anchor.referrerPolicy = 'no-referrer';
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
      } else {
        previewKey.current = key;
        setUrl(ticket.ticket_url);
      }
    } catch {
      if (current.current === lease && resource.current === key)
        setError('Audio could not open. Your saved artifact is retained; try opening it again.');
    } finally {
      if (current.current === lease && resource.current === key) setBusy(false);
    }
  };
  return (
    <div className="space-y-2 rounded border border-border p-2 text-xs">
      <p className="break-all">
        {artifact.artifact_type}: {artifact.artifact_id}
      </p>
      {artifact.mime_type?.startsWith('audio/') ? (
        <Button size="small" disabled={busy} onClick={() => void open(false)}>
          Load audio preview
        </Button>
      ) : null}
      <Button size="small" disabled={busy} onClick={() => void open(true)}>
        Download {artifact.artifact_type}
      </Button>
      {url && previewKey.current === resourceKey ? (
        <audio
          className="w-full"
          aria-label={`Audio ${artifact.artifact_id}`}
          ref={(element) => element?.setAttribute('referrerpolicy', 'no-referrer')}
          src={url}
          controls
          onError={() => setError('Playback failed. Load the preview again to renew access.')}
        />
      ) : null}
      {error ? <p role="alert">{error}</p> : null}
    </div>
  );
};
