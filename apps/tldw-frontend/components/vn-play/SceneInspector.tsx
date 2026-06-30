import React, { FormEvent, useState } from 'react';
import { BookmarkPlus, RotateCcw } from 'lucide-react';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import type { VNPlayBranch, VNPlayCheckpoint, VNPlaySceneState, VNPlaySession } from '@web/types/vn-play';

export interface SceneInspectorProps {
  branches?: VNPlayBranch[];
  checkpoints?: VNPlayCheckpoint[];
  isCreatingCheckpoint?: boolean;
  restoringCheckpointId?: number | null;
  sceneState: VNPlaySceneState;
  session: VNPlaySession;
  onCreateCheckpoint?: (label: string) => void | Promise<void>;
  onRestoreCheckpoint?: (checkpointId: number) => void | Promise<void>;
}

export default function SceneInspector({
  branches = [],
  checkpoints = [],
  isCreatingCheckpoint = false,
  restoringCheckpointId = null,
  sceneState,
  session,
  onCreateCheckpoint,
  onRestoreCheckpoint,
}: SceneInspectorProps) {
  const [checkpointLabel, setCheckpointLabel] = useState('');
  const sceneVersion = sceneState.scene_version ?? session.scene_version ?? 0;
  const warningCount = sceneState.warnings?.length ?? 0;

  const submitCheckpoint = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const label = checkpointLabel.trim();
    if (!label || !onCreateCheckpoint) return;
    void Promise.resolve(onCreateCheckpoint(label))
      .then(() => setCheckpointLabel(''))
      .catch(() => undefined);
  };

  return (
    <aside className="rounded-md border border-border bg-surface p-4">
      <h2 className="mb-4 text-lg font-semibold">Runtime inspector</h2>
      <dl className="grid gap-3 text-sm">
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Mode</dt>
          <dd className="font-medium">{session.mode === 'story' ? 'Story/CYOA' : 'Freeform'}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Character</dt>
          <dd className="font-medium">Character {session.primary_character_id}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Asset pack</dt>
          <dd className="font-medium">Pack {session.vn_asset_pack_id}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Scene version</dt>
          <dd className="font-medium">{sceneVersion}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Warnings</dt>
          <dd className="font-medium">{warningCount}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Branches</dt>
          <dd className="font-medium">{branches.length}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Checkpoints</dt>
          <dd className="font-medium">{checkpoints.length}</dd>
        </div>
      </dl>

      <section className="mt-5 border-t border-border pt-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-normal text-text-muted">
          Checkpoints
        </h3>
        {onCreateCheckpoint && (
          <form className="grid gap-2" onSubmit={submitCheckpoint}>
            <Input
              label="Checkpoint label"
              value={checkpointLabel}
              onChange={(event) => setCheckpointLabel(event.target.value)}
            />
            <Button
              className="gap-2"
              disabled={!checkpointLabel.trim()}
              loading={isCreatingCheckpoint}
              type="submit"
            >
              <BookmarkPlus aria-hidden className="h-4 w-4" />
              Create checkpoint
            </Button>
          </form>
        )}
        {checkpoints.length > 0 ? (
          <ul className="mt-3 grid gap-2">
            {checkpoints.map((checkpoint) => (
              <li key={checkpoint.id} className="rounded-md border border-border bg-bg p-2 text-sm">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-medium">{checkpoint.label}</p>
                    <p className="text-xs text-text-muted">Scene {checkpoint.scene_version}</p>
                  </div>
                  {onRestoreCheckpoint && (
                    <Button
                      aria-label={`Restore checkpoint: ${checkpoint.label}`}
                      className="gap-1"
                      loading={restoringCheckpointId === checkpoint.id}
                      onClick={() => void onRestoreCheckpoint(checkpoint.id)}
                      size="xs"
                      type="button"
                      variant="secondary"
                    >
                      <RotateCcw aria-hidden className="h-3.5 w-3.5" />
                      Restore
                    </Button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-text-muted">No checkpoints.</p>
        )}
      </section>

      <section className="mt-5 border-t border-border pt-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-normal text-text-muted">
          Branches
        </h3>
        {branches.length > 0 ? (
          <ul className="grid gap-2">
            {branches.map((branch) => (
              <li key={branch.id} className="rounded-md border border-border bg-bg p-2 text-sm">
                <p className="font-medium">{branch.branch_label || `Branch ${branch.id}`}</p>
                <p className="text-xs text-text-muted">{branch.status || 'unknown'}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-text-muted">No branches.</p>
        )}
      </section>
    </aside>
  );
}
