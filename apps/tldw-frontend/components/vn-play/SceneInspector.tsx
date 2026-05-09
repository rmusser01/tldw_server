import React from 'react';
import { RotateCcw, Save } from 'lucide-react';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import type { VNPlayBranch, VNPlayCheckpoint, VNPlaySceneState, VNPlaySession } from '@web/types/vn-play';

export interface SceneInspectorProps {
  branches?: VNPlayBranch[];
  checkpointLabel?: string;
  checkpoints?: VNPlayCheckpoint[];
  isCreatingCheckpoint?: boolean;
  restoringCheckpointId?: number | null;
  sceneState: VNPlaySceneState;
  session: VNPlaySession;
  onCheckpointLabelChange?: (label: string) => void;
  onCreateCheckpoint?: () => void;
  onRestoreCheckpoint?: (checkpointId: number) => void;
}

export default function SceneInspector({
  branches = [],
  checkpointLabel = '',
  checkpoints = [],
  isCreatingCheckpoint = false,
  restoringCheckpointId = null,
  sceneState,
  session,
  onCheckpointLabelChange,
  onCreateCheckpoint,
  onRestoreCheckpoint,
}: SceneInspectorProps) {
  const sceneVersion = sceneState.scene_version ?? session.scene_version ?? 0;
  const warningCount = sceneState.warnings?.length ?? 0;

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
          <dd className="font-medium" data-testid="vn-play-scene-version">{sceneVersion}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Warnings</dt>
          <dd className="font-medium">{warningCount}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Branches</dt>
          <dd className="font-medium" data-testid="vn-play-branch-count">{branches.length}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-normal text-text-muted">Checkpoints</dt>
          <dd className="font-medium" data-testid="vn-play-checkpoint-count">{checkpoints.length}</dd>
        </div>
      </dl>
      <section className="mt-5 border-t border-border pt-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-normal text-text-muted">Branches</h3>
        {branches.length > 0 ? (
          <ul className="grid gap-2 text-sm" aria-label="VN play branches">
            {branches.map((branch) => (
              <li key={branch.id} className="flex items-center justify-between gap-2">
                <span className="font-medium">{branch.branch_label || `Branch ${branch.id}`}</span>
                {branch.status && <span className="text-xs text-text-muted">{branch.status}</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-text-muted">No branches.</p>
        )}
      </section>
      <section className="mt-5 border-t border-border pt-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-normal text-text-muted">Checkpoints</h3>
        <form
          className="grid gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            onCreateCheckpoint?.();
          }}
        >
          <Input
            label="Checkpoint label"
            value={checkpointLabel}
            onChange={(event) => onCheckpointLabelChange?.(event.target.value)}
          />
          <Button
            className="gap-2"
            disabled={!onCreateCheckpoint}
            loading={isCreatingCheckpoint}
            type="submit"
            variant="secondary"
          >
            <Save aria-hidden className="h-4 w-4" />
            Create checkpoint
          </Button>
        </form>
        {checkpoints.length > 0 ? (
          <ul className="mt-4 grid gap-3 text-sm" aria-label="VN play checkpoints">
            {checkpoints.map((checkpoint) => (
              <li key={checkpoint.id} className="grid gap-2 border-t border-border pt-3 first:border-t-0 first:pt-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium">{checkpoint.label}</span>
                  <span className="text-xs text-text-muted">Scene {checkpoint.scene_version}</span>
                </div>
                <Button
                  className="gap-2 justify-self-start"
                  disabled={!onRestoreCheckpoint}
                  loading={restoringCheckpointId === checkpoint.id}
                  onClick={() => onRestoreCheckpoint?.(checkpoint.id)}
                  type="button"
                  variant="secondary"
                >
                  <RotateCcw aria-hidden className="h-4 w-4" />
                  Restore checkpoint {checkpoint.label}
                </Button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-text-muted">No checkpoints.</p>
        )}
      </section>
    </aside>
  );
}
