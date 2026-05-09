import React from 'react';
import type { VNPlayBranch, VNPlayCheckpoint, VNPlaySceneState, VNPlaySession } from '@web/types/vn-play';

export interface SceneInspectorProps {
  branches?: VNPlayBranch[];
  checkpoints?: VNPlayCheckpoint[];
  sceneState: VNPlaySceneState;
  session: VNPlaySession;
}

export default function SceneInspector({
  branches = [],
  checkpoints = [],
  sceneState,
  session,
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
    </aside>
  );
}
