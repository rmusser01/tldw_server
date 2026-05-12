import React from 'react';
import { GitBranch, RotateCcw } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import type {
  VNPlayBranchNavigationNode,
  VNPlayBranchNavigationResponse,
  VNPlayBranchRestoreTarget,
} from '@web/types/vn-play';

export interface BranchTimelinePanelProps {
  isLoading?: boolean;
  navigation: VNPlayBranchNavigationResponse | null;
  onRestoreBranch?: (branchId: number, target: VNPlayBranchRestoreTarget) => void | Promise<void>;
  restoreTarget?: VNPlayBranchRestoreTarget | null;
  restoringBranchId?: number | null;
}

function branchLabel(branch: VNPlayBranchNavigationNode): string {
  return branch.branch_label || branch.choice_text || `Branch ${branch.branch_id}`;
}

function targetLabel(target: VNPlayBranchRestoreTarget): string {
  return target === 'choice_point' ? 'Return to choice' : 'Resume branch';
}

function eventRangeLabel(branch: VNPlayBranchNavigationNode): string | null {
  const start = branch.event_range.start_sequence_number;
  const latest = branch.event_range.latest_sequence_number;
  if (typeof start === 'number' && typeof latest === 'number') {
    return `Events ${start}-${latest}`;
  }
  if (typeof latest === 'number') {
    return `Latest event ${latest}`;
  }
  return null;
}

export default function BranchTimelinePanel({
  isLoading = false,
  navigation,
  onRestoreBranch,
  restoreTarget = null,
  restoringBranchId = null,
}: BranchTimelinePanelProps) {
  const branches = navigation?.branches ?? [];
  const activePath = navigation?.active_path ?? [];

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <GitBranch aria-hidden className="h-4 w-4 text-text-muted" />
          <h2 className="text-lg font-semibold">Branch timeline</h2>
        </div>
        {navigation && <Badge variant="neutral">Scene {navigation.scene_version}</Badge>}
      </div>

      {isLoading && <p className="text-sm text-text-muted">Loading branch timeline...</p>}

      {!isLoading && branches.length === 0 && (
        <p className="text-sm text-text-muted">
          No Story branches are available yet. Branches appear after Story choices are selected.
        </p>
      )}

      {!isLoading && activePath.length > 0 && (
        <div className="mb-4 rounded-md border border-border bg-bg px-3 py-2">
          <p className="mb-2 text-xs uppercase tracking-normal text-text-muted">Active path</p>
          <ol className="flex flex-wrap gap-2 text-sm">
            {activePath.map((step) => (
              <li key={`${step.branch_id}-${step.depth}`} className="flex items-center gap-2">
                <span className="rounded-sm bg-surface px-2 py-1">
                  {step.choice_text || step.branch_label || `Branch ${step.branch_id}`}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {navigation?.warnings?.length ? (
        <ul className="mb-4 grid gap-2">
          {navigation.warnings.map((warning) => (
            <li key={`${warning.code}-${warning.branch_id ?? 'global'}`} className="text-sm text-warn">
              {warning.message || warning.code}
            </li>
          ))}
        </ul>
      ) : null}

      {branches.length > 0 && (
        <ul className="grid gap-3">
          {branches.map((branch) => {
            const label = branchLabel(branch);
            const rangeLabel = eventRangeLabel(branch);
            const restoreTargets = branch.restore.supported ? branch.restore.target_names : [];

            return (
              <li key={branch.branch_id} className="rounded-md border border-border bg-bg p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="font-medium">{label}</p>
                      {branch.is_active && <Badge variant="success">Active</Badge>}
                      {branch.is_on_active_path && <Badge variant="neutral">On path</Badge>}
                    </div>
                    {branch.choice_text && branch.choice_text !== label && (
                      <p className="mt-1 text-sm text-text-muted">{branch.choice_text}</p>
                    )}
                    <p className="mt-1 text-xs text-text-muted">
                      Depth {branch.depth}
                      {rangeLabel ? ` · ${rangeLabel}` : ''}
                    </p>
                    {branch.warnings.length > 0 && (
                      <ul className="mt-2 grid gap-1">
                        {branch.warnings.map((warning) => (
                          <li key={`${branch.branch_id}-${warning.code}`} className="text-xs text-warn">
                            {warning.message || warning.code}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                  {restoreTargets.length > 0 && onRestoreBranch && (
                    <div className="flex flex-wrap gap-2">
                      {restoreTargets.map((target) => (
                        <Button
                          key={target}
                          aria-label={`${targetLabel(target)}: ${label}`}
                          className="gap-1"
                          loading={restoringBranchId === branch.branch_id && restoreTarget === target}
                          onClick={() => void onRestoreBranch(branch.branch_id, target)}
                          size="xs"
                          type="button"
                          variant={target === branch.restore.default_target ? 'secondary' : 'ghost'}
                        >
                          <RotateCcw aria-hidden className="h-3.5 w-3.5" />
                          {targetLabel(target)}
                        </Button>
                      ))}
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
