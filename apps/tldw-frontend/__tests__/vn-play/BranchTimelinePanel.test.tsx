import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import BranchTimelinePanel from '@web/components/vn-play/BranchTimelinePanel';
import type { VNPlayBranchNavigationResponse } from '@web/types/vn-play';

function navigation(overrides: Partial<VNPlayBranchNavigationResponse> = {}): VNPlayBranchNavigationResponse {
  return {
    session_id: 1,
    mode: 'story',
    scene_version: 6,
    last_event_id: 41,
    active_branch_node_id: 12,
    active_path: [
      {
        branch_id: 8,
        branch_label: 'Open the archive door',
        choice_id: 'open-door',
        choice_text: 'Open the archive door',
        depth: 1,
      },
      {
        branch_id: 12,
        branch_label: 'Step inside',
        choice_id: 'step-inside',
        choice_text: 'Step inside',
        depth: 2,
      },
    ],
    branches: [
      {
        branch_id: 12,
        parent_branch_id: 8,
        parent_event_id: 32,
        choice_selected_event_id: 33,
        branch_label: 'Step inside',
        choice_id: 'step-inside',
        choice_text: 'Step inside',
        branch_path: [],
        depth: 2,
        status: 'active',
        is_active: true,
        is_on_active_path: true,
        event_range: {
          start_event_id: 33,
          start_sequence_number: 18,
          latest_event_id: 41,
          latest_sequence_number: 26,
        },
        subtree_event_range: {
          start_event_id: 33,
          start_sequence_number: 18,
          latest_event_id: 41,
          latest_sequence_number: 26,
        },
        restore: {
          supported: true,
          default_target: 'branch_latest',
          target_names: ['branch_latest', 'choice_point'],
          targets: {
            branch_latest: { event_id: 41, sequence_number: 26 },
            choice_point: { event_id: 32 },
          },
        },
        warnings: [
          {
            code: 'parent_branch_unresolved',
            severity: 'warning',
            recoverable: true,
            message: 'Parent branch could not be resolved from branch path prefix.',
            branch_id: 12,
          },
        ],
      },
      {
        branch_id: 14,
        branch_label: 'Read the plaque',
        choice_id: 'read-plaque',
        choice_text: 'Read the plaque',
        branch_path: [],
        depth: 2,
        status: 'active',
        is_active: false,
        is_on_active_path: false,
        event_range: {},
        subtree_event_range: {},
        restore: {
          supported: true,
          default_target: 'branch_latest',
          target_names: ['branch_latest'],
          targets: {
            branch_latest: { event_id: 44, sequence_number: 29 },
          },
        },
        warnings: [],
      },
    ],
    warnings: [],
    ...overrides,
  };
}

describe('BranchTimelinePanel', () => {
  it('renders an empty state when no branch navigation is available', () => {
    render(<BranchTimelinePanel navigation={null} />);

    expect(screen.getByText('Branch timeline')).toBeInTheDocument();
    expect(screen.getByText(/No Story branches are available yet/i)).toBeInTheDocument();
  });

  it('renders active path and backend-provided branch restore targets', () => {
    render(<BranchTimelinePanel navigation={navigation()} onRestoreBranch={vi.fn()} />);

    expect(screen.getByText('Open the archive door')).toBeInTheDocument();
    expect(screen.getAllByText('Step inside')).toHaveLength(2);
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(screen.getByText('On path')).toBeInTheDocument();
    expect(screen.getByText(/Events 18-26/i)).toBeInTheDocument();
    expect(screen.getByText('Parent branch could not be resolved from branch path prefix.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /resume branch: step inside/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /return to choice: step inside/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /resume branch: read the plaque/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /return to choice: read the plaque/i })).not.toBeInTheDocument();
  });

  it('submits restore target selected by the user', async () => {
    const user = userEvent.setup();
    const onRestoreBranch = vi.fn();

    render(<BranchTimelinePanel navigation={navigation()} onRestoreBranch={onRestoreBranch} />);

    await user.click(screen.getByRole('button', { name: /return to choice: step inside/i }));

    expect(onRestoreBranch).toHaveBeenCalledWith(12, 'choice_point');
  });
});
