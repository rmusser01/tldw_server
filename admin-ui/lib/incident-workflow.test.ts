import { describe, expect, it } from 'vitest';
import type { IncidentItem } from '@/types/incidents';
import {
  addIncidentActionItem,
  buildPostmortemTimelineMessage,
  ensureIncidentWorkflowState,
  upsertIncidentWorkflowState,
  updateIncidentActionItem,
} from './incident-workflow';

const makeIncident = (overrides: Partial<IncidentItem> = {}): IncidentItem => ({
  id: 'inc-1',
  version: 1,
  title: 'Queue latency spike',
  status: 'open',
  severity: 'high',
  summary: 'Elevated queue latency',
  tags: ['queue'],
  created_at: '2026-03-12T12:00:00Z',
  updated_at: '2026-03-12T12:00:00Z',
  resolved_at: null,
  timeline: [],
  assigned_to_user_id: null,
  assigned_to_label: null,
  root_cause: null,
  impact: null,
  action_items: [],
  ...overrides,
});

describe('incident workflow state helpers', () => {
  it('creates default incident workflow state when missing', () => {
    const state = ensureIncidentWorkflowState({}, makeIncident());
    expect(state.assignedTo).toBeUndefined();
    expect(state.rootCause).toBe('');
    expect(state.impact).toBe('');
    expect(state.actionItems).toEqual([]);
  });

  it('supports action item add and update operations', () => {
    let map = upsertIncidentWorkflowState({}, 'inc-1', {
      rootCause: 'Database lock contention',
      impact: 'Elevated latency for writes',
    });
    map = addIncidentActionItem(map, 'inc-1');
    const added = map['inc-1'].actionItems[0];
    map = updateIncidentActionItem(map, 'inc-1', added.id, {
      text: 'Add lock wait observability dashboard',
      done: true,
    });
    expect(map['inc-1'].actionItems[0].text).toBe('Add lock wait observability dashboard');
    expect(map['inc-1'].actionItems[0].done).toBe(true);
  });

  it('preserves a default runbook url in fallback workflow state', () => {
    const map = addIncidentActionItem({}, 'inc-1');

    expect(map['inc-1'].runbookUrl).toBe('');
    expect(map['inc-1'].actionItems).toHaveLength(1);
  });

  it('builds a post-mortem timeline summary message', () => {
    const message = buildPostmortemTimelineMessage({
      assignedTo: '2',
      rootCause: 'API pool exhausted',
      impact: 'Requests failed for 4 minutes',
      actionItems: [
        { id: 'a1', text: 'Raise pool limits', done: false },
        { id: 'a2', text: 'Add saturation alert', done: false },
      ],
    });
    expect(message).toContain('Post-mortem updated');
    expect(message).toContain('2 action items');
  });
});
