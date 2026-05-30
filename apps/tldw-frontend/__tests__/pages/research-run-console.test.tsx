import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { renderWithProviders } from '@web/__tests__/testUtils/renderWithProviders';

const mocks = vi.hoisted(() => ({
  listResearchRuns: vi.fn(),
  createResearchRun: vi.fn(),
  getResearchRun: vi.fn(),
  pauseResearchRun: vi.fn(),
  resumeResearchRun: vi.fn(),
  cancelResearchRun: vi.fn(),
  approveResearchCheckpoint: vi.fn(),
  getResearchArtifact: vi.fn(),
  getResearchBundle: vi.fn(),
  subscribeResearchRunEvents: vi.fn(),
}));

vi.mock('@web/lib/api/researchRuns', () => ({
  listResearchRuns: (...args: unknown[]) => mocks.listResearchRuns(...args),
  createResearchRun: (...args: unknown[]) => mocks.createResearchRun(...args),
  getResearchRun: (...args: unknown[]) => mocks.getResearchRun(...args),
  pauseResearchRun: (...args: unknown[]) => mocks.pauseResearchRun(...args),
  resumeResearchRun: (...args: unknown[]) => mocks.resumeResearchRun(...args),
  cancelResearchRun: (...args: unknown[]) => mocks.cancelResearchRun(...args),
  approveResearchCheckpoint: (...args: unknown[]) => mocks.approveResearchCheckpoint(...args),
  getResearchArtifact: (...args: unknown[]) => mocks.getResearchArtifact(...args),
  getResearchBundle: (...args: unknown[]) => mocks.getResearchBundle(...args),
  subscribeResearchRunEvents: (...args: unknown[]) => mocks.subscribeResearchRunEvents(...args),
}));

import ResearchRunsPage from '@web/pages/research';

function makeRun(overrides: Record<string, unknown> = {}) {
  return {
    id: 'rs_1',
    query: 'Investigate local evidence',
    status: 'running',
    phase: 'collecting',
    control_state: 'running',
    progress_percent: 35,
    progress_message: 'Planning research',
    active_job_id: '11',
    latest_checkpoint_id: 'chk_1',
    created_at: '2026-03-07T10:00:00Z',
    updated_at: '2026-03-07T10:05:00Z',
    completed_at: null,
    ...overrides,
  };
}

function makeSnapshot(overrides: Record<string, unknown> = {}) {
  return {
    run: makeRun(),
    latest_event_id: 5,
    checkpoint: {
      checkpoint_id: 'chk_1',
      checkpoint_type: 'plan_review',
      status: 'pending',
      proposed_payload: {
        query: 'Investigate local evidence',
        focus_areas: ['background', 'counterevidence'],
        constraints: ['Use cited sources'],
        open_questions: ['What changed most recently?'],
        stop_criteria: {
          min_cited_sections: 1,
          min_sources: 2,
        },
      },
      resolution: null,
    },
    artifacts: [
      {
        artifact_name: 'plan.json',
        artifact_version: 1,
        content_type: 'application/json',
        phase: 'drafting_plan',
        job_id: '11',
      },
    ],
    ...overrides,
  };
}

function makeTrustArtifacts(overrides: Record<string, unknown> = {}) {
  return [
    {
      artifact_name: 'verification_summary.json',
      artifact_version: 1,
      content_type: 'application/json',
      phase: 'synthesizing',
      job_id: '11',
    },
    {
      artifact_name: 'unsupported_claims.json',
      artifact_version: 1,
      content_type: 'application/json',
      phase: 'synthesizing',
      job_id: '11',
    },
    {
      artifact_name: 'contradictions.json',
      artifact_version: 1,
      content_type: 'application/json',
      phase: 'synthesizing',
      job_id: '11',
    },
    {
      artifact_name: 'source_trust.json',
      artifact_version: 1,
      content_type: 'application/json',
      phase: 'synthesizing',
      job_id: '11',
    },
  ].map((artifact) => ({ ...artifact, ...overrides }));
}

function makeTrustBundle(overrides: Record<string, unknown> = {}) {
  return {
    concise_answer: 'Final synthesized answer',
    report: '# Final report',
    verification_summary: {
      supported_claim_count: 2,
      unsupported_claim_count: 1,
      contradiction_count: 1,
      warnings: ['Needs corroboration'],
    },
    unsupported_claims: [
      {
        claim_id: 'clm_1',
        text: 'Unverified assertion',
        focus_area: 'background',
        reason: 'no_supporting_notes',
      },
    ],
    contradictions: [
      {
        note_id: 'note_1',
        text: 'Evidence contradicted by later note.',
        focus_area: 'counterevidence',
        source_id: 'src_2',
      },
    ],
    source_trust: [
      {
        source_id: 'src_1',
        title: 'Primary memo',
        provider: 'local_corpus',
        trust_tier: 'internal',
        snapshot_policy: 'full_artifact',
        trust_labels: ['local_corpus', 'internal'],
      },
    ],
    ...overrides,
  };
}

const TRUST_ARTIFACT_CONTENTS: Record<string, unknown> = {
  'verification_summary.json': {
    supported_claim_count: 2,
    unsupported_claim_count: 1,
    contradiction_count: 1,
    warnings: ['Needs corroboration'],
  },
  'unsupported_claims.json': {
    claims: [
      {
        claim_id: 'clm_1',
        text: 'Unverified assertion',
        focus_area: 'background',
        reason: 'no_supporting_notes',
      },
    ],
  },
  'contradictions.json': {
    contradictions: [
      {
        note_id: 'note_1',
        text: 'Evidence contradicted by later note.',
        focus_area: 'counterevidence',
        source_id: 'src_2',
      },
    ],
  },
  'source_trust.json': {
    sources: [
      {
        source_id: 'src_1',
        title: 'Primary memo',
        provider: 'local_corpus',
        trust_tier: 'internal',
        snapshot_policy: 'full_artifact',
        trust_labels: ['local_corpus', 'internal'],
      },
    ],
  },
};

describe('ResearchRunsPage', () => {
  let currentSnapshot: ReturnType<typeof makeSnapshot>;
  let streamHandlers: Map<string, (event: { event: string; id?: number; payload?: unknown }) => void>;

  function emitStreamEvent(sessionId: string, event: { event: string; id?: number; payload?: unknown }) {
    streamHandlers.get(sessionId)?.(event);
  }

  beforeEach(() => {
    vi.clearAllMocks();
    window.history.replaceState({}, '', '/research');
    currentSnapshot = makeSnapshot();
    streamHandlers = new Map();

    mocks.listResearchRuns.mockResolvedValue([
      makeRun(),
      makeRun({
        id: 'rs_0',
        query: 'Older run',
        latest_checkpoint_id: null,
        created_at: '2026-03-07T09:00:00Z',
        updated_at: '2026-03-07T09:05:00Z',
      }),
    ]);
    mocks.getResearchRun.mockImplementation(async (sessionId: string) => {
      if (sessionId === 'rs_new') {
        return makeRun({
          id: 'rs_new',
          query: 'Newly created run',
          latest_checkpoint_id: null,
        });
      }
      return makeRun({ id: sessionId });
    });
    mocks.createResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_new',
        query: 'Newly created run',
        latest_checkpoint_id: null,
      })
    );
    mocks.pauseResearchRun.mockResolvedValue(makeRun({ control_state: 'pause_requested' }));
    mocks.resumeResearchRun.mockResolvedValue(makeRun({ control_state: 'running' }));
    mocks.cancelResearchRun.mockResolvedValue(makeRun({ control_state: 'cancel_requested' }));
    mocks.approveResearchCheckpoint.mockResolvedValue(
      makeRun({
        phase: 'collecting',
        latest_checkpoint_id: null,
      })
    );
    mocks.getResearchArtifact.mockImplementation(async (_sessionId: string, artifactName: string) => ({
      artifact_name: artifactName,
      content_type: 'application/json',
      content:
        TRUST_ARTIFACT_CONTENTS[artifactName] ?? {
          artifact_summary: 'Loaded artifact body',
        },
    }));
    mocks.getResearchBundle.mockResolvedValue(makeTrustBundle());
    mocks.subscribeResearchRunEvents.mockImplementation((options: {
      sessionId: string;
      onEvent: (event: { event: string; id?: number; payload?: unknown }) => void;
    }) => {
      streamHandlers.set(options.sessionId, options.onEvent);
      if (options.sessionId === 'rs_1') {
        options.onEvent({
          event: 'snapshot',
          id: 5,
          payload: currentSnapshot,
        });
        options.onEvent({
          event: 'progress',
          id: 6,
          payload: {
            id: 'rs_1',
            progress_percent: 55,
            progress_message: 'Collecting sources',
          },
        });
      }
      if (options.sessionId === 'rs_new') {
        options.onEvent({
          event: 'snapshot',
          id: 7,
          payload: makeSnapshot({
            run: makeRun({
              id: 'rs_new',
              query: 'Newly created run',
              latest_checkpoint_id: null,
            }),
            checkpoint: null,
            artifacts: [],
          }),
        });
      }
      return () => {};
    });
  });

  it('renders run history and applies live snapshot updates for the selected run', async () => {
    renderWithProviders(<ResearchRunsPage />);

    expect(await screen.findByText('Investigate local evidence')).toBeInTheDocument();
    expect(await screen.findByText('Collecting sources')).toBeInTheDocument();
    expect(screen.getByText('plan_review')).toBeInTheDocument();
    expect(mocks.subscribeResearchRunEvents).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: 'rs_1',
        afterId: 0,
      })
    );
  });

  it('creates a run and selects the new session', async () => {
    const user = userEvent.setup();

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.type(screen.getByLabelText('Research question'), 'Newly created run');
    await user.click(screen.getByRole('button', { name: 'Start run' }));

    await waitFor(() => {
      expect(mocks.createResearchRun).toHaveBeenCalledWith({
        query: 'Newly created run',
        source_policy: 'balanced',
        autonomy_mode: 'checkpointed',
      });
    });
    expect(await screen.findByText('Selected run: Newly created run')).toBeInTheDocument();
  });

  it('shows a back-to-chat link only for runs with linked chat context', async () => {
    mocks.getResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_1',
        chat_id: 'chat_123',
      })
    );

    renderWithProviders(<ResearchRunsPage />);

    const link = await screen.findByRole('link', { name: 'Back to Chat' });
    expect(link).toHaveAttribute(
      'href',
      '/chat?settingsServerChatId=chat_123&researchReturnRunId=rs_1'
    );
  });

  it('does not show a back-to-chat link for unlinked runs', async () => {
    mocks.getResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_1',
        chat_id: null,
      })
    );

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Selected run: Investigate local evidence');
    expect(screen.queryByRole('link', { name: 'Back to Chat' })).not.toBeInTheDocument();
  });

  it('prefills the research question from launch query params', async () => {
    window.history.replaceState(
      {},
      '',
      '/research?query=Trace%20the%20policy%20timeline&source_policy=local_first'
    );

    renderWithProviders(<ResearchRunsPage />);

    expect(await screen.findByDisplayValue('Trace the policy timeline')).toBeInTheDocument();
    expect(mocks.createResearchRun).not.toHaveBeenCalled();
  });

  it('carries launch follow-up context into manual run creation', async () => {
    const user = userEvent.setup();
    const followUp = {
      question: 'Verify the peer coaching retention hypothesis.',
      background: {
        question: 'Verify the peer coaching retention hypothesis.',
        outline: [{ title: 'Hypothesis 1', focus_area: 'hypothesis_1' }],
        key_claims: [
          {
            claim_id: 'artifact-hypotheses:finding-1',
            text: 'Evidence-supported finding: Paper A reports higher completion.'
          }
        ],
        unresolved_questions: [
          'Which parts of the hypothesis are evidence-supported versus proposed work?'
        ],
        verification_summary: {
          supported_claim_count: 1,
          unsupported_claim_count: 1
        },
        source_trust_summary: {
          high_trust_count: 2,
          low_trust_count: 1
        }
      }
    };
    const params = new URLSearchParams({
      query: 'Verify the peer coaching retention hypothesis.',
      source_policy: 'local_first',
      autonomy_mode: 'checkpointed',
      follow_up: JSON.stringify(followUp),
      from: 'research-workspace'
    });
    window.history.replaceState({}, '', `/research?${params.toString()}`);

    renderWithProviders(<ResearchRunsPage />);

    expect(
      await screen.findByDisplayValue('Verify the peer coaching retention hypothesis.')
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Start run' }));

    await waitFor(() => {
      expect(mocks.createResearchRun).toHaveBeenCalledWith({
        query: 'Verify the peer coaching retention hypothesis.',
        source_policy: 'local_first',
        autonomy_mode: 'checkpointed',
        follow_up: followUp,
      });
    });
  });

  it('keeps launch follow-up background question aligned when the query is edited', async () => {
    const user = userEvent.setup();
    const followUp = {
      question: 'Verify the peer coaching retention hypothesis.',
      background: {
        question: 'Verify the peer coaching retention hypothesis.',
        outline: [{ title: 'Hypothesis 1', focus_area: 'hypothesis_1' }],
        key_claims: [
          {
            claim_id: 'artifact-hypotheses:finding-1',
            text: 'Evidence-supported finding: Paper A reports higher completion.'
          }
        ],
        unresolved_questions: [
          'Which parts of the hypothesis are evidence-supported versus proposed work?'
        ],
        verification_summary: {
          supported_claim_count: 1,
          unsupported_claim_count: 1
        },
        source_trust_summary: {
          high_trust_count: 2,
          low_trust_count: 1
        }
      }
    };
    const params = new URLSearchParams({
      query: 'Verify the peer coaching retention hypothesis.',
      source_policy: 'local_first',
      autonomy_mode: 'checkpointed',
      follow_up: JSON.stringify(followUp),
      from: 'research-workspace'
    });
    window.history.replaceState({}, '', `/research?${params.toString()}`);

    renderWithProviders(<ResearchRunsPage />);

    const questionField = await screen.findByLabelText('Research question');
    await user.clear(questionField);
    await user.type(questionField, 'Refined peer coaching retention question.');
    await user.click(screen.getByRole('button', { name: 'Start run' }));

    await waitFor(() => {
      expect(mocks.createResearchRun).toHaveBeenCalledWith({
        query: 'Refined peer coaching retention question.',
        source_policy: 'local_first',
        autonomy_mode: 'checkpointed',
        follow_up: {
          ...followUp,
          question: 'Refined peer coaching retention question.',
          background: {
            ...followUp.background,
            question: 'Refined peer coaching retention question.',
          },
        },
      });
    });
  });

  it('bounds unresolved question entries parsed from launch follow-up context', async () => {
    const user = userEvent.setup();
    const oversizedQuestion = 'x'.repeat(2200);
    const followUp = {
      question: 'Verify bounded unresolved questions.',
      background: {
        question: 'Verify bounded unresolved questions.',
        outline: [{ title: 'Hypothesis 1', focus_area: 'hypothesis_1' }],
        key_claims: [
          {
            claim_id: 'artifact-hypotheses:finding-1',
            text: 'Evidence-supported finding: Paper A reports higher completion.'
          }
        ],
        unresolved_questions: [
          oversizedQuestion,
          'Which claims still need a primary source?',
          oversizedQuestion
        ],
        verification_summary: {
          supported_claim_count: 1,
          unsupported_claim_count: 1
        },
        source_trust_summary: {
          high_trust_count: 2,
          low_trust_count: 1
        }
      }
    };
    const params = new URLSearchParams({
      query: 'Verify bounded unresolved questions.',
      source_policy: 'local_first',
      autonomy_mode: 'checkpointed',
      follow_up: JSON.stringify(followUp),
      from: 'research-workspace'
    });
    window.history.replaceState({}, '', `/research?${params.toString()}`);

    renderWithProviders(<ResearchRunsPage />);

    expect(
      await screen.findByDisplayValue('Verify bounded unresolved questions.')
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Start run' }));

    await waitFor(() => {
      expect(mocks.createResearchRun).toHaveBeenCalledWith(
        expect.objectContaining({
          follow_up: expect.objectContaining({
            background: expect.objectContaining({
              unresolved_questions: [
                'x'.repeat(1800),
                'Which claims still need a primary source?',
              ],
            }),
          }),
        })
      );
    });
  });

  it('auto-creates a run from launch params when autorun is enabled', async () => {
    mocks.createResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_launch',
        query: 'Trace the policy timeline',
        latest_checkpoint_id: null,
      })
    );
    window.history.replaceState(
      {},
      '',
      '/research?query=Trace%20the%20policy%20timeline&source_policy=local_first&autonomy_mode=autonomous&autorun=1&from=chat&chat_id=chat_123'
    );

    renderWithProviders(<ResearchRunsPage />);

    await waitFor(() => {
      expect(mocks.createResearchRun).toHaveBeenCalledWith({
        query: 'Trace the policy timeline',
        source_policy: 'local_first',
        autonomy_mode: 'autonomous',
        chat_handoff: {
          chat_id: 'chat_123',
        },
      });
    });
    await waitFor(() => {
      expect(window.location.search).not.toContain('autorun=1');
      expect(window.location.search).not.toContain('query=');
      expect(window.location.search).not.toContain('chat_id=');
    });
  });

  it('submits an edited plan_review checkpoint patch', async () => {
    const user = userEvent.setup();

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    const focusAreasField = await screen.findByLabelText('Focus areas');
    await user.clear(focusAreasField);
    await user.type(focusAreasField, 'background{enter}counterevidence{enter}primary sources');
    await user.clear(screen.getByLabelText('Minimum sources'));
    await user.type(screen.getByLabelText('Minimum sources'), '3');
    await user.click(screen.getByRole('button', { name: 'Approve checkpoint' }));

    await waitFor(() => {
      expect(mocks.approveResearchCheckpoint).toHaveBeenCalledWith('rs_1', 'chk_1', {
        focus_areas: ['background', 'counterevidence', 'primary sources'],
        constraints: ['Use cited sources'],
        open_questions: ['What changed most recently?'],
        stop_criteria: {
          min_cited_sections: 1,
          min_sources: 3,
        },
      });
    });
  });

  it('submits a sources_review patch with curation and recollection directives', async () => {
    const user = userEvent.setup();
    currentSnapshot = makeSnapshot({
      checkpoint: {
        checkpoint_id: 'chk_sources',
        checkpoint_type: 'sources_review',
        status: 'pending',
        proposed_payload: {
          query: 'Investigate local evidence',
          focus_areas: ['background'],
          source_inventory: [
            { source_id: 'src_1', title: 'Primary memo', provider: 'local_corpus', focus_area: 'background' },
            { source_id: 'src_2', title: 'Counter note', provider: 'kagi', focus_area: 'background' },
          ],
          collection_summary: { source_count: 2 },
        },
        resolution: null,
      },
    });
    mocks.approveResearchCheckpoint.mockResolvedValue(
      makeRun({
        phase: 'collecting',
        latest_checkpoint_id: null,
      })
    );

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.click(await screen.findByRole('button', { name: 'Pin Primary memo' }));
    await user.click(screen.getByRole('button', { name: 'Drop Counter note' }));
    await user.click(screen.getByRole('button', { name: 'Prioritize Primary memo' }));
    await user.click(screen.getByLabelText('Recollect sources'));
    await user.click(screen.getByLabelText('Need more primary sources'));
    await user.click(screen.getByLabelText('Need more contradictions'));
    await user.type(screen.getByLabelText('Recollection guidance'), 'Look for newer contradictory primary sources.');
    await user.click(screen.getByRole('button', { name: 'Approve checkpoint' }));

    await waitFor(() => {
      expect(mocks.approveResearchCheckpoint).toHaveBeenCalledWith('rs_1', 'chk_sources', {
        pinned_source_ids: ['src_1'],
        dropped_source_ids: ['src_2'],
        prioritized_source_ids: ['src_1'],
        recollect: {
          enabled: true,
          need_primary_sources: true,
          need_contradictions: true,
          guidance: 'Look for newer contradictory primary sources.',
        },
      });
    });
  });

  it('blocks invalid sources_review edits and submits an edited outline_review patch', async () => {
    const user = userEvent.setup();
    currentSnapshot = makeSnapshot({
      checkpoint: {
        checkpoint_id: 'chk_outline',
        checkpoint_type: 'outline_review',
        status: 'pending',
        proposed_payload: {
          outline: {
            sections: [
              { title: 'Background', focus_area: 'background' },
              { title: 'Counterevidence', focus_area: 'counterevidence' },
            ],
          },
          focus_areas: ['background', 'counterevidence'],
        },
        resolution: null,
      },
    });

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.clear(screen.getByLabelText('Section title 1'));
    await user.click(screen.getByRole('button', { name: 'Approve checkpoint' }));

    expect(screen.getByText('Every outline section needs a title and focus area.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Approve checkpoint' })).toBeDisabled();

    await user.type(screen.getByLabelText('Section title 1'), 'Background context');
    await user.click(screen.getByRole('button', { name: 'Move section down: background' }));
    await user.click(screen.getByRole('button', { name: 'Approve checkpoint' }));

    await waitFor(() => {
      expect(mocks.approveResearchCheckpoint).toHaveBeenCalledWith('rs_1', 'chk_outline', {
        sections: [
          { title: 'Counterevidence', focus_area: 'counterevidence' },
          { title: 'Background context', focus_area: 'background' },
        ],
      });
    });
  });

  it('shows a trust empty state before synthesis', async () => {
    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');

    expect(screen.getByText('Research Trust')).toBeInTheDocument();
    expect(screen.getByText('Trust signals will appear after synthesis')).toBeInTheDocument();
  });

  it('renders trust details from a loaded bundle', async () => {
    const user = userEvent.setup();

    mocks.getResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_1',
        status: 'completed',
        phase: 'completed',
        progress_message: 'Completed',
        latest_checkpoint_id: null,
        completed_at: '2026-03-07T10:10:00Z',
      })
    );
    mocks.getResearchBundle.mockResolvedValue(makeTrustBundle());

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.click(screen.getByRole('button', { name: 'Load bundle' }));

    expect(await screen.findByText('Supported claims: 2')).toBeInTheDocument();
    expect(screen.getByText('Unsupported claims: 1')).toBeInTheDocument();
    expect(screen.getByText('Contradictions: 1')).toBeInTheDocument();
    expect(screen.getByText('Needs corroboration')).toBeInTheDocument();
    expect(screen.getByText('Unverified assertion')).toBeInTheDocument();
    expect(screen.getByText('Primary memo')).toBeInTheDocument();
    expect(
      screen.getByText(/local_corpus\s+·\s+internal\s+·\s+full_artifact/i),
    ).toBeInTheDocument();
  });

  it('lazy-loads trust artifacts when they are available but no bundle is loaded', async () => {
    const user = userEvent.setup();
    mocks.getResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_1',
        status: 'waiting_human',
        phase: 'awaiting_outline_review',
        latest_checkpoint_id: null,
      })
    );
    currentSnapshot = makeSnapshot({
      checkpoint: null,
      run: makeRun({
        phase: 'awaiting_outline_review',
        status: 'waiting_human',
        latest_checkpoint_id: null,
      }),
      artifacts: makeTrustArtifacts(),
    });

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.click(screen.getByRole('button', { name: 'Load trust details' }));

    await waitFor(() => {
      expect(mocks.getResearchArtifact).toHaveBeenCalledWith('rs_1', 'verification_summary.json');
      expect(mocks.getResearchArtifact).toHaveBeenCalledWith('rs_1', 'unsupported_claims.json');
      expect(mocks.getResearchArtifact).toHaveBeenCalledWith('rs_1', 'contradictions.json');
      expect(mocks.getResearchArtifact).toHaveBeenCalledWith('rs_1', 'source_trust.json');
    });
  });

  it('reuses trust artifacts that were already loaded through the raw artifact viewer', async () => {
    const user = userEvent.setup();
    mocks.getResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_1',
        status: 'waiting_human',
        phase: 'awaiting_outline_review',
        latest_checkpoint_id: null,
      })
    );
    currentSnapshot = makeSnapshot({
      checkpoint: null,
      run: makeRun({
        phase: 'awaiting_outline_review',
        status: 'waiting_human',
        latest_checkpoint_id: null,
      }),
      artifacts: makeTrustArtifacts(),
    });

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.click(screen.getByRole('button', { name: 'Load verification_summary.json' }));
    await user.click(screen.getByRole('button', { name: 'Load unsupported_claims.json' }));
    await user.click(screen.getByRole('button', { name: 'Load contradictions.json' }));
    await user.click(screen.getByRole('button', { name: 'Load source_trust.json' }));

    expect(await screen.findByText('Supported claims: 2')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load trust details' })).not.toBeInTheDocument();
  });

  it('clears stale trust details when the run returns to collecting', async () => {
    const user = userEvent.setup();
    mocks.getResearchRun.mockResolvedValue(
      makeRun({
        id: 'rs_1',
        status: 'waiting_human',
        phase: 'awaiting_outline_review',
        latest_checkpoint_id: null,
      })
    );
    currentSnapshot = makeSnapshot({
      checkpoint: null,
      run: makeRun({
        phase: 'awaiting_outline_review',
        status: 'waiting_human',
        latest_checkpoint_id: null,
      }),
      artifacts: makeTrustArtifacts(),
    });

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.click(screen.getByRole('button', { name: 'Load verification_summary.json' }));
    await user.click(screen.getByRole('button', { name: 'Load unsupported_claims.json' }));
    await user.click(screen.getByRole('button', { name: 'Load contradictions.json' }));
    await user.click(screen.getByRole('button', { name: 'Load source_trust.json' }));
    expect(await screen.findByText('Supported claims: 2')).toBeInTheDocument();

    emitStreamEvent('rs_1', {
      event: 'status',
      id: 7,
      payload: {
        id: 'rs_1',
        status: 'running',
        phase: 'collecting',
        progress_message: 'Collecting sources',
      },
    });

    await waitFor(() => {
      expect(screen.queryByText('Supported claims: 2')).not.toBeInTheDocument();
    });
    expect(screen.getByText('Trust signals will appear after synthesis')).toBeInTheDocument();
  });

  it('lazy-loads artifacts and completed bundles', async () => {
    const user = userEvent.setup();

    renderWithProviders(<ResearchRunsPage />);

    await screen.findByText('Investigate local evidence');
    await user.click(screen.getByRole('button', { name: 'Load plan.json' }));
    await waitFor(() => {
      expect(mocks.getResearchArtifact).toHaveBeenCalledWith('rs_1', 'plan.json');
    });
    expect(await screen.findByText(/Loaded artifact body/)).toBeInTheDocument();

    mocks.getResearchRun.mockResolvedValueOnce(
      makeRun({
        id: 'rs_1',
        status: 'completed',
        phase: 'completed',
        progress_message: 'Completed',
        latest_checkpoint_id: null,
        completed_at: '2026-03-07T10:10:00Z',
      })
    );

    await user.click(screen.getByRole('button', { name: 'Refresh selected run' }));
    await user.click(screen.getByRole('button', { name: 'Load bundle' }));
    await waitFor(() => {
      expect(mocks.getResearchBundle).toHaveBeenCalledWith('rs_1');
    });
    expect(await screen.findByText(/Final synthesized answer/)).toBeInTheDocument();
  });
});
