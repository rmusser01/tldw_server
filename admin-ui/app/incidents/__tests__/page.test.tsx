/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import IncidentsPage from '../page';
import { api } from '@/lib/api-client';
import { usePagedResource } from '@/lib/use-paged-resource';

const confirmMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const reloadMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => confirmMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/use-url-state', () => ({
  useUrlPagination: () => ({
    page: 1,
    pageSize: 20,
    setPage: vi.fn(),
    setPageSize: vi.fn(),
    resetPagination: vi.fn(),
  }),
}));

vi.mock('@/lib/use-paged-resource', () => ({
  usePagedResource: vi.fn(),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getUsers: vi.fn(),
    createIncident: vi.fn(),
    updateIncident: vi.fn(),
    addIncidentEvent: vi.fn(),
    deleteIncident: vi.fn(),
    notifyIncidentStakeholders: vi.fn(),
    getIncidentSlaMetrics: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);
const usePagedResourceMock = vi.mocked(usePagedResource);

beforeEach(() => {
  localStorage.clear();
  confirmMock.mockResolvedValue(true);
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
  reloadMock.mockResolvedValue(undefined);

  apiMock.getUsers.mockResolvedValue([
    { id: 1, email: 'alice@example.com', username: 'alice' },
    { id: 2, email: 'bob@example.com', username: 'bob' },
  ]);
  apiMock.createIncident.mockResolvedValue({});
  apiMock.updateIncident.mockResolvedValue({});
  apiMock.addIncidentEvent.mockResolvedValue({});
  apiMock.deleteIncident.mockResolvedValue({});
  apiMock.notifyIncidentStakeholders.mockResolvedValue({
    incident_id: '',
    notifications: [],
  });
  apiMock.getIncidentSlaMetrics.mockResolvedValue({
    total_incidents: 0,
    resolved_count: 0,
    acknowledged_count: 0,
    avg_mtta_minutes: null,
    avg_mttr_minutes: null,
    p95_mtta_minutes: null,
    p95_mttr_minutes: null,
  });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  localStorage.clear();
});

describe('IncidentsPage Stage 3 workflows', () => {
  it('updates incident assignment through the backend incident patch route', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-1',
        title: 'Queue latency spike',
        status: 'resolved',
        severity: 'high',
        summary: 'Elevated queue latency',
        tags: ['queue'],
        created_at: '2026-02-17T10:00:00Z',
        updated_at: '2026-02-17T10:00:00Z',
        resolved_at: '2026-02-17T10:00:00Z',
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });
    apiMock.updateIncident.mockResolvedValue({
      id: 'inc-1',
      title: 'Queue latency spike',
      status: 'resolved',
      severity: 'high',
      summary: 'Elevated queue latency',
      tags: ['queue'],
      created_at: '2026-02-17T10:00:00Z',
      updated_at: '2026-02-17T10:05:00Z',
      resolved_at: '2026-02-17T10:00:00Z',
      assigned_to_user_id: 2,
      assigned_to_label: 'bob@example.com',
      root_cause: null,
      impact: null,
      action_items: [],
      timeline: [],
    });

    const user = userEvent.setup();
    render(<IncidentsPage />);

    await waitFor(() => {
      expect(apiMock.getUsers).toHaveBeenCalledWith({ limit: '100', admin_capable: 'true' });
    });

    const assigneeSelect = await screen.findByTestId('incident-assigned-to-inc-1');
    await user.selectOptions(assigneeSelect, '2');

    await waitFor(() => {
      expect(apiMock.updateIncident).toHaveBeenCalledWith('inc-1', {
        assigned_to_user_id: 2,
        update_message: 'Assigned to bob',
      });
    });
    expect(apiMock.addIncidentEvent).not.toHaveBeenCalled();
    expect(reloadMock).toHaveBeenCalled();
  });

  it('preserves unsaved post-mortem draft when assignment changes', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-1',
        title: 'Queue latency spike',
        status: 'resolved',
        severity: 'high',
        summary: 'Elevated queue latency',
        tags: ['queue'],
        created_at: '2026-02-17T10:00:00Z',
        updated_at: '2026-02-17T10:00:00Z',
        resolved_at: '2026-02-17T10:00:00Z',
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });
    apiMock.updateIncident.mockResolvedValue({
      id: 'inc-1',
      title: 'Queue latency spike',
      status: 'resolved',
      severity: 'high',
      summary: 'Elevated queue latency',
      tags: ['queue'],
      created_at: '2026-02-17T10:00:00Z',
      updated_at: '2026-02-17T10:05:00Z',
      resolved_at: '2026-02-17T10:00:00Z',
      assigned_to_user_id: 2,
      assigned_to_label: 'bob@example.com',
      root_cause: null,
      impact: null,
      action_items: [],
      timeline: [],
    });

    const user = userEvent.setup();
    render(<IncidentsPage />);

    const rootCause = await screen.findByTestId('incident-root-cause-inc-1');
    const impact = screen.getByTestId('incident-impact-inc-1');

    await user.type(rootCause, 'Keep this draft root cause');
    await user.type(impact, 'Keep this draft impact');
    await user.click(screen.getByRole('button', { name: 'Add Action Item' }));
    const actionItemInput = screen.getByPlaceholderText('Describe follow-up action');
    await user.type(actionItemInput, 'Keep this draft action item');

    const assigneeSelect = screen.getByTestId('incident-assigned-to-inc-1');
    await user.selectOptions(assigneeSelect, '2');

    await waitFor(() => {
      expect(apiMock.updateIncident).toHaveBeenCalledWith('inc-1', {
        assigned_to_user_id: 2,
        update_message: 'Assigned to bob',
      });
    });

    expect((rootCause as HTMLTextAreaElement).value).toBe('Keep this draft root cause');
    expect((impact as HTMLTextAreaElement).value).toBe('Keep this draft impact');
    expect((actionItemInput as HTMLInputElement).value).toBe('Keep this draft action item');
  });

  it('preserves unsaved post-mortem draft when assignment update fails', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-1',
        title: 'Queue latency spike',
        status: 'resolved',
        severity: 'high',
        summary: 'Elevated queue latency',
        tags: ['queue'],
        created_at: '2026-02-17T10:00:00Z',
        updated_at: '2026-02-17T10:00:00Z',
        resolved_at: '2026-02-17T10:00:00Z',
        assigned_to_user_id: 1,
        assigned_to_label: 'alice@example.com',
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });
    apiMock.updateIncident.mockRejectedValue(new Error('Failed to update assignment'));

    const user = userEvent.setup();
    render(<IncidentsPage />);

    const rootCause = await screen.findByTestId('incident-root-cause-inc-1');
    const impact = screen.getByTestId('incident-impact-inc-1');

    await user.type(rootCause, 'Retain this root cause draft');
    await user.type(impact, 'Retain this impact draft');
    await user.click(screen.getByRole('button', { name: 'Add Action Item' }));
    const actionItemInput = screen.getByPlaceholderText('Describe follow-up action');
    await user.type(actionItemInput, 'Retain this action item draft');

    const assigneeSelect = screen.getByTestId('incident-assigned-to-inc-1');
    await user.selectOptions(assigneeSelect, '2');

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith('Failed to update assignment');
    });

    expect((assigneeSelect as HTMLSelectElement).value).toBe('1');
    expect((rootCause as HTMLTextAreaElement).value).toBe('Retain this root cause draft');
    expect((impact as HTMLTextAreaElement).value).toBe('Retain this impact draft');
    expect((actionItemInput as HTMLInputElement).value).toBe('Retain this action item draft');
  });

  it('renders backend workflow fields and assignee labels from the incident payload', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-2',
        title: 'Database outage',
        status: 'resolved',
        severity: 'critical',
        summary: 'Primary DB unavailable for 4 minutes',
        tags: ['database'],
        created_at: '2026-02-17T08:00:00Z',
        updated_at: '2026-02-17T09:00:00Z',
        resolved_at: '2026-02-17T09:00:00Z',
        assigned_to_user_id: 77,
        assigned_to_label: 'oncall-admin@example.com',
        root_cause: 'Connection pool exhaustion under failover.',
        impact: 'Writes failed for 4 minutes in one region.',
        action_items: [
          { id: 'ai_keep', text: 'Add failover pool saturation alert.', done: false },
        ],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });

    render(<IncidentsPage />);

    const rootCause = await screen.findByTestId('incident-root-cause-inc-2');
    const impact = screen.getByTestId('incident-impact-inc-2');
    const assigneeSelect = screen.getByTestId('incident-assigned-to-inc-2');
    const actionItemInput = screen.getByDisplayValue('Add failover pool saturation alert.');

    expect((assigneeSelect as HTMLSelectElement).value).toBe('77');
    expect(screen.getByRole('option', { name: 'oncall-admin@example.com' })).toBeInTheDocument();
    expect((rootCause as HTMLTextAreaElement).value).toBe('Connection pool exhaustion under failover.');
    expect((impact as HTMLTextAreaElement).value).toBe('Writes failed for 4 minutes in one region.');
    expect(actionItemInput).toBeInTheDocument();
  });

  it('saves structured post-mortem fields through the backend incident patch route', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-2',
        title: 'Database outage',
        status: 'resolved',
        severity: 'critical',
        summary: 'Primary DB unavailable for 4 minutes',
        tags: ['database'],
        created_at: '2026-02-17T08:00:00Z',
        updated_at: '2026-02-17T09:00:00Z',
        resolved_at: '2026-02-17T09:00:00Z',
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });
    apiMock.updateIncident.mockResolvedValue({
      id: 'inc-2',
      title: 'Database outage',
      status: 'resolved',
      severity: 'critical',
      summary: 'Primary DB unavailable for 4 minutes',
      tags: ['database'],
      created_at: '2026-02-17T08:00:00Z',
      updated_at: '2026-02-17T09:05:00Z',
      resolved_at: '2026-02-17T09:00:00Z',
      assigned_to_user_id: null,
      assigned_to_label: null,
      root_cause: 'Connection pool exhaustion under failover.',
      impact: 'Writes failed for 4 minutes in one region.',
      action_items: [
        { id: 'ai_keep', text: 'Add failover pool saturation alert.', done: false },
      ],
      timeline: [],
    });

    const user = userEvent.setup();
    render(<IncidentsPage />);

    const rootCause = await screen.findByTestId('incident-root-cause-inc-2');
    const impact = screen.getByTestId('incident-impact-inc-2');

    await user.type(rootCause, 'Connection pool exhaustion under failover.');
    await user.type(impact, 'Writes failed for 4 minutes in one region.');
    await user.click(screen.getByRole('button', { name: 'Add Action Item' }));
    const actionItemInput = screen.getByPlaceholderText('Describe follow-up action');
    await user.type(actionItemInput, 'Add failover pool saturation alert.');

    await user.click(screen.getByTestId('incident-save-postmortem-inc-2'));

    await waitFor(() => {
      expect(apiMock.updateIncident).toHaveBeenCalledWith(
        'inc-2',
        expect.objectContaining({
          root_cause: 'Connection pool exhaustion under failover.',
          impact: 'Writes failed for 4 minutes in one region.',
          action_items: [
            expect.objectContaining({
              text: 'Add failover pool saturation alert.',
              done: false,
            }),
          ],
          update_message: expect.stringContaining('Post-mortem updated'),
        }),
      );
    });
    expect(apiMock.addIncidentEvent).not.toHaveBeenCalled();
    expect(localStorage.getItem('admin.incidents.workflow.v1')).toBeNull();
  });

  it('opens notify dialog, sends notification, and displays delivery results', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-3',
        title: 'API gateway timeout',
        status: 'investigating',
        severity: 'high',
        summary: 'Gateway returning 504s',
        tags: ['gateway'],
        created_at: '2026-03-27T08:00:00Z',
        updated_at: '2026-03-27T08:00:00Z',
        resolved_at: null,
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });
    apiMock.notifyIncidentStakeholders.mockResolvedValue({
      incident_id: 'inc-3',
      notifications: [
        { email: 'alice@example.com', status: 'sent' },
        { email: 'bob@example.com', status: 'failed', error: 'SMTP timeout' },
      ],
    });

    const user = userEvent.setup();
    render(<IncidentsPage />);

    // Click the notify button
    const notifyBtn = await screen.findByTestId('incident-notify-inc-3');
    await user.click(notifyBtn);

    // Dialog should be open
    expect(screen.getByTestId('notify-dialog-overlay')).toBeInTheDocument();

    // Fill in recipients and message
    const recipientsInput = screen.getByTestId('notify-recipients-input');
    const messageInput = screen.getByTestId('notify-message-input');
    await user.type(recipientsInput, 'alice@example.com, bob@example.com');
    await user.type(messageInput, 'Please investigate urgently');

    // Send
    await user.click(screen.getByTestId('notify-send-button'));

    await waitFor(() => {
      expect(apiMock.notifyIncidentStakeholders).toHaveBeenCalledWith('inc-3', {
        recipients: ['alice@example.com', 'bob@example.com'],
        message: 'Please investigate urgently',
      });
    });

    // Delivery results should display
    await waitFor(() => {
      expect(screen.getByTestId('notify-results')).toBeInTheDocument();
    });
    expect(screen.getByText('alice@example.com')).toBeInTheDocument();
    expect(screen.getByText('bob@example.com')).toBeInTheDocument();
    expect(toastSuccessMock).toHaveBeenCalledWith('Notification sent to 1/2 recipient(s)');
    expect(reloadMock).toHaveBeenCalled();
  });

  it('renders SLA metric cards with avg MTTA, avg MTTR, P95, and resolved count', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-sla',
        title: 'SLA test incident',
        status: 'resolved',
        severity: 'medium',
        summary: 'Testing SLA metrics',
        tags: [],
        created_at: '2026-03-27T08:00:00Z',
        updated_at: '2026-03-27T09:00:00Z',
        resolved_at: '2026-03-27T09:00:00Z',
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });
    apiMock.getIncidentSlaMetrics.mockResolvedValue({
      total_incidents: 10,
      resolved_count: 7,
      acknowledged_count: 9,
      avg_mtta_minutes: 5,
      avg_mttr_minutes: 45,
      p95_mtta_minutes: 12,
      p95_mttr_minutes: 120,
    });

    render(<IncidentsPage />);

    // SLA cards render the formatted values
    expect(await screen.findByText('Avg. Time to Acknowledge')).toBeInTheDocument();
    expect(screen.getByText('Avg. Time to Resolve')).toBeInTheDocument();
    expect(screen.getByText('P95 Resolution Time')).toBeInTheDocument();

    // Verify computed values
    expect(screen.getByText('5m')).toBeInTheDocument();       // avg MTTA
    expect(screen.getByText('45m')).toBeInTheDocument();      // avg MTTR
    expect(screen.getByText('2h')).toBeInTheDocument();       // P95 MTTR = 120m = 2h
    expect(screen.getByText('7 / 10')).toBeInTheDocument();   // resolved / total
  });

  it('renders runbook URL link when incident has runbook_url set', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-rb',
        title: 'Runbook test',
        status: 'investigating',
        severity: 'high',
        summary: 'Testing runbook link',
        tags: ['infra'],
        created_at: '2026-03-27T08:00:00Z',
        updated_at: '2026-03-27T08:00:00Z',
        resolved_at: null,
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
        runbook_url: 'https://wiki.example.com/runbooks/infra-outage',
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });

    render(<IncidentsPage />);

    const runbookLink = await screen.findByRole('link', { name: 'Runbook' });
    expect(runbookLink).toBeInTheDocument();
    expect(runbookLink.getAttribute('href')).toBe('https://wiki.example.com/runbooks/infra-outage');
    expect(runbookLink.getAttribute('target')).toBe('_blank');
    expect(runbookLink.getAttribute('rel')).toBe('noopener noreferrer');
  });

  it('does not render runbook link when runbook_url is not set', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-no-rb',
        title: 'No runbook test',
        status: 'open',
        severity: 'low',
        summary: 'No runbook here',
        tags: [],
        created_at: '2026-03-27T08:00:00Z',
        updated_at: '2026-03-27T08:00:00Z',
        resolved_at: null,
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });

    render(<IncidentsPage />);

    await screen.findByText('No runbook test');
    expect(screen.queryByRole('link', { name: 'Runbook' })).not.toBeInTheDocument();
  });

  it('rejects unsafe runbook URLs before create and does not render unsafe links', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-unsafe',
        title: 'Unsafe runbook test',
        status: 'open',
        severity: 'medium',
        summary: 'Unsafe runbook should not render',
        tags: [],
        created_at: '2026-03-27T08:00:00Z',
        updated_at: '2026-03-27T08:00:00Z',
        resolved_at: null,
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
        runbook_url: 'javascript:alert(1)',
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });

    const user = userEvent.setup();
    render(<IncidentsPage />);

    await screen.findByText('Unsafe runbook test');
    expect(screen.queryByRole('link', { name: 'Runbook' })).not.toBeInTheDocument();

    await user.type(screen.getByLabelText('Title'), 'Queue outage');
    await user.type(screen.getByLabelText('Runbook URL (optional)'), 'javascript:alert(1)');
    await user.click(screen.getByRole('button', { name: 'Create Incident' }));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith('Runbook URL must start with http:// or https://');
    });
    expect(apiMock.createIncident).not.toHaveBeenCalled();
  });

  it('notify button opens dialog overlay', async () => {
    usePagedResourceMock.mockReturnValue({
      items: [{
        id: 'inc-notify',
        title: 'Notify dialog test',
        status: 'investigating',
        severity: 'medium',
        summary: 'Testing notify dialog open',
        tags: [],
        created_at: '2026-03-27T08:00:00Z',
        updated_at: '2026-03-27T08:00:00Z',
        resolved_at: null,
        assigned_to_user_id: null,
        assigned_to_label: null,
        root_cause: null,
        impact: null,
        action_items: [],
        timeline: [],
      }],
      total: 1,
      loading: false,
      error: '',
      reload: reloadMock,
    });

    const user = userEvent.setup();
    render(<IncidentsPage />);

    // Dialog should not be open initially
    expect(screen.queryByTestId('notify-dialog-overlay')).not.toBeInTheDocument();

    // Click the notify button
    const notifyBtn = await screen.findByTestId('incident-notify-inc-notify');
    await user.click(notifyBtn);

    // Dialog should now be visible with recipients and message inputs
    expect(screen.getByTestId('notify-dialog-overlay')).toBeInTheDocument();
    expect(screen.getByTestId('notify-recipients-input')).toBeInTheDocument();
    expect(screen.getByTestId('notify-message-input')).toBeInTheDocument();
    expect(screen.getByTestId('notify-send-button')).toBeInTheDocument();
    expect(screen.getByText('Notify Stakeholders')).toBeInTheDocument();
  });
});
