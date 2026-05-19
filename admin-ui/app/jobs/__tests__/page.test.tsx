/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import JobsPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
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

vi.mock('@/lib/api-client', () => ({
  ApiError: class extends Error {
    status: number;

    constructor(status: number, message?: string) {
      super(message);
      this.status = status;
    }
  },
  api: {
    getJobsStats: vi.fn(),
    getJobs: vi.fn(),
    getJobsStale: vi.fn(),
    getJobSlaPolicies: vi.fn(),
    getJobSlaBreaches: vi.fn(),
    getMonitoringMetrics: vi.fn(),
    getMetricsText: vi.fn(),
    getJobDetail: vi.fn(),
    getJobAttachments: vi.fn(),
    createJobSlaPolicy: vi.fn(),
    deleteJobSlaPolicy: vi.fn(),
    cancelJobs: vi.fn(),
    retryJobsNow: vi.fn(),
    requeueQuarantinedJobs: vi.fn(),
  },
}));

type ApiMock = {
  getJobsStats: ReturnType<typeof vi.fn>;
  getJobs: ReturnType<typeof vi.fn>;
  getJobsStale: ReturnType<typeof vi.fn>;
  getJobSlaPolicies: ReturnType<typeof vi.fn>;
  getJobSlaBreaches: ReturnType<typeof vi.fn>;
  getMonitoringMetrics: ReturnType<typeof vi.fn>;
  getMetricsText: ReturnType<typeof vi.fn>;
  getJobDetail: ReturnType<typeof vi.fn>;
  getJobAttachments: ReturnType<typeof vi.fn>;
  createJobSlaPolicy: ReturnType<typeof vi.fn>;
  deleteJobSlaPolicy: ReturnType<typeof vi.fn>;
  cancelJobs: ReturnType<typeof vi.fn>;
  retryJobsNow: ReturnType<typeof vi.fn>;
  requeueQuarantinedJobs: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const jobsResponse = [
  {
    id: 101,
    uuid: 'job-101',
    domain: 'exports',
    queue: 'default',
    job_type: 'root-task',
    status: 'processing',
    retry_count: 0,
    max_retries: 3,
    created_at: '2026-02-17T12:00:00Z',
    started_at: '2026-02-17T12:01:00Z',
  },
  {
    id: 102,
    uuid: 'job-102',
    domain: 'exports',
    queue: 'default',
    job_type: 'child-task',
    status: 'queued',
    retry_count: 0,
    max_retries: 3,
    created_at: '2026-02-17T12:02:00Z',
    parent_id: 101,
  },
];

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getJobsStats.mockResolvedValue([
    { domain: 'exports', queue: 'default', job_type: 'root-task', queued: 1, scheduled: 0, processing: 1, quarantined: 0 },
  ]);
  apiMock.getJobs.mockResolvedValue(jobsResponse);
  apiMock.getJobsStale.mockResolvedValue([]);
  apiMock.getJobSlaPolicies.mockResolvedValue([]);
  apiMock.getJobSlaBreaches.mockResolvedValue([]);
  apiMock.getMonitoringMetrics.mockResolvedValue([
    { timestamp: '2026-02-17T11:00:00Z', queue_depth: 4 },
    { timestamp: '2026-02-17T12:00:00Z', queue_depth: 7 },
    { timestamp: '2026-02-17T13:00:00Z', queue_depth: 3 },
  ]);
  apiMock.getMetricsText.mockResolvedValue('jobs_queue_depth 3');
  apiMock.getJobAttachments.mockResolvedValue([]);
  apiMock.getJobDetail.mockImplementation(async (jobId: string | number) => {
    if (Number(jobId) === 101) {
      return {
        ...jobsResponse[0],
        payload: { child_job_ids: [102] },
        result: null,
      };
    }
    return {
      ...jobsResponse[1],
      payload: { parent_id: 101 },
      result: null,
    };
  });
  apiMock.createJobSlaPolicy.mockResolvedValue({});
  apiMock.deleteJobSlaPolicy.mockResolvedValue({});
  apiMock.cancelJobs.mockResolvedValue({});
  apiMock.retryJobsNow.mockResolvedValue({});
  apiMock.requeueQuarantinedJobs.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('JobsPage', () => {
  it('renders queue depth chart and queue throughput metrics', async () => {
    render(<JobsPage />);

    expect(await screen.findByTestId('queue-depth-chart')).toBeInTheDocument();
    const liveRegion = screen.getByTestId('jobs-queue-stats-live-region');
    expect(liveRegion.getAttribute('aria-live')).toBe('polite');
    expect(liveRegion.getAttribute('role')).toBe('status');
    expect(screen.getByText('Queue stats table with 1 rows.')).toBeInTheDocument();
    expect(screen.getByTestId('queue-depth-current').textContent).toContain('3');
    expect(screen.getByTestId('queue-depth-peak').textContent).toContain('7');
    expect(screen.getByTestId('queue-throughput-completed').textContent).toContain('0');
  });

  it('does not fabricate queue depth history from a single metrics text snapshot', async () => {
    apiMock.getMonitoringMetrics.mockResolvedValue([]);
    apiMock.getMetricsText.mockResolvedValue('jobs_queue_depth 91');

    render(<JobsPage />);

    expect(await screen.findByText('Queue depth history unavailable.')).toBeInTheDocument();
    expect(screen.queryByTestId('queue-depth-chart')).not.toBeInTheDocument();
    expect(screen.queryByTestId('queue-depth-current')).not.toBeInTheDocument();
    expect(screen.queryByTestId('queue-depth-peak')).not.toBeInTheDocument();
  });

  it('renders dependency rows for related parent/child jobs', async () => {
    render(<JobsPage />);

    await screen.findByText('Job Dependencies');
    expect(await screen.findByTestId('job-dependency-row-101-102')).toBeInTheDocument();
    expect(screen.getByText('Child references parent')).toBeInTheDocument();
  });

  it('shows related jobs in detail modal when a job has dependencies', async () => {
    const user = userEvent.setup();
    render(<JobsPage />);

    const viewButtons = await screen.findAllByRole('button', { name: /view/i });
    await user.click(viewButtons[0]);

    await waitFor(() => {
      expect(apiMock.getJobDetail).toHaveBeenCalledWith(101, { domain: 'exports' });
    });

    const relatedJobsSection = await screen.findByTestId('job-related-jobs');
    expect(within(relatedJobsSection).getByText('Related Jobs')).toBeInTheDocument();
    expect(within(relatedJobsSection).getByRole('button', { name: /Job 102/i })).toBeInTheDocument();
  });

  it('shows SLA breach badges on breaching jobs', async () => {
    apiMock.getJobSlaBreaches.mockResolvedValue([
      {
        job_id: 101,
        domain: 'exports',
        queue: 'default',
        job_type: 'root-task',
        status: 'processing',
        breach_kinds: ['duration'],
        processing_seconds: 7200,
        max_processing_seconds: 3600,
      },
    ]);

    render(<JobsPage />);

    const breachBadge = await screen.findByTestId('sla-breach-badge-101');
    expect(breachBadge).toBeInTheDocument();
    expect(breachBadge.textContent).toBe('SLA');
  });

  it('shows breach count in the SLA Policies card header', async () => {
    apiMock.getJobSlaBreaches.mockResolvedValue([
      {
        job_id: 101,
        domain: 'exports',
        queue: 'default',
        job_type: 'root-task',
        status: 'processing',
        breach_kinds: ['duration'],
      },
      {
        job_id: 102,
        domain: 'exports',
        queue: 'default',
        job_type: 'child-task',
        status: 'queued',
        breach_kinds: ['queue_latency'],
      },
    ]);

    render(<JobsPage />);

    const breachCount = await screen.findByTestId('sla-breach-count');
    expect(breachCount).toBeInTheDocument();
    expect(breachCount.textContent).toBe('2 breaches');
  });

  it('does not show breach badges when there are no breaches', async () => {
    apiMock.getJobSlaBreaches.mockResolvedValue([]);

    render(<JobsPage />);

    await screen.findByText('SLA Policies');
    expect(screen.queryByTestId('sla-breach-count')).not.toBeInTheDocument();
    expect(screen.queryByTestId('sla-breach-badge-101')).not.toBeInTheDocument();
  });

  it('renders SLA policies with delete button', async () => {
    apiMock.getJobSlaPolicies.mockResolvedValue([
      {
        domain: 'exports',
        queue: 'default',
        job_type: 'export',
        max_queue_latency_seconds: 300,
        max_duration_seconds: 3600,
        enabled: true,
      },
    ]);

    render(<JobsPage />);

    await screen.findByText('export');
    expect(screen.getByRole('button', { name: /delete policy/i })).toBeInTheDocument();
  });

  it('prefers job-specific SLA policies over the generic fallback', async () => {
    apiMock.getJobs.mockResolvedValue([
      {
        id: 201,
        uuid: 'job-201',
        domain: 'exports',
        queue: 'default',
        job_type: 'root-task',
        status: 'completed',
        retry_count: 0,
        max_retries: 3,
        created_at: '2026-02-17T12:00:00Z',
        started_at: '2026-02-17T12:01:00Z',
        completed_at: '2026-02-17T12:04:00Z',
      },
    ]);
    apiMock.getJobSlaPolicies.mockResolvedValue([
      { id: 1, enabled: true, job_type: null, max_processing_time_seconds: 60 },
      { id: 2, enabled: true, job_type: 'root-task', max_processing_time_seconds: 600 },
    ]);

    render(<JobsPage />);

    expect(await screen.findByText('Jobs')).toBeInTheDocument();
    await waitFor(() => {
      expect(apiMock.getJobs).toHaveBeenCalled();
      expect(apiMock.getJobSlaPolicies).toHaveBeenCalled();
      expect(screen.queryByText('SLA')).not.toBeInTheDocument();
    });
  });

  it('requires an explicit job type and sends name separately when creating an SLA policy', async () => {
    const user = userEvent.setup();
    render(<JobsPage />);

    await screen.findByText('SLA Policies');
    await user.click(screen.getByRole('button', { name: 'New Policy' }));

    await user.type(screen.getByLabelText('Policy Name'), 'Standard Export');
    await user.click(screen.getByRole('button', { name: 'Create Policy' }));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith('Job type required', 'Please enter a backend job type');
    });
    expect(apiMock.createJobSlaPolicy).not.toHaveBeenCalled();

    await user.type(screen.getByLabelText('Job Type'), 'export_job');
    await user.click(screen.getByRole('button', { name: 'Create Policy' }));

    await waitFor(() => {
      expect(apiMock.createJobSlaPolicy).toHaveBeenCalledWith(expect.objectContaining({
        name: 'Standard Export',
        job_type: 'export_job',
      }));
    });
  });
});
