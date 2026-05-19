/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ACPSessionsPage from '../page';
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
    getACPSessions: vi.fn(),
    closeACPSession: vi.fn(),
    getUsersPage: vi.fn(),
    setSessionBudget: vi.fn(),
  },
}));

type ApiMock = {
  getACPSessions: ReturnType<typeof vi.fn>;
  closeACPSession: ReturnType<typeof vi.fn>;
  getUsersPage: ReturnType<typeof vi.fn>;
  setSessionBudget: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

beforeEach(() => {
  apiMock.getACPSessions.mockResolvedValue({
    sessions: [],
    total: 0,
  });
  apiMock.closeACPSession.mockResolvedValue({});
  apiMock.getUsersPage.mockResolvedValue({ items: [] });
  apiMock.setSessionBudget.mockResolvedValue({});
  confirmMock.mockResolvedValue(true);
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('ACPSessionsPage filters', () => {
  it('does not refetch while typing filters until apply is clicked', async () => {
    const user = userEvent.setup();
    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(apiMock.getACPSessions).toHaveBeenCalledTimes(1);
    });

    await user.type(screen.getByPlaceholderText('Agent type...'), 'assistant');
    await user.type(screen.getByPlaceholderText('User ID...'), '42');

    expect(apiMock.getACPSessions).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: 'Apply' }));

    await waitFor(() => {
      expect(apiMock.getACPSessions).toHaveBeenCalledTimes(2);
    });
    expect(apiMock.getACPSessions).toHaveBeenLastCalledWith({
      agent_type: 'assistant',
      user_id: '42',
    });
  });

  it('ignores slower stale responses from older session loads', async () => {
    const initialRequest = deferred<{ sessions: ReturnType<typeof makeSession>[]; total: number }>();
    const filteredRequest = deferred<{ sessions: ReturnType<typeof makeSession>[]; total: number }>();
    apiMock.getACPSessions
      .mockReturnValueOnce(initialRequest.promise)
      .mockReturnValueOnce(filteredRequest.promise);

    const user = userEvent.setup();
    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(apiMock.getACPSessions).toHaveBeenCalledTimes(1);
    });

    await user.type(screen.getByPlaceholderText('Agent type...'), 'assistant');
    await user.click(screen.getByRole('button', { name: 'Apply' }));

    await waitFor(() => {
      expect(apiMock.getACPSessions).toHaveBeenCalledTimes(2);
    });

    filteredRequest.resolve({
      sessions: [makeSession({ session_id: 'sess-fresh', name: 'Fresh Session', agent_type: 'assistant' })],
      total: 1,
    });

    await screen.findByText('Fresh Session');

    initialRequest.resolve({
      sessions: [makeSession({ session_id: 'sess-stale', name: 'Stale Session' })],
      total: 1,
    });

    await waitFor(() => {
      expect(screen.getByText('Fresh Session')).toBeInTheDocument();
      expect(screen.queryByText('Stale Session')).not.toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// Helpers for extended tests
// ---------------------------------------------------------------------------

const makeSession = (overrides: Record<string, unknown> = {}) => ({
  session_id: 'sess-aaaa-bbbb-cccc-dddddddddddd',
  user_id: 1,
  agent_type: 'codex',
  name: 'Test Session',
  status: 'active',
  created_at: '2026-03-20T10:00:00Z',
  last_activity_at: '2026-03-27T12:00:00Z',
  message_count: 15,
  usage: { prompt_tokens: 5000, completion_tokens: 3000, total_tokens: 8000 },
  tags: [],
  has_websocket: false,
  model: 'claude-opus-4-6',
  estimated_cost_usd: 0.42,
  token_budget: null,
  auto_terminate_at_budget: false,
  budget_exhausted: false,
  budget_remaining: null,
  ...overrides,
});

describe('ACPSessionsPage - budget progress bar', () => {
  it('renders budget progress bar when token_budget is set', async () => {
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [
        makeSession({
          token_budget: 100000,
          usage: { prompt_tokens: 30000, completion_tokens: 10000, total_tokens: 40000 },
          auto_terminate_at_budget: true,
        }),
      ],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByTestId('budget-progress')).toBeInTheDocument();
    });

    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toBeInTheDocument();
    expect(progressBar.getAttribute('aria-valuenow')).toBe('40');
    expect(screen.getByText('40%')).toBeInTheDocument();
    // Auto-terminate indicator
    expect(screen.getByText('(auto)')).toBeInTheDocument();
  });

  it('renders exhausted badge when budget is used up', async () => {
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [
        makeSession({
          token_budget: 10000,
          usage: { prompt_tokens: 7000, completion_tokens: 5000, total_tokens: 12000 },
          budget_exhausted: true,
        }),
      ],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByTestId('budget-exhausted')).toBeInTheDocument();
    });
    expect(screen.getByText('Exhausted')).toBeInTheDocument();
  });
});

describe('ACPSessionsPage - Set Budget dialog', () => {
  it('opens Set Budget dialog when budget button is clicked', async () => {
    const user = userEvent.setup();
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [makeSession()],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Set budget' }));

    await waitFor(() => {
      expect(screen.getByText('Set Token Budget')).toBeInTheDocument();
    });

    expect(screen.getByTestId('budget-input')).toBeInTheDocument();
    expect(screen.getByTestId('budget-auto-terminate')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /set budget/i })).toBeInTheDocument();
  });
});

describe('ACPSessionsPage - auto-refresh indicator', () => {
  it('shows auto-refresh toggle and last updated label', async () => {
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [makeSession()],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });

    // Auto-refresh toggle should be present
    const toggleBtn = screen.getByTestId('auto-refresh-toggle');
    expect(toggleBtn).toBeInTheDocument();
    expect(toggleBtn.getAttribute('aria-label')).toBe('Pause auto-refresh');

    // Last updated label should appear after data loads
    expect(screen.getByTestId('last-updated-label')).toBeInTheDocument();
  });

  it('toggles auto-refresh label on click', async () => {
    const user = userEvent.setup();
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [makeSession()],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });

    const toggleBtn = screen.getByTestId('auto-refresh-toggle');
    expect(toggleBtn.getAttribute('aria-label')).toBe('Pause auto-refresh');

    await user.click(toggleBtn);

    expect(toggleBtn.getAttribute('aria-label')).toBe('Resume auto-refresh');
  });
});

describe('ACPSessionsPage - cost column', () => {
  it('renders estimated cost for sessions', async () => {
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [
        makeSession({ estimated_cost_usd: 1.75, model: 'claude-opus-4-6' }),
      ],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByText('$1.75')).toBeInTheDocument();
    });

    // Model name should be shown
    expect(screen.getByText('claude-opus-4-6')).toBeInTheDocument();
  });

  it('renders em-dash for null cost', async () => {
    apiMock.getACPSessions.mockResolvedValue({
      sessions: [
        makeSession({ estimated_cost_usd: null, model: null }),
      ],
      total: 1,
    });

    render(<ACPSessionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Session')).toBeInTheDocument();
    });

    // The cost cell should contain an em-dash character
    const costCells = screen.getAllByText('\u2014');
    expect(costCells.length).toBeGreaterThan(0);
  });
});
