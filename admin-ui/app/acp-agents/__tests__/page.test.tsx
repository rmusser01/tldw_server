/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ACPAgentsPage from '../page';
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
    getACPAgentConfigs: vi.fn(),
    getACPPermissionPolicies: vi.fn(),
    getACPAgentMetrics: vi.fn(),
    getACPAgentUsage: vi.fn(),
    getMCPTools: vi.fn(),
    createACPAgentConfig: vi.fn(),
    updateACPAgentConfig: vi.fn(),
    deleteACPAgentConfig: vi.fn(),
    createACPPermissionPolicy: vi.fn(),
    updateACPPermissionPolicy: vi.fn(),
    deleteACPPermissionPolicy: vi.fn(),
  },
}));

type ApiMock = {
  getACPAgentConfigs: ReturnType<typeof vi.fn>;
  getACPPermissionPolicies: ReturnType<typeof vi.fn>;
  getACPAgentMetrics: ReturnType<typeof vi.fn>;
  getACPAgentUsage: ReturnType<typeof vi.fn>;
  getMCPTools: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  apiMock.getACPAgentConfigs.mockResolvedValue({
    agents: [],
    total: 0,
  });
  apiMock.getACPPermissionPolicies.mockResolvedValue({
    policies: [],
    total: 0,
  });
  apiMock.getACPAgentMetrics.mockResolvedValue({ items: [] });
  apiMock.getACPAgentUsage.mockResolvedValue({
    agents: [],
    total: 0,
  });
  apiMock.getMCPTools.mockResolvedValue({
    tools: [],
  });
  confirmMock.mockResolvedValue(true);
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('ACPAgentsPage', () => {
  it('loads agent configurations on mount', async () => {
    apiMock.getACPAgentConfigs.mockResolvedValue({
      agents: [
        {
          id: 1,
          type: 'codex',
          name: 'Code Agent',
          description: 'Handles implementation tasks',
          system_prompt: null,
          allowed_tools: ['fs.read', 'fs.write'],
          denied_tools: ['bash.run'],
          parameters: { model: 'gpt-5-codex' },
          requires_api_key: null,
          org_id: null,
          team_id: null,
          enabled: true,
          is_configured: true,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });

    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(apiMock.getACPAgentConfigs).toHaveBeenCalledTimes(1);
      expect(apiMock.getACPPermissionPolicies).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByText('Code Agent')).toBeInTheDocument();
    expect(screen.getByText('Handles implementation tasks')).toBeInTheDocument();
    expect(screen.getByText('Enabled')).toBeInTheDocument();
    expect(screen.getByText('Configured')).toBeInTheDocument();
  });

  it('opens the edit dialog with tag inputs for tools when editing an agent', async () => {
    const user = userEvent.setup();
    apiMock.getACPAgentConfigs.mockResolvedValue({
      agents: [
        {
          id: 1,
          type: 'codex',
          name: 'Code Agent',
          description: 'Handles tasks',
          system_prompt: null,
          allowed_tools: ['fs.read', 'fs.write'],
          denied_tools: ['bash.run'],
          parameters: { model: 'gpt-5-codex' },
          requires_api_key: null,
          org_id: null,
          team_id: null,
          enabled: true,
          is_configured: true,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });

    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Code Agent')).toBeInTheDocument();
    });

    await user.click(screen.getAllByRole('button', { name: 'Edit' })[0]);

    await waitFor(() => {
      expect(screen.getByText('Edit Agent Configuration')).toBeInTheDocument();
    });

    // Tag chips should be rendered for allowed and denied tools
    expect(screen.getByText('fs.read')).toBeInTheDocument();
    expect(screen.getByText('fs.write')).toBeInTheDocument();
    expect(screen.getByText('bash.run')).toBeInTheDocument();

    // Remove buttons should exist for each tag
    expect(screen.getByRole('button', { name: 'Remove fs.read' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove bash.run' })).toBeInTheDocument();
  });

  it('shows an empty state for the policies tab when no policies exist', async () => {
    const user = userEvent.setup();
    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(apiMock.getACPPermissionPolicies).toHaveBeenCalledTimes(1);
    });

    await user.click(screen.getByRole('button', { name: /Permission Policies \(0\)/i }));

    expect(screen.getByText('No Permission Policies')).toBeInTheDocument();
    expect(
      screen.getByText('Create policies to control which tools require approval.')
    ).toBeInTheDocument();
  });

  it('renders policy rules and actions when policies are loaded', async () => {
    const user = userEvent.setup();
    apiMock.getACPPermissionPolicies.mockResolvedValue({
      policies: [
        {
          id: 7,
          name: 'Write Guard',
          description: 'Require explicit approval for file writes',
          rules: [{ tool_pattern: 'fs.write*', tier: 'individual' }],
          org_id: null,
          team_id: null,
          priority: 10,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });

    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(apiMock.getACPPermissionPolicies).toHaveBeenCalledTimes(1);
    });

    await user.click(screen.getByRole('button', { name: /Permission Policies \(1\)/i }));

    expect(screen.getByText('Write Guard')).toBeInTheDocument();
    expect(screen.getByText('Require explicit approval for file writes')).toBeInTheDocument();
    expect(screen.getByText('fs.write*: individual')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Edit' }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('button', { name: 'Delete' }).length).toBeGreaterThan(0);
  });

  it('renders agent table with config data including tools summary', async () => {
    apiMock.getACPAgentConfigs.mockResolvedValue({
      agents: [
        {
          id: 1,
          type: 'codex',
          name: 'Code Agent',
          description: 'Writes code',
          system_prompt: null,
          allowed_tools: ['fs.read', 'fs.write'],
          denied_tools: ['bash.run'],
          parameters: { model: 'gpt-5-codex' },
          requires_api_key: null,
          org_id: null,
          team_id: null,
          enabled: true,
          is_configured: true,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });

    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Code Agent')).toBeInTheDocument();
    });

    expect(screen.getByText('codex')).toBeInTheDocument();
    expect(screen.getByText('gpt-5-codex')).toBeInTheDocument();
    // Tools column combines allowed/denied in the same cell
    expect(screen.getByText(/2 allowed/)).toBeInTheDocument();
    expect(screen.getByText(/1 denied/)).toBeInTheDocument();
  });

  it('shows metrics columns when data is available', async () => {
    apiMock.getACPAgentConfigs.mockResolvedValue({
      agents: [
        {
          id: 1,
          type: 'codex',
          name: 'Code Agent',
          description: 'Writes code',
          system_prompt: null,
          allowed_tools: null,
          denied_tools: null,
          parameters: { model: 'gpt-5-codex' },
          requires_api_key: null,
          org_id: null,
          team_id: null,
          enabled: true,
          is_configured: true,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });
    apiMock.getACPAgentMetrics.mockResolvedValue({
      items: [
        {
          agent_type: 'codex',
          session_count: 42,
          active_sessions: 3,
          total_prompt_tokens: 1_500_000,
          total_completion_tokens: 500_000,
          total_tokens: 2_000_000,
          total_messages: 120,
          last_used_at: '2026-03-26T10:00:00Z',
          total_estimated_cost_usd: 12.50,
        },
      ],
    });

    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Code Agent')).toBeInTheDocument();
    });

    // Wait for async metrics to resolve
    await waitFor(() => {
      expect(screen.getByText('42')).toBeInTheDocument();
    });
    expect(screen.getByText('2.0M')).toBeInTheDocument();
    expect(screen.getByText('$12.50')).toBeInTheDocument();
    expect(screen.getByText('3 active')).toBeInTheDocument();
  });

  it('shows budget fields in edit dialog', async () => {
    const user = userEvent.setup();
    apiMock.getACPAgentConfigs.mockResolvedValue({
      agents: [
        {
          id: 1,
          type: 'codex',
          name: 'Code Agent',
          description: 'Writes code',
          system_prompt: null,
          allowed_tools: null,
          denied_tools: null,
          parameters: {
            model: 'gpt-5-codex',
            default_token_budget: 100000,
            default_auto_terminate_at_budget: true,
          },
          requires_api_key: null,
          org_id: null,
          team_id: null,
          enabled: true,
          is_configured: true,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });

    render(<ACPAgentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Code Agent')).toBeInTheDocument();
    });

    await user.click(screen.getAllByRole('button', { name: 'Edit' })[0]);

    await waitFor(() => {
      expect(screen.getByText('Edit Agent Configuration')).toBeInTheDocument();
    });

    const budgetInput = screen.getByLabelText('Default Token Budget');
    expect(budgetInput).toBeInTheDocument();
    expect((budgetInput as HTMLInputElement).value).toBe('100000');

    expect(screen.getByLabelText('Auto-terminate at budget')).toBeInTheDocument();
  });

  it('handles metrics fetch failure gracefully without showing error', async () => {
    apiMock.getACPAgentConfigs.mockResolvedValue({
      agents: [
        {
          id: 1,
          type: 'codex',
          name: 'Code Agent',
          description: 'Writes code',
          system_prompt: null,
          allowed_tools: null,
          denied_tools: null,
          parameters: { model: 'gpt-5-codex' },
          requires_api_key: null,
          org_id: null,
          team_id: null,
          enabled: true,
          is_configured: true,
          created_at: '2024-01-01T00:00:00.000Z',
          updated_at: null,
        },
      ],
      total: 1,
    });
    apiMock.getACPAgentMetrics.mockRejectedValue(new Error('metrics unavailable'));

    render(<ACPAgentsPage />);

    // Page should still load agents fine
    await waitFor(() => {
      expect(screen.getByText('Code Agent')).toBeInTheDocument();
    });

    // No error alert should appear (metrics failure is silently ignored)
    expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/metrics unavailable/i)).not.toBeInTheDocument();
  });
});
