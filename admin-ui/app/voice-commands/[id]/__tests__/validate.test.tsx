/* @vitest-environment jsdom */
import { Suspense, type ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VoiceCommandDetailPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const pushMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

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

vi.mock('@/lib/api-client', () => ({
  api: {
    getVoiceCommand: vi.fn(),
    updateVoiceCommand: vi.fn(),
    deleteVoiceCommand: vi.fn(),
    toggleVoiceCommand: vi.fn(),
    getVoiceCommandUsage: vi.fn(),
    validateVoiceCommand: vi.fn(),
  },
}));

type ApiMock = {
  getVoiceCommand: ReturnType<typeof vi.fn>;
  updateVoiceCommand: ReturnType<typeof vi.fn>;
  deleteVoiceCommand: ReturnType<typeof vi.fn>;
  toggleVoiceCommand: ReturnType<typeof vi.fn>;
  getVoiceCommandUsage: ReturnType<typeof vi.fn>;
  validateVoiceCommand: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const MOCK_COMMAND = {
  id: 'cmd-1',
  user_id: 1,
  name: 'Search Media',
  phrases: ['search for', 'find'],
  action_type: 'mcp_tool' as const,
  action_config: { tool_name: 'media.search' },
  priority: 10,
  enabled: true,
  requires_confirmation: false,
  description: 'Search through media',
};

/**
 * Create a pre-resolved promise that React.use() can consume synchronously.
 *
 * NOTE: React's `use()` hook inspects `.status` and `.value` on promise
 * objects to avoid suspending when the result is already available.  This
 * is an *internal* implementation detail (not public API) but is the only
 * reliable way to test components that call `use()` without wrapping every
 * render in a Suspense boundary.  If a future React version changes this
 * mechanism, this helper will need to be updated accordingly.
 */
function resolvedPromise<T>(value: T): Promise<T> {
  const p = Promise.resolve(value);
  (p as unknown as Record<string, unknown>).status = 'fulfilled';
  (p as unknown as Record<string, unknown>).value = value;
  return p;
}

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  pushMock.mockClear();
  toastSuccessMock.mockReset();
  toastErrorMock.mockReset();

  apiMock.getVoiceCommand.mockResolvedValue(MOCK_COMMAND);
  apiMock.getVoiceCommandUsage.mockRejectedValue(new Error('no usage'));
  apiMock.validateVoiceCommand.mockResolvedValue({
    command_id: 'cmd-1',
    command_name: 'Search Media',
    action_type: 'mcp_tool',
    valid: true,
    steps: [
      { name: 'config_schema', passed: true, message: "Action config is well-formed for 'mcp_tool'" },
      { name: 'action_target', passed: true, message: "MCP tool 'media.search' is available" },
      { name: 'phrases', passed: true, message: '2 trigger phrase(s) configured' },
    ],
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function renderPage() {
  let result: ReturnType<typeof render>;
  await act(async () => {
    result = render(
      <Suspense fallback={<div>Loading...</div>}>
        <VoiceCommandDetailPage params={resolvedPromise({ id: 'cmd-1' })} />
      </Suspense>
    );
  });
  return result!;
}

describe('Voice Command Validate (dry-run)', () => {
  it('renders a Validate button', async () => {
    await renderPage();
    await waitFor(() => {
      expect(screen.getByTestId('validate-button')).toBeInTheDocument();
    });
  });

  it('shows all-passed validation report on success', async () => {
    const user = userEvent.setup();
    await renderPage();

    await waitFor(() => {
      expect(screen.getByTestId('validate-button')).toBeInTheDocument();
    });

    await user.click(screen.getByTestId('validate-button'));

    await waitFor(() => {
      expect(screen.getByTestId('validation-report')).toBeInTheDocument();
    });

    expect(screen.getByText('All checks passed')).toBeInTheDocument();
    expect(screen.getByTestId('validation-step-config_schema')).toBeInTheDocument();
    expect(screen.getByTestId('validation-step-action_target')).toBeInTheDocument();
    expect(screen.getByTestId('validation-step-phrases')).toBeInTheDocument();

    expect(apiMock.validateVoiceCommand).toHaveBeenCalledWith('cmd-1');
  });

  it('shows failure state when validation has a failing step', async () => {
    apiMock.validateVoiceCommand.mockResolvedValue({
      command_id: 'cmd-1',
      command_name: 'Search Media',
      action_type: 'mcp_tool',
      valid: false,
      steps: [
        { name: 'config_schema', passed: true, message: "Action config is well-formed for 'mcp_tool'" },
        {
          name: 'action_target',
          passed: false,
          message: "MCP tool 'media.search' not found in available tools",
          details: { available_tools_sample: ['notes.create', 'notes.search'] },
        },
        { name: 'phrases', passed: true, message: '2 trigger phrase(s) configured' },
      ],
    });

    const user = userEvent.setup();
    await renderPage();

    await waitFor(() => {
      expect(screen.getByTestId('validate-button')).toBeInTheDocument();
    });

    await user.click(screen.getByTestId('validate-button'));

    await waitFor(() => {
      expect(screen.getByTestId('validation-report')).toBeInTheDocument();
    });

    expect(screen.getByText('Validation failed')).toBeInTheDocument();
    expect(screen.getByText(/not found in available tools/)).toBeInTheDocument();
  });

  it('shows error toast when API call fails', async () => {
    apiMock.validateVoiceCommand.mockRejectedValue(new Error('Server error'));

    const user = userEvent.setup();
    await renderPage();

    await waitFor(() => {
      expect(screen.getByTestId('validate-button')).toBeInTheDocument();
    });

    await user.click(screen.getByTestId('validate-button'));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith('Validation failed', 'Server error');
    });
  });
});
