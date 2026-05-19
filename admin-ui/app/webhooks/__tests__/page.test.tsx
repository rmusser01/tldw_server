/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import WebhooksPage from '../page';
import { api } from '@/lib/api-client';

const privilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
  usePermissions: () => ({
    user: { id: 1, uuid: 'u-1', username: 'Admin', role: 'admin' },
    permissions: [],
    permissionHints: [],
    roles: ['admin'],
    loading: false,
    hasPermission: () => true,
    hasRole: () => true,
    hasAnyPermission: () => true,
    hasAllPermissions: () => true,
    isAdmin: () => true,
    isSuperAdmin: () => false,
    refresh: async () => {},
  }),
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => privilegedActionMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => '/webhooks',
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getWebhooks: vi.fn(),
    createWebhook: vi.fn(),
    updateWebhook: vi.fn(),
    deleteWebhook: vi.fn(),
    getWebhookDeliveries: vi.fn(),
    testWebhook: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);

const makeWebhook = (overrides: Partial<Record<string, unknown>> = {}) => ({
  id: 'wh-1',
  url: 'https://example.com/hook',
  events: ['user.created', 'incident.created'],
  enabled: true,
  created_at: '2025-06-01T00:00:00Z',
  updated_at: '2025-06-01T00:00:00Z',
  ...overrides,
});

beforeEach(() => {
  privilegedActionMock.mockResolvedValue({
    reason: 'Testing',
    adminPassword: 'Pass123!',
  });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getWebhooks.mockResolvedValue({
    items: [
      makeWebhook(),
      makeWebhook({
        id: 'wh-2',
        url: 'https://other.com/webhook',
        events: ['user.deleted'],
        enabled: false,
      }),
    ],
    total: 2,
  });

  apiMock.getWebhookDeliveries.mockResolvedValue({
    items: [
      {
        id: 'del-1',
        webhook_id: 'wh-1',
        event_type: 'user.created',
        status_code: 200,
        response_time_ms: 42,
        success: true,
        error: null,
        attempted_at: '2025-06-01T01:00:00Z',
        payload_preview: null,
      },
    ],
    total: 1,
  });

  apiMock.createWebhook.mockResolvedValue({
    id: 'wh-new',
    url: 'https://new.com/hook',
    events: ['user.created'],
    enabled: true,
    created_at: '2025-06-02T00:00:00Z',
    updated_at: '2025-06-02T00:00:00Z',
    secret: 'whsec_abc123',
  });

  apiMock.deleteWebhook.mockResolvedValue({});
  apiMock.updateWebhook.mockResolvedValue({});
  apiMock.testWebhook.mockResolvedValue({
    id: 'del-test',
    webhook_id: 'wh-1',
    event_type: 'webhook.test',
    status_code: 200,
    response_time_ms: 55,
    success: true,
    error: null,
    attempted_at: '2025-06-02T00:00:00Z',
    payload_preview: null,
  });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('WebhooksPage', () => {
  it('renders webhook list from API', async () => {
    render(<WebhooksPage />);

    expect(await screen.findByText('https://example.com/hook')).toBeInTheDocument();
    expect(screen.getByText('https://other.com/webhook')).toBeInTheDocument();
    expect(screen.getByText('user.created')).toBeInTheDocument();
    expect(screen.getByText('incident.created')).toBeInTheDocument();
    expect(screen.getByText('user.deleted')).toBeInTheDocument();
  });

  it('shows empty state when no webhooks exist', async () => {
    apiMock.getWebhooks.mockResolvedValue({ items: [], total: 0 });
    render(<WebhooksPage />);

    expect(await screen.findByText('No webhooks configured')).toBeInTheDocument();
  });

  it('opens create dialog on Add Webhook button click', async () => {
    const user = userEvent.setup();
    render(<WebhooksPage />);

    await screen.findByText('https://example.com/hook');
    await user.click(screen.getByRole('button', { name: /add webhook/i }));

    const dialog = await screen.findByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(screen.getByLabelText('Endpoint URL')).toBeInTheDocument();
    // The dialog should show available events including incident.resolved
    // which doesn't exist in the table data, so it's unique to the dialog
    expect(screen.getByText('incident.resolved')).toBeInTheDocument();
    expect(screen.getByText('incident.updated')).toBeInTheDocument();
  });

  it('calls deleteWebhook through PrivilegedActionDialog', async () => {
    const user = userEvent.setup();
    render(<WebhooksPage />);

    await screen.findByText('https://example.com/hook');

    // Find the first delete button by its accessible label
    const trashButtons = screen.getAllByRole('button', { name: 'Delete webhook' });
    expect(trashButtons.length).toBeGreaterThan(0);
    await user.click(trashButtons[0]);

    await waitFor(() => {
      expect(privilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete Webhook',
          confirmText: 'Delete Webhook',
        })
      );
    });

    await waitFor(() => {
      expect(apiMock.deleteWebhook).toHaveBeenCalledWith('wh-1');
    });
  });

  it('shows delivery expand toggle and fetches deliveries', async () => {
    const user = userEvent.setup();
    render(<WebhooksPage />);

    await screen.findByText('https://example.com/hook');

    const toggleButtons = screen.getAllByRole('button', { name: /toggle deliveries/i });
    expect(toggleButtons.length).toBeGreaterThan(0);

    await user.click(toggleButtons[0]);

    expect(await screen.findByText('Delivery History')).toBeInTheDocument();

    await waitFor(() => {
      expect(apiMock.getWebhookDeliveries).toHaveBeenCalledWith('wh-1', 50);
    });
  });

  it('handles API error gracefully', async () => {
    apiMock.getWebhooks.mockRejectedValue(new Error('Network failure'));
    render(<WebhooksPage />);

    expect(await screen.findByText('Network failure')).toBeInTheDocument();
  });

  it('shows enabled/disabled badge for each webhook', async () => {
    render(<WebhooksPage />);

    await screen.findByText('https://example.com/hook');
    expect(screen.getByText('Enabled')).toBeInTheDocument();
    expect(screen.getByText('Disabled')).toBeInTheDocument();
  });

  it('displays the webhook secret after creation', async () => {
    const user = userEvent.setup();
    render(<WebhooksPage />);

    await screen.findByText('https://example.com/hook');
    await user.click(screen.getByRole('button', { name: /add webhook/i }));

    // Fill in the create form
    await user.type(screen.getByLabelText('Endpoint URL'), 'https://new.com/hook');

    // Select at least one event inside the dialog
    const dialog = screen.getByRole('dialog');
    const eventCheckboxes = within(dialog).queryAllByRole('checkbox');
    if (eventCheckboxes.length > 0) {
      await user.click(eventCheckboxes[0]);
    } else {
      // Fall back to native checkbox selector if ARIA roles aren't set
      const nativeCheckboxes = dialog.querySelectorAll('input[type="checkbox"]');
      expect(nativeCheckboxes.length).toBeGreaterThan(0);
      await user.click(nativeCheckboxes[0] as HTMLElement);
    }

    await user.click(screen.getByRole('button', { name: /create webhook/i }));

    await waitFor(() => {
      expect(apiMock.createWebhook).toHaveBeenCalled();
    });

    // The secret dialog should appear
    expect(await screen.findByText('Webhook Secret')).toBeInTheDocument();
    const secretInput = screen.getByTestId('webhook-secret-value') as HTMLInputElement;
    expect(secretInput.value).toBe('whsec_abc123');
  });
});
