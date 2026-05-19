/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AuditPage from '../page';
import { api } from '@/lib/api-client';
import { formatAxeViolations, getCriticalAndSeriousAxeViolations } from '@/test-utils/axe';
import { getScopedItem, setScopedItem } from '@/lib/scoped-storage';

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

vi.mock('@/components/OrgContextSwitcher', () => ({
  useOrgContext: () => ({ selectedOrg: null }),
  OrgContextSwitcher: () => <div data-testid="org-switcher" />,
}));

vi.mock('next/navigation', () => ({
  usePathname: () => '/audit',
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/use-url-state', async () => {
  const React = await import('react');
  return {
    useUrlMultiState: <T extends Record<string, string>>(defaults: T) => {
      const [value, setValue] = React.useState<T>(defaults);
      const update = (updates: Partial<T>) => {
        setValue((previous) => ({ ...previous, ...updates }));
      };
      const clear = () => setValue(defaults);
      return [value, update, clear] as const;
    },
    useUrlPagination: () => {
      const [page, setPage] = React.useState(1);
      const [pageSize, setPageSize] = React.useState(20);
      return {
        page,
        pageSize,
        setPage,
        setPageSize,
        resetPagination: () => setPage(1),
      };
    },
  };
});

vi.mock('@/lib/api-client', () => ({
  api: {
    getAuditLogs: vi.fn(),
    testNotification: vi.fn(),
    getUsersPage: vi.fn(),
  },
}));

type ApiMock = {
  getAuditLogs: ReturnType<typeof vi.fn>;
  testNotification: ReturnType<typeof vi.fn>;
  getUsersPage: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;
let createObjectURLMock: ReturnType<typeof vi.fn>;
let revokeObjectURLMock: ReturnType<typeof vi.fn>;
let anchorClickSpy: ReturnType<typeof vi.spyOn>;

const baseEntry = {
  id: 'evt-1',
  timestamp: '2026-02-17T10:00:00Z',
  user_id: 42,
  action: 'user.create',
  resource: 'user',
  details: { username: 'alice' },
};

beforeEach(() => {
  window.localStorage.clear();
  toastSuccessMock.mockReset();
  toastErrorMock.mockReset();
  createObjectURLMock = vi.fn(() => 'blob:compliance-report');
  revokeObjectURLMock = vi.fn();
  Object.defineProperty(window.URL, 'createObjectURL', {
    configurable: true,
    writable: true,
    value: createObjectURLMock,
  });
  Object.defineProperty(window.URL, 'revokeObjectURL', {
    configurable: true,
    writable: true,
    value: revokeObjectURLMock,
  });
  anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

  apiMock.getAuditLogs.mockResolvedValue({
    entries: [baseEntry],
    total: 1,
    limit: 20,
    offset: 0,
  });
  apiMock.testNotification.mockResolvedValue({});
  apiMock.getUsersPage.mockResolvedValue({
    items: [{ id: 1, username: 'admin', role: 'admin' }],
    total: 1,
    limit: 200,
    page: 1,
    pages: 1,
  });
});

afterEach(() => {
  cleanup();
  anchorClickSpy.mockRestore();
  vi.resetAllMocks();
});

describe('AuditPage', () => {
  it('has no critical/serious axe violations in the default state', async () => {
    const { container } = render(<AuditPage />);
    await screen.findByText('Audit Logs');

    const violations = await getCriticalAndSeriousAxeViolations(container);
    expect(violations, formatAxeViolations(violations)).toEqual([]);
  });

  it('supports saved search save, apply, and delete', async () => {
    const user = userEvent.setup();
    render(<AuditPage />);

    expect(await screen.findByText('Audit Logs')).toBeInTheDocument();

    await user.type(screen.getByLabelText('Action (exact or prefix*)'), 'user.create');
    await user.type(screen.getByLabelText('Saved search name'), 'Create events');
    await user.click(screen.getByRole('button', { name: 'Save Current Filters' }));

    await waitFor(() => {
      const persisted = getScopedItem('admin.audit.saved-searches.v1');
      expect(persisted).toContain('Create events');
      expect(persisted).toContain('user.create');
    });

    await user.click(screen.getByRole('button', { name: 'Clear Filters' }));
    await user.click(screen.getByRole('button', { name: 'Create events' }));

    await waitFor(() => {
      expect(apiMock.getAuditLogs).toHaveBeenLastCalledWith(
        expect.objectContaining({
          action: 'user.create',
        })
      );
    });

    await user.click(screen.getByRole('button', { name: 'Delete saved search Create events' }));
    expect(screen.queryByRole('button', { name: 'Create events' })).not.toBeInTheDocument();
  });

  it('alerts on pattern when enabled saved search finds a new event', async () => {
    const savedSearches = [
      {
        id: 'saved-1',
        name: 'Failed logins',
        filters: {
          user: '',
          action: 'login.failed',
          resource: '',
          start: '',
          end: '',
        },
        alertOnPattern: true,
        createdAt: '2026-02-17T09:00:00Z',
        updatedAt: '2026-02-17T09:00:00Z',
        lastMatchedEventId: 'evt-1',
      },
    ];
    setScopedItem('admin.audit.saved-searches.v1', JSON.stringify(savedSearches));

    apiMock.getAuditLogs.mockImplementation(async (params?: Record<string, string>) => {
      if (params?.action === 'login.failed') {
        return {
          entries: [
            {
              id: 'evt-2',
              timestamp: '2026-02-17T11:00:00Z',
              user_id: 99,
              action: 'login.failed',
              resource: 'auth',
              details: {},
            },
          ],
          total: 2,
          limit: 1,
          offset: 0,
        };
      }
      return {
        entries: [baseEntry],
        total: 1,
        limit: 20,
        offset: 0,
      };
    });

    render(<AuditPage />);

    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalledWith(
        'Audit pattern matched',
        'Failed logins found new matching events (2).'
      );
    });
    expect(apiMock.testNotification).toHaveBeenCalled();
  });

  it('generates compliance report HTML download for selected date range and type', async () => {
    const user = userEvent.setup();
    render(<AuditPage />);

    expect(await screen.findByText('Compliance Reports')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Report Type'), 'access_review');
    await user.clear(screen.getByLabelText('Report Start Date'));
    await user.type(screen.getByLabelText('Report Start Date'), '2026-02-01');
    await user.clear(screen.getByLabelText('Report End Date'));
    await user.type(screen.getByLabelText('Report End Date'), '2026-02-17');
    await user.click(screen.getByRole('button', { name: 'Generate Compliance Report' }));

    await waitFor(() => {
      expect(apiMock.getAuditLogs).toHaveBeenCalledWith(
        expect.objectContaining({
          start: '2026-02-01',
          end: '2026-02-17',
          limit: '5000',
          offset: '0',
        })
      );
    });
    expect(createObjectURLMock).toHaveBeenCalled();
    expect(anchorClickSpy).toHaveBeenCalled();
    expect(revokeObjectURLMock).toHaveBeenCalledWith('blob:compliance-report');
    expect(toastSuccessMock).toHaveBeenCalledWith(
      'Compliance report generated',
      expect.stringContaining('Access Review')
    );
  });
});
