/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LogsPage from '../page';
import { api } from '@/lib/api-client';

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

let mockSearchParams = new URLSearchParams();
vi.mock('next/navigation', () => ({
  useSearchParams: () => mockSearchParams,
}));

vi.mock('@/lib/use-url-state', async () => {
  const React = await import('react');
  return {
    useUrlPagination: () => {
      const [page, setPage] = React.useState(1);
      const [pageSize, setPageSize] = React.useState(25);
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
    getSystemLogs: vi.fn(),
  },
}));

type ApiMock = {
  getSystemLogs: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const mockLogsResponse = {
  items: [
    {
      timestamp: '2026-02-17T12:00:00.000Z',
      level: 'ERROR',
      message: 'Primary failure for request',
      logger: 'worker-a',
      request_id: 'req-1',
      org_id: 101,
      user_id: 501,
    },
    {
      timestamp: '2026-02-17T11:59:00.000Z',
      level: 'INFO',
      message: 'Secondary event for same request',
      logger: 'worker-b',
      request_id: 'req-1',
      org_id: 101,
      user_id: 502,
    },
    {
      timestamp: '2026-02-17T11:58:00.000Z',
      level: 'WARNING',
      message: 'Different request activity',
      logger: 'worker-c',
      request_id: 'req-2',
      org_id: 102,
      user_id: 503,
    },
  ],
  total: 3,
};

beforeEach(() => {
  mockSearchParams = new URLSearchParams();
  apiMock.getSystemLogs.mockResolvedValue(mockLogsResponse);
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('LogsPage', () => {
  it('shows regex validation error and avoids API reload with invalid regex', async () => {
    const user = userEvent.setup();
    render(<LogsPage />);

    await screen.findByText('System Logs');
    expect(apiMock.getSystemLogs).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('checkbox', { name: /treat search as regex/i }));
    fireEvent.change(screen.getByLabelText(/^search$/i), { target: { value: '[' } });

    await waitFor(() => {
      expect(screen.getByText(/Regex pattern is invalid:/i)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(apiMock.getSystemLogs).toHaveBeenCalledTimes(1);
    });
  });

  it('supports request-id correlation from request link and row action', async () => {
    const user = userEvent.setup();
    render(<LogsPage />);

    await screen.findByText('Primary failure for request', { selector: 'summary' });
    expect(screen.getByText('Different request activity', { selector: 'summary' })).toBeInTheDocument();

    await user.click(screen.getAllByRole('button', { name: 'req-1' })[0]);

    await waitFor(() => {
      expect(apiMock.getSystemLogs).toHaveBeenLastCalledWith(
        expect.objectContaining({ request_id: 'req-1' }),
        expect.any(Object)
      );
    });

    await waitFor(() => {
      expect(screen.queryByText('Different request activity', { selector: 'summary' })).not.toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /clear correlation filter/i }));
    await screen.findByText('Different request activity', { selector: 'summary' });

    const differentRequestRow = screen.getByText('Different request activity', { selector: 'summary' }).closest('tr');
    expect(differentRequestRow).not.toBeNull();

    await user.click(
      within(differentRequestRow as HTMLElement).getByRole('button', { name: /view correlated logs/i })
    );

    await waitFor(() => {
      expect(apiMock.getSystemLogs).toHaveBeenLastCalledWith(
        expect.objectContaining({ request_id: 'req-2' }),
        expect.any(Object)
      );
    });
    await waitFor(() => {
      expect(screen.queryByText('Primary failure for request', { selector: 'summary' })).not.toBeInTheDocument();
      expect(screen.getByText('Different request activity', { selector: 'summary' })).toBeInTheDocument();
    });
  });

  it('keeps the request id filter in sync with client-side URL changes', async () => {
    const { rerender } = render(<LogsPage />);

    await waitFor(() => {
      expect(apiMock.getSystemLogs).toHaveBeenCalledWith(
        expect.not.objectContaining({ request_id: expect.anything() }),
        expect.any(Object)
      );
    });

    mockSearchParams = new URLSearchParams('request_id=req-2');
    rerender(<LogsPage />);

    await waitFor(() => {
      expect((screen.getByLabelText('Request ID') as HTMLInputElement).value).toBe('req-2');
    });
    await waitFor(() => {
      expect(apiMock.getSystemLogs).toHaveBeenLastCalledWith(
        expect.objectContaining({ request_id: 'req-2' }),
        expect.any(Object)
      );
    });
  });
});
