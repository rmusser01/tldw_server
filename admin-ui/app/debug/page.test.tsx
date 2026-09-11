/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import DebugPage from './page';
import { api } from '@/lib/api-client';

const permissionGuardMock = vi.hoisted(() => vi.fn(({ children }: { children: ReactNode }) => <>{children}</>));
const isSingleUserModeMock = vi.hoisted(() => vi.fn(() => false));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: permissionGuardMock,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    debugResolveApiKey: vi.fn(),
    debugGetBudgetSummary: vi.fn(),
    getUser: vi.fn(),
    debugResolvePermissions: vi.fn(),
  },
}));

vi.mock('@/lib/auth', () => ({
  isSingleUserMode: isSingleUserModeMock,
}));

describe('DebugPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    isSingleUserModeMock.mockReturnValue(false);
  });

  it('requires super-admin or owner access in multi-user mode', () => {
    render(<DebugPage />);

    expect(permissionGuardMock).toHaveBeenCalledWith(
      expect.objectContaining({
        variant: 'route',
        requireAuth: true,
        role: ['super_admin', 'owner'],
      }),
      undefined
    );
  });

  it('allows admin access in single-user mode', () => {
    isSingleUserModeMock.mockReturnValue(true);

    render(<DebugPage />);

    expect(permissionGuardMock).toHaveBeenCalledWith(
      expect.objectContaining({
        variant: 'route',
        requireAuth: true,
        role: ['admin', 'super_admin', 'owner'],
      }),
      undefined
    );
  });

  it('rejects non-numeric user IDs before lookup or permission resolution', async () => {
    const user = userEvent.setup();
    render(<DebugPage />);

    const userIdInputs = screen.getAllByLabelText('User ID').filter(
      (input) => input.id !== 'rl-user-id'
    );
    expect(userIdInputs).toHaveLength(3);

    for (const input of userIdInputs) {
      await user.type(input, '12abc{enter}');
    }

    expect(screen.getAllByText('Enter a valid positive user ID')).toHaveLength(3);
    expect(api.getUser).not.toHaveBeenCalled();
    expect(api.debugResolvePermissions).not.toHaveBeenCalled();
  });
});
