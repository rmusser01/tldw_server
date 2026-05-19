/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { ResponsiveLayout } from './ResponsiveLayout';

let mockPathname = '/';
const isSingleUserModeMock = vi.hoisted(() => vi.fn(() => false));

vi.mock('next/link', () => ({
  default: ({ href, children, ...props }: { href: string; children: ReactNode }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

vi.mock('next/navigation', () => ({
  usePathname: () => mockPathname,
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock('@/lib/auth', () => ({
  logout: vi.fn().mockResolvedValue(undefined),
  isSingleUserMode: isSingleUserModeMock,
}));

vi.mock('@/components/PermissionGuard', () => ({
  usePermissions: () => ({
    user: {
      id: 1,
      uuid: 'user-1',
      username: 'Admin',
      email: 'admin@example.com',
      role: 'admin',
      is_active: true,
      is_verified: true,
      storage_quota_mb: 100,
      storage_used_mb: 10,
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    },
    hasPermission: () => true,
    hasRole: (role: string | string[]) => {
      const requiredRoles = Array.isArray(role) ? role : [role];
      return requiredRoles.includes('admin');
    },
    loading: false,
    refresh: async () => {},
  }),
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    error: vi.fn(),
  }),
}));

vi.mock('@/components/ThemeToggle', () => ({
  ThemeToggle: () => <div data-testid="theme-toggle" />,
}));

vi.mock('@/components/OrgContextSwitcher', () => ({
  OrgContextSwitcher: () => <div data-testid="org-switcher" />,
  OrgContextBanner: () => null,
}));

afterEach(() => {
  cleanup();
  mockPathname = '/';
  isSingleUserModeMock.mockReturnValue(false);
  localStorage.clear();
});

beforeEach(() => {
  isSingleUserModeMock.mockReturnValue(false);
});

describe('ResponsiveLayout mobile navigation', () => {
  it('renders a skip link targeting main content', () => {
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    const skipLink = screen.getByRole('link', { name: 'Skip to main content' });
    expect(skipLink.getAttribute('href')).toBe('#main-content');

    const main = screen.getByRole('main');
    expect(main.getAttribute('id')).toBe('main-content');
    expect(main.getAttribute('tabindex')).toBe('-1');
  });

  it('opens as a modal dialog and closes on Escape', () => {
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    expect(screen.getByRole('dialog', { name: 'Main navigation' })).toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Main navigation' })).toBeNull();
  });

  it('shows no-results state when nav search has no match', () => {
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    const dialog = screen.getByRole('dialog', { name: 'Main navigation' });

    fireEvent.change(within(dialog).getByLabelText('Find navigation page'), {
      target: { value: 'no-such-admin-route' },
    });

    expect(within(dialog).getByText('No navigation matches your search.')).toBeInTheDocument();
  });

  it('renders breadcrumbs for nested routes and updates document title', () => {
    mockPathname = '/users/123';
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    const breadcrumbs = screen.getByTestId('breadcrumbs-nav');
    expect(within(breadcrumbs).getByRole('link', { name: 'Users' }).getAttribute('href')).toBe('/users');
    expect(within(breadcrumbs).getByText('User 123')).toBeTruthy();
    expect(document.title).toBe('User 123 | Admin Dashboard');
  });

  it('moves focus to main content on route changes', async () => {
    const { rerender } = render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    const previousFocus = document.createElement('button');
    document.body.appendChild(previousFocus);
    previousFocus.focus();

    mockPathname = '/users';
    rerender(
      <ResponsiveLayout>
        <div>Users content</div>
      </ResponsiveLayout>
    );

    await waitFor(() => {
      expect(document.activeElement).toBe(screen.getByRole('main'));
    });

    previousFocus.remove();
  });

  it('shows shortcuts tip banner once and persists dismissal', () => {
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    expect(screen.getByTestId('shortcuts-tip-banner')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByTestId('shortcuts-tip-banner')).toBeNull();

    cleanup();

    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );
    expect(screen.queryByTestId('shortcuts-tip-banner')).toBeNull();
  });

  it('shows sidebar keyboard hint text', () => {
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    expect(screen.getByTestId('sidebar-shortcuts-hint').textContent).toContain('Press ? for shortcuts');
  });

  it('hides debug navigation from plain admins in multi-user mode', () => {
    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    expect(screen.queryByRole('link', { name: 'Debug' })).toBeNull();
  });

  it('shows debug navigation for single-user admins', () => {
    isSingleUserModeMock.mockReturnValue(true);

    render(
      <ResponsiveLayout>
        <div>Page content</div>
      </ResponsiveLayout>
    );

    expect(screen.getByRole('link', { name: 'Debug' })).toBeInTheDocument();
  });
});
