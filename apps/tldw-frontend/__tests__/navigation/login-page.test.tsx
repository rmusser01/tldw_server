import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import LoginPage from '@web/pages/login';

const mockReplace = vi.fn();
const mockPrefetch = vi.fn().mockResolvedValue(undefined);
const mockTrackRouteAliasRedirect = vi.fn().mockResolvedValue(undefined);
const mockRouter = {
  asPath: '/login?next=%2Faccount',
  pathname: '/login',
  prefetch: mockPrefetch,
  replace: mockReplace,
};

vi.mock('next/router', () => ({
  useRouter: () => mockRouter,
}));

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock('next/dynamic', async () => {
  const ReactModule = await vi.importActual<typeof import('react')>('react');

  return {
    default: () => {
      const DynamicTldwSettings = () =>
        ReactModule.createElement(
          'section',
          null,
          ReactModule.createElement('h1', null, 'Sign in'),
          ReactModule.createElement('button', { type: 'button' }, 'Sign in')
        );
      return DynamicTldwSettings;
    },
  };
});

vi.mock('@/utils/route-alias-telemetry', () => ({
  trackRouteAliasRedirect: (...args: unknown[]) => mockTrackRouteAliasRedirect(...args),
}));

describe('LoginPage deployment policy', () => {
  beforeEach(() => {
    mockReplace.mockReset();
    mockPrefetch.mockReset();
    mockPrefetch.mockResolvedValue(undefined);
    mockTrackRouteAliasRedirect.mockReset();
    mockTrackRouteAliasRedirect.mockResolvedValue(undefined);
    mockRouter.asPath = '/login?next=%2Faccount';
    mockRouter.pathname = '/login';
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('renders an explicit self-host redirect panel instead of a blank login page', async () => {
    vi.stubEnv('NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE', 'self_host');

    render(<LoginPage />);

    expect(
      screen.getByRole('heading', { level: 1, name: 'Login is managed in local settings' })
    ).toBeVisible();
    expect(screen.getByText('/login?next=%2Faccount')).toBeVisible();
    expect(screen.getByText('/settings/tldw?next=%2Faccount')).toBeVisible();
    expect(screen.getByTestId('route-redirect-open-updated-page')).toHaveAttribute(
      'href',
      '/settings/tldw?next=%2Faccount'
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/settings/tldw?next=%2Faccount');
    });
  });

  it('keeps hosted login on the auth form', async () => {
    vi.stubEnv('NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE', 'hosted');

    render(<LoginPage />);

    expect(screen.queryByTestId('route-redirect-panel')).not.toBeInTheDocument();
    expect(await screen.findByRole('heading', { level: 1, name: 'Sign in' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeVisible();
    expect(mockReplace).not.toHaveBeenCalled();
  });
});
