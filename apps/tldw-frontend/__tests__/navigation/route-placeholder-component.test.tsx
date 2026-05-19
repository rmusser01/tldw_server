import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { RoutePlaceholder } from '@web/components/navigation/RoutePlaceholder';
import ConfigRedirectPage from '@web/pages/config';
import ProfileRedirectPage from '@web/pages/profile';

const mockBack = vi.fn();
const mockRouter = {
  asPath: '/connectors/jobs',
  back: mockBack,
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

describe('RoutePlaceholder recovery', () => {
  beforeEach(() => {
    mockBack.mockReset();
    mockRouter.asPath = '/connectors/jobs';
  });

  it('shows route context and fallback actions', () => {
    render(
      <RoutePlaceholder
        title="Connector Jobs Is Coming Soon"
        description="Connector job orchestration is planned for this route."
        plannedPath="/connectors/jobs"
        primaryCtaHref="/connectors"
        primaryCtaLabel="Open Connectors Hub"
      />
    );

    expect(screen.getByRole('heading', { name: 'Connector Jobs Is Coming Soon' })).toBeVisible();
    expect(screen.getAllByText('/connectors/jobs')).toHaveLength(2);
    expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute('href', '/connectors');
    expect(screen.getByTestId('route-placeholder-open-settings')).toHaveAttribute('href', '/settings');
    expect(screen.getByTestId('route-placeholder-go-back')).toBeVisible();
  });

  it('supports keyboard traversal across recovery actions', async () => {
    const user = userEvent.setup();
    render(
      <RoutePlaceholder
        title="Connector Jobs Is Coming Soon"
        description="Connector job orchestration is planned for this route."
        plannedPath="/connectors/jobs"
        primaryCtaHref="/connectors"
        primaryCtaLabel="Open Connectors Hub"
      />
    );

    const openPrimary = screen.getByTestId('route-placeholder-primary');
    const openSettings = screen.getByTestId('route-placeholder-open-settings');
    const goBack = screen.getByTestId('route-placeholder-go-back');

    await user.tab();
    expect(openPrimary).toHaveFocus();

    await user.tab();
    expect(openSettings).toHaveFocus();

    await user.tab();
    expect(goBack).toHaveFocus();
  });

  it('falls back to root CTA and triggers router back action', async () => {
    const user = userEvent.setup();
    mockRouter.asPath = '/profile';

    render(
      <RoutePlaceholder
        title="Profile Page Is Coming Soon"
        description="Dedicated profile management is not yet available on this route."
      />
    );

    expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute('href', '/');
    expect(screen.getByTestId('route-placeholder-primary')).toHaveTextContent('Open Home');
    expect(screen.queryByText('Planned route:')).not.toBeInTheDocument();

    await user.click(screen.getByTestId('route-placeholder-go-back'));
    expect(mockBack).toHaveBeenCalledTimes(1);
  });

  it('suppresses the secondary settings link when the primary CTA already opens settings', async () => {
    const user = userEvent.setup();
    mockRouter.asPath = '/config';

    render(
      <RoutePlaceholder
        title="Configuration Center Is Coming Soon"
        description="Unified configuration workflows are planned for this route."
        plannedPath="/config"
        primaryCtaHref="/settings"
        primaryCtaLabel="Open Settings"
      />
    );

    const primaryCta = screen.getByTestId('route-placeholder-primary');
    const goBack = screen.getByTestId('route-placeholder-go-back');

    expect(primaryCta).toHaveAttribute('href', '/settings');
    expect(screen.queryByTestId('route-placeholder-open-settings')).not.toBeInTheDocument();

    await user.tab();
    expect(primaryCta).toHaveFocus();

    await user.tab();
    expect(goBack).toHaveFocus();
  });

  it('suppresses the secondary settings link when the primary CTA opens a settings child route', () => {
    mockRouter.asPath = '/account';

    render(
      <RoutePlaceholder
        title="Hosted Account Pages Live In The Private Distribution"
        description="The OSS web client does not ship the hosted account surface."
        plannedPath="/account"
        primaryCtaHref="/settings/tldw"
        primaryCtaLabel="Open Local Auth Settings"
      />
    );

    expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute(
      'href',
      '/settings/tldw'
    );
    expect(screen.queryByTestId('route-placeholder-open-settings')).not.toBeInTheDocument();
  });

  it('profile page names its route context and opens settings as the primary recovery path', () => {
    mockRouter.asPath = '/profile';

    render(<ProfileRedirectPage />);

    expect(screen.getByRole('heading', { level: 1, name: 'Profile Page Is Coming Soon' })).toBeVisible();
    expect(screen.getAllByText('/profile')).toHaveLength(2);
    expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute('href', '/settings');
    expect(screen.getByTestId('route-placeholder-primary')).toHaveTextContent('Open Settings');
    expect(screen.queryByTestId('route-placeholder-open-settings')).not.toBeInTheDocument();
  });

  it('config page names its route context and opens settings as the primary recovery path', () => {
    mockRouter.asPath = '/config';

    render(<ConfigRedirectPage />);

    expect(screen.getByRole('heading', { level: 1, name: 'Configuration Center Is Coming Soon' })).toBeVisible();
    expect(screen.getAllByText('/config')).toHaveLength(2);
    expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute('href', '/settings');
    expect(screen.getByTestId('route-placeholder-primary')).toHaveTextContent('Open Settings');
    expect(screen.queryByTestId('route-placeholder-open-settings')).not.toBeInTheDocument();
  });
});
