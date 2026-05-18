import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AccountPage from '@web/pages/account/index';
import BillingPage from '@web/pages/billing/index';
import BillingCancelPage from '@web/pages/billing/cancel';
import BillingSuccessPage from '@web/pages/billing/success';
import MagicLinkPage from '@web/pages/auth/magic-link';
import ResetPasswordPage from '@web/pages/auth/reset-password';
import SignupPage from '@web/pages/signup';
import VerifyEmailPage from '@web/pages/auth/verify-email';

const mockBack = vi.fn();
const mockRouter = {
  asPath: '/account',
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

const hostedOnlyRoutes = [
  {
    path: '/account',
    title: 'Hosted Account Pages Live In The Private Distribution',
    Component: AccountPage,
  },
  {
    path: '/billing',
    title: 'Hosted Billing Lives In The Private Distribution',
    Component: BillingPage,
  },
  {
    path: '/billing/cancel',
    title: 'Hosted Billing Redirects Live In The Private Distribution',
    Component: BillingCancelPage,
  },
  {
    path: '/billing/success',
    title: 'Hosted Billing Redirects Live In The Private Distribution',
    Component: BillingSuccessPage,
  },
  {
    path: '/signup',
    title: 'Signup Is Not Part Of The OSS Web Surface',
    Component: SignupPage,
  },
  {
    path: '/auth/magic-link',
    title: 'Magic Link Sign-In Is Not Active Here',
    Component: MagicLinkPage,
  },
  {
    path: '/auth/reset-password',
    title: 'Password Reset Is Not Active Here',
    Component: ResetPasswordPage,
  },
  {
    path: '/auth/verify-email',
    title: 'Email Verification Is Not Active Here',
    Component: VerifyEmailPage,
  },
];

describe('hosted-only placeholder pages', () => {
  beforeEach(() => {
    mockBack.mockReset();
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it.each(hostedOnlyRoutes)(
    '$path points self-host users directly to local auth settings',
    ({ path, title, Component }) => {
      vi.stubEnv('NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE', 'self_host');
      mockRouter.asPath = path;

      render(<Component />);

      expect(screen.getByRole('heading', { level: 1, name: title })).toBeVisible();
      expect(screen.getAllByText(path)).toHaveLength(2);
      expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute(
        'href',
        '/settings/tldw'
      );
      expect(screen.getByTestId('route-placeholder-primary')).toHaveTextContent(
        'Open Local Auth Settings'
      );
      expect(screen.queryByTestId('route-placeholder-open-settings')).not.toBeInTheDocument();
    }
  );

  it.each(hostedOnlyRoutes)(
    '$path keeps the hosted login CTA when running in hosted mode',
    ({ path, Component }) => {
      vi.stubEnv('NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE', 'hosted');
      mockRouter.asPath = path;

      render(<Component />);

      expect(screen.getAllByText(path)).toHaveLength(2);
      expect(screen.getByTestId('route-placeholder-primary')).toHaveAttribute('href', '/login');
      expect(screen.getByTestId('route-placeholder-primary')).toHaveTextContent('Open Login');
      expect(screen.getByTestId('route-placeholder-open-settings')).toHaveAttribute(
        'href',
        '/settings'
      );
    }
  );
});
