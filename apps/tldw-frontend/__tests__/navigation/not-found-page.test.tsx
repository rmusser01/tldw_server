import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import NotFoundPage from '@web/pages/404';

const mockPush = vi.fn();
const mockBack = vi.fn();
const mockRouter = {
  asPath: '/missing-route?foo=bar',
  push: mockPush,
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

describe('404 recovery controls', () => {
  beforeEach(() => {
    mockPush.mockReset();
    mockBack.mockReset();
    mockRouter.asPath = '/missing-route?foo=bar';
  });

  it('shows route context and provides recovery destinations', () => {
    render(<NotFoundPage />);

    expect(screen.getByRole('heading', { name: 'We could not find that route' })).toBeVisible();
    expect(screen.getByText('/missing-route?foo=bar')).toBeVisible();
    expect(screen.getByTestId('not-found-open-home')).toHaveTextContent('Open Home');
    expect(screen.getByTestId('not-found-open-research')).toBeVisible();
    expect(screen.getByTestId('not-found-open-media')).toBeVisible();
    expect(screen.getByTestId('not-found-open-settings')).toBeVisible();
    expect(screen.getByTestId('not-found-go-back')).toBeVisible();
  });

  it('supports keyboard traversal across recovery actions', async () => {
    const user = userEvent.setup();
    render(<NotFoundPage />);

    const openHome = screen.getByTestId('not-found-open-home');
    const openResearch = screen.getByTestId('not-found-open-research');
    const openMedia = screen.getByTestId('not-found-open-media');
    const openSettings = screen.getByTestId('not-found-open-settings');
    const goBack = screen.getByTestId('not-found-go-back');

    await user.tab();
    expect(openHome).toHaveFocus();

    await user.tab();
    expect(openResearch).toHaveFocus();

    await user.tab();
    expect(openMedia).toHaveFocus();

    await user.tab();
    expect(openSettings).toHaveFocus();

    await user.tab();
    expect(goBack).toHaveFocus();
  });

  it('triggers primary recovery actions', async () => {
    const user = userEvent.setup();
    render(<NotFoundPage />);

    await user.click(screen.getByTestId('not-found-open-home'));
    expect(mockPush).toHaveBeenCalledWith('/');

    await user.click(screen.getByTestId('not-found-go-back'));
    expect(mockBack).toHaveBeenCalledTimes(1);
  });
});
