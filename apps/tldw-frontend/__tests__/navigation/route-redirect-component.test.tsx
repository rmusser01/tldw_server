import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { RouteRedirect } from '@web/components/navigation/RouteRedirect';

const mockReplace = vi.fn();
const mockPrefetch = vi.fn().mockResolvedValue(undefined);
const mockTrackRouteAliasRedirect = vi.fn().mockResolvedValue(undefined);
const mockRouter = {
  asPath: '/search?q=rag#examples',
  pathname: '/search',
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

vi.mock('@/utils/route-alias-telemetry', () => ({
  trackRouteAliasRedirect: (...args: unknown[]) => mockTrackRouteAliasRedirect(...args),
}));

describe('RouteRedirect telemetry', () => {
  beforeEach(() => {
    mockReplace.mockReset();
    mockPrefetch.mockReset();
    mockPrefetch.mockResolvedValue(undefined);
    mockTrackRouteAliasRedirect.mockReset();
    mockTrackRouteAliasRedirect.mockResolvedValue(undefined);
    mockRouter.asPath = '/search?q=rag#examples';
    mockRouter.pathname = '/search';
  });

  it('tracks alias redirect payload before navigation', async () => {
    render(<RouteRedirect to="/knowledge" preserveParams />);

    await waitFor(() => {
      expect(mockTrackRouteAliasRedirect).toHaveBeenCalledWith({
        sourcePath: '/search?q=rag#examples',
        destinationPath: '/knowledge?q=rag#examples',
        preserveParams: true,
      });
    });

    expect(mockPrefetch).toHaveBeenCalledWith('/knowledge?q=rag#examples');
    expect(mockReplace).toHaveBeenCalledWith('/knowledge?q=rag#examples');
    expect(mockPrefetch.mock.invocationCallOrder[0]).toBeLessThan(
      mockReplace.mock.invocationCallOrder[0]
    );
  });

  it('uses pathname fallback when asPath is unavailable', async () => {
    mockRouter.asPath = '';
    mockRouter.pathname = '/claims-review';

    render(<RouteRedirect to="/content-review" preserveParams={false} />);

    await waitFor(() => {
      expect(mockTrackRouteAliasRedirect).toHaveBeenCalledWith({
        sourcePath: '/claims-review',
        destinationPath: '/content-review',
        preserveParams: false,
      });
    });

    expect(mockPrefetch).toHaveBeenCalledWith('/content-review');
    expect(mockReplace).toHaveBeenCalledWith('/content-review');
  });

  it('keeps redirect recovery actions keyboard-focusable in predictable order', async () => {
    const user = userEvent.setup();
    render(<RouteRedirect to="/knowledge" preserveParams={false} />);

    const openUpdatedPage = screen.getByTestId('route-redirect-open-updated-page');
    const goToChat = screen.getByTestId('route-redirect-go-chat');
    const openSettings = screen.getByTestId('route-redirect-open-settings');

    await user.tab();
    expect(openUpdatedPage).toHaveFocus();

    await user.tab();
    expect(goToChat).toHaveFocus();

    await user.tab();
    expect(openSettings).toHaveFocus();
  });

  it('redirects even when route prefetch does not settle', async () => {
    mockRouter.asPath = '/audio';
    mockRouter.pathname = '/audio';
    mockPrefetch.mockImplementation(() => new Promise<void>(() => {}));

    render(<RouteRedirect to="/speech" preserveParams={false} />);

    await waitFor(() => {
      expect(mockTrackRouteAliasRedirect).toHaveBeenCalledWith({
        sourcePath: '/audio',
        destinationPath: '/speech',
        preserveParams: false,
      });
    });

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/speech');
    });
  });

  it('redirects even when alias telemetry storage does not settle', async () => {
    mockRouter.asPath = '/audio';
    mockRouter.pathname = '/audio';
    mockTrackRouteAliasRedirect.mockImplementation(() => new Promise<void>(() => {}));

    render(<RouteRedirect to="/speech" preserveParams={false} />);

    await waitFor(() => {
      expect(mockTrackRouteAliasRedirect).toHaveBeenCalledWith({
        sourcePath: '/audio',
        destinationPath: '/speech',
        preserveParams: false,
      });
    });

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/speech');
    });
  });
});
