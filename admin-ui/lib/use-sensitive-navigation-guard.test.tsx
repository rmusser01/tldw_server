import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useSensitiveNavigationGuard } from './use-sensitive-navigation-guard';

describe('useSensitiveNavigationGuard fallback history protection', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, '', '/');
  });

  it('restores the protected URL after fallback Back navigation', () => {
    Object.defineProperty(window, 'navigation', {
      configurable: true,
      value: undefined,
    });
    window.history.replaceState({ page: 'webhooks' }, '', '/webhooks');
    const rawPushState = window.history.pushState.bind(window.history);
    const onBlocked = vi.fn();
    const { unmount } = renderHook(() => {
      useSensitiveNavigationGuard(true, onBlocked);
    });

    act(() => {
      rawPushState({ page: 'incidents' }, '', '/incidents');
      window.dispatchEvent(new PopStateEvent('popstate', {
        state: { page: 'incidents' },
      }));
    });

    expect(window.location.pathname).toBe('/webhooks');
    expect(onBlocked).toHaveBeenCalledOnce();
    unmount();
  });
});
