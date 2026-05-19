/* @vitest-environment jsdom */
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useUserFilters } from './use-user-filters';
import { _scopedKeyForTesting, getScopedItem, setScopedItem } from '@/lib/scoped-storage';

vi.mock('@/lib/use-url-state', async () => {
  const React = await import('react');
  return {
    useUrlState: (_key: string, options?: { defaultValue?: unknown }) => {
      const [value, setValue] = React.useState(options?.defaultValue);
      return [value, setValue];
    },
  };
});

describe('useUserFilters', () => {
  const resetPagination = vi.fn();

  beforeEach(() => {
    localStorage.clear();
    resetPagination.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('persists, reapplies, and removes saved views', async () => {
    const { result } = renderHook(() => useUserFilters({ resetPagination }));

    act(() => {
      result.current.handleSearchChange('bob');
    });

    expect(resetPagination).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.setSaveViewName('Bob only');
    });

    let savedViewId = '';
    act(() => {
      const saveResult = result.current.saveCurrentView();
      expect(saveResult.ok).toBe(true);
      if (saveResult.ok) {
        savedViewId = saveResult.view.id;
      }
    });

    await waitFor(() => {
      expect(result.current.savedViews).toHaveLength(1);
      expect(result.current.activeViewId).toBe(savedViewId);
    });
    expect(getScopedItem('admin_users_saved_views')).toContain('Bob only');

    act(() => {
      result.current.handleSearchChange('');
      result.current.handleApplySavedView(savedViewId);
    });

    await waitFor(() => {
      expect(result.current.searchQuery).toBe('bob');
    });

    act(() => {
      const removed = result.current.removeSavedView(savedViewId);
      expect(removed).toMatchObject({ name: 'Bob only', query: 'bob' });
    });

    await waitFor(() => {
      expect(result.current.savedViews).toHaveLength(0);
      expect(result.current.activeViewId).toBe('');
    });
    expect(getScopedItem('admin_users_saved_views')).toBe('[]');
  });

  it('reacts to scoped storage events from another tab', async () => {
    document.cookie = 'admin_session=abcdef123456';
    const { result } = renderHook(() => useUserFilters({ resetPagination }));

    act(() => {
      setScopedItem('admin_users_saved_views', JSON.stringify([
        {
          id: 'shared-view',
          name: 'Shared view',
          query: 'alice',
          role: '',
          status: '',
          verification: '',
          orgId: '',
          mfaEnabled: '',
        },
      ]));
      window.dispatchEvent(new StorageEvent('storage', {
        key: _scopedKeyForTesting('admin_users_saved_views'),
      }));
    });

    await waitFor(() => {
      expect(result.current.savedViews).toEqual([
        expect.objectContaining({ id: 'shared-view', name: 'Shared view', query: 'alice' }),
      ]);
    });
  });
});
