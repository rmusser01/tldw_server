'use client';

import { useCallback, useMemo, useState, useSyncExternalStore } from 'react';
import { useUrlState } from '@/lib/use-url-state';
import { getScopedItem, setScopedItem } from '@/lib/scoped-storage';
import { logger } from '@/lib/logger';

export type SavedUserView = {
  id: string;
  name: string;
  query: string;
};

export type UserStatusFilter = 'all' | 'active' | 'inactive';
export type UserVerifiedFilter = 'all' | 'verified' | 'unverified';
export type UserMfaFilter = 'all' | 'enabled' | 'disabled';

const SAVED_VIEWS_STORAGE_KEY = 'admin_users_saved_views';
const savedViewsListeners = new Set<() => void>();
let cachedSavedViewsRaw: string | null = null;
let cachedSavedViews: SavedUserView[] = [];

const readSavedViews = (): SavedUserView[] => {
  if (typeof window === 'undefined') return [];
  let stored: string | null = null;
  try {
    stored = getScopedItem(SAVED_VIEWS_STORAGE_KEY);
    if (stored === cachedSavedViewsRaw) {
      return cachedSavedViews;
    }
    if (!stored) {
      cachedSavedViewsRaw = null;
      cachedSavedViews = [];
      return cachedSavedViews;
    }
    const parsed = JSON.parse(stored);
    cachedSavedViewsRaw = stored;
    cachedSavedViews = Array.isArray(parsed) ? (parsed as SavedUserView[]) : [];
    return cachedSavedViews;
  } catch (error) {
    logger.warn('Failed to load saved user views', { component: 'useUserFilters', error: error instanceof Error ? error.message : String(error) });
    cachedSavedViewsRaw = stored;
    cachedSavedViews = [];
    return cachedSavedViews;
  }
};

const emitSavedViewsChange = () => {
  savedViewsListeners.forEach((listener) => listener());
};

const subscribeToSavedViews = (listener: () => void) => {
  savedViewsListeners.add(listener);

  if (typeof window === 'undefined') {
    return () => {
      savedViewsListeners.delete(listener);
    };
  }

  const handleStorage = (event: StorageEvent) => {
    if (
      event.key === null
      || event.key === SAVED_VIEWS_STORAGE_KEY
      || event.key.endsWith(SAVED_VIEWS_STORAGE_KEY)
    ) {
      listener();
    }
  };

  window.addEventListener('storage', handleStorage);

  return () => {
    savedViewsListeners.delete(listener);
    window.removeEventListener('storage', handleStorage);
  };
};

interface UseUserFiltersOptions {
  resetPagination: () => void;
}

export function useUserFilters({ resetPagination }: UseUserFiltersOptions) {
  const [showSaveViewDialog, setShowSaveViewDialog] = useState(false);
  const [saveViewName, setSaveViewName] = useState('');
  const [saveViewError, setSaveViewError] = useState('');
  const [searchQuery, setSearchQuery] = useUrlState<string>('q', { defaultValue: '' });
  const [statusFilter, setStatusFilter] = useUrlState<UserStatusFilter>('status', { defaultValue: 'all' });
  const [verifiedFilter, setVerifiedFilter] = useUrlState<UserVerifiedFilter>('verified', { defaultValue: 'all' });
  const [mfaFilter, setMfaFilter] = useUrlState<UserMfaFilter>('mfa', { defaultValue: 'all' });
  const savedViews = useSyncExternalStore(subscribeToSavedViews, readSavedViews, () => []);

  const persistSavedViews = useCallback((views: SavedUserView[]) => {
    if (typeof window === 'undefined') return;
    try {
      const serialized = JSON.stringify(views);
      cachedSavedViewsRaw = serialized;
      cachedSavedViews = views;
      setScopedItem(SAVED_VIEWS_STORAGE_KEY, serialized);
      emitSavedViewsChange();
    } catch (error) {
      logger.warn('Failed to persist saved user views', { component: 'useUserFilters', error: error instanceof Error ? error.message : String(error) });
    }
  }, []);

  const activeViewId = useMemo(() => {
    const match = savedViews.find((view) => view.query === (searchQuery || ''));
    return match ? match.id : '';
  }, [savedViews, searchQuery]);

  const hasActiveFilters = (statusFilter || 'all') !== 'all'
    || (verifiedFilter || 'all') !== 'all'
    || (mfaFilter || 'all') !== 'all';

  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value || undefined);
    resetPagination();
  }, [resetPagination, setSearchQuery]);

  const handleStatusFilterChange = useCallback((value: UserStatusFilter) => {
    setStatusFilter(value === 'all' ? undefined : value);
    resetPagination();
  }, [resetPagination, setStatusFilter]);

  const handleVerifiedFilterChange = useCallback((value: UserVerifiedFilter) => {
    setVerifiedFilter(value === 'all' ? undefined : value);
    resetPagination();
  }, [resetPagination, setVerifiedFilter]);

  const handleMfaFilterChange = useCallback((value: UserMfaFilter) => {
    setMfaFilter(value === 'all' ? undefined : value);
    resetPagination();
  }, [resetPagination, setMfaFilter]);

  const handleClearFilters = useCallback(() => {
    setStatusFilter(undefined);
    setVerifiedFilter(undefined);
    setMfaFilter(undefined);
    resetPagination();
  }, [resetPagination, setMfaFilter, setStatusFilter, setVerifiedFilter]);

  const handleApplySavedView = useCallback((viewId: string) => {
    if (!viewId) {
      setSearchQuery(undefined);
      resetPagination();
      return;
    }
    const view = savedViews.find((item) => item.id === viewId);
    if (!view) return;
    setSearchQuery(view.query || undefined);
    resetPagination();
  }, [resetPagination, savedViews, setSearchQuery]);

  const clearSaveViewForm = useCallback(() => {
    setSaveViewError('');
    setSaveViewName('');
  }, []);

  const saveCurrentView = useCallback(() => {
    const name = saveViewName.trim();
    if (!name) {
      setSaveViewError('Provide a name for this view.');
      return { ok: false as const };
    }

    const nextView: SavedUserView = {
      id: `${Date.now()}`,
      name,
      query: searchQuery || '',
    };
    persistSavedViews([nextView, ...savedViews]);
    clearSaveViewForm();
    setShowSaveViewDialog(false);
    return { ok: true as const, view: nextView };
  }, [clearSaveViewForm, persistSavedViews, saveViewName, savedViews, searchQuery]);

  const removeSavedView = useCallback((viewId: string) => {
    const view = savedViews.find((item) => item.id === viewId);
    if (!view) return null;
    const next = savedViews.filter((item) => item.id !== viewId);
    persistSavedViews(next);
    return view;
  }, [persistSavedViews, savedViews]);

  return {
    savedViews,
    showSaveViewDialog,
    saveViewName,
    saveViewError,
    searchQuery,
    statusFilter,
    verifiedFilter,
    mfaFilter,
    activeViewId,
    hasActiveFilters,
    setShowSaveViewDialog,
    setSaveViewName,
    clearSaveViewForm,
    handleSearchChange,
    handleStatusFilterChange,
    handleVerifiedFilterChange,
    handleMfaFilterChange,
    handleClearFilters,
    handleApplySavedView,
    saveCurrentView,
    removeSavedView,
  };
}
