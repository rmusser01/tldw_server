import { useCallback, useState } from 'react';
import { logger } from '@/lib/logger';
import type { Watchlist, WatchlistDraft } from './types';

type ConfirmVariant = 'danger' | 'warning' | 'default';
type ConfirmIcon = 'delete' | 'warning' | 'rotate' | 'remove-user' | 'key';

type ConfirmOptions = {
  title: string;
  message: string;
  confirmText?: string;
  variant?: ConfirmVariant;
  icon?: ConfirmIcon;
};

type ConfirmFn = (options: ConfirmOptions) => Promise<boolean>;

export type WatchlistApiClient = {
  createWatchlist: (data: WatchlistDraft) => Promise<unknown>;
  deleteWatchlist: (watchlistId: string) => Promise<unknown>;
};

type UseWatchlistActionsArgs = {
  apiClient: WatchlistApiClient;
  confirm: ConfirmFn;
  setError: (message: string) => void;
  setSuccess: (message: string) => void;
  onReloadRequested: () => void | Promise<void>;
};

export const DEFAULT_WATCHLIST_DRAFT: WatchlistDraft = {
  name: '',
  description: '',
  target: '',
  type: 'resource',
  threshold: 80,
};

const createDefaultWatchlistDraft = (): WatchlistDraft => ({
  ...DEFAULT_WATCHLIST_DRAFT,
});

export const useWatchlistActions = ({
  apiClient,
  confirm,
  setError,
  setSuccess,
  onReloadRequested,
}: UseWatchlistActionsArgs) => {
  const [showCreateWatchlist, setShowCreateWatchlist] = useState(false);
  const [newWatchlist, setNewWatchlist] = useState<WatchlistDraft>(() => createDefaultWatchlistDraft());
  const [deletingWatchlistId, setDeletingWatchlistId] = useState<string | null>(null);

  const handleCreateWatchlist = useCallback(async () => {
    if (!newWatchlist.name || !newWatchlist.target) {
      setError('Name and target are required');
      return;
    }

    try {
      setError('');
      await apiClient.createWatchlist(newWatchlist);
      setSuccess('Watchlist created successfully');
      setShowCreateWatchlist(false);
      setNewWatchlist(createDefaultWatchlistDraft());
      void onReloadRequested();
    } catch (err: unknown) {
      logger.error('Failed to create watchlist', { component: 'useWatchlistActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to create watchlist');
    }
  }, [apiClient, newWatchlist, onReloadRequested, setError, setSuccess]);

  const handleDeleteWatchlist = useCallback(async (watchlist: Watchlist) => {
    const watchlistId = String(watchlist.id);
    if (deletingWatchlistId === watchlistId) return;

    const confirmed = await confirm({
      title: 'Delete Watchlist',
      message: `Delete watchlist "${watchlist.name}"?`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setError('');
      setDeletingWatchlistId(watchlistId);
      await apiClient.deleteWatchlist(watchlist.id);
      setSuccess('Watchlist deleted');
      void onReloadRequested();
    } catch (err: unknown) {
      logger.error('Failed to delete watchlist', { component: 'useWatchlistActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to delete watchlist');
    } finally {
      setDeletingWatchlistId((prev) => (prev === watchlistId ? null : prev));
    }
  }, [apiClient, confirm, deletingWatchlistId, onReloadRequested, setError, setSuccess]);

  return {
    showCreateWatchlist,
    setShowCreateWatchlist,
    newWatchlist,
    setNewWatchlist,
    deletingWatchlistId,
    handleCreateWatchlist,
    handleDeleteWatchlist,
  };
};
