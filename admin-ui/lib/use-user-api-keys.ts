import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api-client';
import { logger } from '@/lib/logger';
import { ApiKey, User } from '@/types';

type UseUserApiKeysOptions = {
  onError?: (message: string) => void;
};

type UseUserApiKeysResult = {
  user: User | null;
  apiKeys: ApiKey[];
  loading: boolean;
  reload: () => Promise<void>;
};

export const useUserApiKeys = (
  userId: string,
  { onError }: UseUserApiKeysOptions = {}
): UseUserApiKeysResult => {
  const [user, setUser] = useState<User | null>(null);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      onError?.('');
      const [userData, keysData] = await Promise.all([
        api.getUser(userId),
        api.getUserApiKeys(userId),
      ]);
      setUser(userData);
      setApiKeys(Array.isArray(keysData) ? keysData : []);
    } catch (err: unknown) {
      logger.error('Failed to load data', { component: 'use-user-api-keys', error: err instanceof Error ? err.message : String(err) });
      const message = err instanceof Error && err.message ? err.message : 'Failed to load data';
      onError?.(message);
    } finally {
      setLoading(false);
    }
  }, [userId, onError]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  return { user, apiKeys, loading, reload: loadData };
};
