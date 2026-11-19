import { useCallback, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@web/lib/api';

type BackendsMap = Record<string, { available: boolean } | boolean>;
type RagCapabilities = Record<string, unknown>;

interface UseVlmBackendsResult {
  loading: boolean;
  error: string | null;
  endpoint: string | null;
  backends: Record<string, boolean> | null;
  reload: () => void;
}

/** Shared capability discovery query; other RAG capability consumers reuse this key. */
const CAPABILITIES_QUERY_KEY = ['rag/capabilities'] as const;
const BACKENDS_QUERY_PREFIX = 'vlm-backends';
/** WP-54: mounts inside this window reuse cached reads instead of refetching. */
const STALE_TIME_MS = 15 * 60_000;
const DEFAULT_BACKENDS_ENDPOINT = '/api/v1/rag/vlm/backends';

const resolveBackendsEndpoint = (caps: RagCapabilities | undefined): string => {
  const features = caps?.features as Record<string, unknown> | undefined;
  const lateChunking = features?.vlm_late_chunking as Record<string, unknown> | undefined;
  const ep = lateChunking?.backends_endpoint;
  return typeof ep === 'string' && ep ? ep : DEFAULT_BACKENDS_ENDPOINT;
};

// Normalize endpoint to be relative to api base (remove /api/v1 prefix if present)
const toBackendsPath = (endpoint: string): string => {
  const rel = '/' + endpoint.replace(/^\/?api\/?v1\//, '/').replace(/^\/+/, '');
  return rel.startsWith('/rag/') ? rel : `/rag/${rel.replace(/^rag\//, '')}`;
};

const toErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Failed to load VLM backends';

/**
 * Fetches RAG capabilities to discover the VLM backends endpoint, then fetches
 * and returns the available/unavailable map. Both reads are cached react-query
 * queries on the app-wide QueryClient; use `reload` to force fresh reads.
 */
export function useVlmBackends(): UseVlmBackendsResult {
  const queryClient = useQueryClient();

  const capabilitiesQuery = useQuery({
    queryKey: CAPABILITIES_QUERY_KEY,
    queryFn: async (): Promise<RagCapabilities> => apiClient.get('/rag/capabilities'),
    staleTime: STALE_TIME_MS,
    gcTime: STALE_TIME_MS,
    retry: false,
  });

  // A 200-with-null body keeps the pre-query behavior: derive from {} so the
  // default backends endpoint still loads instead of a silent empty state.
  const endpoint = capabilitiesQuery.data !== undefined
    ? resolveBackendsEndpoint(capabilitiesQuery.data ?? {})
    : null;

  const backendsQuery = useQuery({
    queryKey: [BACKENDS_QUERY_PREFIX, endpoint],
    queryFn: async (): Promise<{ backends?: BackendsMap }> =>
      apiClient.get(toBackendsPath(endpoint ?? DEFAULT_BACKENDS_ENDPOINT)),
    enabled: endpoint !== null,
    staleTime: STALE_TIME_MS,
    gcTime: STALE_TIME_MS,
    retry: false,
  });

  const loading =
    capabilitiesQuery.isPending || (endpoint !== null && backendsQuery.isPending);
  const error = capabilitiesQuery.isError
    ? toErrorMessage(capabilitiesQuery.error)
    : backendsQuery.isError
      ? toErrorMessage(backendsQuery.error)
      : null;

  const backends = useMemo(() => {
    const raw = backendsQuery.data;
    if (!raw) return null;
    const out: Record<string, boolean> = {};
    for (const [k, v] of Object.entries(raw.backends || {})) {
      out[k] = typeof v === 'object' && v !== null ? !!v.available : !!v;
    }
    return out;
  }, [backendsQuery.data]);

  const reload = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: CAPABILITIES_QUERY_KEY });
    void queryClient.invalidateQueries({ queryKey: [BACKENDS_QUERY_PREFIX] });
  }, [queryClient]);

  return { loading, error, endpoint, backends, reload };
}

export default useVlmBackends;
