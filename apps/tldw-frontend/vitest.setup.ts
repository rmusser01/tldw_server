import '@testing-library/jest-dom/vitest';
import React from 'react';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

const readBlobAsText = (blob: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        resolve(typeof result === 'string' ? result : '');
      };
      reader.onerror = () => {
        reject(reader.error ?? new Error('Failed to read blob as text'));
      };
      reader.readAsText(blob);
    } catch (error) {
      reject(error);
    }
  });

if (typeof window !== 'undefined') {
  // Mock next/dynamic to render the actual component instead of a loading wrapper
  // This is needed because dynamic({ ssr: false }) doesn't render in jsdom tests
  vi.doMock('next/dynamic', () => ({
    default: (
      importFn: () => Promise<{ default: React.ComponentType<Record<string, unknown>> }>
    ) => {
      // Return a component that uses React.lazy internally
      const LazyComponent = React.lazy(importFn);

      return function DynamicWrapper(props: Record<string, unknown>) {
        return React.createElement(
          React.Suspense,
          { fallback: null },
          React.createElement(LazyComponent, props)
        );
      };
    },
  }));

  // Mock the Dexie database to avoid IndexedDB issues in tests
  vi.doMock('@/db/dexie/schema', () => {
    const mockTable = {
      toArray: vi.fn().mockResolvedValue([]),
      get: vi.fn().mockResolvedValue(undefined),
      put: vi.fn().mockResolvedValue(undefined),
      add: vi.fn().mockResolvedValue(undefined),
      delete: vi.fn().mockResolvedValue(undefined),
      where: vi.fn().mockReturnThis(),
      equals: vi.fn().mockReturnThis(),
      first: vi.fn().mockResolvedValue(undefined),
      count: vi.fn().mockResolvedValue(0),
      bulkPut: vi.fn().mockResolvedValue(undefined),
      bulkDelete: vi.fn().mockResolvedValue(undefined),
    };

    return {
      db: {
        chatHistories: mockTable,
        messages: mockTable,
        prompts: mockTable,
        webshares: mockTable,
        sessionFiles: mockTable,
        userSettings: mockTable,
        customModels: mockTable,
        modelNickname: mockTable,
        processedMedia: mockTable,
        compareStates: mockTable,
        contentDrafts: mockTable,
        draftBatches: mockTable,
        draftAssets: mockTable,
        folders: mockTable,
        keywords: mockTable,
        folderKeywordLinks: mockTable,
        conversationKeywordLinks: mockTable,
        audiobookProjects: mockTable,
        audiobookChapterAssets: mockTable,
        ttsClips: mockTable,
        transaction: vi.fn().mockImplementation(async (_mode, _tables, cb) => cb()),
      },
      PageAssistDexieDB: vi.fn(),
    };
  });

  const hasMissingQueryClientError = (error: unknown): boolean =>
    error instanceof Error && /No QueryClient set/i.test(error.message);

  // Mock React Query with graceful fallback for tests that don't mount QueryClientProvider.
  // When provider is present, preserve real hook behavior to avoid diverging package-ui tests.
  vi.doMock('@tanstack/react-query', async () => {
    const actual = await vi.importActual<typeof import('@tanstack/react-query')>(
      '@tanstack/react-query'
    );
    const actualHooks = actual as typeof import('@tanstack/react-query');

    return {
      ...actual,
      useQuery: ((...args: Parameters<typeof actualHooks.useQuery>) => {
        try {
          return actualHooks.useQuery(...args);
        } catch (error) {
          if (!hasMissingQueryClientError(error)) throw error;
          return {
            data: undefined,
            isLoading: false,
            isFetching: false,
            isError: false,
            isPending: false,
            isSuccess: false,
            isFetched: false,
            isFetchedAfterMount: false,
            isLoadingError: false,
            isPaused: false,
            isPlaceholderData: false,
            isRefetchError: false,
            isRefetching: false,
            isStale: true,
            fetchStatus: 'idle',
            status: 'pending',
            error: null,
            dataUpdatedAt: 0,
            errorUpdatedAt: 0,
            failureCount: 0,
            failureReason: null,
            refetch: vi.fn(),
          } as unknown as ReturnType<typeof actualHooks.useQuery>;
        }
      }) as typeof actualHooks.useQuery,
      useMutation: ((...args: Parameters<typeof actualHooks.useMutation>) => {
        try {
          return actualHooks.useMutation(...args);
        } catch (error) {
          if (!hasMissingQueryClientError(error)) throw error;
          return {
            data: undefined,
            mutate: vi.fn(),
            mutateAsync: vi.fn().mockResolvedValue(undefined),
            isPending: false,
            isLoading: false,
            isIdle: true,
            isSuccess: false,
            isError: false,
            isPaused: false,
            error: null,
            reset: vi.fn(),
            status: 'idle',
            failureCount: 0,
            failureReason: null,
            submittedAt: 0,
            variables: undefined,
            context: undefined,
          } as unknown as ReturnType<typeof actualHooks.useMutation>;
        }
      }) as typeof actualHooks.useMutation,
      useQueryClient: (() => {
        try {
          return actualHooks.useQueryClient();
        } catch (error) {
          if (!hasMissingQueryClientError(error)) throw error;
          return {
            invalidateQueries: vi.fn(),
            setQueryData: vi.fn(),
            getQueryData: vi.fn(),
            getQueriesData: vi.fn().mockReturnValue([]),
            setQueriesData: vi.fn(),
            removeQueries: vi.fn(),
            cancelQueries: vi.fn(),
            resetQueries: vi.fn(),
            refetchQueries: vi.fn(),
            fetchQuery: vi.fn().mockResolvedValue(undefined),
            prefetchQuery: vi.fn().mockResolvedValue(undefined),
            ensureQueryData: vi.fn().mockResolvedValue(undefined),
            isFetching: vi.fn().mockReturnValue(0),
            isMutating: vi.fn().mockReturnValue(0),
            clear: vi.fn(),
          } as unknown as ReturnType<typeof actualHooks.useQueryClient>;
        }
      }) as typeof actualHooks.useQueryClient,
    };
  });

  await import('../packages/ui/vitest.setup');

  // jsdom 27 lacks Blob/File.text(); characters import preview depends on it.
  if (typeof Blob !== 'undefined' && typeof (Blob.prototype as any).text !== 'function') {
    Object.defineProperty(Blob.prototype, 'text', {
      configurable: true,
      writable: true,
      value: function text(this: Blob): Promise<string> {
        return readBlobAsText(this);
      },
    });
  }

  if (typeof File !== 'undefined' && typeof (File.prototype as any).text !== 'function') {
    Object.defineProperty(File.prototype, 'text', {
      configurable: true,
      writable: true,
      value: function text(this: File): Promise<string> {
        return readBlobAsText(this);
      },
    });
  }

  afterEach(() => {
    cleanup();
  });
}
