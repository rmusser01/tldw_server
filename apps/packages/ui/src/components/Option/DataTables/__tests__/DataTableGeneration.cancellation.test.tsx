import React from 'react';
import { ConfigProvider, message } from 'antd';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDataTablesStore } from '@/store/data-tables';
import type { DataTable } from '@/types/data-tables';
import { TablePreview } from '../TablePreview';
import { TableDetailModal } from '../TableDetailModal';

const api = vi.hoisted(() => ({
  generateDataTable: vi.fn(),
  regenerateDataTable: vi.fn(),
  getDataTable: vi.fn(),
  getDataTableJob: vi.fn(),
  cancelDataTableJob: vi.fn(),
}));
const capturedScope = vi.hoisted(() => ({
  config: { serverUrl: 'http://original-s03.test', authMode: 'multi-user' },
  userId: 42,
}));
vi.mock('@/services/service-prompts', () => ({
  resolveServicePromptScope: vi.fn(async () => capturedScope),
  createServicePromptScopeLease: (signal: AbortSignal) => ({
    signal,
    bind: vi.fn(),
    release: vi.fn(),
  }),
}));
const polling = vi.hoisted(() => ({ poll: vi.fn() }));
vi.mock('@/services/tldw/TldwApiClient', () => ({ tldwClient: api }));
vi.mock('@/utils/data-tables-jobs', () => ({ pollDataTableJob: polling.poll }));
vi.mock('@/utils/data-table-export', () => ({ exportAndDownload: vi.fn() }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback || _key }),
}));
vi.mock('@dnd-kit/react', () => ({
  DragDropProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));
vi.mock('@dnd-kit/react/sortable', () => ({
  useSortable: () => ({ ref: vi.fn(), handleRef: vi.fn() }),
}));
vi.mock('@dnd-kit/collision', () => ({ closestCenter: vi.fn() }));
vi.mock('../EditableCell', () => ({
  EditableCell: ({ value }: { value: string }) => <span>{value}</span>,
}));
vi.mock('../EditableDataTable', () => ({ EditableDataTable: () => <div>Saved table editor</div> }));

const saved: DataTable = {
  id: '83569797-3613-42e0-b2a2-498794fc1cd0',
  name: 'S03 saved facts',
  prompt: 'Rowan facts',
  columns: [{ id: 'value', name: 'Value', type: 'text' }],
  rows: [{ Value: 'Selene' }],
  sources: [],
  row_count: 1,
  created_at: '2026-10-26T00:00:00Z',
  updated_at: '2026-10-26T00:00:00Z',
};
const replacement = { ...saved, rows: [{ Value: 'Replacement' }] };
const receipt = { job_id: 71, status: 'queued', table: { uuid: saved.id } };
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((accept, refuse) => {
    resolve = accept;
    reject = refuse;
  });
  return { promise, resolve, reject };
}
let client: QueryClient;
let start: ReturnType<typeof deferred<typeof receipt>>;
let complete: ReturnType<typeof deferred<{ status: string; table_uuid: string }>>;

beforeEach(() => {
  vi.resetAllMocks();
  useDataTablesStore.getState().resetStore();
  useDataTablesStore.getState().setTables([{ ...saved, column_count: 1, source_count: 0 }]);
  start = deferred();
  complete = deferred();
  api.generateDataTable.mockReturnValue(start.promise);
  api.regenerateDataTable.mockReturnValue(start.promise);
  api.getDataTable.mockResolvedValue(saved);
  api.cancelDataTableJob.mockResolvedValue({ success: true, status: 'cancelled' });
  polling.poll.mockReturnValue(complete.promise);
});
afterEach(() => {
  cleanup();
  client?.clear();
  message.destroy();
  useDataTablesStore.getState().resetStore();
});

async function open(mode: string) {
  client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const view = render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <QueryClientProvider client={client}>
        {mode === 'wizard' ? (
          <TablePreview />
        ) : (
          <TableDetailModal open tableId={saved.id} onClose={vi.fn()} />
        )}
      </QueryClientProvider>
    </ConfigProvider>
  );
  if (mode === 'detail') fireEvent.click(await screen.findByRole('button', { name: 'Regenerate' }));
  await waitFor(() =>
    expect(
      mode === 'wizard' ? api.generateDataTable : api.regenerateDataTable
    ).toHaveBeenCalledOnce()
  );
  return view;
}
async function finish() {
  api.getDataTable.mockResolvedValue(replacement);
  await act(async () => complete.resolve({ status: 'completed', table_uuid: saved.id }));
}
function result(mode: string) {
  return mode === 'wizard'
    ? useDataTablesStore.getState().generatedTable
    : useDataTablesStore.getState().currentTable;
}

describe.each(['wizard', 'detail'])('%s generation cancellation', (mode) => {
  it('cancels the accepted server job and refuses a late completion without changing saved content', async () => {
    await open(mode);
    await act(async () => start.resolve(receipt));
    await waitFor(() => expect(polling.poll).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));
    await waitFor(() =>
      expect(api.cancelDataTableJob).toHaveBeenCalledWith(71, { requestScope: capturedScope })
    );
    await finish();
    expect(result(mode)).toEqual(mode === 'wizard' ? null : saved);
    expect(useDataTablesStore.getState().tables[0].name).toBe(saved.name);
    expect(screen.queryByText('Table generated successfully!')).not.toBeInTheDocument();
    expect(screen.queryByText('Table regenerated successfully!')).not.toBeInTheDocument();
  });

  it('cancels a late start receipt after Cancel during submission', async () => {
    await open(mode);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));
    expect(api.cancelDataTableJob).not.toHaveBeenCalled();
    await act(async () => start.resolve(receipt));
    await waitFor(() =>
      expect(api.cancelDataTableJob).toHaveBeenCalledWith(71, { requestScope: capturedScope })
    );
    await finish();
    expect(result(mode)).toEqual(mode === 'wizard' ? null : saved);
  });

  it('reports a refused cancellation and continues to the actual completed result', async () => {
    api.cancelDataTableJob.mockRejectedValue(new Error('Cancellation unavailable'));
    await open(mode);
    await act(async () => start.resolve(receipt));
    await waitFor(() => expect(polling.poll).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));
    await screen.findByText('Cancellation unavailable');
    await finish();
    await waitFor(() => expect(result(mode)?.rows).toEqual(replacement.rows));
  });

  it('keeps Cancel retryable when polling fails before the pending cancellation fails', async () => {
    const poll = deferred<{ status: string; table_uuid: string }>();
    const deletion = deferred<{ success: boolean; status: string }>();
    polling.poll.mockReturnValue(poll.promise);
    api.cancelDataTableJob.mockReturnValue(deletion.promise);
    await open(mode);
    await act(async () => start.resolve(receipt));
    await waitFor(() => expect(polling.poll).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));
    await act(async () => poll.reject(new Error('Status unavailable')));
    await act(async () => deletion.reject(new Error('Cancellation unavailable')));
    await screen.findByText('Cancellation unavailable');
    expect(screen.getByRole('button', { name: 'Cancel generation' })).toBeEnabled();
    api.cancelDataTableJob.mockResolvedValue({ success: true, status: 'cancelled' });
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));
    await waitFor(() => expect(api.cancelDataTableJob).toHaveBeenCalledTimes(2));
    expect(result(mode)).toEqual(mode === 'wizard' ? null : saved);
  });

  it('does not publish a table GET that completes after cancellation', async () => {
    await open(mode);
    const detail = deferred<DataTable>();
    api.getDataTable.mockReturnValue(detail.promise);
    await act(async () => start.resolve(receipt));
    await act(async () => complete.resolve({ status: 'completed', table_uuid: saved.id }));
    await waitFor(() => expect(api.getDataTable).toHaveBeenCalledTimes(mode === 'wizard' ? 1 : 2));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));
    await waitFor(() => expect(api.cancelDataTableJob).toHaveBeenCalledOnce());
    await act(async () => detail.resolve(replacement));
    expect(result(mode)).toEqual(mode === 'wizard' ? null : saved);
  });

  it('retires its late accepted job after unmount without publishing a result', async () => {
    const view = await open(mode);
    view.unmount();
    await act(async () => start.resolve(receipt));
    await waitFor(() =>
      expect(api.cancelDataTableJob).toHaveBeenCalledWith(71, { requestScope: capturedScope })
    );
    await finish();
    expect(result(mode)).toBeNull();
  });
});

it('starts only one server job under StrictMode effect replay', async () => {
  client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <React.StrictMode>
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QueryClientProvider client={client}>
          <TablePreview />
        </QueryClientProvider>
      </ConfigProvider>
    </React.StrictMode>
  );
  await waitFor(() => expect(api.generateDataTable).toHaveBeenCalledOnce());
  await act(async () => start.resolve(receipt));
  await finish();
  await waitFor(() =>
    expect(useDataTablesStore.getState().generatedTable?.rows).toEqual(replacement.rows)
  );
  expect(api.generateDataTable).toHaveBeenCalledOnce();
});

it.each(['switch', 'close'])('ignores a held detail fetch after %s', async (action) => {
  const first = deferred<DataTable>();
  const second = {
    ...saved,
    id: '0ca3184a-a81a-4f60-9dbd-869665a6c6bf',
    name: 'Second saved table',
  };
  api.getDataTable.mockReturnValueOnce(first.promise).mockResolvedValue(second);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const drawer = (id: string, visible = true) => (
    <ConfigProvider theme={{ token: { motion: false } }}>
      <QueryClientProvider client={client}>
        <TableDetailModal open={visible} tableId={id} onClose={vi.fn()} />
      </QueryClientProvider>
    </ConfigProvider>
  );
  const view = render(drawer(saved.id));
  await waitFor(() => expect(api.getDataTable).toHaveBeenCalledOnce());
  view.rerender(drawer(second.id, action !== 'close'));
  if (action === 'switch')
    await waitFor(() => expect(useDataTablesStore.getState().currentTable?.id).toBe(second.id));
  await act(async () => first.resolve(saved));
  expect(useDataTablesStore.getState().currentTable).toEqual(action === 'close' ? null : second);
});
