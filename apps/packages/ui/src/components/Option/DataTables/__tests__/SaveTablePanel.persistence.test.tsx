import React from 'react';
import { ConfigProvider, message } from 'antd';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDataTablesStore } from '@/store/data-tables';
import type { DataTable } from '@/types/data-tables';
import { SaveTablePanel } from '../SaveTablePanel';

const api = vi.hoisted(() => ({
  updateDataTable: vi.fn(),
  saveDataTableContent: vi.fn(),
}));
vi.mock('@/services/tldw/TldwApiClient', () => ({ tldwClient: api }));
vi.mock('@/utils/data-table-export', () => ({ exportAndDownload: vi.fn() }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, value?: string | { defaultValue?: string }) =>
      typeof value === 'string' ? value : value?.defaultValue || key,
  }),
}));

const original: DataTable = {
  id: '83569797-3613-42e0-b2a2-498794fc1cd0',
  name: 'S03 Rowan facts',
  prompt: 'Extract the three facts',
  columns: [
    { id: 'item', name: 'Item', type: 'text' },
    { id: 'value', name: 'Value', type: 'text' },
  ],
  rows: [
    { Item: 'Telescope', Value: 'Selene' },
    { Item: 'Opening', Value: '1988' },
    { Item: 'Tours', Value: 'Friday 18:00' },
  ],
  sources: [],
  row_count: 3,
  created_at: '2026-10-26T00:00:00Z',
  updated_at: '2026-10-26T00:00:00Z',
};
let persisted: DataTable;
let client: QueryClient;

function renderPreview(id = original.id, rowLimit = 3) {
  const preview = structuredClone(original);
  preview.id = id;
  preview.rows[2].Value = 'Friday at 18:00';
  preview.rows = preview.rows.slice(0, rowLimit);
  useDataTablesStore.getState().setGeneratedTable(preview);
  client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <QueryClientProvider client={client}>
        <SaveTablePanel />
      </QueryClientProvider>
    </ConfigProvider>
  );
  return preview;
}

beforeEach(() => {
  vi.clearAllMocks();
  useDataTablesStore.getState().resetStore();
  persisted = structuredClone(original);
  api.updateDataTable.mockImplementation(async (_id, metadata) => {
    persisted = { ...persisted, ...metadata };
    return structuredClone(persisted);
  });
  api.saveDataTableContent.mockImplementation(async (_id, content) => {
    persisted = { ...persisted, ...structuredClone(content), row_count: content.rows.length };
    return structuredClone(persisted);
  });
});
afterEach(() => {
  cleanup();
  client?.clear();
  message.destroy();
  useDataTablesStore.getState().resetStore();
});

describe('SaveTablePanel persistence', () => {
  it('saves edited cells and columns through the content API before reporting library success', async () => {
    const preview = renderPreview();
    fireEvent.click(screen.getByRole('button', { name: 'Save to Library' }));
    await screen.findByText('Table Saved!');
    expect({ columns: persisted.columns, rows: persisted.rows }).toEqual({
      columns: preview.columns,
      rows: preview.rows,
    });
    expect(useDataTablesStore.getState().tables).toHaveLength(1);
  });

  it('uses the persisted row count after deleting a preview row', async () => {
    renderPreview(original.id, 2);
    fireEvent.click(screen.getByRole('button', { name: 'Save to Library' }));
    await screen.findByText('Table Saved!');
    expect(useDataTablesStore.getState().tables[0].row_count).toBe(2);
  });

  it('retains the draft and refuses success when content persistence fails', async () => {
    api.saveDataTableContent.mockRejectedValue(new Error('Storage unavailable'));
    const preview = renderPreview();
    fireEvent.click(screen.getByRole('button', { name: 'Save to Library' }));
    await screen.findByText('Storage unavailable');
    expect(screen.queryByText('Table Saved!')).not.toBeInTheDocument();
    expect(api.updateDataTable).not.toHaveBeenCalled();
    expect(persisted).toEqual(original);
    expect(useDataTablesStore.getState().generatedTable).toEqual(preview);
    expect(useDataTablesStore.getState().tables).toEqual([]);
  });

  it('keeps saved content and the exportable draft when the metadata update fails', async () => {
    api.updateDataTable.mockRejectedValue(new Error('Metadata unavailable'));
    const preview = renderPreview();
    fireEvent.change(screen.getByPlaceholderText('Enter a name for your table'), {
      target: { value: 'S03 renamed facts' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save to Library' }));
    await screen.findByText('Metadata unavailable');
    expect({ columns: persisted.columns, rows: persisted.rows }).toEqual({
      columns: preview.columns,
      rows: preview.rows,
    });
    expect(persisted.name).toBe(original.name);
    expect(screen.queryByText('Table Saved!')).not.toBeInTheDocument();
    expect(useDataTablesStore.getState().tables).toEqual([]);
    expect(useDataTablesStore.getState().generatedTable).toEqual(preview);
    expect(screen.getByRole('button', { name: 'CSV' })).toBeEnabled();
  });

  it.each(['artifact-1', 'preview-1', 'local-1'])(
    'does not claim server persistence for %s',
    async (id) => {
      const preview = renderPreview(id);
      fireEvent.click(screen.getByRole('button', { name: 'Save to Library' }));
      await waitFor(() =>
        expect(screen.getByRole('button', { name: 'Save to Library' })).not.toHaveClass(
          /ant-btn-loading/
        )
      );
      expect(screen.queryByText('Table Saved!')).not.toBeInTheDocument();
      expect(api.saveDataTableContent).not.toHaveBeenCalled();
      expect(api.updateDataTable).not.toHaveBeenCalled();
      expect(useDataTablesStore.getState().tables).toEqual([]);
      expect(useDataTablesStore.getState().generatedTable).toEqual(preview);
      expect(screen.getByRole('button', { name: 'CSV' })).toBeEnabled();
    }
  );
});
