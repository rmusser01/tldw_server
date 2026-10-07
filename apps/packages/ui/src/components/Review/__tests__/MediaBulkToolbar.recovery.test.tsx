import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { MediaBulkToolbar } from '../MediaBulkToolbar'

// Existing controlled Modal pattern isolates recovery state from CSS animation.
vi.mock('antd', () => ({
  Modal: ({ open, onOk, okText, children }: { open: boolean; onOk: () => Promise<void>; okText: string; children: React.ReactNode }) =>
    open ? <div role="dialog"><div>{children}</div><button onClick={() => void onOk()}>{okText}</button></div> : null
}))
afterEach(cleanup)

it('keeps earlier Note recovery across successive batches and retains only failed restores', async () => {
  const firstRestore = vi.fn().mockResolvedValueOnce(1).mockResolvedValueOnce(0)
  const secondRestore = vi.fn().mockResolvedValue(0)
  const deleteSelection = vi.fn()
    .mockResolvedValueOnce({ mediaCount: 0, noteCount: 1, restoreNotes: firstRestore })
    .mockResolvedValueOnce({ mediaCount: 1, noteCount: 1, restoreNotes: secondRestore })
  const selection: React.ComponentProps<typeof MediaBulkToolbar>['selection'] = {
    bulkSelectedItems: [{ kind: 'note', id: 'note-1', title: 'Note', raw: {} }],
    bulkKeywordsDraft: '', setBulkKeywordsDraft: vi.fn(), handleBulkAddKeywords: vi.fn(), handleBulkDelete: deleteSelection,
    collectionDraftName: '', setCollectionDraftName: vi.fn(), handleAddSelectionToCollection: vi.fn(), handleOpenSelectionInMultiReview: vi.fn(),
    bulkExportFormat: 'json', setBulkExportFormat: vi.fn(), handleBulkExport: vi.fn(), handleSelectAllVisibleItems: vi.fn(), handleClearBulkSelection: vi.fn()
  }
  const t = ((_key: string, options: { defaultValue?: string; count?: number }) => (options?.defaultValue || _key).replace('{{count}}', String(options?.count))) as React.ComponentProps<typeof MediaBulkToolbar>['t']
  render(<MemoryRouter><MediaBulkToolbar selection={selection} t={t} /></MemoryRouter>)
  for (const count of [1, 2]) {
    fireEvent.click(screen.getByTestId('media-bulk-delete'))
    fireEvent.click(await screen.findByRole('button', { name: 'Move to trash' }))
    expect(await screen.findByRole('button', { name: count === 1 ? 'Restore 1 note' : 'Restore 2 notes' })).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  }
  expect(screen.getByRole('button', { name: 'Open Trash' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Restore 2 notes' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Restore 1 note' }))
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Restore 1 note' })).not.toBeInTheDocument())
  expect(firstRestore).toHaveBeenCalledTimes(2)
  expect(secondRestore).toHaveBeenCalledTimes(1)
  expect(screen.getByRole('button', { name: 'Open Trash' })).toBeInTheDocument()
})
