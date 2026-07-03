import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// TODO: These integration tests require a comprehensive test harness with:
// - QueryClientProvider (React Query)
// - Router context (react-router-dom)
// - Dexie/IndexedDB mock
// - Various UI providers (Toast, Layout, Theme)
// Skip until a shared test wrapper is created.
const SKIP_INTEGRATION_TESTS = true;

const mocks = vi.hoisted(() => ({
  showToast: vi.fn(),
  apiClient: {
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
  },
  loadDrafts: vi.fn(),
  persistDraft: vi.fn(),
  persistDraftFileAsset: vi.fn(),
  requestFileHandleForFile: vi.fn(),
  getDraftFileAsset: vi.fn(),
  resolveDraftFileAssetStatus: vi.fn(),
  getDraftFileForUpload: vi.fn(),
}));

vi.mock('dexie', () => ({
  default: class DexieMock {},
}), { virtual: true });

vi.mock('@web/components/ui/ToastProvider', () => ({
  useToast: () => ({ show: mocks.showToast }),
}));

vi.mock('@web/lib/api', () => ({
  apiClient: mocks.apiClient,
}));

vi.mock('@web/lib/drafts', () => ({
  loadDrafts: mocks.loadDrafts,
  persistDraft: mocks.persistDraft,
  persistDraftFileAsset: mocks.persistDraftFileAsset,
  requestFileHandleForFile: mocks.requestFileHandleForFile,
  getDraftFileAsset: mocks.getDraftFileAsset,
  resolveDraftFileAssetStatus: mocks.resolveDraftFileAssetStatus,
  getDraftFileForUpload: mocks.getDraftFileForUpload,
}));

const {
  showToast,
  apiClient,
  loadDrafts,
  persistDraft,
  persistDraftFileAsset,
  requestFileHandleForFile,
  getDraftFileAsset,
  resolveDraftFileAssetStatus,
  getDraftFileForUpload,
} = mocks;

import ContentReviewPage from '@web/pages/content-review';

const setNavigatorOnline = (value: boolean) => {
  Object.defineProperty(window.navigator, 'onLine', {
    value,
    configurable: true,
  });
};

const waitForEditor = async () => {
  await screen.findByText('Batch');
  return screen.findByLabelText('Draft content editor');
};

beforeEach(() => {
  vi.clearAllMocks();
  setNavigatorOnline(true);
  loadDrafts.mockResolvedValue([]);
  persistDraft.mockResolvedValue(undefined);
  persistDraftFileAsset.mockResolvedValue({
    assetStatus: 'present',
    assetNote: 'File stored locally.',
    source: { kind: 'file', filename: 'clip.mp3' },
    storedAs: 'blob',
  });
  requestFileHandleForFile.mockResolvedValue(null);
  getDraftFileAsset.mockResolvedValue(null);
  resolveDraftFileAssetStatus.mockResolvedValue({
    assetStatus: 'missing',
    assetNote: 'Source file missing. Reattach before commit.',
  });
  getDraftFileForUpload.mockResolvedValue({
    assetStatus: 'missing',
    assetNote: 'File handle unavailable. Reattach before commit.',
    file: null,
  });
  apiClient.post.mockResolvedValue({});
  apiClient.put.mockResolvedValue({});
  apiClient.patch.mockResolvedValue({});
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('ContentReviewPage', () => {
  it('prompts before switching drafts when there are unsaved changes', async () => {
    const user = userEvent.setup();

    render(<ContentReviewPage />);
    const editor = await waitForEditor();

    await user.type(editor, ' extra');
    expect(screen.getByText(/Unsaved changes/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Research Notes/i }));
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText(/Discard unsaved changes\?/i)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /Interview Transcript/i })).toBeInTheDocument();

    await user.click(within(dialog).getByRole('button', { name: /Keep editing/i }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: /Research Notes/i }));
    const confirmDialog = screen.getByRole('dialog');
    await user.click(within(confirmDialog).getByRole('button', { name: /Discard changes/i }));
    expect(screen.getByRole('heading', { name: /Research Notes/i })).toBeInTheDocument();
  });

  it('saves draft changes and shows success toast', async () => {
    const user = userEvent.setup();
    render(<ContentReviewPage />);
    const editor = await waitForEditor();

    await user.clear(editor);
    await user.type(editor, 'Updated content');

    const keywordsInput = screen.getByPlaceholderText('comma-separated');
    await user.clear(keywordsInput);
    await user.type(keywordsInput, 'alpha, beta');

    const saveButton = screen.getByRole('button', { name: /Save Draft/i });
    expect(saveButton).toBeEnabled();

    await user.click(saveButton);

    await waitFor(() => {
      expect(persistDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          content: 'Updated content',
          keywords: ['alpha', 'beta'],
        })
      );
    });

    expect(showToast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Draft saved', variant: 'success' }));
    expect(screen.getByText(/Saved/i)).toBeInTheDocument();
  });

  it('blocks commit when source is missing and disables the commit button', async () => {
    render(<ContentReviewPage />);
    await waitForEditor();

    expect(screen.getByText(/Source required to commit this item/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Commit/i })).toBeDisabled();
  });

  it('warns when title is missing on commit', async () => {
    loadDrafts.mockResolvedValueOnce([
      {
        id: 'draft-1',
        title: '',
        status: 'in_progress',
        mediaType: 'audio',
        content: 'Has content',
        keywords: [],
        assetStatus: 'present',
        source: { kind: 'url', value: 'https://example.com' },
      },
    ]);

    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitFor(() => expect(screen.getByRole('button', { name: /Commit/i })).toBeEnabled());

    await user.click(screen.getByRole('button', { name: /Commit/i }));

    expect(showToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Title required',
        variant: 'warning',
      })
    );
  });

  it('warns when content is missing on commit', async () => {
    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitForEditor();

    await user.click(screen.getByRole('button', { name: /Research Notes/i }));
    const editor = await screen.findByLabelText('Draft content editor');
    await user.clear(editor);

    await user.click(screen.getByRole('button', { name: /Commit/i }));

    expect(showToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Content required',
        variant: 'warning',
      })
    );
  });

  it('handles missing file source during commit', async () => {
    resolveDraftFileAssetStatus.mockResolvedValueOnce({ assetStatus: 'present' });
    loadDrafts.mockResolvedValueOnce([
      {
        id: 'draft-1',
        title: 'File Draft',
        status: 'in_progress',
        mediaType: 'audio',
        content: 'File content',
        keywords: [],
        assetStatus: 'present',
        source: { kind: 'file', filename: 'clip.mp3' },
      },
    ]);

    getDraftFileForUpload.mockResolvedValueOnce({
      assetStatus: 'missing',
      assetNote: 'File handle unavailable. Reattach before commit.',
      source: { kind: 'file', filename: 'clip.mp3' },
      file: null,
    });

    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitFor(() => expect(screen.getByRole('button', { name: /Commit/i })).toBeEnabled());

    await user.click(screen.getByRole('button', { name: /Commit/i }));

    expect(showToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Source missing',
        variant: 'warning',
      })
    );
  });

  it('commits successfully and triggers media calls', async () => {
    apiClient.post.mockImplementation((url: string) => {
      if (url === '/media/add') {
        return Promise.resolve({ results: [{ db_id: 42, status: 'Success' }] });
      }
      return Promise.resolve({});
    });

    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitForEditor();

    await user.click(screen.getByRole('button', { name: /Research Notes/i }));
    await user.click(screen.getByRole('button', { name: /Commit/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/media/add', expect.any(FormData));
      expect(apiClient.put).toHaveBeenCalledWith('/media/42', expect.objectContaining({ title: 'Research Notes' }));
      expect(apiClient.patch).toHaveBeenCalledWith('/media/42/metadata', expect.any(Object));
      expect(apiClient.post).toHaveBeenCalledWith('/media/42/reprocess', expect.any(Object));
    });

    expect(showToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Committed', variant: 'success' })
    );
    expect(screen.getAllByText(/reviewed/i).length).toBeGreaterThan(0);
  });

  it('shows errors for invalid URL attachment', async () => {
    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitForEditor();

    await user.click(screen.getByRole('button', { name: /Reattach Source/i }));
    await user.click(screen.getByRole('button', { name: /Provide URL/i }));

    const dialog = screen.getByRole('dialog');
    const urlInput = within(dialog).getByPlaceholderText('https://...');
    await user.type(urlInput, 'not-a-url');
    await user.click(within(dialog).getByRole('button', { name: /Attach Source/i }));

    expect(within(dialog).getByText(/URL format looks invalid/i)).toBeInTheDocument();
  });

  it('handles large file re-selection flow', async () => {
    persistDraftFileAsset.mockResolvedValueOnce({
      assetStatus: 'pending',
      assetNote: 'Large file not stored locally. Re-select before commit.',
      source: { kind: 'file', filename: 'big.mp4' },
      storedAs: 'metadata',
    });

    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitForEditor();

    await user.click(screen.getByRole('button', { name: /Reattach Source/i }));

    const dialog = screen.getByRole('dialog');
    const fileInput = dialog.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput).toBeTruthy();

    const largeFile = new File(['x'], 'big.mp4', { type: 'video/mp4' });
    Object.defineProperty(largeFile, 'size', { value: 101 * 1024 * 1024, configurable: true });

    await user.upload(fileInput, largeFile);
    await user.click(within(dialog).getByRole('button', { name: /Attach Source/i }));

    await waitFor(() => {
      expect(requestFileHandleForFile).toHaveBeenCalledWith(largeFile);
      expect(persistDraftFileAsset).toHaveBeenCalled();
    });

    expect(showToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Large file requires re-selection', variant: 'warning' })
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('updates UI for offline state and disables commit', async () => {
    setNavigatorOnline(false);
    const user = userEvent.setup();
    render(<ContentReviewPage />);
    await waitForEditor();

    await user.click(screen.getByRole('button', { name: /Research Notes/i }));

    expect(screen.getByText(/You are offline/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Commit/i })).toBeDisabled();
  });

  it('surfaces save and commit errors with toasts', async () => {
    loadDrafts.mockResolvedValueOnce([
      {
        id: 'draft-1',
        title: 'Review Draft',
        status: 'pending',
        mediaType: 'document',
        content: 'Content to save',
        keywords: [],
        assetStatus: 'present',
        source: { kind: 'url', value: 'https://example.com/source' },
      },
    ]);
    persistDraft.mockRejectedValueOnce(new Error('Disk full'));
    apiClient.post.mockRejectedValueOnce(new Error('Upload failed'));

    const user = userEvent.setup();
    render(<ContentReviewPage />);
    const editor = await waitForEditor();

    await user.type(editor, ' updated');
    await user.click(screen.getByRole('button', { name: /Save Draft/i }));

    await waitFor(() => {
      expect(showToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Save failed', variant: 'danger' })
      );
    });

    await user.click(screen.getByRole('button', { name: /Commit/i }));

    await waitFor(() => {
      expect(showToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Commit failed', variant: 'danger' })
      );
    });
  });
});
