import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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
  buildAuthHeaders: vi.fn(() => ({ Authorization: 'Bearer test-token' })),
  getApiBaseUrl: vi.fn(() => 'http://example.com/api/v1'),
}));

vi.mock('@web/components/ui/ToastProvider', () => ({
  useToast: () => ({ show: mocks.showToast }),
}));

vi.mock('@web/lib/api', () => ({
  buildAuthHeaders: (...args: string[]) => mocks.buildAuthHeaders(...args),
  getApiBaseUrl: () => mocks.getApiBaseUrl(),
}));

import AdminMaintenancePage from '@web/pages/admin/maintenance';

describe.skipIf(SKIP_INTEGRATION_TESTS)('AdminMaintenancePage effective config fetch', () => {
  let originalFetch: typeof globalThis.fetch | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    if (originalFetch) {
      globalThis.fetch = originalFetch;
    } else {
      delete (globalThis as { fetch?: typeof globalThis.fetch }).fetch;
    }
  });

  it('requests effective config and renders the payload', async () => {
    const payload = {
      config_root: '/config',
      config_file: '/config/config.txt',
      prompts_dir: '/config/Prompts',
      module_yaml: { tts: null },
      values: {
        tts: {
          default_provider: { value: 'openai', source: 'env', redacted: false },
        },
      },
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      text: vi.fn().mockResolvedValue(JSON.stringify(payload)),
    });
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const user = userEvent.setup();
    render(<AdminMaintenancePage />);

    await user.click(screen.getByRole('button', { name: 'Fetch Effective Config' }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'http://example.com/api/v1/admin/config/effective?include_defaults=false',
        expect.objectContaining({ method: 'GET' })
      );
    });

    expect(await screen.findByText(/config_root/)).toBeInTheDocument();
  });
});
