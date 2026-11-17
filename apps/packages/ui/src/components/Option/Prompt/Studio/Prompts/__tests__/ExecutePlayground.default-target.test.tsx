import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { notification } from 'antd';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExecutePlayground } from '../ExecutePlayground';

const mocks = vi.hoisted(() => ({
  getPrompt: vi.fn(),
  executePrompt: vi.fn(),
  getLlmProviders: vi.fn(),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key,
  }),
}));
vi.mock('@/services/prompt-studio', () => ({
  getPrompt: (...args: unknown[]) => mocks.getPrompt(...args),
  executePrompt: (...args: unknown[]) => mocks.executePrompt(...args),
  getLlmProviders: (...args: unknown[]) => mocks.getLlmProviders(...args),
}));

const source = 'Rowan Observatory is directed by Dr. Mira Vale.';
const catalog = {
  default_provider: 'ollama',
  providers: [
    {
      name: 'ollama',
      display_name: 'Local Ollama',
      default_model: 'qwen2.5:0.5b',
      models: ['ollama-first', 'qwen2.5:0.5b', 'ollama-selected'],
    },
    {
      name: 'anthropic',
      display_name: 'Anthropic',
      default_model: 'claude-default',
      models: ['claude-default', 'claude-selected'],
    },
  ],
};

beforeEach(() => {
  mocks.getPrompt.mockReset().mockResolvedValue({
    ok: true,
    status: 200,
    data: {
      data: {
        id: 42,
        name: 'Observatory prompt',
        version_number: 2,
        user_prompt: 'Source: {{source}}',
      },
    },
  });
  mocks.getLlmProviders.mockReset().mockResolvedValue({ ok: true, status: 200, data: catalog });
  mocks.executePrompt.mockReset().mockResolvedValue({
    ok: true,
    status: 200,
    data: { output: 'Dr. Mira Vale', tokens_used: 151, execution_time: 0.1 },
  });
});
afterEach(() => vi.restoreAllMocks());

const mount = async (emptyCatalog = false) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const mounted = render(
    <QueryClientProvider client={queryClient}>
      <ExecutePlayground open promptId={42} onClose={vi.fn()} />
    </QueryClientProvider>
  );
  await screen.findByText('Variables detected in prompt');
  if (emptyCatalog) await screen.findAllByText('Use server default', { exact: true });
  else await screen.findByText(`Default: ${catalog.default_provider}`, { exact: true });
  fireEvent.change(screen.getByLabelText('Inputs (JSON)'), {
    target: { value: JSON.stringify({ source }) },
  });
  return { ...mounted, queryClient, user: userEvent.setup() };
};
const select = async (user: ReturnType<typeof userEvent.setup>, label: string, option: string) => {
  await user.click(screen.getByRole('combobox', { name: label }));
  await user.click(
    await screen.findByText(option, { selector: '.ant-select-item-option-content', exact: true })
  );
};
const execute = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole('button', { name: 'Execute' }));
  await screen.findByText('Dr. Mira Vale');
};
const expected = (provider?: string, model?: string) => ({
  prompt_id: 42,
  inputs: { source },
  provider,
  model,
});

describe('ExecutePlayground effective submitted target', () => {
  it('sends the displayed catalog defaults when both selectors are blank', async () => {
    const { unmount, queryClient, user } = await mount();
    try {
      expect(screen.getByText('Default: qwen2.5:0.5b', { exact: true })).toBeInTheDocument();
      await execute(user);
      expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith(
        expected('ollama', 'qwen2.5:0.5b')
      );
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('sends the displayed provider with a model-only user choice', async () => {
    const { unmount, queryClient, user } = await mount();
    try {
      await select(user, 'Model (optional)', 'ollama-selected');
      await execute(user);
      expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith(
        expected('ollama', 'ollama-selected')
      );
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('uses an explicitly selected provider own displayed default model', async () => {
    const { unmount, queryClient, user } = await mount();
    try {
      await select(user, 'Provider (optional)', 'Anthropic');
      await screen.findByText('Default: claude-default', { exact: true });
      await execute(user);
      expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith(
        expected('anthropic', 'claude-default')
      );
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('preserves an explicit provider and model pair', async () => {
    const { unmount, queryClient, user } = await mount();
    try {
      await select(user, 'Provider (optional)', 'Anthropic');
      await select(user, 'Model (optional)', 'claude-selected');
      await execute(user);
      expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith(
        expected('anthropic', 'claude-selected')
      );
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('uses the existing first-model default when default_model is absent', async () => {
    mocks.getLlmProviders.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        default_provider: 'ollama',
        providers: [{ name: 'ollama', models: ['ollama-first', 'ollama-selected'] }],
      },
    });
    const { unmount, queryClient, user } = await mount();
    try {
      await screen.findByText('Default: ollama-first', { exact: true });
      await execute(user);
      expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith(
        expected('ollama', 'ollama-first')
      );
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('preserves backend compatibility omission when the catalog has no target', async () => {
    mocks.getLlmProviders.mockResolvedValue({
      ok: true,
      status: 200,
      data: { default_provider: null, providers: [] },
    });
    const { unmount, queryClient, user } = await mount(true);
    try {
      await execute(user);
      expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith(expected(undefined, undefined));
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('keeps the existing provider guard for an unavailable catalog default', async () => {
    const notify = vi.spyOn(notification, 'error').mockImplementation(() => undefined);
    mocks.getLlmProviders.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ...catalog, default_provider: 'unavailable-provider' },
    });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const { unmount } = render(
      <QueryClientProvider client={queryClient}>
        <ExecutePlayground open promptId={42} onClose={vi.fn()} />
      </QueryClientProvider>
    );
    try {
      await screen.findByText('Default: unavailable-provider', { exact: true });
      fireEvent.change(screen.getByLabelText('Inputs (JSON)'), {
        target: { value: JSON.stringify({ source }) },
      });
      await userEvent.setup().click(screen.getByRole('button', { name: 'Execute' }));
      await waitFor(() =>
        expect(notify).toHaveBeenCalledExactlyOnceWith({
          message: 'Provider is not available on this server',
        })
      );
      expect(mocks.executePrompt).not.toHaveBeenCalled();
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('keeps the existing model guard for an unavailable catalog default model', async () => {
    const notify = vi.spyOn(notification, 'error').mockImplementation(() => undefined);
    mocks.getLlmProviders.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        default_provider: 'ollama',
        providers: [
          { name: 'ollama', models: ['ollama-first'], default_model: 'unavailable-model' },
        ],
      },
    });
    const { unmount, queryClient, user } = await mount();
    try {
      await screen.findByText('Default: unavailable-model', { exact: true });
      await user.click(screen.getByRole('button', { name: 'Execute' }));
      await waitFor(() =>
        expect(notify).toHaveBeenCalledExactlyOnceWith({
          message: 'Model is not available for the selected provider',
        })
      );
      expect(mocks.executePrompt).not.toHaveBeenCalled();
    } finally {
      unmount();
      queryClient.clear();
    }
  });
});
