import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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

const success = {
  ok: true,
  status: 200,
  data: { output: 'Dr. Mira Vale', tokens_used: 151, execution_time: 0.1 },
};
const missingInput = {
  ok: false,
  status: 400,
  error: 'Required prompt inputs are missing.',
  data: {
    detail: {
      error_code: 'missing_required_variable',
      message: 'Required prompt inputs are missing.',
    },
  },
};

const mountPlayground = async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const mounted = render(
    <QueryClientProvider client={queryClient}>
      <ExecutePlayground open promptId={42} onClose={vi.fn()} />
    </QueryClientProvider>
  );
  await screen.findByText('Variables detected in prompt');
  await screen.findByText('Default: ollama', { exact: true });
  return { ...mounted, queryClient };
};
const execute = () => fireEvent.click(screen.getByRole('button', { name: 'Execute' }));

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
  mocks.getLlmProviders.mockReset().mockResolvedValue({
    ok: true,
    status: 200,
    data: { default_provider: 'ollama', providers: [{ name: 'ollama', models: ['qwen2.5:0.5b'] }] },
  });
  mocks.executePrompt.mockReset();
});

afterEach(() => vi.restoreAllMocks());

describe('ExecutePlayground resolved execution failures', () => {
  it.each([
    [missingInput, 'Required prompt inputs are missing.'],
    [{ ok: false, status: 400 }, 'Prompt execution failed.'],
  ])(
    'shows an error and no output for a resolved failed envelope %#',
    async (response, description) => {
      const notify = vi.spyOn(notification, 'error').mockImplementation(() => undefined);
      mocks.executePrompt.mockResolvedValue(response);
      const { unmount, queryClient } = await mountPlayground();
      try {
        execute();
        await waitFor(() => expect(notify).toHaveBeenCalledWith({ message: 'Error', description }));
        expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith({
          prompt_id: 42,
          inputs: {},
          provider: 'ollama',
          model: 'qwen2.5:0.5b',
        });
        expect(screen.queryByText('Output')).not.toBeInTheDocument();
        expect(screen.queryByText('No output')).not.toBeInTheDocument();
      } finally {
        unmount();
        queryClient.clear();
      }
    }
  );

  it('clears earlier output and stats on failure and allows a corrected input to recover', async () => {
    const notify = vi.spyOn(notification, 'error').mockImplementation(() => undefined);
    mocks.executePrompt
      .mockResolvedValueOnce(success)
      .mockResolvedValueOnce(missingInput)
      .mockResolvedValueOnce(success);
    const { unmount, queryClient } = await mountPlayground();
    try {
      execute();
      await screen.findByText('Dr. Mira Vale');
      expect(screen.getByText('151 tokens')).toBeInTheDocument();
      expect(screen.getByText('0.10s')).toBeInTheDocument();
      execute();
      await waitFor(() =>
        expect(notify).toHaveBeenCalledWith({ message: 'Error', description: missingInput.error })
      );
      expect(screen.queryByText('Dr. Mira Vale')).not.toBeInTheDocument();
      expect(screen.queryByText('151 tokens')).not.toBeInTheDocument();
      expect(screen.queryByText('0.10s')).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText('Inputs (JSON)'), {
        target: {
          value: JSON.stringify({ source: 'Rowan Observatory is directed by Dr. Mira Vale.' }),
        },
      });
      execute();
      await screen.findByText('Dr. Mira Vale');
      expect(mocks.executePrompt).toHaveBeenLastCalledWith({
        prompt_id: 42,
        inputs: { source: 'Rowan Observatory is directed by Dr. Mira Vale.' },
        provider: 'ollama',
        model: 'qwen2.5:0.5b',
      });
      expect(notify).toHaveBeenCalledTimes(1);
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('preserves successful output and statistics', async () => {
    const notify = vi.spyOn(notification, 'error').mockImplementation(() => undefined);
    mocks.executePrompt.mockResolvedValue(success);
    const { unmount, queryClient } = await mountPlayground();
    try {
      execute();
      await screen.findByText('Dr. Mira Vale');
      expect(screen.getByText('151 tokens')).toBeInTheDocument();
      expect(screen.getByText('0.10s')).toBeInTheDocument();
      expect(notify).not.toHaveBeenCalled();
    } finally {
      unmount();
      queryClient.clear();
    }
  });

  it('preserves the existing rejected transport error path', async () => {
    const notify = vi.spyOn(notification, 'error').mockImplementation(() => undefined);
    mocks.executePrompt.mockRejectedValue(new Error('Connection unavailable.'));
    const { unmount, queryClient } = await mountPlayground();
    try {
      execute();
      await waitFor(() =>
        expect(notify).toHaveBeenCalledWith({
          message: 'Error',
          description: 'Connection unavailable.',
        })
      );
      expect(screen.queryByText('Output')).not.toBeInTheDocument();
    } finally {
      unmount();
      queryClient.clear();
    }
  });
});
