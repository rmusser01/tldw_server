import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { notification } from 'antd';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExecutePlayground } from '../ExecutePlayground';

const mocks = vi.hoisted(() => ({
  getPrompt: vi.fn(),
  getLlmProviders: vi.fn(),
  executePrompt: vi.fn(),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key,
  }),
}));
vi.mock('@/services/prompt-studio', () => ({
  getPrompt: (...args: unknown[]) => mocks.getPrompt(...args),
  getLlmProviders: (...args: unknown[]) => mocks.getLlmProviders(...args),
  executePrompt: (...args: unknown[]) => mocks.executePrompt(...args),
}));
const success = {
  ok: true,
  status: 200,
  data: { output: 'Dr. Mira Vale', tokens_used: 159, execution_time: 0.06 },
};
const missing = { ok: false, status: 400, error: 'Required prompt inputs are missing.' };
const catalog = {
  default_provider: 'ollama',
  providers: [{ name: 'ollama', models: ['qwen2.5:0.5b'] }],
};
const deferred = () => {
  let resolve!: (value: unknown) => void;
  const promise = new Promise<unknown>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

beforeEach(() => {
  mocks.getPrompt.mockReset().mockImplementation(async (id: number) => ({
    ok: true,
    status: 200,
    data: {
      data: { id, name: `Prompt ${id}`, version_number: 2, user_prompt: 'Source: {{source}}' },
    },
  }));
  mocks.getLlmProviders.mockReset().mockResolvedValue({ ok: true, status: 200, data: catalog });
  mocks.executePrompt.mockReset();
  vi.spyOn(notification, 'error').mockImplementation(() => undefined);
});
afterEach(() => vi.restoreAllMocks());

const mount = async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onClose = vi.fn();
  const view = (open: boolean, promptId: number) => (
    <QueryClientProvider client={queryClient}>
      <ExecutePlayground open={open} promptId={promptId} onClose={onClose} />
    </QueryClientProvider>
  );
  const mounted = render(view(true, 42));
  await screen.findByText('Prompt 42');
  await screen.findByText('Default: ollama', { exact: true });
  return {
    ...mounted,
    queryClient,
    onClose,
    user: userEvent.setup(),
    update: (open: boolean, promptId = 42) => mounted.rerender(view(open, promptId)),
    dispose: () => {
      mounted.unmount();
      queryClient.clear();
    },
  };
};
const execute = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole('button', { name: 'Execute' }));
const drawerErrors = () =>
  within(screen.getByRole('dialog'))
    .queryAllByRole('alert')
    .filter((element) => element.getAttribute('data-ds-component') === 'Alert');
const visibleError = async (message: string) => {
  await waitFor(() => {
    const alerts = drawerErrors();
    expect(alerts).toHaveLength(1);
    expect(alerts[0]).toBeVisible();
    expect(alerts[0]).toHaveTextContent(message);
  });
};
const settle = async (pending: ReturnType<typeof deferred>, result: unknown) => {
  await act(async () => {
    pending.resolve(result);
    await pending.promise;
  });
  await waitFor(() => expect(screen.queryByText('Executing...')).not.toBeInTheDocument());
};

describe('ExecutePlayground persistent drawer refusal', () => {
  it.each(['envelope', 'throw'] as const)(
    'keeps %s failures in the actual drawer Alert',
    async (mode) => {
      const message = mode === 'envelope' ? missing.error : 'Connection unavailable.';
      if (mode === 'envelope') mocks.executePrompt.mockResolvedValue(missing);
      else mocks.executePrompt.mockRejectedValue(new Error(message));
      const mounted = await mount();
      try {
        await execute(mounted.user);
        await visibleError(message);
        expect(screen.queryByText('Output')).not.toBeInTheDocument();
        expect(mocks.executePrompt).toHaveBeenCalledExactlyOnceWith({
          prompt_id: 42,
          inputs: {},
          provider: 'ollama',
          model: 'qwen2.5:0.5b',
        });
      } finally {
        mounted.dispose();
      }
    }
  );

  it('clears previous successful output and statistics when the retained refusal arrives', async () => {
    mocks.executePrompt.mockResolvedValueOnce(success).mockResolvedValueOnce(missing);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await screen.findByText('Dr. Mira Vale');
      expect(screen.getByText('159 tokens')).toBeInTheDocument();
      expect(screen.getByText('0.06s')).toBeInTheDocument();
      await execute(mounted.user);
      await visibleError(missing.error);
      expect(screen.queryByText('Dr. Mira Vale')).not.toBeInTheDocument();
      expect(screen.queryByText('159 tokens')).not.toBeInTheDocument();
      expect(screen.queryByText('0.06s')).not.toBeInTheDocument();
    } finally {
      mounted.dispose();
    }
  });

  it('clears the prior error at the next execute and retains subsequent success', async () => {
    const pending = deferred();
    mocks.executePrompt.mockResolvedValueOnce(missing).mockReturnValueOnce(pending.promise);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await visibleError(missing.error);
      fireEvent.change(screen.getByLabelText('Inputs (JSON)'), {
        target: { value: JSON.stringify({ source: 'Dr. Mira Vale directs Rowan Observatory.' }) },
      });
      await execute(mounted.user);
      await screen.findByText('Executing...');
      expect(drawerErrors()).toHaveLength(0);
      await settle(pending, success);
      await screen.findByText('Dr. Mira Vale');
      expect(drawerErrors()).toHaveLength(0);
      expect(screen.getByText('159 tokens')).toBeInTheDocument();
    } finally {
      mounted.dispose();
    }
  });

  it('clears the retained error and inputs on a changed prompt', async () => {
    mocks.executePrompt.mockResolvedValue(missing);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await visibleError(missing.error);
      mounted.update(true, 43);
      await screen.findByText('Prompt 43');
      expect(drawerErrors()).toHaveLength(0);
      expect(screen.getByLabelText('Inputs (JSON)')).toHaveValue('{}');
    } finally {
      mounted.dispose();
    }
  });

  it('clears retained errors through actual close and same-prompt reopen', async () => {
    mocks.executePrompt.mockResolvedValue(missing);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await visibleError(missing.error);
      await mounted.user.click(screen.getByRole('button', { name: 'Close' }));
      expect(mounted.onClose).toHaveBeenCalledTimes(1);
      mounted.update(false);
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      mounted.update(true);
      await screen.findByText('Prompt 42');
      expect(drawerErrors()).toHaveLength(0);
      expect(screen.queryByText('Dr. Mira Vale')).not.toBeInTheDocument();
    } finally {
      mounted.dispose();
    }
  });

  it.each(['success', 'failure'] as const)(
    'retires a late %s after close and same-prompt reopen',
    async (kind) => {
      const pending = deferred();
      mocks.executePrompt.mockReturnValue(pending.promise);
      const mounted = await mount();
      try {
        await execute(mounted.user);
        await screen.findByText('Executing...');
        mounted.update(false);
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        mounted.update(true);
        await screen.findByText('Prompt 42');
        await settle(pending, kind === 'success' ? success : missing);
        expect(drawerErrors()).toHaveLength(0);
        expect(screen.queryByText('Dr. Mira Vale')).not.toBeInTheDocument();
        expect(screen.queryByText('159 tokens')).not.toBeInTheDocument();
        expect(notification.error).not.toHaveBeenCalled();
      } finally {
        mounted.dispose();
      }
    }
  );

  it('retires late successful output after a changed prompt', async () => {
    const pending = deferred();
    mocks.executePrompt.mockReturnValue(pending.promise);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await screen.findByText('Executing...');
      mounted.update(true, 43);
      await screen.findByText('Prompt 43');
      await settle(pending, success);
      expect(drawerErrors()).toHaveLength(0);
      expect(screen.queryByText('Dr. Mira Vale')).not.toBeInTheDocument();
      expect(screen.queryByText('159 tokens')).not.toBeInTheDocument();
    } finally {
      mounted.dispose();
    }
  });

  it('does not notify a late refusal after the caller unmounts the drawer', async () => {
    const pending = deferred();
    mocks.executePrompt.mockReturnValue(pending.promise);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await screen.findByText('Executing...');
      mounted.unmount();
      await act(async () => {
        pending.resolve(missing);
        await pending.promise;
      });
      await waitFor(() => expect(mocks.executePrompt).toHaveBeenCalledTimes(1));
      expect(notification.error).not.toHaveBeenCalled();
    } finally {
      mounted.dispose();
    }
  });

  it('retains actual Form JSON validation feedback in the drawer Alert', async () => {
    const mounted = await mount();
    try {
      fireEvent.change(screen.getByLabelText('Inputs (JSON)'), { target: { value: '{broken' } });
      await execute(mounted.user);
      await visibleError('Invalid JSON format');
      expect(mocks.executePrompt).not.toHaveBeenCalled();
    } finally {
      mounted.dispose();
    }
  });

  it.each(['provider', 'model'] as const)(
    'retains the effective-default %s refusal without dispatch',
    async (field) => {
      mocks.getLlmProviders.mockResolvedValue({
        ok: true,
        status: 200,
        data:
          field === 'provider'
            ? { ...catalog, default_provider: 'unavailable-provider' }
            : {
                default_provider: 'ollama',
                providers: [
                  { name: 'ollama', models: ['qwen2.5:0.5b'], default_model: 'unavailable-model' },
                ],
              },
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
        await screen.findByText(
          `Default: ${field === 'provider' ? 'unavailable-provider' : 'unavailable-model'}`,
          { exact: true }
        );
        await execute(userEvent.setup());
        await visibleError(
          field === 'provider'
            ? 'Provider is not available on this server'
            : 'Model is not available for the selected provider'
        );
        expect(mocks.executePrompt).not.toHaveBeenCalled();
      } finally {
        unmount();
        queryClient.clear();
      }
    }
  );

  it('uses the existing bounded sanitizer before persistent display and notification', async () => {
    mocks.executePrompt.mockRejectedValue(
      new Error(
        'Provider unavailable /Users/private/config.txt token=sk_test_secret https://private.example/api\nraw provider traceback'
      )
    );
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await visibleError(
        'Provider unavailable [redacted-path] token=[redacted-secret] [server-url]'
      );
      expect(document.body.textContent).not.toContain('/Users/private');
      expect(document.body.textContent).not.toContain('sk_test_secret');
      expect(document.body.textContent).not.toContain('private.example');
      expect(document.body.textContent).not.toContain('raw provider traceback');
      expect(notification.error).toHaveBeenCalledWith({
        message: 'Error',
        description: 'Provider unavailable [redacted-path] token=[redacted-secret] [server-url]',
      });
    } finally {
      mounted.dispose();
    }
  });

  it('preserves positive output and stats without an error Alert', async () => {
    mocks.executePrompt.mockResolvedValue(success);
    const mounted = await mount();
    try {
      await execute(mounted.user);
      await screen.findByText('Dr. Mira Vale');
      expect(screen.getByText('159 tokens')).toBeInTheDocument();
      expect(screen.getByText('0.06s')).toBeInTheDocument();
      expect(drawerErrors()).toHaveLength(0);
      expect(notification.error).not.toHaveBeenCalled();
    } finally {
      mounted.dispose();
    }
  });
});
