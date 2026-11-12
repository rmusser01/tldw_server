import React from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, queryAllByRole, render, screen, waitFor } from '@testing-library/react';
import type { Page } from '@playwright/test';
import { App, ConfigProvider } from 'antd';
import { createInstance } from 'i18next';
import { I18nextProvider } from 'react-i18next';
import { MemoryRouter } from 'react-router-dom';
import { ServerConnectionCard } from '@/components/Common/ServerConnectionCard';
import { dismissConnectionModals, waitForConnection } from '../helpers';

vi.mock('@/hooks/useConnectionState', () => ({
  useConnectionState: () => ({ phase: 'error', isChecking: false, serverUrl: 'http://127.0.0.1:8000' }),
  useConnectionUxState: () => ({ uxState: 'error_unreachable', hasCompletedFirstRun: true }),
  useConnectionActions: () => ({ checkOnce: vi.fn(), setDemoMode: vi.fn() }),
}));
vi.mock('@/context/demo-mode', () => ({
  useDemoMode: () => ({ setDemoEnabled: vi.fn() }),
}));

// Substitute only Playwright's browser boundary: selectors resolve against the
// mounted DOM, clicks invoke its handlers, and evaluate runs the real callback.
function domPage(): Page {
  const locator = (resolve: () => HTMLElement[]) => ({
    getByRole: (role: Parameters<typeof queryAllByRole>[1], options: Parameters<typeof queryAllByRole>[2]) =>
      locator(() => resolve().flatMap(element => queryAllByRole(element, role, options))),
    isVisible: async () => {
      const elements = resolve();
      if (elements.length > 1) throw new Error('Strict locator matched multiple elements');
      return elements.length === 1 && !elements[0].hidden;
    },
    click: async () => {
      const elements = resolve();
      if (elements.length !== 1) throw new Error('Expected exactly one click target');
      await act(async () => { fireEvent.click(elements[0]); });
    },
    waitFor: async ({ state }: { state: string }) => {
      await waitFor(() => {
        expect(resolve().length === 0).toBe(state === 'hidden');
      });
    },
  });
  return {
    locator: (selector: string) => locator(() => Array.from(document.querySelectorAll<HTMLElement>(selector))),
    getByRole: (role: Parameters<typeof queryAllByRole>[1], options: Parameters<typeof queryAllByRole>[2]) =>
      locator(() => queryAllByRole(document.body, role, options)),
    evaluate: async (callback: () => unknown) => callback(),
    waitForLoadState: async () => {},
    waitForFunction: async (callback: () => boolean) => { expect(callback()).toBe(true); },
  } as unknown as Page;
}

afterEach(() => {
  cleanup();
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

it('preserves the unread inbox item while waiting for an established connection', async () => {
  document.body.innerHTML = '<div id="__next"><p>Unread: 1</p><ul><li>Run completed <button>Dismiss</button></li></ul></div>';
  document.querySelector('button')!.onclick = () => {
    document.querySelector('li')!.remove();
    document.querySelector('p')!.textContent = 'Unread: 0';
  };
  vi.stubGlobal('__tldw_useConnectionStore', {
    getState: () => ({ state: { isConnected: true, phase: 'connected' } }),
  });

  await waitForConnection(domPage());

  expect(screen.getByText('Run completed')).toBeInTheDocument();
  expect(screen.getByText('Unread: 1')).toBeInTheDocument();
});

it.each(['Dismiss', 'Dismiss error', 'Dismiss workflow error'])(
  'preserves unrelated feature controls named %s and their portals',
  async label => {
    document.body.innerHTML = `<div id="tldw-portal-root"><div class="ant-modal-root"><div class="ant-modal-mask"></div><div class="ant-modal-wrap"><section role="dialog" aria-label="Feature details"><button>${label}</button></section></div></div></div><nextjs-portal><div>Development tools</div></nextjs-portal>`;
    document.querySelector('button')!.onclick = () => document.querySelector('section')!.remove();

    await dismissConnectionModals(domPage());

    expect(screen.getByRole('dialog', { name: 'Feature details' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    expect(document.querySelector('.ant-modal-mask')).not.toBeNull();
    expect(screen.getByText('Development tools')).toBeInTheDocument();
  }
);

it('dismisses the real connection toast through its control when feature Dismiss buttons coexist', async () => {
  const i18n = createInstance();
  await i18n.init({ lng: 'en', resources: {} });
  render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter>
        <ConfigProvider theme={{ token: { motion: false } }}>
          <App><ServerConnectionCard showToastOnError /><button>Dismiss</button></App>
        </ConfigProvider>
      </MemoryRouter>
    </I18nextProvider>
  );
  await waitFor(() => expect(document.querySelector('.tldw-connection-toast')).not.toBeNull());

  await dismissConnectionModals(domPage());

  await waitFor(() => expect(document.querySelector('.tldw-connection-toast')).toBeNull());
  expect(screen.getByRole('button', { name: /^Dismiss$/ })).toBeInTheDocument();
});
