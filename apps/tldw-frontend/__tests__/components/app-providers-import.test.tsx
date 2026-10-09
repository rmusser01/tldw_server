import React, { useState } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clearSetting, setSetting } from '@/services/settings/registry';
import { SPLASH_DISABLED_SETTING } from '@/services/settings/ui-settings';
import { SPLASH_TRIGGER_EVENT } from '@/services/splash-events';
import { useSetting } from '@/hooks/useSetting';
import type { SplashCard } from '@/components/Common/SplashScreen/engine/types';

const mocks = vi.hoisted(() => ({
  loadVisual: vi.fn(),
  waitForVisual: () => Promise.resolve(),
}));

function SplashPreview({
  card,
  message,
  onDismiss,
}: {
  card: SplashCard;
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div role="dialog" aria-label="Splash preview">
      <p>{card.title}</p>
      <p>{message}</p>
      <button onClick={onDismiss}>Dismiss splash</button>
    </div>
  );
}

function WorkingChat() {
  const [clicks, setClicks] = useState(0);
  const [disabled] = useSetting(SPLASH_DISABLED_SETTING);
  return (
    <>
      <button onClick={() => setClicks(clicks + 1)}>Continue chat {clicks}</button>
      <output>{disabled ? 'Splash disabled' : 'Splash enabled'}</output>
    </>
  );
}

vi.mock('antd', () => ({
  App: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  ConfigProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Empty: () => <div data-testid="empty-state" />,
}));

vi.mock('@ant-design/cssinjs', () => ({
  StyleProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('@tanstack/react-query', () => ({
  QueryClientProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? _key,
    i18n: {
      resolvedLanguage: 'en',
      language: 'en',
      dir: () => 'ltr',
    },
  }),
}));

vi.mock('@/hooks/useTheme', () => ({
  useTheme: () => ({ antdTheme: {} }),
}));

vi.mock('@/components/Common/PageAssistProvider', () => ({
  PageAssistProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('@/context/FontSizeProvider', () => ({
  FontSizeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('@/context/demo-mode', () => ({
  DemoModeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('@/services/query-client', () => ({
  getQueryClient: () => ({}),
}));

vi.mock('@web/components/notifications/NotificationToastBridge', () => ({
  NotificationToastBridge: () => <div data-testid="notification-toast-bridge" />,
}));

vi.mock('@/utils/antd-notification-compat', () => ({
  patchStaticAntdNotificationCompat: vi.fn(),
}));

vi.mock('@web/components/ui/ToastProvider', () => ({
  ToastProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe('AppProviders', () => {
  beforeEach(async () => {
    vi.resetModules();
    // Use Next's actual optional-chunk error path instead of the global React.lazy double.
    vi.doUnmock('next/dynamic');
    mocks.loadVisual.mockReset();
    mocks.waitForVisual = () => Promise.resolve();
    await clearSetting(SPLASH_DISABLED_SETTING);
    localStorage.clear();
    vi.doMock('@/components/Common/SplashScreen', () => {
      mocks.loadVisual();
      return { SplashOverlay: SplashPreview };
    });
    vi.doMock('@/components/Common/SplashScreen/SplashOverlay', async () => {
      mocks.loadVisual();
      await mocks.waitForVisual();
      return { default: SplashPreview };
    });
  });

  it('imports and renders with the web notification bridge path', async () => {
    const { AppProviders } = await import('@web/components/AppProviders');

    render(
      <AppProviders>
        <div>child content</div>
      </AppProviders>
    );

    expect(screen.getByText('child content')).toBeInTheDocument();
    expect(screen.getByTestId('notification-toast-bridge')).toBeInTheDocument();
  });

  it('keeps the notification bridge unmounted when notifications are disabled', async () => {
    const { AppProviders } = await import('@web/components/AppProviders');

    render(
      <AppProviders enableNotifications={false}>
        <div>child content</div>
      </AppProviders>
    );

    expect(screen.getByText('child content')).toBeInTheDocument();
    expect(screen.queryByTestId('notification-toast-bridge')).not.toBeInTheDocument();
  });

  it('does not request hidden splash visuals', async () => {
    const { AppProviders } = await import('@web/components/AppProviders');
    render(
      <AppProviders>
        <WorkingChat />
      </AppProviders>
    );
    expect(screen.getByRole('button', { name: 'Continue chat 0' })).toBeEnabled();
    expect(mocks.loadVisual).not.toHaveBeenCalled();
  });

  it('keeps chat usable while visuals load, then displays and dismisses the requested splash', async () => {
    let release!: () => void;
    mocks.waitForVisual = () =>
      new Promise<void>((resolve) => {
        release = resolve;
      });
    const { AppProviders } = await import('@web/components/AppProviders');
    render(
      <AppProviders>
        <WorkingChat />
      </AppProviders>
    );
    act(() => window.dispatchEvent(new CustomEvent(SPLASH_TRIGGER_EVENT)));
    await waitFor(() => expect(mocks.loadVisual).toHaveBeenCalledOnce());
    expect(screen.queryByRole('dialog', { name: 'Splash preview' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Continue chat 0' }));
    expect(screen.getByRole('button', { name: 'Continue chat 1' })).toBeEnabled();
    await act(async () => release());
    expect(await screen.findByRole('dialog', { name: 'Splash preview' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss splash' }));
    expect(screen.queryByRole('dialog', { name: 'Splash preview' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Continue chat 1' })).toBeEnabled();
  });

  it('keeps chat usable and reports rejected optional loading', async () => {
    const failure = new Error('Optional splash chunk unavailable');
    mocks.waitForVisual = () => Promise.reject(failure);
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const { AppProviders } = await import('@web/components/AppProviders');
      render(
        <AppProviders>
          <WorkingChat />
        </AppProviders>
      );
      act(() => window.dispatchEvent(new CustomEvent(SPLASH_TRIGGER_EVENT)));
      await waitFor(() =>
        expect(warning).toHaveBeenCalledWith(
          'Unable to load splash screen',
          expect.objectContaining({ cause: failure })
        )
      );
      expect(screen.queryByRole('dialog', { name: 'Splash preview' })).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Continue chat 0' }));
      expect(screen.getByRole('button', { name: 'Continue chat 1' })).toBeEnabled();
    } finally {
      warning.mockRestore();
    }
  });

  it('preserves disabled preferences and explicit forced display', async () => {
    await setSetting(SPLASH_DISABLED_SETTING, true);
    const { AppProviders } = await import('@web/components/AppProviders');
    render(
      <AppProviders>
        <WorkingChat />
      </AppProviders>
    );
    await screen.findByText('Splash disabled');
    act(() => window.dispatchEvent(new CustomEvent(SPLASH_TRIGGER_EVENT)));
    expect(mocks.loadVisual).not.toHaveBeenCalled();
    act(() =>
      window.dispatchEvent(
        new CustomEvent(SPLASH_TRIGGER_EVENT, {
          detail: { force: true },
        })
      )
    );
    expect(await screen.findByRole('dialog', { name: 'Splash preview' })).toBeInTheDocument();
    expect(mocks.loadVisual).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss splash' }));
    expect(screen.queryByRole('dialog', { name: 'Splash preview' })).not.toBeInTheDocument();
  });
});
