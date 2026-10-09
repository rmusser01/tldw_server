import { createElement } from 'react';
import { render, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import i18n from '../lib/i18n-web';

vi.mock('wxt/browser', () => ({ browser: { storage: { local: { get: async () => ({}) } } } }));
vi.mock('@plasmohq/storage/hook', () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback],
}));

it('loads the tutorial namespace with real Guides before Help while keeping startup unloaded', async () => {
  await i18n.changeLanguage('en');
  expect(i18n.hasResourceBundle('en', 'tutorials')).toBe(false);
  const { QuickChatGuidesPanel } =
    await import('@/components/Common/QuickChatHelper/QuickChatGuidesPanel');
  const view = render(
    createElement(QuickChatGuidesPanel, {
      currentRoute: '/chat',
      onAskGuide: vi.fn(),
      onOpenRoute: vi.fn(),
    })
  );
  expect(
    within(view.getByTestId('quick-chat-guides-tutorial-playground-basics')).getByText('5 steps')
  ).toBeVisible();
  await import('@/components/Common/PageHelpModal');
  expect(i18n.t('tutorials:steps', { count: 5 })).toBe('5 steps');
});
