import { expect, it, vi } from 'vitest';
import i18n from 'i18next';
import ICU from '@tldw/ui/i18n/icu-format';

vi.mock('wxt/browser', () => ({ browser: { storage: { local: { get: async () => ({}) } } } }));

it('registers optional Help before initialization without replacing existing English resources', async () => {
  expect(i18n.isInitialized).toBeFalsy();
  await import('@/components/Common/PageHelpModal');
  await i18n.use(ICU).init({
    lng: 'en',
    fallbackLng: 'en',
    resources: { en: { tutorials: { actions: { start: 'Custom start' } } } },
  });
  expect(i18n.t('tutorials:steps', { count: 1 })).toBe('1 step');
  expect(i18n.t('tutorials:actions.start')).toBe('Custom start');
});
