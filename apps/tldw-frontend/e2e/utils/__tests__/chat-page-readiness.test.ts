import { afterEach, expect, it, vi } from 'vitest';
import type { Page } from '@playwright/test';
import { ChatPage } from '../page-objects/ChatPage';

// Only the browser boundary is substituted. The real page-object readiness
// method runs against a mounted handoff and development-tool portal.
function pageWithHandoff() {
  document.body.innerHTML = `<textarea placeholder="Type a message">Owned QA handoff</textarea>
    <nextjs-portal></nextjs-portal><div class="ant-modal-root">Recovery dialog</div>`;
  const locator = {
    first: () => locator,
    filter: () => locator,
    or: () => locator,
    waitFor: vi.fn(async () => {}),
    count: vi.fn(async () => 1),
  };
  const reload = vi.fn(async () => {
    document.querySelector('textarea')!.value = '';
  });
  const page = {
    locator: () => locator,
    getByRole: () => locator,
    getByTestId: () => locator,
    getByPlaceholder: () => locator,
    reload,
    evaluate: async (callback: () => void) => callback(),
  } as unknown as Page;
  return { page, reload, locator };
}

afterEach(() => { document.body.innerHTML = ''; });

it('preserves an in-memory handoff and visible dialogs when Next dev tools are present', async () => {
  const { page, reload } = pageWithHandoff();
  await new ChatPage(page).waitForReady();
  expect(document.querySelector('textarea')!.value).toBe('Owned QA handoff');
  expect(document.querySelector('.ant-modal-root')?.textContent).toBe('Recovery dialog');
  expect(reload).not.toHaveBeenCalled();
});

it('propagates a missing chat surface without reloading or hiding recovery UI', async () => {
  const { page, locator, reload } = pageWithHandoff();
  locator.waitFor.mockRejectedValue(new Error('Chat surface missing'));
  await expect(new ChatPage(page).waitForReady()).rejects.toThrow('Chat surface missing');
  expect(document.querySelector('.ant-modal-root')).not.toBeNull();
  expect(reload).not.toHaveBeenCalled();
});
