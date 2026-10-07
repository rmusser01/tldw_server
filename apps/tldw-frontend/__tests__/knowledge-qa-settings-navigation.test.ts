import { describe, expect, it, vi } from 'vitest';
import type { Page } from '@playwright/test';
import { KnowledgeQAPage } from '../e2e/utils/page-objects/KnowledgeQAPage';
vi.mock('@playwright/test', () => ({ expect: (locator: { isVisible: () => Promise<boolean> }) => ({ toBeVisible: async () => { if (!await locator.isVisible()) throw new Error('RAG dialog not open'); } }) }));
vi.mock('../e2e/utils/helpers', () => ({}));
function setup(missing = false, initiallyOpen = false) {
  let open = initiallyOpen;
  const dialog = { isVisible: async () => open };
  const globalClick = vi.fn(async () => {});
  const global = { click: globalClick, isVisible: async () => false, last: () => global };
  const missingControl = { click: async () => { throw new Error('QA settings control missing'); }, isVisible: async () => false };
  const click = vi.fn(async () => { open = true; });
  const local = { click, isVisible: async () => true };
  const shell = { getByRole: (_role: string, options: { name: string }) => options.name === 'Open Knowledge QA settings' && !missing ? local : missingControl };
  const page = { getByTestId: () => shell, getByRole: (role: string) => role === 'dialog' ? dialog : global };
  return { qa: new KnowledgeQAPage(page as unknown as Page), click, globalClick, dialog };
}
describe('Knowledge QA settings navigation', () => {
  it('opens the scoped QA dialog despite an unrelated global Settings control', async () => {
    const {qa,click,globalClick,dialog}=setup(); await qa.openSettings();
    expect(click).toHaveBeenCalledOnce(); expect(globalClick).not.toHaveBeenCalled(); expect(await dialog.isVisible()).toBe(true);
  });
  it('fails when the required QA control is absent instead of opening global Settings', async () => {
    const {qa,globalClick}=setup(true); await expect(qa.openSettings()).rejects.toThrow('QA settings control missing');
    expect(globalClick).not.toHaveBeenCalled();
  });
  it('keeps an already open QA settings dialog', async () => {
    const {qa,click,globalClick}=setup(false,true); await qa.openSettings(); expect(click).not.toHaveBeenCalled(); expect(globalClick).not.toHaveBeenCalled();
  });
});
