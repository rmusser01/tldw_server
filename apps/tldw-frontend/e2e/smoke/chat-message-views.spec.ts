import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import type { useStoreMessageOption } from '../../../packages/ui/src/store/option';

const prepare = async (page: Page, theme: string) => {
  await page.route('http://127.0.0.1:8000/**', route => route.fulfill({status:503,contentType:'application/json',body:'{"detail":"Offline preview"}'}));
  await page.route('**/api/v1/llm/models/metadata**', route => route.fulfill({status:200,contentType:'application/json',body:'{"models":[]}'}));
  await page.route('**/api/v1/llm/providers**', route => route.fulfill({status:200,contentType:'application/json',body:'{"providers":[]}'}));
  await page.addInitScript(theme => {
    localStorage.setItem('theme', theme);
    localStorage.setItem('assistant_setup_dismissed','true');
    localStorage.setItem('__tldw_first_run_complete','true');
    localStorage.setItem('__tldw_test_bypass','true');
    localStorage.setItem('playgroundChatContextRailVisible','false');
    localStorage.setItem('playgroundChatRuntimeRailVisible','false');
    localStorage.setItem('tldwConfig',JSON.stringify({serverUrl:'http://127.0.0.1:8000',authMode:'single-user',apiKey:'THIS-IS-A-SECURE-KEY-123-FAKE-KEY'}));
  },theme);
  await page.goto('/chat',{waitUntil:'domcontentloaded'});
  await expect(page.getByTestId('chat-input')).toBeVisible({timeout:60000});
  const exitFocus = page.getByRole('button', { name: /Exit focus/i });
  if (await exitFocus.isVisible()) await exitFocus.click();
  await expect(page.getByRole('combobox', { name: 'Message view', exact: true })).toBeVisible();
  await page.evaluate(() => {
    const store=(window as unknown as {__tldw_useStoreMessageOption: typeof useStoreMessageOption}).__tldw_useStoreMessageOption;
    store.getState().setMessages([
      {id:'example-question',isBot:false,role:'user',name:'You',createdAt:1792785600000,sources:[],message:'Synthetic preview: What should we verify before using these sources in a research brief?'},
      {id:'example-answer',isBot:true,role:'assistant',name:'Example assistant',modelName:'Example model',createdAt:1792785610000,message:'## Check the evidence first\n\nStart with the **original publication**, then compare its claims with an independent source. Keep the distinction between a documented finding and your interpretation visible.\n\n| Check | What to look for |\n| --- | --- |\n| Provenance | Author, publication date, and original data |\n| Corroboration | An independent account of the same claim |\n\nA useful next step is to collect the passages that support the brief, then note any unresolved disagreement.',sources:[{name:'Example field notes',url:'https://example.com/field-notes',content:'Synthetic source used for interface preview.'},{name:'Example research methods',url:'https://example.com/methods',content:'Synthetic source used for interface preview.'}],generationInfo:{usage:{prompt_tokens:1280,completion_tokens:342,total_tokens:1622}}}
    ]);
  });
  await expect(page.getByText('Check the evidence first',{exact:true})).toBeVisible();
  await page.evaluate(async()=>{await document.fonts.ready;await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame)});
};

const chooseView = async (page: Page, label: string) => {
  const picker = page.getByRole('combobox', { name: 'Message view', exact: true });
  await picker.press('ArrowDown');
  await page.getByRole('option', { name: new RegExp(`^${label}`) }).click();
  await expect(picker).toHaveAttribute('aria-expanded', 'false');
};

for (const width of [1440, 390]) {
  for (const theme of ['light', 'dark']) {
    for (const [view, label] of [['transcript', 'Transcript'], ['conversation', 'Conversation'], ['research', 'Research']]) {
      test(`${view} ${width} ${theme}`, async ({ page }, info) => {
        const errors: string[] = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
        await prepare(page, theme);
        if (view !== 'transcript') await chooseView(page, label);
        const answer = page.locator('[data-testid="chat-message"][data-role="assistant"]');
        await expect(answer).toHaveAttribute('data-message-view', view);
        const table = answer.getByRole('table');
        await expect(table).toBeVisible();
        if (width === 1440) {
          await expect.poll(() => table.evaluate(element => {
            const frame = element.closest('[role="region"]');
            const width = element.getBoundingClientRect().width;
            return frame && width > 0
              ? Math.abs(frame.getBoundingClientRect().width - width)
              : Infinity;
          })).toBeLessThanOrEqual(2);
        }
        await expect(page.getByTestId('message-sources').getByRole('button', { name: /2 sources/ })).toHaveAttribute('aria-expanded', String(view === 'research'));
        if (view === 'research') {
          await expect(answer.getByText(/1280 prompt/)).toContainText('1622 tokens');
          await expect(answer.getByText('Example field notes', { exact: true })).toBeVisible();
        }
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        const axe = await new AxeBuilder({ page }).include('[data-testid="chat-message"]').analyze();
        expect(axe.violations).toEqual([]);
        expect(errors).toEqual([]);
        await answer.evaluate(element => {
          let parent = element.parentElement;
          while (parent) {
            if (getComputedStyle(parent).overflowY === 'auto') { parent.scrollTop = 0; break; }
            parent = parent.parentElement;
          }
        });
        await page.mouse.move(0, 0);
        await page.screenshot({ path: info.outputPath(`${view}-${width}-${theme}.png`), animations: 'disabled' });
        if (view === 'research') {
          const feedback = answer.locator('[data-source-feedback]').first();
          await feedback.scrollIntoViewIfNeeded();
          const targets = await feedback.locator('button').evaluateAll(buttons => buttons.map(button => button.getBoundingClientRect().height));
          expect(targets.every(height => height >= 44)).toBe(true);
          await page.screenshot({ path: info.outputPath(`${view}-${width}-${theme}-sources.png`), animations: 'disabled' });
        }
      });
    }
  }
}

test('switching views preserves messages, drafts and explicit source disclosure', async ({ page }) => {
  await prepare(page, 'light');
  const input = page.getByTestId('chat-input');
  await input.fill('An unfinished follow-up');
  const before = await page.evaluate(() => (window as unknown as { __tldw_useStoreMessageOption: typeof useStoreMessageOption }).__tldw_useStoreMessageOption.getState().messages);
  await chooseView(page, 'Research');
  const sources = page.getByTestId('message-sources').getByRole('button', { name: /2 sources/ });
  await sources.click();
  await expect(sources).toHaveAttribute('aria-expanded', 'false');
  await chooseView(page, 'Conversation');
  await chooseView(page, 'Research');
  await expect(sources).toHaveAttribute('aria-expanded', 'false');
  await expect(input).toHaveValue('An unfinished follow-up');
  const after = await page.evaluate(() => (window as unknown as { __tldw_useStoreMessageOption: typeof useStoreMessageOption }).__tldw_useStoreMessageOption.getState().messages);
  expect(after).toEqual(before);
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Message view', exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Message view', exact: true }).locator('..')).toContainText('Research');
});
