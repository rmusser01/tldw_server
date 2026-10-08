// Run against an existing, authenticated Research Workspace tab with a selected
// workspace: node e2e/workflows/source-view-cancel-layout.cdp.mjs CDP_URL PAGE_URL
// No server, browser, profile, config, source or saved-view mutation is created.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const [cdpUrl, pageUrl] = process.argv.slice(2);
assert(cdpUrl && pageUrl, 'Pass the owned CDP endpoint and existing page URL');
const browser = await chromium.connectOverCDP(cdpUrl);
const receipt = { geometry: [], events: [], writes: 0 };
try {
  const pages = browser.contexts().flatMap((context) => context.pages());
  const matches = pages.filter((page) => page.url() === pageUrl);
  assert.equal(matches.length, 1, 'The existing workspace tab must be unique');
  const page = matches[0];
  page.on('request', (request) => {
    if (request.method() !== 'GET' && new URL(request.url()).pathname.includes('/source-views'))
      receipt.writes += 1;
  });
  const input = page.getByRole('textbox', { name: 'View name', exact: true });
  const modal = input.locator('xpath=ancestor::*[@role="dialog"][1]');
  const cancel = modal.getByRole('button', { name: 'Cancel', exact: true });
  const open = async () => {
    await page.getByRole('button', { name: 'Save source view', exact: true }).click();
    await input.waitFor();
    // Observe native enter completion and the host's normal initial focus.
    await page.waitForFunction(() => {
      const input = document.querySelector('input[aria-label="View name"]');
      return (
        input === document.activeElement &&
        input
          .closest('[role="dialog"]')
          .getAnimations({ subtree: true })
          .every((animation) => animation.playState === 'finished')
      );
    });
    assert.equal(await input.inputValue(), '');
  };
  const dismiss = async () => {
    await page.keyboard.press('Escape');
    await modal.waitFor({ state: 'hidden' });
  };
  if (await input.isVisible()) await dismiss();
  await open();
  const before = await cancel.boundingBox();
  // Distinct diagnostic: keyboard blur, never a repeated failed pointer click.
  await input.press('Tab');
  await modal.getByRole('alert').waitFor();
  const after = await cancel.boundingBox();
  receipt.geometry.push({ before, after });
  assert.deepEqual(after, before, 'Name validation must not move the native Cancel target');
  assert.equal(await input.getAttribute('aria-invalid'), 'true');
  assert(await input.getAttribute('aria-describedby'));
  await dismiss();

  // Only reached once the geometry regression passes: real coordinate delivery
  // from an untouched draft, including the input blur between down and up.
  await open();
  const target = await cancel.boundingBox();
  await page.evaluate(() => {
    window.__sourceViewCancelEvents = [];
    window.__sourceViewCancelListener = (event) => {
      window.__sourceViewCancelEvents.push({
        type: event.type,
        trusted: event.isTrusted,
        button: event.target.closest?.('button')?.textContent.trim() ?? null,
      });
    };
    for (const type of ['pointerdown', 'pointerup', 'click'])
      document.addEventListener(type, window.__sourceViewCancelListener, true);
  });
  try {
    await page.mouse.move(target.x + target.width / 2, target.y + target.height / 2);
    await page.mouse.down();
    try {
      await modal.getByRole('alert').waitFor();
      assert.deepEqual(await cancel.boundingBox(), target);
    } finally {
      await page.mouse.up();
    }
    await modal.waitFor({ state: 'hidden' });
    receipt.events = await page.evaluate(() => window.__sourceViewCancelEvents);
    assert.deepEqual(
      receipt.events,
      ['pointerdown', 'pointerup', 'click'].map((type) => ({
        type,
        trusted: true,
        button: 'Cancel',
      }))
    );
    assert.equal(receipt.writes, 0);
  } finally {
    await page.evaluate(() => {
      for (const type of ['pointerdown', 'pointerup', 'click'])
        document.removeEventListener(type, window.__sourceViewCancelListener, true);
      delete window.__sourceViewCancelEvents;
      delete window.__sourceViewCancelListener;
    });
  }
} finally {
  console.log(JSON.stringify(receipt));
  await browser.close();
}
