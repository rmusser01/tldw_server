import type { Route } from '@playwright/test';
import { test, expect, seedAuth } from './smoke.setup';

const image = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC',
  'base64',
);
const statuses = ['draft', 'approved', 'rejected', 'hidden'];
const json = (route: Route, body: unknown, status = 200) => route.fulfill({ status, json: body });

for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
  test(`VN review keeps keyboard selection through a failed write at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await seedAuth(page);
    await page.addInitScript(() => localStorage.setItem('assistant_setup_dismissed', 'true'));
    await page.route(/\/api\/v1\/users\/me\/profile(?:\?.*)?$/, (route) =>
      json(route, { user: { id: 1, is_active: true } }));
    await page.route(/\/api\/v1\/persona\/profiles(?:\?.*)?$/, (route) => json(route, []));

    let items = Array.from({ length: 24 }, (_, index) => ({
      id: index + 1,
      pack_id: 7,
      slot_id: Math.floor(index / 3) + 1,
      variant_index: index % 3,
      generated_file_id: 200 + index,
      mime_type: 'image/png',
      width: 512,
      height: 768,
      review_status: statuses[index % statuses.length],
      preferred: false,
      source: 'generated',
    }));
    const writes: unknown[] = [];
    let imageRequests = 0;
    await page.route(/\/api\/v1\/vn\/vn-assets(?:\/.*)?$/, (route) => {
      const path = new URL(route.request().url()).pathname.replace('/api/v1/vn/vn-assets', '');
      if (path === '/starter-matrices') return json(route, { matrices: [] });
      if (path === '/packs') return json(route, [
        { id: 7, owner_user_id: 1, title: 'Review evidence pack', primary_character_id: 42, status: 'draft' },
      ]);
      if (path === '/packs/7/slots') return json(route, []);
      if (path === '/packs/7/items') return json(route, items);
      if (path === '/packs/7/readiness') return json(route, { ready: false, status: 'not_ready', warnings: [], errors: [] });
      if (path === '/packs/7/generation') return json(route, { status: 'idle' });
      if (path === '/packs/7/generation/preflight') return json(route, { warnings: [], slots: [] });
      if (path === '/packs/7/items/1/preview') {
        imageRequests += 1;
        return route.fulfill({ status: 200, contentType: 'image/png', body: image });
      }
      if (path === '/packs/7/items/bulk-review') {
        writes.push(route.request().postDataJSON());
        if (writes.length === 1) return json(route, { detail: 'Review temporarily unavailable' }, 409);
        items = items.map((item) => item.id === 1 ? { ...item, review_status: 'approved' } : item);
        return json(route, [items[0]]);
      }
      return json(route, { detail: 'Unexpected VN review test request' }, 404);
    });

    await page.goto('/vn-assets');
    const heading = page.getByRole('heading', { name: 'Review board' });
    const board = heading.locator('xpath=ancestor::section[1]');
    await expect(board.getByRole('checkbox')).toHaveCount(24);
    await heading.evaluate((element) => {
      element.scrollIntoView({ block: 'start' });
      window.scrollBy(0, -72);
    });
    await page.screenshot({ path: testInfo.outputPath('review-board.png') });

    const filter = board.getByRole('combobox', { name: 'Review status' });
    await filter.selectOption('draft');
    await expect(board.getByRole('checkbox')).toHaveCount(6);
    expect(imageRequests).toBe(0);
    await board.getByRole('button', { name: 'Preview item 1', exact: true }).click();
    const preview = board.getByRole('img', { name: 'Slot 1 · Variant 1 preview' });
    await expect(preview).toBeVisible();
    await expect.poll(() => preview.evaluate((element: HTMLImageElement) => element.naturalWidth)).toBeGreaterThan(0);

    const checkbox = board.getByRole('checkbox', { name: 'Select item 1', exact: true });
    await checkbox.focus();
    await checkbox.press('Space');
    await expect(checkbox).toBeChecked();
    const approve = board.getByRole('button', { name: 'Approve selected' });
    await approve.focus();
    await approve.press('Enter');
    await expect(board.getByRole('alert')).toContainText('Review temporarily unavailable');
    await expect(checkbox).toBeChecked();
    await expect(approve).toBeFocused();
    await page.screenshot({ path: testInfo.outputPath('review-retry.png') });

    await approve.press('Enter');
    await expect(checkbox).toHaveCount(0);
    await expect(filter).toBeFocused();
    expect(writes).toEqual([
      { item_ids: [1], review_status: 'approved' },
      { item_ids: [1], review_status: 'approved' },
    ]);
    expect(await board.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  });
}
