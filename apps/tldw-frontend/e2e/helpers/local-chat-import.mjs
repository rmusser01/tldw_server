import assert from 'node:assert/strict';

/** Exercise the normal FileReader/Dexie import without a database or route double. */
export default async function runLocalChatImport(page, file) {
  await page
    .getByRole('button', { name: 'Import Data', exact: true })
    .waitFor({ state: 'visible', timeout: 15000 });
  assert.equal(
    await page
      .locator('.ant-notification-notice')
      .filter({ hasText: /Imported data successfully|Import error/ })
      .count(),
    0,
    'Require a fresh import outcome'
  );
  const end = Date.now() + 15000;
  const terminal = page.waitForFunction(
    () =>
      [...document.querySelectorAll('.ant-notification-notice')].some((e) =>
        /Imported data successfully|Import error/.test(e.textContent || '')
      ),
    null,
    { timeout: 15000 }
  );
  terminal.catch(() => {});
  const chooser = page.waitForEvent('filechooser', { timeout: Math.max(1, end - Date.now()) });
  chooser.catch(() => {});
  await page
    .getByRole('button', { name: 'Import Data', exact: true })
    .click({ timeout: Math.max(1, end - Date.now()) });
  await (await chooser).setFiles(file, { timeout: Math.max(1, end - Date.now()) });
  await terminal;
  assert.ok(Date.now() <= end, 'Import acknowledgement deadline');
  const notice = page
    .locator('.ant-notification-notice')
    .filter({ hasText: /Imported data successfully|Import error/ })
    .last();
  const message = await notice.innerText();
  assert.match(message, /Imported data successfully/, 'Normal chat restore must commit atomically');
  assert.doesNotMatch(message, /Import error|SubTransaction|not included in parent transaction/);
  // Cancel only the scheduled page refresh; the acknowledged import is committed.
  await notice
    .getByRole('button', { name: 'Cancel', exact: true })
    .click({ timeout: Math.max(1, end - Date.now()) });
  return { importCommitted: true, noTransactionScopeError: true };
}
