import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, describe, expect, test } from 'vitest';

import VitestExecutionOrderReporter from '../../../Helper_Scripts/ci/vitest_execution_order_reporter.mjs';

const temporaryDirectories: string[] = [];

async function reportPath(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'vitest-order-reporter-'));
  temporaryDirectories.push(directory);
  return join(directory, 'nested', 'execution-order.json');
}

afterEach(async () => {
  delete process.env.TLDW_VITEST_ORDER_REPORT;
  await Promise.all(temporaryDirectories.splice(0).map(
    (directory) => rm(directory, { recursive: true, force: true }),
  ));
});

describe('VitestExecutionOrderReporter', () => {
  test('atomically replaces the report with normalized module order', async () => {
    const path = await reportPath();
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, '{"stale":true}\n', 'utf-8');
    const reporter = new VitestExecutionOrderReporter(path);

    reporter.onTestModuleStart({ relativeModuleId: 'app\\alpha.test.ts' });
    reporter.onTestModuleStart({ relativeModuleId: 'app/beta.test.ts' });
    reporter.onTestRunEnd([{}, {}]);

    await expect(readFile(path, 'utf-8')).resolves.toBe(
      '{"schemaVersion":1,"moduleCount":2,"modules":["app/alpha.test.ts","app/beta.test.ts"]}\n',
    );
    await expect(readdir(dirname(path))).resolves.toEqual(['execution-order.json']);
  });

  test('uses the environment path when Vitest supplies reporter options', async () => {
    const path = await reportPath();
    process.env.TLDW_VITEST_ORDER_REPORT = path;
    const reporter = new VitestExecutionOrderReporter({});

    reporter.onTestModuleStart({ relativeModuleId: 'app/example.test.ts' });
    reporter.onTestRunEnd([{}]);

    await expect(readFile(path, 'utf-8')).resolves.toContain(
      '"modules":["app/example.test.ts"]',
    );
  });
});
