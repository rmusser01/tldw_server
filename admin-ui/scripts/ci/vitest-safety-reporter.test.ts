import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { afterEach, describe, expect, test } from 'vitest';

import VitestSafetyReporter from './vitest-safety-reporter.mjs';

const temporaryDirectories: string[] = [];
const execFileAsync = promisify(execFile);
const adminUiRoot = process.cwd();
const reporterPath = resolve(adminUiRoot, 'scripts/ci/vitest-safety-reporter.mjs');

async function reportPath(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'vitest-safety-reporter-'));
  temporaryDirectories.push(directory);
  return join(directory, 'report.json');
}

function testModule(testCount: number, moduleErrorCount = 0) {
  const tests = Array.from({ length: testCount }, () => ({}));
  const errors = Array.from({ length: moduleErrorCount }, () => ({}));
  return {
    errors: () => errors,
    children: {
      allTests: () => tests.values(),
      allSuites: () => [].values(),
    },
  };
}

async function readReport(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, 'utf-8'));
}

afterEach(async () => {
  delete process.env.TLDW_VITEST_SAFETY_REPORT;
  await Promise.all(temporaryDirectories.splice(0).map(
    (directory) => rm(directory, { recursive: true, force: true }),
  ));
});

describe('VitestSafetyReporter', () => {
  test('uses the environment path when Vitest supplies reporter options', async () => {
    const path = await reportPath();
    process.env.TLDW_VITEST_SAFETY_REPORT = path;
    const reporter = new VitestSafetyReporter({});

    await reporter.onTestRunEnd([testModule(1)], [], 'passed');

    await expect(readReport(path)).resolves.toMatchObject({ testCount: 1 });
  });

  test('records non-zero discovery and a clean completed run', async () => {
    const path = await reportPath();
    const reporter = new VitestSafetyReporter(path);

    await reporter.onTestRunEnd([testModule(3)], [], 'passed');

    await expect(readReport(path)).resolves.toEqual({
      schemaVersion: 1,
      reason: 'passed',
      moduleCount: 1,
      testCount: 3,
      unhandledErrorCount: 0,
      moduleErrorCount: 0,
      hookErrorCount: 0,
    });
  });

  test('counts unhandled and module-level errors without serializing details', async () => {
    const path = await reportPath();
    const reporter = new VitestSafetyReporter(path);

    await reporter.onTestRunEnd(
      [testModule(1, 2)],
      [{ message: 'sensitive unhandled detail' }],
      'failed',
    );

    const serializedReport = await readFile(path, 'utf-8');
    expect(JSON.parse(serializedReport)).toMatchObject({
      reason: 'failed',
      unhandledErrorCount: 1,
      moduleErrorCount: 2,
    });
    expect(serializedReport).not.toContain('sensitive unhandled detail');
  });

  test.each(['beforeEach', 'afterEach', 'beforeAll', 'afterAll'])(
    'reports a %s failure from a real Vitest lifecycle',
    async (hookName) => {
      const directory = await mkdtemp(join(tmpdir(), 'vitest-hook-failure-'));
      temporaryDirectories.push(directory);
      const safetyReportPath = join(directory, 'safety-report.json');
      await writeFile(
        join(directory, 'hook-failure.test.mjs'),
        [
          `${hookName}(() => { throw new Error('hook failed'); });`,
          "test('exercises the hook lifecycle', () => { expect(true).toBe(true); });",
          '',
        ].join('\n'),
        { encoding: 'utf-8' },
      );

      await expect(execFileAsync(
        process.execPath,
        [
          join(adminUiRoot, 'node_modules/vitest/vitest.mjs'),
          'run',
          'hook-failure.test.mjs',
          '--root',
          directory,
          '--globals',
          `--reporter=${reporterPath}`,
        ],
        {
          cwd: adminUiRoot,
          encoding: 'utf-8',
          env: {
            ...process.env,
            TLDW_VITEST_SAFETY_REPORT: safetyReportPath,
          },
        },
      )).rejects.toMatchObject({ code: 1 });

      await expect(readReport(safetyReportPath)).resolves.toMatchObject({
        reason: 'failed',
        hookErrorCount: 1,
      });
    },
  );
});
