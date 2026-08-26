import { execFile } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { afterEach, describe, expect, test } from 'vitest';

import VitestSafetyReporter from './vitest-safety-reporter.mjs';

const temporaryDirectories: string[] = [];
const execFileAsync = promisify(execFile);
const adminUiRoot = process.cwd();
const reporterPath = resolve(adminUiRoot, 'scripts/ci/vitest-safety-reporter.mjs');

function reportPath(): string {
  const directory = mkdtempSync(join(tmpdir(), 'vitest-safety-reporter-'));
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

function readReport(path: string) {
  return JSON.parse(readFileSync(path, 'utf-8'));
}

afterEach(() => {
  delete process.env.TLDW_VITEST_SAFETY_REPORT;
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe('VitestSafetyReporter', () => {
  test('uses the environment path when Vitest supplies reporter options', async () => {
    const path = reportPath();
    process.env.TLDW_VITEST_SAFETY_REPORT = path;
    const reporter = new VitestSafetyReporter({});

    await reporter.onTestRunEnd([testModule(1)], [], 'passed');

    expect(readReport(path).testCount).toBe(1);
  });

  test('records non-zero discovery and a clean completed run', async () => {
    const path = reportPath();
    const reporter = new VitestSafetyReporter(path);

    await reporter.onTestRunEnd([testModule(3)], [], 'passed');

    expect(readReport(path)).toEqual({
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
    const path = reportPath();
    const reporter = new VitestSafetyReporter(path);

    await reporter.onTestRunEnd(
      [testModule(1, 2)],
      [{ message: 'sensitive unhandled detail' }],
      'failed',
    );

    const serializedReport = readFileSync(path, 'utf-8');
    expect(JSON.parse(serializedReport)).toMatchObject({
      reason: 'failed',
      unhandledErrorCount: 1,
      moduleErrorCount: 2,
    });
    expect(serializedReport).not.toContain('sensitive unhandled detail');
  });

  test('reports a setup-hook failure from a real Vitest lifecycle', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'vitest-hook-failure-'));
    temporaryDirectories.push(directory);
    const safetyReportPath = join(directory, 'safety-report.json');
    writeFileSync(
      join(directory, 'hook-failure.test.mjs'),
      [
        "beforeEach(() => { throw new Error('setup failed'); });",
        "test('never reaches its assertion', () => { expect(true).toBe(true); });",
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

    expect(readReport(safetyReportPath)).toMatchObject({
      reason: 'failed',
      hookErrorCount: 1,
    });
  });
});
