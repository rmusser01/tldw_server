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

type FakeSuite = {
  state: 'passed' | 'failed' | 'skipped' | 'pending';
  mode: 'run' | 'only' | 'skip' | 'todo';
};

function testModule(
  testCount: number,
  moduleErrorCount = 0,
  suites: FakeSuite[] = [],
  moduleState: 'passed' | 'failed' | 'skipped' | 'pending' = 'passed',
  testState: 'passed' | 'failed' | 'skipped' | 'pending' = 'passed',
) {
  const tests = Array.from({ length: testCount }, () => ({
    result: () => ({ state: testState }),
  }));
  const errors = Array.from({ length: moduleErrorCount }, () => ({}));
  return {
    state: () => moduleState,
    errors: () => errors,
    children: {
      allTests: () => tests.values(),
      allSuites: () => suites.map((suite) => ({
        state: () => suite.state,
        options: { mode: suite.mode },
        errors: () => [],
      })).values(),
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

    await reporter.onTestRunEnd([testModule(3, 0, [
      { state: 'passed', mode: 'run' },
      { state: 'skipped', mode: 'skip' },
    ])], [], 'passed');

    await expect(readReport(path)).resolves.toEqual({
      schemaVersion: 2,
      reason: 'passed',
      moduleCount: 1,
      suiteCount: 3,
      passedSuiteCount: 3,
      failedSuiteCount: 0,
      pendingSuiteCount: 0,
      incompleteSuiteCount: 0,
      incompleteTestCount: 0,
      testCount: 3,
      unhandledErrorCount: 0,
      moduleErrorCount: 0,
      hookErrorCount: 0,
    });
  });

  test('records unfinished runtime suites and tests separately from todo work', async () => {
    const path = await reportPath();
    const reporter = new VitestSafetyReporter(path);

    await reporter.onTestRunEnd([testModule(
      1,
      0,
      [{ state: 'pending', mode: 'run' }],
      'pending',
      'pending',
    )], [], 'passed');

    await expect(readReport(path)).resolves.toMatchObject({
      suiteCount: 2,
      passedSuiteCount: 0,
      failedSuiteCount: 0,
      pendingSuiteCount: 2,
      incompleteSuiteCount: 2,
      incompleteTestCount: 1,
    });
  });

  test('counts unhandled and module-level errors without serializing details', async () => {
    const path = await reportPath();
    const reporter = new VitestSafetyReporter(path);

    await reporter.onTestRunEnd(
      [testModule(1, 2, [], 'failed')],
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

  test('matches JSON suite counters for hidden skipped and todo suites', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'vitest-hidden-suite-status-'));
    temporaryDirectories.push(directory);
    const safetyReportPath = join(directory, 'safety-report.json');
    const jsonReportPath = join(directory, 'vitest-report.json');
    await writeFile(
      join(directory, 'hidden-suites.test.mjs'),
      [
        "describe('visible suite', () => { test('passes', () => expect(true).toBe(true)); });",
        "describe.skip('empty skipped suite', () => {});",
        "describe.todo('empty todo suite', () => {});",
        "describe('todo-only parent', () => { test.todo('future behavior'); });",
        "describe('mixed parent', () => {",
        "  test('current behavior', () => expect(true).toBe(true));",
        "  test.todo('future behavior');",
        "});",
        '',
      ].join('\n'),
      { encoding: 'utf-8' },
    );

    await execFileAsync(
      process.execPath,
      [
        join(adminUiRoot, 'node_modules/vitest/vitest.mjs'),
        'run',
        'hidden-suites.test.mjs',
        '--root',
        directory,
        '--globals',
        '--reporter=json',
        `--reporter=${reporterPath}`,
        `--outputFile.json=${jsonReportPath}`,
      ],
      {
        cwd: adminUiRoot,
        encoding: 'utf-8',
        env: {
          ...process.env,
          TLDW_VITEST_SAFETY_REPORT: safetyReportPath,
        },
      },
    );

    const jsonReport = await readReport(jsonReportPath) as Record<string, number>;
    await expect(readReport(safetyReportPath)).resolves.toMatchObject({
      reason: 'passed',
      suiteCount: 6,
      passedSuiteCount: 5,
      failedSuiteCount: 0,
      pendingSuiteCount: 1,
      incompleteSuiteCount: 0,
      incompleteTestCount: 0,
    });
    expect({
      suiteCount: jsonReport.numTotalTestSuites,
      passedSuiteCount: jsonReport.numPassedTestSuites,
      failedSuiteCount: jsonReport.numFailedTestSuites,
      pendingSuiteCount: jsonReport.numPendingTestSuites,
    }).toEqual({
      suiteCount: 6,
      passedSuiteCount: 5,
      failedSuiteCount: 0,
      pendingSuiteCount: 1,
    });
  });
});
