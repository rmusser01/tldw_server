import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, describe, expect, test } from 'vitest';

import VitestExecutionOrderReporter from '../../../Helper_Scripts/ci/vitest_execution_order_reporter.mjs';

const temporaryDirectories: string[] = [];

function collection(...children: object[]) {
  return { array: () => children };
}

function suite(name: string, state: string, mode: string, ...children: object[]) {
  return {
    type: 'suite',
    name,
    state: () => state,
    options: { mode },
    children: collection(...children),
  };
}

function testCase(name: string, state: string, errors: object[] = []) {
  return {
    type: 'test',
    name,
    result: () => ({ state, errors }),
  };
}

function testModule(relativeModuleId: string, state: string, ...children: object[]) {
  return {
    relativeModuleId,
    state: () => state,
    children: collection(...children),
  };
}

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
  test('atomically replaces the report with normalized order and runtime suites', async () => {
    const path = await reportPath();
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, '{"stale":true}\n', 'utf-8');
    const reporter = new VitestExecutionOrderReporter(path);

    const alpha = testModule(
      'app\\alpha.test.ts',
      'failed',
      testCase('passes directly', 'passed'),
      suite(
        'failure suite',
        'failed',
        'run',
        testCase('rejects invalid input', 'failed', [
          {
            name: 'AssertionError',
            message: 'expected true to be false',
            stack: 'raw stack must not be serialized',
            stacks: [
              {
                method: 'rejectInput',
                file: '/checkout/app/alpha.test.ts',
                line: 42,
                column: 7,
              },
            ],
          },
        ]),
      ),
      suite('todo suite', 'skipped', 'todo'),
    );
    const beta = testModule('app/beta.test.ts', 'passed');

    reporter.onTestModuleStart(alpha);
    reporter.onTestModuleStart(beta);
    reporter.onTestRunEnd([alpha, beta]);

    await expect(readFile(path, 'utf-8')).resolves.toBe(
      '{"schemaVersion":3,"moduleCount":2,"modules":["app/alpha.test.ts","app/beta.test.ts"],"suiteCount":4,"suites":[{"module":"app/alpha.test.ts","path":[],"name":"app/alpha.test.ts","state":"failed","mode":null},{"module":"app/alpha.test.ts","path":[1],"name":"failure suite","state":"failed","mode":"run"},{"module":"app/alpha.test.ts","path":[2],"name":"todo suite","state":"skipped","mode":"todo"},{"module":"app/beta.test.ts","path":[],"name":"app/beta.test.ts","state":"passed","mode":null}],"failureCount":1,"failures":[{"module":"app/alpha.test.ts","ancestorTitles":["failure suite"],"title":"rejects invalid input","fullName":"failure suite rejects invalid input","errors":[{"name":"AssertionError","message":"expected true to be false","stacks":[{"method":"rejectInput","file":"/checkout/app/alpha.test.ts","line":42,"column":7}]}]}]}\n',
    );
    await expect(readdir(dirname(path))).resolves.toEqual(['execution-order.json']);
  });

  test('uses the environment path when Vitest supplies reporter options', async () => {
    const path = await reportPath();
    process.env.TLDW_VITEST_ORDER_REPORT = path;
    const reporter = new VitestExecutionOrderReporter({});

    const exampleModule = testModule('app/example.test.ts', 'passed');
    reporter.onTestModuleStart(exampleModule);
    reporter.onTestRunEnd([exampleModule]);

    await expect(readFile(path, 'utf-8')).resolves.toContain(
      '"modules":["app/example.test.ts"]',
    );
  });
});
