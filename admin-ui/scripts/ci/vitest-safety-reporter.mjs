import { mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

const REPORT_PATH_ENV = 'TLDW_VITEST_SAFETY_REPORT';

function moduleErrorCount(testModules) {
  let count = 0;
  for (const testModule of testModules) {
    count += testModule.errors().length;
    for (const suite of testModule.children.allSuites()) {
      count += suite.errors().length;
    }
  }
  return count;
}

function suiteStatusCounts(testModules) {
  const counts = { passed: 0, failed: 0, pending: 0, incomplete: 0 };
  for (const testModule of testModules) {
    const suites = [testModule, ...testModule.children.allSuites()];
    for (const suite of suites) {
      const state = suite.state();
      let status = 'passed';
      if (state === 'failed') {
        status = 'failed';
      } else if (state === 'pending' || state === 'queued' || suite.options?.mode === 'todo') {
        status = 'pending';
      }
      if (state === 'pending' || state === 'queued') {
        counts.incomplete += 1;
      }
      counts[status] += 1;
    }
  }
  return counts;
}

function incompleteTestCount(testModules) {
  return testModules.reduce(
    (count, testModule) => count + Array.from(testModule.children.allTests())
      .filter((test) => test.result().state === 'pending').length,
    0,
  );
}

export default class VitestSafetyReporter {
  constructor(optionsOrReportPath) {
    const reportPath = typeof optionsOrReportPath === 'string'
      ? optionsOrReportPath
      : process.env[REPORT_PATH_ENV];
    if (!reportPath) {
      throw new Error(`${REPORT_PATH_ENV} must name the safety report output`);
    }
    this.reportPath = reportPath;
    this.activeHookCount = 0;
    this.hookProtocolErrorCount = 0;
  }

  onHookStart() {
    this.activeHookCount += 1;
  }

  onHookEnd() {
    if (this.activeHookCount === 0) {
      this.hookProtocolErrorCount += 1;
      return;
    }
    this.activeHookCount -= 1;
  }

  async onTestRunEnd(testModules, unhandledErrors, reason) {
    const suiteCounts = suiteStatusCounts(testModules);
    const report = {
      schemaVersion: 2,
      reason,
      moduleCount: testModules.length,
      suiteCount: suiteCounts.passed + suiteCounts.failed + suiteCounts.pending,
      passedSuiteCount: suiteCounts.passed,
      failedSuiteCount: suiteCounts.failed,
      pendingSuiteCount: suiteCounts.pending,
      incompleteSuiteCount: suiteCounts.incomplete,
      incompleteTestCount: incompleteTestCount(testModules),
      testCount: testModules.reduce(
        (count, testModule) => count + Array.from(testModule.children.allTests()).length,
        0,
      ),
      unhandledErrorCount: unhandledErrors.length,
      moduleErrorCount: moduleErrorCount(testModules),
      hookErrorCount: this.activeHookCount + this.hookProtocolErrorCount,
    };
    const temporaryPath = `${this.reportPath}.tmp-${process.pid}`;
    mkdirSync(dirname(this.reportPath), { recursive: true });
    writeFileSync(temporaryPath, `${JSON.stringify(report)}\n`, { encoding: 'utf-8' });
    renameSync(temporaryPath, this.reportPath);
  }
}
