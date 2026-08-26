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
    const report = {
      schemaVersion: 1,
      reason,
      moduleCount: testModules.length,
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
