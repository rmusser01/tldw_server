import { mkdirSync, renameSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"

const REPORT_PATH_ENV = "TLDW_VITEST_ORDER_REPORT"

export default class VitestExecutionOrderReporter {
  constructor(optionsOrReportPath) {
    const reportPath =
      typeof optionsOrReportPath === "string"
        ? optionsOrReportPath
        : process.env[REPORT_PATH_ENV]
    if (!reportPath) {
      throw new Error(`${REPORT_PATH_ENV} must name the order report output`)
    }
    this.reportPath = reportPath
    this.modules = []
  }

  onTestModuleStart(testModule) {
    this.modules.push(testModule.relativeModuleId.replaceAll("\\", "/"))
  }

  onTestRunEnd(testModules) {
    const report = {
      schemaVersion: 1,
      moduleCount: testModules.length,
      modules: this.modules,
    }
    const temporaryPath = `${this.reportPath}.tmp-${process.pid}`
    mkdirSync(dirname(this.reportPath), { recursive: true })
    writeFileSync(temporaryPath, `${JSON.stringify(report)}\n`, {
      encoding: "utf-8",
    })
    renameSync(temporaryPath, this.reportPath)
  }
}
