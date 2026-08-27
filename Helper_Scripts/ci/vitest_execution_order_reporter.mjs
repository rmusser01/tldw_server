import { mkdirSync, renameSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"

const REPORT_PATH_ENV = "TLDW_VITEST_ORDER_REPORT"

function normalizedModuleId(testModule) {
  return testModule.relativeModuleId.replaceAll("\\", "/")
}

function runtimeSuites(testModule) {
  const module = normalizedModuleId(testModule)
  const suites = [
    { module, path: [], name: module, state: testModule.state(), mode: null },
  ]

  const visit = (children, parentPath) => {
    children.array().forEach((child, index) => {
      if (child.type !== "suite") {
        return
      }
      const path = [...parentPath, index]
      suites.push({
        module,
        path,
        name: child.name,
        state: child.state(),
        mode: child.options.mode,
      })
      visit(child.children, path)
    })
  }
  visit(testModule.children, [])
  return suites
}

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
    this.modules.push(normalizedModuleId(testModule))
  }

  onTestRunEnd(testModules) {
    const suites = testModules.flatMap(runtimeSuites)
    const report = {
      schemaVersion: 2,
      moduleCount: testModules.length,
      modules: this.modules,
      suiteCount: suites.length,
      suites,
    }
    const temporaryPath = `${this.reportPath}.tmp-${process.pid}`
    mkdirSync(dirname(this.reportPath), { recursive: true })
    writeFileSync(temporaryPath, `${JSON.stringify(report)}\n`, {
      encoding: "utf-8",
    })
    renameSync(temporaryPath, this.reportPath)
  }
}
