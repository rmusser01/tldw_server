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

function structuredError(error) {
  if (!error || typeof error !== "object") {
    throw new Error("failed Vitest test contains an invalid error")
  }
  const name = error.name == null ? null : error.name
  if (name !== null && (typeof name !== "string" || !name.trim())) {
    throw new Error("failed Vitest test contains an invalid error name")
  }
  if (typeof error.message !== "string" || !error.message.trim()) {
    throw new Error("failed Vitest test contains an invalid error message")
  }
  const rawStacks = error.stacks ?? []
  if (!Array.isArray(rawStacks)) {
    throw new Error("failed Vitest test contains invalid parsed stacks")
  }
  const stacks = rawStacks.map((frame) => {
    if (
      !frame ||
      typeof frame !== "object" ||
      typeof frame.method !== "string" ||
      typeof frame.file !== "string" ||
      !frame.file.trim() ||
      !Number.isInteger(frame.line) ||
      frame.line < 0 ||
      !Number.isInteger(frame.column) ||
      frame.column < 0
    ) {
      throw new Error("failed Vitest test contains an invalid parsed stack frame")
    }
    return {
      method: frame.method,
      file: frame.file,
      line: frame.line,
      column: frame.column,
    }
  })
  return { name, message: error.message, stacks }
}

function structuredFailures(testModule) {
  const module = normalizedModuleId(testModule)
  const failures = []

  const visit = (children, ancestorTitles) => {
    children.array().forEach((child) => {
      if (child.type === "suite") {
        visit(child.children, [...ancestorTitles, child.name])
        return
      }
      if (child.type !== "test") {
        return
      }
      const result = child.result()
      if (result?.state !== "failed") {
        return
      }
      if (typeof child.name !== "string" || !child.name.trim()) {
        throw new Error("failed Vitest test contains an invalid title")
      }
      if (!Array.isArray(result.errors) || result.errors.length === 0) {
        throw new Error("failed Vitest test is missing structured errors")
      }
      failures.push({
        module,
        ancestorTitles,
        title: child.name,
        fullName: [...ancestorTitles, child.name].join(" "),
        errors: result.errors.map(structuredError),
      })
    })
  }
  visit(testModule.children, [])
  return failures
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
    const failures = testModules.flatMap(structuredFailures)
    const report = {
      schemaVersion: 3,
      moduleCount: testModules.length,
      modules: this.modules,
      suiteCount: suites.length,
      suites,
      failureCount: failures.length,
      failures,
    }
    const temporaryPath = `${this.reportPath}.tmp-${process.pid}`
    mkdirSync(dirname(this.reportPath), { recursive: true })
    writeFileSync(temporaryPath, `${JSON.stringify(report)}\n`, {
      encoding: "utf-8",
    })
    renameSync(temporaryPath, this.reportPath)
  }
}
