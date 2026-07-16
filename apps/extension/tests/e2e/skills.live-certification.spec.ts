import {
  expect,
  test,
  type BrowserContext,
  type Page,
} from "@playwright/test"

import {
  runSkillsLiveCertification,
  SKILLS_CERT_ARGUMENTS,
  SKILLS_CERT_RENDERED,
} from "../../../tldw-frontend/e2e/utils/skills-live-certification"
import { launchWithBuiltExtension } from "./utils/extension-build"
import { createSkillsRelayObserver } from "./utils/skills-certification-relay"

const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim()
  if (!value) throw new Error(`${name} is required for Skills extension certification`)
  return value
}

const serverUrl = requireEnv("TLDW_E2E_SERVER_URL")
const apiKey = requireEnv("TLDW_E2E_API_KEY")
const skillName = requireEnv("TLDW_SKILLS_CERT_SKILL_NAME")
const profileRoot = requireEnv("TLDW_SKILLS_CERT_EXTENSION_PROFILE_ROOT")
const resultPath = requireEnv("TLDW_SKILLS_CERT_EXTENSION_RESULT")
const ledgerPath = requireEnv("TLDW_SKILLS_CERT_EXTENSION_LEDGER")

if (skillName !== "skills-cert-extension") {
  throw new Error("TLDW_SKILLS_CERT_SKILL_NAME must equal skills-cert-extension")
}

type Phase =
  | "extension_launch"
  | "extension_worker"
  | "extension_workflow"
  | "extension_relay"
type SanitizedJsonWriter = (filePath: string, value: unknown) => unknown

const workerUrlPattern = /^chrome-extension:\/\/[^/]+\/background\.js$/

const boundedDetail = (
  categories: Set<Phase>,
  errors: unknown[],
  pageErrorCount: number,
  relayEntryCount: number,
): Record<string, number | Phase[]> => ({
  error_count: errors.length,
  page_error_count: pageErrorCount,
  relay_entry_count: relayEntryCount,
  categories: Array.from(categories).slice(0, 4),
})

test("certifies the complete live Skills extension lifecycle", async ({}, testInfo) => {
  let phase: Phase = "extension_launch"
  const categories = new Set<Phase>()
  const errors: unknown[] = []
  let originalError: unknown
  let hasOriginalError = false
  let context: BrowserContext | undefined
  let page: Page | undefined
  let relayObserver: ReturnType<typeof createSkillsRelayObserver> | undefined
  let observedWorkerUrl: string | undefined
  let pageErrorCount = 0
  let writeSanitizedJson: SanitizedJsonWriter | undefined

  const onPageError = (): void => {
    pageErrorCount += 1
  }
  const retainOriginalError = (error: unknown): void => {
    if (!hasOriginalError) {
      originalError = error
      hasOriginalError = true
    }
    categories.add(phase)
    errors.push(error)
  }
  const retainFinalizationError = (error: unknown): void => {
    categories.add(phase)
    errors.push(error)
  }

  try {
    const evidence = await import(
      "../../../tldw-frontend/scripts/skills-certification/evidence.mjs"
    )
    writeSanitizedJson = evidence.writeSanitizedJson
    writeSanitizedJson(resultPath, { status: "running" })

    const launch = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl,
        authMode: "single-user",
        apiKey,
      },
      allowOffline: false,
      optionsTarget: "/skills",
      profileRoot,
      prepareOptionsPage: async ({ context: preparedContext, page: preparedPage }) => {
        phase = "extension_worker"
        context = preparedContext
        page = preparedPage

        const workers = preparedContext
          .serviceWorkers()
          .filter((worker) => workerUrlPattern.test(worker.url()))
        if (workers.length !== 1) {
          throw new Error("Expected exactly one extension background worker")
        }

        observedWorkerUrl = workers[0].url()
        relayObserver = createSkillsRelayObserver(preparedContext, observedWorkerUrl)
        preparedPage.on("pageerror", onPageError)
      },
    })

    context = launch.context
    page = launch.page
    if (
      observedWorkerUrl !== `chrome-extension://${launch.extensionId}/background.js`
    ) {
      throw new Error("Observed worker does not belong to the launched extension")
    }

    phase = "extension_workflow"
    await runSkillsLiveCertification({
      page,
      expect,
      initialExpectation: "target-absent",
      name: skillName,
      arguments: SKILLS_CERT_ARGUMENTS,
      expectedRenderedPrompt: SKILLS_CERT_RENDERED,
      step: test.step,
    })

    if (pageErrorCount > 0) {
      throw new Error("Extension page reported errors during the Skills workflow")
    }
  } catch (error) {
    retainOriginalError(error)
  } finally {
    if (hasOriginalError && page && !page.isClosed()) {
      try {
        await page.screenshot({
          path: testInfo.outputPath("skills-live-certification-failure.png"),
        })
      } catch (error) {
        retainFinalizationError(error)
      }
    }

    if (context) {
      try {
        await context.close()
      } catch (error) {
        retainFinalizationError(error)
      }
    }

    if (page) {
      try {
        page.off("pageerror", onPageError)
      } catch (error) {
        retainFinalizationError(error)
      }
    }

    if (relayObserver) {
      try {
        relayObserver.dispose()
      } catch (error) {
        retainFinalizationError(error)
      }
    }

    phase = "extension_relay"
    if (relayObserver) {
      try {
        relayObserver.assertValid()
      } catch (error) {
        retainFinalizationError(error)
      }
    }

    const relayEntries = relayObserver?.entries ?? []
    if (writeSanitizedJson) {
      try {
        writeSanitizedJson(ledgerPath, relayEntries)
      } catch (error) {
        retainFinalizationError(error)
      }

      try {
        writeSanitizedJson(resultPath, {
          status: hasOriginalError || errors.length > 0 ? "failed" : "passed",
          categories: Array.from(categories),
          detail: boundedDetail(categories, errors, pageErrorCount, relayEntries.length),
        })
      } catch (error) {
        retainFinalizationError(error)
      }
    }
  }

  if (hasOriginalError) throw originalError
  if (errors.length === 1) throw errors[0]
  if (errors.length > 1) {
    throw new AggregateError(errors, "Skills extension certification finalization failed")
  }
})
