import { expect, test } from "@playwright/test"

import { forceConnected, waitForConnectionStore } from "./utils/connection"
import { launchWithBuiltExtensionOrSkip } from "./utils/real-server"

const SERVER_URL = "http://127.0.0.1:8000"
const WEBUI_URL = "https://webui.example.test/tldw/app/"

type OpenCall = {
  url: string
  target: string | null
  features: string | null
}

const installOpenCapture = async (page: import("@playwright/test").Page) => {
  await page.evaluate(() => {
    ;(window as any).__presentationStudioOpenCalls = []
    window.open = ((
      url?: string | URL,
      target?: string,
      features?: string
    ) => {
      const renderedUrl =
        typeof url === "string"
          ? url
          : url instanceof URL
            ? url.toString()
            : ""
      ;(window as any).__presentationStudioOpenCalls.push({
        url: renderedUrl,
        target: target ?? null,
        features: features ?? null
      })
      return null
    }) as typeof window.open
  })
}

const readFirstOpenCall = async (
  page: import("@playwright/test").Page
): Promise<OpenCall | null> =>
  page.evaluate(
    () => (window as any).__presentationStudioOpenCalls?.[0] ?? null
  )

const installSourceBearingIpcTripwires = async (
  page: import("@playwright/test").Page
) => {
  await page.evaluate(async () => {
    ;(window as any).__presentationStudioSourceIpc = []
    ;(window as any).__presentationStudioIpcTraffic = []
    ;(window as any).__presentationStudioIpcInstrumentation = {
      runtimeAvailable: [],
      runtimeWrapped: [],
      storageAvailable: [],
      storageWrapped: [],
      storageCalibrated: []
    }
    const sourceIpc = (window as any).__presentationStudioSourceIpc as Array<{
      kind: string
    }>
    const traffic = (window as any).__presentationStudioIpcTraffic as Array<{
      kind: string
      metadata: boolean
      sourceBearing: boolean
    }>
    const instrumentation = (window as any)
      .__presentationStudioIpcInstrumentation as {
      runtimeAvailable: string[]
      runtimeWrapped: string[]
      storageAvailable: string[]
      storageWrapped: string[]
      storageCalibrated: string[]
    }
    const normalizeKey = (key: string) =>
      key
        .slice(0, 128)
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "")
    const isForbiddenSourceKey = (key: string) => {
      const normalized = normalizeKey(key)
      return (
        normalized.endsWith("htmldocument") ||
        normalized.endsWith("htmlsource") ||
        normalized.includes("draftattachment") ||
        normalized.includes("draftbody") ||
        normalized.includes("draftsource") ||
        normalized.includes("versioncontent")
      )
    }
    const hasStandaloneMarkup = (value: string) => {
      if (value.length > 1_048_576) return true
      const sample = value.slice(0, 8192).trimStart()
      const lowered = sample.toLowerCase()
      if (lowered.startsWith("<!doctype html") || sample.startsWith("<!--")) {
        return true
      }
      if (sample[0] !== "<") return false

      const isAsciiLetter = (character: string | undefined) => {
        if (!character) return false
        const code = character.charCodeAt(0)
        return (code >= 65 && code <= 90) || (code >= 97 && code <= 122)
      }
      const isTagNameCharacter = (character: string | undefined) => {
        if (!character) return false
        const code = character.charCodeAt(0)
        return (
          isAsciiLetter(character) ||
          (code >= 48 && code <= 57) ||
          character === "-" ||
          character === ":"
        )
      }

      let index = 1
      if (sample[index] === "/") index += 1
      if (!isAsciiLetter(sample[index])) return false
      while (index < sample.length && isTagNameCharacter(sample[index])) {
        index += 1
      }
      const boundary = sample[index]
      if (
        boundary !== ">" &&
        boundary !== "/" &&
        boundary !== " " &&
        boundary !== "\t" &&
        boundary !== "\r" &&
        boundary !== "\n" &&
        boundary !== "\f"
      ) {
        return false
      }

      let quote: '"' | "'" | null = null
      const scanLimit = Math.min(sample.length, index + 2048)
      for (; index < scanLimit; index += 1) {
        const character = sample[index]
        if (quote) {
          if (character === quote) quote = null
          continue
        }
        if (character === '"' || character === "'") {
          quote = character
          continue
        }
        if (character === ">") return true
        if (character === "<") return false
      }
      return false
    }
    const isSourceBearing = (value: unknown) => {
      const stack: Array<{ value: unknown; key: string; depth: number }> = [
        { value, key: "", depth: 0 }
      ]
      const seen = new WeakSet<object>()
      let visited = 0
      while (stack.length > 0) {
        const current = stack.pop()
        if (!current) break
        visited += 1
        if (visited > 512 || current.depth > 8) return true
        if (isForbiddenSourceKey(current.key)) return true
        if (typeof current.value === "string") {
          if (hasStandaloneMarkup(current.value)) return true
          const lowered = current.value.slice(0, 512).toLowerCase()
          if (
            lowered.includes("draft-attachment") ||
            lowered.includes("version-content") ||
            lowered.includes("presentation.html")
          ) {
            return true
          }
          continue
        }
        if (current.value === null || typeof current.value !== "object") {
          continue
        }
        if (seen.has(current.value)) continue
        seen.add(current.value)
        const entries = Object.entries(current.value)
        if (entries.length > 128) return true
        for (let index = entries.length - 1; index >= 0; index -= 1) {
          const [key, nestedValue] = entries[index]
          stack.push({ value: nestedValue, key, depth: current.depth + 1 })
        }
      }
      return false
    }
    const record = (kind: string, value: unknown) => {
      let serialized = ""
      try {
        serialized = JSON.stringify(value)
      } catch {
        serialized = ""
      }
      const sourceBearing = isSourceBearing(value)
      traffic.push({
        kind,
        metadata: serialized.includes("/metadata"),
        sourceBearing
      })
      if (sourceBearing) {
        sourceIpc.push({ kind })
      }
    }

    const replaceMethod = (
      owner: Record<string, unknown>,
      key: string,
      wrapper: (...args: unknown[]) => unknown
    ) => {
      try {
        owner[key] = wrapper
      } catch {
        // Fall through to a configurable own-property replacement.
      }
      if (owner[key] !== wrapper) {
        try {
          Object.defineProperty(owner, key, {
            configurable: true,
            writable: true,
            value: wrapper
          })
        } catch {
          return false
        }
      }
      return owner[key] === wrapper
    }
    const seenRuntimes = new WeakSet<object>()
    const seenStorageAreas = new WeakSet<object>()
    const calibrationTargets: Array<{
      area: Record<string, any>
      kind: string
    }> = []
    const wrapApi = (namespace: "chrome" | "browser", api: any) => {
      const runtime = api?.runtime
      if (
        runtime &&
        typeof runtime === "object" &&
        !seenRuntimes.has(runtime) &&
        typeof runtime.sendMessage === "function"
      ) {
        seenRuntimes.add(runtime)
        const kind = `${namespace}.runtime.sendMessage`
        instrumentation.runtimeAvailable.push(kind)
        const original = runtime.sendMessage
        const wrapper = (...args: unknown[]) => {
          record(kind, args)
          return Reflect.apply(original, runtime, args)
        }
        if (replaceMethod(runtime, "sendMessage", wrapper)) {
          instrumentation.runtimeWrapped.push(kind)
        }
      }

      for (const areaName of ["local", "sync", "session"] as const) {
        const area = api?.storage?.[areaName]
        if (
          !area ||
          typeof area !== "object" ||
          seenStorageAreas.has(area) ||
          typeof area.set !== "function"
        ) {
          continue
        }
        seenStorageAreas.add(area)
        const kind = `${namespace}.storage.${areaName}.set`
        instrumentation.storageAvailable.push(kind)
        const original = area.set
        const wrapper = (...args: unknown[]) => {
          record(kind, args)
          return Reflect.apply(original, area, args)
        }
        if (replaceMethod(area, "set", wrapper)) {
          instrumentation.storageWrapped.push(kind)
          calibrationTargets.push({ area, kind })
        }
      }
    }

    wrapApi("chrome", (globalThis as any).chrome)
    wrapApi("browser", (globalThis as any).browser)

    for (const { area, kind } of calibrationTargets) {
      const key = `__tldw_task16_ipc_calibration_${kind.replace(/[^a-z0-9]/gi, "_")}`
      const before = traffic.length
      try {
        await Promise.resolve(area.set({ [key]: "metadata-only" }))
        if (
          traffic
            .slice(before)
            .some(
              (entry) => entry.kind === kind && entry.sourceBearing === false
            )
        ) {
          instrumentation.storageCalibrated.push(kind)
        }
      } catch {
        // The assertions fail closed when an available wrapper cannot calibrate.
      } finally {
        if (typeof area.remove === "function") {
          try {
            await Promise.resolve(area.remove(key))
          } catch {
            // Cleanup failure must not hide the failed calibration state.
          }
        }
      }
    }
  })
}

test.describe("Presentation Studio quick start", () => {
  test("audits every distinct extension write surface without flagging benign traffic", async ({
    page
  }) => {
    await page.goto("about:blank")
    await page.evaluate(() => {
      const nativeWrites: Array<{ kind: string; value: unknown }> = []
      const makeStorageArea = (kind: string) => ({
        set: async (value: unknown) => {
          nativeWrites.push({ kind, value })
        },
        remove: async () => undefined
      })
      const makeRuntime = (kind: string) => ({
        sendMessage: async (value: unknown) => {
          nativeWrites.push({ kind, value })
          return undefined
        }
      })
      const chromeApi = (globalThis as any).chrome ?? {}
      Object.defineProperties(chromeApi, {
        runtime: {
          configurable: true,
          writable: true,
          value: makeRuntime("chrome.runtime.sendMessage")
        },
        storage: {
          configurable: true,
          writable: true,
          value: {
            local: makeStorageArea("chrome.storage.local.set"),
            sync: makeStorageArea("chrome.storage.sync.set"),
            session: makeStorageArea("chrome.storage.session.set")
          }
        }
      })
      ;(globalThis as any).browser = {
        runtime: makeRuntime("browser.runtime.sendMessage"),
        storage: {
          local: makeStorageArea("browser.storage.local.set"),
          sync: makeStorageArea("browser.storage.sync.set"),
          session: makeStorageArea("browser.storage.session.set")
        }
      }
      ;(window as any).__presentationStudioNativeWrites = nativeWrites
    })

    await installSourceBearingIpcTripwires(page)

    const expectedStorageSurfaces = [
      "chrome.storage.local.set",
      "chrome.storage.sync.set",
      "chrome.storage.session.set",
      "browser.storage.local.set",
      "browser.storage.sync.set",
      "browser.storage.session.set"
    ]
    const expectedRuntimeSurfaces = [
      "chrome.runtime.sendMessage",
      "browser.runtime.sendMessage"
    ]
    const instrumentation = await page.evaluate(
      () => (window as any).__presentationStudioIpcInstrumentation
    )
    expect(instrumentation).toEqual({
      runtimeAvailable: expectedRuntimeSurfaces,
      runtimeWrapped: expectedRuntimeSurfaces,
      storageAvailable: expectedStorageSurfaces,
      storageWrapped: expectedStorageSurfaces,
      storageCalibrated: expectedStorageSurfaces
    })

    await page.evaluate(async () => {
      const chromeApi = (globalThis as any).chrome
      const browserApi = (globalThis as any).browser
      const benign = {
        title: "Metadata only",
        source: "prompt",
        payload: { status: "ready" }
      }
      await chromeApi.storage.local.set(benign)
      await chromeApi.storage.sync.set(benign)
      await chromeApi.storage.session.set(benign)
      await browserApi.storage.local.set(benign)
      await browserApi.storage.sync.set(benign)
      await browserApi.storage.session.set(benign)
      await chromeApi.runtime.sendMessage(benign)
      await browserApi.runtime.sendMessage(benign)
      await chromeApi.storage.local.set({ source: "章/節 の説明" })
      await browserApi.runtime.sendMessage({ source: "版本 2.1：説明" })
    })
    expect(
      await page.evaluate(
        () => (window as any).__presentationStudioSourceIpc ?? []
      )
    ).toEqual([])

    await page.evaluate(async () => {
      const chromeApi = (globalThis as any).chrome
      const browserApi = (globalThis as any).browser
      await chromeApi.storage.local.set({
        source: "<!doctype html><html><body>local source</body></html>"
      })
      await chromeApi.storage.sync.set({ htmlDocument: "standalone source" })
      await chromeApi.storage.session.set({ htmlSource: "standalone source" })
      await browserApi.storage.local.set({
        payload: { versionContent: "standalone source" }
      })
      await browserApi.storage.sync.set({
        type: "presentation-version",
        payload: "<!doctype html><html><body>version source</body></html>"
      })
      await browserApi.storage.session.set({
        draftAttachment: { body: "standalone source" }
      })
      await chromeApi.runtime.sendMessage({ htmlSource: "standalone source" })
      await browserApi.runtime.sendMessage({
        source: "<!doctype html><html><body>runtime source</body></html>"
      })
    })

    expect(
      await page.evaluate(() =>
        ((window as any).__presentationStudioSourceIpc ?? [])
          .map((entry: { kind: string }) => entry.kind)
          .sort()
      )
    ).toEqual([...expectedStorageSurfaces, ...expectedRuntimeSurfaces].sort())

    await page.evaluate(async () => {
      ;((window as any).__presentationStudioSourceIpc as unknown[]).length = 0
      const chromeApi = (globalThis as any).chrome
      const browserApi = (globalThis as any).browser
      await chromeApi.storage.local.set({ source: "<div/>" })
      await browserApi.runtime.sendMessage({
        source: "<!-- lead --><section>fragment source</section>"
      })
    })
    expect(
      await page.evaluate(() =>
        ((window as any).__presentationStudioSourceIpc ?? [])
          .map((entry: { kind: string }) => entry.kind)
          .sort()
      )
    ).toEqual(
      ["chrome.storage.local.set", "browser.runtime.sendMessage"].sort()
    )
  })

  test("creates a seeded project and opens the WebUI editor", async () => {
    let createdPayload: Record<string, any> | null = null

    const { context, page, optionsUrl } = await launchWithBuiltExtensionOrSkip(test, {
      seedConfig: {
        __tldw_first_run_complete: true,
        tldwConfig: {
          serverUrl: SERVER_URL,
          webUiUrl: WEBUI_URL,
          authMode: "single-user",
          apiKey: "test-key"
        }
      }
    })

    await context.route(`${SERVER_URL}/api/v1/slides/presentations`, async (route) => {
      if (route.request().method() !== "POST") {
        await route.fulfill({ status: 204 })
        return
      }

      createdPayload = route.request().postDataJSON() as Record<string, any>
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        headers: {
          "access-control-allow-origin": "*"
        },
        body: JSON.stringify({
          id: "presentation-quickstart-1",
          title: createdPayload?.title ?? "Untitled Presentation",
          description: null,
          theme: "black",
          studio_data: createdPayload?.studio_data ?? null,
          slides: createdPayload?.slides ?? [],
          created_at: "2026-03-13T00:00:00Z",
          last_modified: "2026-03-13T00:00:00Z",
          deleted: false,
          client_id: "1",
          version: 1
        })
      })
    })

    try {
      await page.goto(`${optionsUrl}#/presentation-studio/start`, {
        waitUntil: "domcontentloaded"
      })
      await waitForConnectionStore(page, "presentation-studio-start")
      await forceConnected(page, { serverUrl: SERVER_URL }, "presentation-studio-start")

      await installOpenCapture(page)

      await expect(
        page.getByRole("heading", { name: /Presentation Studio Quick Start/i })
      ).toBeVisible()

      await page.getByLabel("Project title").fill("Extension storyboard")
      await page
        .getByLabel("Narration seed")
        .fill("Open with the problem statement, then walk through the proposed workflow.")
      await page.getByRole("button", { name: "Create seeded project" }).click()

      await expect
        .poll(() => createdPayload, {
          message: "presentation create payload should be captured"
        })
        .not.toBeNull()

      expect(createdPayload).toMatchObject({
        title: "Extension storyboard",
        studio_data: {
          origin: "extension_capture",
          entry_surface: "extension_start",
          has_narration_seed: true,
          has_image_seed: false
        }
      })
      expect(createdPayload?.slides?.[0]).toMatchObject({
        order: 0,
        layout: "content",
        title: "Extension storyboard",
        speaker_notes: "Open with the problem statement, then walk through the proposed workflow."
      })

      await expect
        .poll(() => readFirstOpenCall(page))
        .toEqual({
          url: `${WEBUI_URL}presentation-studio/presentation-quickstart-1`,
          target: "_blank",
          features: "noopener,noreferrer"
        })
    } finally {
      await context.close()
    }
  })

  test("hands standalone HTML metadata to the fixed WebUI without requesting source", async () => {
    const presentationId = "html-project-1"
    const unexpectedSourceRequests: string[] = []

    const { context, page, optionsUrl } = await launchWithBuiltExtensionOrSkip(test, {
      seedConfig: {
        __tldw_first_run_complete: true,
        tldwConfig: {
          serverUrl: SERVER_URL,
          webUiUrl: `${WEBUI_URL}?ignored=1#ignored`,
          authMode: "single-user",
          apiKey: "test-key"
        }
      }
    })

    await context.route(
      `${SERVER_URL}/api/v1/slides/presentations/**`,
      async (route) => {
        const request = route.request()
        const url = new URL(request.url())
        const metadataPath =
          `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}/metadata`
        if (request.method() === "GET" && url.pathname === metadataPath) {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            headers: {
              "access-control-allow-origin": "*",
              "cache-control": "private, no-store"
            },
            body: JSON.stringify({
              id: presentationId,
              title: "Architecture briefing",
              description: "Metadata only.",
              theme: "black",
              content_kind: "standalone_html",
              html_slide_count: 7,
              html_bytes: 12345,
              created_at: "2026-08-01T00:00:00Z",
              last_modified: "2026-08-02T00:00:00Z",
              deleted: false,
              version: 1,
              provenance: {
                source_kind: "prompt",
                provider: "openai",
                model: "gpt-5"
              }
            })
          })
          return
        }

        unexpectedSourceRequests.push(`${request.method()} ${url.pathname}`)
        await route.fulfill({
          status: 418,
          contentType: "application/json",
          body: JSON.stringify({ detail: "source-bearing route tripwire" })
        })
      }
    )

    try {
      await page.goto(optionsUrl, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "presentation-studio-html-handoff")
      await forceConnected(
        page,
        { serverUrl: SERVER_URL },
        "presentation-studio-html-handoff"
      )
      await installOpenCapture(page)
      await installSourceBearingIpcTripwires(page)
      await page.evaluate((projectId) => {
        window.location.hash = `#/presentation-studio/${encodeURIComponent(projectId)}`
      }, presentationId)

      await expect(
        page.getByRole("heading", { name: "Architecture briefing" })
      ).toBeVisible()
      await expect(page.getByText("Standalone HTML + JavaScript")).toBeVisible()
      await expect(page.getByText("Prompt")).toBeVisible()

      await page.getByRole("button", { name: "Open in WebUI" }).click()
      await expect
        .poll(() => readFirstOpenCall(page))
        .toEqual({
          url: `${WEBUI_URL}presentation-studio/${presentationId}`,
          target: "_blank",
          features: "noopener,noreferrer"
        })

      expect(unexpectedSourceRequests).toEqual([])
      expect(
        await page.evaluate(
          () => (window as any).__presentationStudioSourceIpc ?? []
        )
      ).toEqual([])
      const ipcAudit = await page.evaluate(() => ({
        instrumentation:
          (window as any).__presentationStudioIpcInstrumentation ?? {},
        traffic: (window as any).__presentationStudioIpcTraffic ?? []
      }))
      const instrumentation = ipcAudit.instrumentation as {
        runtimeAvailable: string[]
        runtimeWrapped: string[]
        storageAvailable: string[]
        storageWrapped: string[]
        storageCalibrated: string[]
      }
      expect(instrumentation.runtimeAvailable.length).toBeGreaterThan(0)
      expect(instrumentation.runtimeWrapped).toEqual(
        instrumentation.runtimeAvailable
      )
      expect(instrumentation.storageAvailable).toContain(
        "chrome.storage.local.set"
      )
      expect(instrumentation.storageAvailable).toContain(
        "chrome.storage.sync.set"
      )
      expect(instrumentation.storageWrapped).toEqual(
        instrumentation.storageAvailable
      )
      expect(instrumentation.storageCalibrated).toEqual(
        instrumentation.storageAvailable
      )
      expect(
        ipcAudit.traffic.some(
          (entry: { kind: string; metadata: boolean }) =>
            entry.kind.endsWith(".runtime.sendMessage") && entry.metadata
        )
      ).toBe(true)
      expect(
        ipcAudit.traffic.every(
          (entry: { sourceBearing: boolean }) => !entry.sourceBearing
        )
      ).toBe(true)
    } finally {
      await context.close()
    }
  })

  test("resolves structured metadata before mounting the existing structured detail", async () => {
    const presentationId = "structured-project-1"
    const requestOrder: string[] = []
    const { context, page, optionsUrl } = await launchWithBuiltExtensionOrSkip(test, {
      seedConfig: {
        __tldw_first_run_complete: true,
        tldwConfig: {
          serverUrl: SERVER_URL,
          webUiUrl: WEBUI_URL,
          authMode: "single-user",
          apiKey: "test-key"
        }
      }
    })

    await context.route(
      `${SERVER_URL}/api/v1/slides/presentations/**`,
      async (route) => {
        const request = route.request()
        const url = new URL(request.url())
        const detailPath =
          `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}`
        const metadataPath = `${detailPath}/metadata`
        if (request.method() === "GET" && url.pathname === metadataPath) {
          requestOrder.push("metadata")
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            headers: { "access-control-allow-origin": "*" },
            body: JSON.stringify({
              id: presentationId,
              title: "Structured extension deck",
              description: null,
              theme: "black",
              content_kind: "structured_slides",
              slide_count: 1,
              created_at: "2026-08-01T00:00:00Z",
              last_modified: "2026-08-02T00:00:00Z",
              deleted: false,
              version: 1,
              provenance: {
                source_kind: "prompt",
                provider: null,
                model: null
              }
            })
          })
          return
        }
        if (request.method() === "GET" && url.pathname === detailPath) {
          requestOrder.push("detail")
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            headers: { "access-control-allow-origin": "*" },
            body: JSON.stringify({
              id: presentationId,
              title: "Structured extension deck",
              description: null,
              theme: "black",
              content_kind: "structured_slides",
              slides: [
                {
                  id: "slide-1",
                  order: 0,
                  layout: "title",
                  title: "Structured title",
                  content: "",
                  speaker_notes: "",
                  metadata: {}
                }
              ],
              studio_data: null,
              settings: null,
              created_at: "2026-08-01T00:00:00Z",
              last_modified: "2026-08-02T00:00:00Z",
              deleted: false,
              client_id: "1",
              version: 1
            })
          })
          return
        }
        await route.continue()
      }
    )

    try {
      await page.goto(optionsUrl, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "presentation-studio-structured-detail")
      await forceConnected(
        page,
        { serverUrl: SERVER_URL },
        "presentation-studio-structured-detail"
      )
      await page.evaluate((projectId) => {
        window.location.hash = `#/presentation-studio/${encodeURIComponent(projectId)}`
      }, presentationId)

      await expect(
        page.getByRole("heading", { name: "Presentation Studio" })
      ).toBeVisible()
      await expect(page.getByText(/Structured extension deck/)).toBeVisible()
      await expect.poll(() => requestOrder.includes("detail")).toBe(true)
      expect(requestOrder[0]).toBe("metadata")
      expect(requestOrder.indexOf("metadata")).toBeLessThan(
        requestOrder.indexOf("detail")
      )
    } finally {
      await context.close()
    }
  })
})
