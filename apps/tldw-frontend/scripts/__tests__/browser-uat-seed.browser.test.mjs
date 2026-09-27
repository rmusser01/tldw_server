import assert from "node:assert/strict"
import test from "node:test"
import { chromium } from "@playwright/test"
import { seedManualUatBrowser } from "../browser-uat-seed.mjs"

test("manual UAT keeps bootstrap and application storage writes out of native Chromium storage", async () => {
  const browser = await chromium.launch()
  try {
    const context = await browser.newContext()
    const settings = {
      webUrl: "https://uat.example.test",
      serverUrl: "https://api.example.test",
      apiKey: "chromium-uat-private-key",
      legacyBootstrap: true
    }
    // One init script guarantees that native handles are captured before the
    // real serialized UAT callback, and that both run before page scripts.
    await context.addInitScript({ content: `
      globalThis.__nativeUatStorage = {
        local: localStorage, session: sessionStorage,
        descriptors: ["localStorage", "sessionStorage"].map(name => {
          const descriptor = Object.getOwnPropertyDescriptor(window, name)
          return { name, configurable: descriptor.configurable, getter: typeof descriptor.get }
        })
      };
      (${seedManualUatBrowser.toString()})(${JSON.stringify(settings)});
    ` })
    await context.route("**/*", route => route.fulfill({
      contentType: "text/html",
      body: "<!doctype html><title>Disposable UAT storage probe</title>"
    }))
    const page = await context.newPage()
    await page.goto(settings.webUrl)
    const boot = await page.evaluate(() => ({
      nativeLengths: [globalThis.__nativeUatStorage.local.length, globalThis.__nativeUatStorage.session.length],
      descriptors: globalThis.__nativeUatStorage.descriptors,
      config: JSON.parse(localStorage.getItem("tldwConfig")),
      legacy: localStorage.getItem("apiKey")
    }))
    assert.deepEqual(boot.nativeLengths, [0, 0])
    assert.deepEqual(boot.descriptors, [
      { name: "localStorage", configurable: true, getter: "function" },
      { name: "sessionStorage", configurable: true, getter: "function" }
    ])
    assert.equal(boot.config.apiKey, settings.apiKey)
    assert.equal(boot.legacy, settings.apiKey)

    const contract = await page.evaluate(() => {
      const results = []
      for (const area of [localStorage, sessionStorage]) {
        area.clear()
        area.setItem("number", 12)
        area.setItem("nullable", null)
        area.named = "application-private-key"
        results.push({
          number: area.getItem("number"), nullable: area.getItem("nullable"),
          missing: area.getItem("missing"), named: area.getItem("named"),
          property: area.named, keys: Object.keys(area), length: area.length,
          first: area.key(0), outside: area.key(99)
        })
        area.removeItem("number")
        delete area.named
        if (area.length !== 1) throw new Error("removeItem/delete did not remove keys")
        area.clear()
        if (area.length !== 0) throw new Error("clear did not empty memory")
        area.setItem("app-auth", "application-private-key")
      }
      return { results, nativeLengths: [globalThis.__nativeUatStorage.local.length, globalThis.__nativeUatStorage.session.length] }
    })
    assert.deepEqual(contract.nativeLengths, [0, 0])
    assert.deepEqual(contract.results, Array.from({ length: 2 }, () => ({
      number: "12", nullable: "null", missing: null,
      named: "application-private-key", property: "application-private-key",
      keys: ["number", "nullable", "named"], length: 3, first: "number", outside: null
    })))
    assert.doesNotMatch(JSON.stringify(await context.storageState()), /chromium-uat-private-key|application-private-key/)

    await page.reload()
    assert.deepEqual(await page.evaluate(() => ({
      apiKey: JSON.parse(localStorage.getItem("tldwConfig")).apiKey,
      previous: localStorage.getItem("app-auth"),
      nativeLengths: [globalThis.__nativeUatStorage.local.length, globalThis.__nativeUatStorage.session.length]
    })), { apiKey: settings.apiKey, previous: null, nativeLengths: [0, 0] })

    await page.evaluate(() => {
      for (const src of ["https://uat.example.test/frame", "https://foreign.example.test/frame"]) {
        const frame = document.createElement("iframe")
        frame.src = src
        document.body.append(frame)
      }
    })
    for (const url of ["https://uat.example.test/frame", "https://foreign.example.test/frame"]) {
      await page.waitForFunction(url => [...document.querySelectorAll("iframe")].some(frame => frame.src === url), url)
    }
    await Promise.all(page.frames().filter(frame => frame !== page.mainFrame()).map(frame => frame.waitForLoadState()))
    const frames = await Promise.all(page.frames().filter(frame => frame !== page.mainFrame()).map(frame => frame.evaluate(() => ({
      origin: location.origin,
      native: localStorage === globalThis.__nativeUatStorage.local && sessionStorage === globalThis.__nativeUatStorage.session,
      key: localStorage.getItem("apiKey"),
      nativeLengths: [globalThis.__nativeUatStorage.local.length, globalThis.__nativeUatStorage.session.length]
    }))))
    assert.deepEqual(frames.sort((a, b) => a.origin.localeCompare(b.origin)), [
      { origin: "https://foreign.example.test", native: true, key: null, nativeLengths: [0, 0] },
      { origin: "https://uat.example.test", native: false, key: settings.apiKey, nativeLengths: [0, 0] }
    ])
    await page.goto("https://foreign.example.test")
    assert.equal(await page.evaluate(() => localStorage === globalThis.__nativeUatStorage.local), true)
    assert.doesNotMatch(JSON.stringify(await context.storageState()), /chromium-uat-private-key|application-private-key/)
  } finally {
    await browser.close()
  }
})
