import { createRequire } from "node:module"
import path from "node:path"
import { afterEach, describe, expect, it, vi } from "vitest"

const require = createRequire(import.meta.url)
const { JSDOM } = require("jsdom") as typeof import("jsdom")
const jsdomRoot = path.dirname(require.resolve("jsdom/package.json"))
const { implForWrapper } = require(path.join(jsdomRoot, "lib/generated/idl/utils.js"))

const mounted: Element[] = []
afterEach(() => mounted.splice(0).forEach(element => element.remove()))

const renderStyled = (css: string, html: string) => {
  const style = document.createElement("style")
  style.textContent = css
  const host = document.createElement("section")
  host.innerHTML = html
  document.head.append(style)
  document.body.append(host)
  mounted.push(style, host)
  return host
}

// Backport source: https://github.com/jsdom/jsdom/blob/v29.1.1/lib/jsdom/living/css/helpers/computed-style.js
// The prefilter must reject only impossible subjects; the real matcher still
// decides wildcard, grouped, ancestor and functional-pseudo-class selectors.
describe("jsdom stylesheet subject prefilter", () => {
  it.each([
    ["ancestor", ".ancestor .subject {display:none}", '<div class="ancestor"><button class="subject"></button></div>', "display", "none"],
    ["group", ".wrong, button.subject {display:none}", '<button class="subject"></button>', "display", "none"],
    ["wildcard", "* {visibility:hidden}", "<button></button>", "visibility", "hidden"],
    ["attribute", "[data-target] {display:none}", "<button data-target></button>", "display", "none"],
    ["id", "#target {display:none}", '<button id="target"></button>', "display", "none"],
    ["is", ":is(.subject, #other) {display:none}", '<button class="subject"></button>', "display", "none"],
    ["where", ":where(.subject, #other) {visibility:hidden}", '<button class="subject"></button>', "visibility", "hidden"],
    ["not", ".subject:not(.wrong) {display:none}", '<button class="subject"></button>', "display", "none"],
    ["has", ".subject:has(.child) {display:none}", '<button class="subject"><span class="child"></span></button>', "display", "none"],
    ["combinator", ".ancestor > .subject + button {display:none}", '<div class="ancestor"><span class="subject"></span><button></button></div>', "display", "none"],
    ["escaped class", ".subject\\:active {display:none}", '<button class="subject:active"></button>', "display", "none"],
    ["escaped id", "#subject\\:active {display:none}", '<button id="subject:active"></button>', "display", "none"],
    ["screen media", "@media screen {.subject {display:none}}", '<button class="subject"></button>', "display", "none"],
    ["inherited visibility", ".ancestor {visibility:hidden}", '<div class="ancestor"><button></button></div>', "visibility", "hidden"],
    ["inherited pointer protection", ".ancestor {pointer-events:none}", '<div class="ancestor"><button></button></div>', "pointer-events", "none"],
    ["unrelated subjects", ".other, #wrong, aside {display:none}", "<button></button>", "display", "inline-block"]
  ])("preserves %s computed styles", (_name, css, html, property, expected) => {
    const host = renderStyled(css, html)
    expect(window.getComputedStyle(host.querySelector("button")!).getPropertyValue(property)).toBe(expected)
  })

  it.each([
    ["quirks class casing", '<style>.SUBJECT {display:none}</style><button class="subject"></button>', "button", {}],
    ["XML tag casing", '<html xmlns="http://www.w3.org/1999/xhtml"><head><style>Subject {display:none}</style></head><body><Subject /></body></html>', "Subject", { contentType: "application/xhtml+xml" }]
  ])("retains the full matcher for %s", (_name, html, selector, options) => {
    const dom = new JSDOM(html, options)
    try {
      expect(dom.window.getComputedStyle(dom.window.document.querySelector(selector)!).display).toBe("none")
    } finally {
      dom.window.close()
    }
  })

  it.each([
    ["class", ".SUBJECT", 'class="subject"'],
    ["id", "#TARGET", 'id="target"']
  ])("keeps standards-mode %s subjects case-sensitive", (_attribute, selector, attributes) => {
    expect(document.compatMode).toBe("CSS1Compat")
    const host = renderStyled(`${selector} {visibility:hidden}`, `<button ${attributes}></button>`)
    expect(window.getComputedStyle(host.querySelector("button")!).visibility).toBe("visible")
  })

  it.each([
    ["class", ".subject"],
    ["id", "#subject"]
  ])("re-evaluates the subject after a dynamic %s change", (attribute, selector) => {
    const host = renderStyled(`${selector} {display:none}`, `<button ${attribute}="other"></button>`)
    const button = host.querySelector("button")!
    expect(window.getComputedStyle(button).display).toBe("inline-block")
    button.setAttribute(attribute, "subject")
    expect(window.getComputedStyle(button).display).toBe("none")
    button.setAttribute(attribute, "other")
    expect(window.getComputedStyle(button).display).toBe("inline-block")
  })

  it("re-extracts subjects after a CSSOM selectorText mutation", () => {
    const host = renderStyled(".first {display:none}", '<button class="first"></button>')
    expect(window.getComputedStyle(host.querySelector("button")!).display).toBe("none")
    const style = mounted[0] as HTMLStyleElement
    ;(style.sheet!.cssRules[0] as CSSStyleRule).selectorText = ".second"
    const second = document.createElement("button")
    second.className = "second"
    host.append(second)
    expect(window.getComputedStyle(second).display).toBe("none")
  })

  it("avoids full selector checks for hundreds of unrelated stylesheet subjects", () => {
    const rules = Array.from({ length: 500 }, (_, i) => `.unrelated-${i} .leaf-${i} {display:none}`)
    const dom = new JSDOM(`<!doctype html><style>${rules.join("\n")}</style><button>Visible</button>`)
    try {
      // Count real matcher calls rather than wall-clock time; host load must not
      // decide whether the dependency's stylesheet prefilter works.
      const selector = implForWrapper(dom.window.document)._domSelector
      const check = vi.spyOn(selector, "check")
      expect(dom.window.getComputedStyle(dom.window.document.querySelector("button")!).display).toBe("inline-block")
      expect(check.mock.calls.length).toBeLessThan(rules.length)
    } finally {
      dom.window.close()
    }
  })
})
