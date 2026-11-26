import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  buildTokenAuthority,
  REQUIRED_DEFINITION_PATHS,
  TOKEN_SOURCE_PATHS,
} from "../../../scripts/design-system-drift-tokens.mjs"

const sharedCss = "apps/packages/ui/src/assets/tailwind-shared.css"
const webConfig = "apps/tldw-frontend/tailwind.config.js"
const extensionConfig = "apps/extension/tailwind.config.js"
const adminConfig = "admin-ui/tailwind.config.ts"
const adminCss = "admin-ui/app/globals.css"
const colorsPath = "apps/packages/ui/src/styles/tokens/colors.ts"
const themesPath = "apps/packages/ui/src/themes/presets.ts"

const fixtures: Record<string, string> = {
  [sharedCss]: `:root { --color-primary: 1 2 3; --color-danger: 4 5 6; --radius-sm: 2px; }
    .dark { --color-primary: 7 8 9; }`,
  [webConfig]: `module.exports = { theme: { extend: {
    colors: { primary: { DEFAULT: 'rgb(var(--color-primary))', foreground: 'rgb(var(--color-danger))' },
      danger: 'rgb(var(--color-danger))', red: { 500: '#ef4444' }, bogus: 'var(--not-established)' },
    borderRadius: { sm: 'var(--radius-sm, 2px)', card: '12px', pill: '9999px' }
  } }, plugins: [require('never-evaluate-this')] }`,
  [extensionConfig]: `const shared = require('../tldw-frontend/tailwind.config.js');
    module.exports = { ...shared, content: ['./entrypoints/**/*.{ts,tsx}'] }`,
  [adminConfig]: `const config = { theme: { extend: {
    colors: { destructive: { DEFAULT: 'hsl(var(--destructive))', foreground: 'hsl(var(--destructive-foreground))' } },
    borderRadius: { lg: 'var(--radius)', md: 'calc(var(--radius) - 2px)' }
  } } }; export default config`,
  [adminCss]: `:root { --destructive: 0 90% 50%; --destructive-foreground: 0 0% 100%; --radius: 0.5rem; }
    .dark { --destructive: 0 80% 40%; }`,
  [colorsPath]: `export const bgColor = { base: 'bg-primary' } as const;`,
}

function sources(overrides: Record<string, string> = {}) {
  return TOKEN_SOURCE_PATHS.map(path => ({
    path,
    source: overrides[path] ?? fixtures[path] ?? "export {}",
  }))
}

function covered(result: ReturnType<typeof buildTokenAuthority>, path: string, offset: number) {
  return result.definitionScopes.some(scope =>
    scope.path === path && scope.start <= offset && offset < scope.end)
}

describe("static drift token authority", () => {
  it("flattens nested DEFAULT colors only when backed by established semantic variables", () => {
    const result = buildTokenAuthority({ sources: sources() })
    expect([...result.surfaces.webui.semanticColors].sort()).toEqual([
      "danger", "primary", "primary-foreground",
    ])
    expect(result.surfaces.webui.cssVariables.has("--not-established")).toBe(false)
  })

  it("merges color branches before flattening and drops descendants of replaced objects", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `module.exports = { theme: {
        colors: {
          primary: { DEFAULT: 'var(--color-primary)', foreground: 'var(--color-danger)' },
          surface: { DEFAULT: 'var(--not-established)', muted: 'var(--color-primary)' }
        },
        extend: { colors: { primary: 'var(--color-danger)', surface: { DEFAULT: 'var(--color-danger)' } } }
      } }`,
    }) })
    expect([...result.surfaces.webui.semanticColors]).toEqual(["primary", "surface", "surface-muted"])
  })

  it.each([
    ["", ["added", "primary-foreground", "text-only"]],
    ["textColor: { primary: 'var(--color-primary)', direct: 'var(--color-danger)' },", ["direct", "text-only"]],
  ])("reads the effective text-only palette with direct textColor %s", (direct, expected) => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `module.exports = { theme: {
        colors: { primary: { DEFAULT: 'var(--color-primary)', foreground: 'var(--color-danger)' } },
        ${direct}
        extend: {
          colors: { added: 'var(--color-danger)' },
          textColor: { primary: { DEFAULT: 'red' }, 'text-only': 'var(--color-danger)' }
        }
      } }`,
    }) })
    for (const surface of ["shared-ui", "webui", "extension"] as const) {
      expect([...result.surfaces[surface].semanticColors]).toEqual(["added", "primary", "primary-foreground"])
      expect(result.surfaces[surface].semanticTextColors).toEqual(new Set(expected))
    }
    expect(result.surfaces["admin-ui"].semanticTextColors).toBeUndefined()
  })

  it("accepts a static textColor-only theme and scopes its definition", () => {
    const source = `module.exports = { theme: { textColor: { standalone: 'var(--color-primary)' } } }`
    const result = buildTokenAuthority({ sources: sources({ [webConfig]: source }) })
    expect(result.surfaces.webui.semanticTextColors).toEqual(new Set(["standalone"]))
    expect(covered(result, webConfig, source.indexOf("textColor:"))).toBe(true)
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === "error")).toEqual([])
  })

  it("reports unsupported textColor config without trusting the masked colors", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `module.exports = { theme: {
        colors: { generic: 'var(--color-primary)' }, textColor: loadTextColors(),
        extend: { textColor: { broken: loadColor(), missing: 'var(--not-established)', safe: 'var(--color-danger)' } }
      } }`,
    }) })
    expect(result.surfaces.webui.semanticColors).toEqual(new Set(["generic"]))
    expect(result.surfaces.webui.semanticTextColors).toEqual(new Set(["safe"]))
    expect(result.diagnostics.filter(diagnostic => diagnostic.code === "unsupported-config-expression"))
      .toEqual([1, 2].map(() => expect.objectContaining({
        path: webConfig, severity: "coverage", message: expect.stringContaining("textColor"),
      })))
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      path: webConfig, code: "unresolved-semantic-variable", message: expect.stringContaining("missing"),
    }))
  })

  it("pins extension inheritance without executing require or unrelated code", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `${fixtures[webConfig]}; throw new Error('must not run');`,
    }) })
    for (const surface of ["shared-ui", "extension"] as const) {
      expect(result.surfaces[surface].semanticColors).toEqual(result.surfaces.webui.semanticColors)
      expect(result.surfaces[surface].radiusClasses.has("rounded-card")).toBe(true)
    }
    expect(result.surfaces.extension.definitionPaths).toContain(extensionConfig)
    expect(result.surfaces.webui.definitionPaths).not.toContain(extensionConfig)
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === "error")).toEqual([])
  })

  it("keeps admin's vocabulary separate", () => {
    const result = buildTokenAuthority({ sources: sources() })
    expect([...result.surfaces["admin-ui"].semanticColors].sort()).toEqual([
      "destructive", "destructive-foreground",
    ])
    expect(result.surfaces.webui.semanticColors.has("destructive")).toBe(false)
    expect(result.surfaces["admin-ui"].cssVariables.has("--color-danger")).toBe(false)
    expect(result.surfaces["admin-ui"].definitionPaths).toEqual([adminCss, adminConfig])
  })

  it("scopes individual recognized light/dark CSS declarations, not adjacent product CSS", () => {
    const source = `:root { --color-danger: 1 2 3; --product-local: red; color: red; }
      .dark { --color-danger: 4 5 6; } .notice { color: red; --color-danger: 9 9 9; }`
    const result = buildTokenAuthority({ sources: sources({ [sharedCss]: source }) })
    expect(result.definitionScopes.filter(scope => scope.path === sharedCss)).toHaveLength(2)
    expect(covered(result, sharedCss, source.indexOf("--color-danger"))).toBe(true)
    expect(covered(result, sharedCss, source.indexOf("--product-local"))).toBe(false)
    expect(covered(result, sharedCss, source.indexOf("color: red"))).toBe(false)
    expect(covered(result, sharedCss, source.lastIndexOf("--color-danger"))).toBe(false)
  })

  it("does not trust arbitrary custom properties even when referenced by config", () => {
    const result = buildTokenAuthority({ sources: sources({
      [sharedCss]: `${fixtures[sharedCss]} :root { --not-established: red; }`,
    }) })
    expect(result.surfaces.webui.semanticColors.has("bogus")).toBe(false)
  })

  it("scopes known token exports but leaves extra exports and components eligible", () => {
    const source = `export const bgColor = { base: 'bg-primary' } as const;
      export const productStyles = { color: 'red' };
      export function Product() { return { color: 'red' } }`
    const result = buildTokenAuthority({ sources: sources({ [colorsPath]: source }) })
    expect(covered(result, colorsPath, source.indexOf("bgColor"))).toBe(true)
    expect(covered(result, colorsPath, source.indexOf("productStyles"))).toBe(false)
    expect(covered(result, colorsPath, source.indexOf("function Product"))).toBe(false)
  })

  it("recognizes only named theme objects, not every object in a theme file", () => {
    const source = `const defaultTheme: ThemeDefinition = { palette: { light: { bg: '1 2 3' } } };
      const productStyle = { color: 'red' }; const inventedTheme = { color: 'red' };`
    const result = buildTokenAuthority({ sources: sources({ [themesPath]: source }) })
    expect(covered(result, themesPath, source.indexOf("defaultTheme"))).toBe(true)
    expect(covered(result, themesPath, source.indexOf("productStyle"))).toBe(false)
    expect(covered(result, themesPath, source.indexOf("inventedTheme"))).toBe(false)
  })

  it("limits config exemptions to definition properties", () => {
    const source = `${fixtures[webConfig]}; const product = { color: 'red' };`
    const result = buildTokenAuthority({ sources: sources({ [webConfig]: source }) })
    expect(covered(result, webConfig, source.indexOf("colors:"))).toBe(true)
    expect(covered(result, webConfig, source.indexOf("borderRadius:"))).toBe(true)
    expect(covered(result, webConfig, source.indexOf("plugins:"))).toBe(false)
    expect(covered(result, webConfig, source.indexOf("const product"))).toBe(false)
  })

  it("combines default radius data and overrides without guessing rem-to-px conversions", () => {
    const result = buildTokenAuthority({ sources: sources() })
    const web = result.surfaces.webui
    for (const name of ["rounded", "rounded-none", "rounded-2xl", "rounded-full", "rounded-card", "rounded-pill"]) {
      expect(web.radiusClasses.has(name)).toBe(true)
    }
    for (const value of ["0px", "0.25rem", "1rem", "2px", "12px", "9999px", "var(--radius-sm, 2px)"]) {
      expect(web.radiusValues.has(value)).toBe(true)
    }
    expect(web.radiusValues.has("16px")).toBe(false)
    expect(result.surfaces["admin-ui"].radiusValues.has("calc(var(--radius) - 2px)")).toBe(true)
    expect(result.surfaces["admin-ui"].radiusValues.has("0.5rem")).toBe(true)
    expect(result.diagnostics.some(diagnostic => diagnostic.severity === "coverage" && /radius/i.test(diagnostic.message))).toBe(true)
  })

  it("keeps conditional CSS declarations scoped without authorizing their literal radii", () => {
    const source = `@layer base {
      :root { --radius-sm: 11px; }
      @media (min-width: 1000px) { :root { --radius-sm: 21px; } }
    }
    @supports (display: grid) { @layer base { .dark { --radius-sm: 31px; } } }
    @container (min-width: 500px) { :root { --radius-sm: 41px; } }`
    const result = buildTokenAuthority({ sources: sources({ [sharedCss]: source }) })
    expect(result.surfaces.webui.radiusValues.has("11px")).toBe(true)
    for (const value of ["21px", "31px", "41px"]) {
      expect(result.surfaces.webui.radiusValues.has(value)).toBe(false)
    }
    for (const match of source.matchAll(/--radius-sm:/g)) {
      expect(covered(result, sharedCss, match.index!)).toBe(true)
    }
  })

  it("uses only the winning radius declaration within each separate CSS rule", () => {
    const source = `:root { --radius-sm: 31px; --radius-sm: 32px; }
      .dark { --radius-sm: 41px !important; --radius-sm: 42px; }
      @layer base { :root { --radius-sm: 51px; --radius-sm: 52px !important; --radius-sm: 53px !important; } }`
    const result = buildTokenAuthority({ sources: sources({ [sharedCss]: source }) })
    expect([...result.surfaces.webui.radiusValues].filter(value => /^[345]\dpx$/.test(value)))
      .toEqual(["32px", "41px", "53px"])
    for (const match of source.matchAll(/--radius-sm:/g)) {
      expect(covered(result, sharedCss, match.index!)).toBe(true)
    }
  })

  it("honors direct radius replacement before extend overrides", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `module.exports = { theme: { borderRadius: { tiny: '3px' }, extend: { borderRadius: { card: '12px' } } } }`,
    }) })
    expect([...result.surfaces.webui.radiusClasses].sort()).toEqual(["rounded-card", "rounded-tiny"])
    expect([...result.surfaces.webui.radiusValues].sort()).toEqual(["12px", "3px"])
  })

  it("does not retain inherited theme scales after extension replaces the theme object", () => {
    const result = buildTokenAuthority({ sources: sources({
      [extensionConfig]: `const shared = require('../tldw-frontend/tailwind.config.js');
        module.exports = { ...shared, theme: { extend: { borderRadius: { custom: '3px' } } } }`,
    }) })
    expect(result.surfaces.extension.semanticColors.size).toBe(0)
    expect(result.surfaces.extension.radiusClasses.has("rounded-custom")).toBe(true)
    expect(result.surfaces.extension.radiusClasses.has("rounded-card")).toBe(false)
  })

  it("preserves explicitly inherited theme.extend entries when extension overrides part of a scale", () => {
    const result = buildTokenAuthority({ sources: sources({
      [extensionConfig]: `const shared = require('../tldw-frontend/tailwind.config.js');
        module.exports = { ...shared, theme: { ...shared.theme, extend: {
          ...shared.theme.extend,
          borderRadius: { ...shared.theme.extend.borderRadius, card: '14px' },
          colors: { ...shared.theme.extend.colors, extra: 'rgb(var(--color-danger))' }
        } } }`,
    }) })
    expect([...result.surfaces.extension.semanticColors].sort()).toEqual([
      "danger", "extra", "primary", "primary-foreground",
    ])
    expect(result.surfaces.extension.radiusClasses.has("rounded-pill")).toBe(true)
    expect(result.surfaces.extension.radiusValues.has("14px")).toBe(true)
    expect(result.surfaces.extension.radiusValues.has("12px")).toBe(false)
    expect(result.surfaces.extension.radiusValues.has("2px")).toBe(true)
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === "error")).toEqual([])
  })

  it("honors root spread order when inherited theme overrides an earlier extension property", () => {
    const result = buildTokenAuthority({ sources: sources({
      [extensionConfig]: `const shared = require('../tldw-frontend/tailwind.config.js');
        module.exports = { theme: { borderRadius: { custom: '3px' } }, ...shared }`,
    }) })
    expect(result.surfaces.extension.semanticColors).toEqual(result.surfaces.webui.semanticColors)
    expect(result.surfaces.extension.radiusClasses.has("rounded-card")).toBe(true)
    expect(result.surfaces.extension.radiusClasses.has("rounded-custom")).toBe(false)
  })

  it("does not trust inherited theme authority before an unresolved root spread", () => {
    const result = buildTokenAuthority({ sources: sources({
      [extensionConfig]: `const shared = require('../tldw-frontend/tailwind.config.js');
        module.exports = { ...shared, ...loadConfig() }`,
    }) })
    expect([...result.surfaces.extension.radiusClasses]).toEqual([])
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      path: extensionConfig, code: "unsupported-config-expression", severity: "coverage",
    }))
  })

  it.each(["...loadColors()", "[loadKey()]: 'red'"])(
    "invalidates earlier color entries at %s but retains later explicit and spread keys", unknown => {
      const result = buildTokenAuthority({ sources: sources({
        [webConfig]: `const later = { branch: { ...loadBranch(), muted: 'var(--color-primary)' } };
          module.exports = { theme: {
            colors: { base: 'var(--color-primary)', branch: { keep: 'var(--color-danger)' } },
            extend: { colors: {
              stale: 'var(--color-danger)', ${unknown}, ...later, restored: 'var(--color-danger)'
            } }
          } }`,
      }) })
      expect([...result.surfaces.webui.semanticColors]).toEqual(["branch-muted", "restored"])
      expect(result.diagnostics).toContainEqual(expect.objectContaining({
        path: webConfig, code: "unsupported-config-expression", severity: "coverage",
      }))
    })

  it("propagates unknown keys through known spreads without trusting earlier radius defaults", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `const settings = { ...loadTheme(), colors: { primary: 'var(--color-primary)' } };
        module.exports = { theme: { borderRadius: { stale: '3px' }, ...settings,
          extend: { borderRadius: { custom: '12px' } } } }`,
    }) })
    expect([...result.surfaces.webui.radiusClasses]).toEqual(["rounded-custom"])
    expect([...result.surfaces.webui.semanticColors]).toEqual(["primary"])
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      path: webConfig, code: "unsupported-config-expression", severity: "coverage",
    }))
  })

  it("does not treat an operand of a symbolic radius calculation as its result", () => {
    const result = buildTokenAuthority({ sources: sources({
      [adminConfig]: `export default { theme: { borderRadius: { md: 'calc(var(--radius) - 2px)' } } }`,
    }) })
    expect([...result.surfaces["admin-ui"].radiusValues]).toEqual(["calc(var(--radius) - 2px)"])
  })

  it("reports parse failures in supplied optional definitions", () => {
    const result = buildTokenAuthority({ sources: sources({ [themesPath]: "const defaultTheme = {" }) })
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path: themesPath, severity: "error" }))
  })

  it.each([webConfig, extensionConfig, sharedCss, adminConfig, adminCss])(
    "reports a missing required input: %s", path => {
      const result = buildTokenAuthority({ sources: sources().filter(source => source.path !== path) })
      expect(result.diagnostics).toContainEqual(expect.objectContaining({ path, severity: "error" }))
    })

  it("allows optional token and preset inputs to be absent", () => {
    const result = buildTokenAuthority({ sources: sources().filter(source => REQUIRED_DEFINITION_PATHS.includes(source.path)) })
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === "error")).toEqual([])
  })

  it.each([[webConfig, "module.exports = {"], [sharedCss, ":root { --color-danger: red;"]])(
    "reports parse failures with original locations: %s", (path, source) => {
      const result = buildTokenAuthority({ sources: sources({ [path]: source }) })
      expect(result.diagnostics).toContainEqual(expect.objectContaining({
        path, severity: "error", line: expect.any(Number), column: expect.any(Number),
      }))
    })

  it("reports unsupported config expressions without evaluating them", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `module.exports = { theme: { extend: { colors: {
        primary: 'rgb(var(--color-primary))', danger: dangerous()
      }, borderRadius: makeRadii() } } }`,
    }) })
    expect([...result.surfaces.webui.semanticColors]).toEqual(["primary"])
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path: webConfig, severity: "coverage" }))
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === "error")).toEqual([])
  })

  it.each([
    [webConfig, "const notExported = { theme: { extend: {} } }"],
    [webConfig, "module.exports = makeConfig()"],
    [webConfig, "module.exports = {}"],
    [webConfig, "module.exports = { theme: loadTheme() }"],
    [webConfig, "module.exports = { theme: { extend: loadTheme() } }"],
    [webConfig, "module.exports = { theme: {} }"],
    [webConfig, "module.exports = { theme: { extend: { colors: loadColors() } } }"],
    [webConfig, "module.exports = { theme: { extend: { colors: { danger: loadColor() } } } }"],
    [webConfig, "module.exports = { theme: { extend: { colors: {}, borderRadius: {} } } }"],
    [adminConfig, "const config = {}; export default config"],
    [adminConfig, "export default getConfig()"],
    [extensionConfig, "module.exports = getConfig()"],
    [extensionConfig, "const shared = require('./unknown.js'); module.exports = { ...shared }"],
  ])("marks wholly unusable required config authority incomplete: %s %s", (path, source) => {
    const result = buildTokenAuthority({ sources: sources({ [path]: source }) })
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path, severity: "error" }))
  })

  it.each([sharedCss, adminCss])("rejects a parsed required stylesheet with no recognized definitions: %s", path => {
    const result = buildTokenAuthority({ sources: sources({ [path]: "body { color: red }" }) })
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path, severity: "error" }))
  })

  it("reports a dynamic theme accessor without invoking it", () => {
    const result = buildTokenAuthority({ sources: sources({
      [webConfig]: `module.exports = { get theme() { throw new Error('must not run') } }`,
    }) })
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path: webConfig, severity: "coverage" }))
  })

  it("attributes extension-specific symbolic scale diagnostics to its own config", () => {
    const result = buildTokenAuthority({ sources: sources({
      [extensionConfig]: `const shared = require('../tldw-frontend/tailwind.config.js');
        module.exports = { ...shared, theme: { borderRadius: { custom: 'var(--radius-sm)' } } }`,
    }) })
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      path: extensionConfig, code: "symbolic-radius", severity: "coverage",
    }))
  })

  it("rejects unsupported extension inheritance instead of silently using WebUI authority", () => {
    const result = buildTokenAuthority({ sources: sources({
      [extensionConfig]: `const shared = require('./unknown.js'); module.exports = { ...shared }`,
    }) })
    expect(result.surfaces.extension.semanticColors.size).toBe(0)
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path: extensionConfig, severity: "coverage" }))
  })

  it("ignores arbitrary sources and returns deterministic scopes/diagnostics", () => {
    const input = [...sources(), { path: "apps/tldw-frontend/components/Product.tsx", source: "invalid {" }]
    expect(buildTokenAuthority({ sources: input })).toEqual(buildTokenAuthority({ sources: input.reverse() }))
    expect(buildTokenAuthority({ sources: input }).definitionScopes.every(scope => scope.path !== input[0].path)).toBe(true)
  })

  it("reads the checked-in inputs as syntax only", () => {
    const root = resolve(process.cwd(), "../../..")
    const input = TOKEN_SOURCE_PATHS.map(path => ({ path, source: readFileSync(`${root}/${path}`, "utf8") }))
    const result = buildTokenAuthority({ sources: input })
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === "error")).toEqual([])
    expect(result.surfaces.webui.semanticColors.has("state-permissionDenied")).toBe(true)
    expect(result.surfaces.extension.semanticColors.has("primary-foreground")).toBe(true)
    expect(result.surfaces["admin-ui"].semanticColors.has("destructive-foreground")).toBe(true)
    expect(result.surfaces.webui.radiusValues.has("6px")).toBe(true)
    expect(Object.isFrozen(REQUIRED_DEFINITION_PATHS)).toBe(true)
    expect(Object.isFrozen(TOKEN_SOURCE_PATHS)).toBe(true)
  })

  it("recognizes the checked-in text-on-tint declarations in both shared theme rules", () => {
    const root = resolve(process.cwd(), "../../..")
    const input = TOKEN_SOURCE_PATHS.map(path => ({ path, source: readFileSync(`${root}/${path}`, "utf8") }))
    const css = input.find(({ path }) => path === sharedCss)!.source
    const result = buildTokenAuthority({ sources: input })
    for (const color of ["primary", "success", "warn", "danger"]) {
      const variable = `--color-${color}-text`
      expect(result.surfaces.webui.cssVariables.has(variable)).toBe(true)
      expect(result.surfaces.webui.semanticTextColors?.has(color)).toBe(true)
      const offsets = [...css.matchAll(new RegExp(`${variable}:`, "g"))].map(match => match.index)
      expect(result.definitionScopes.filter(scope => scope.path === sharedCss && offsets.includes(scope.start))
        .map(scope => scope.owner)).toEqual([":root", ".dark"])
    }
    const config = input.find(({ path }) => path === webConfig)!.source
    expect(covered(result, webConfig, config.indexOf("textColor:"))).toBe(true)
  })
})
