import { describe, expect, it } from "vitest"
import { analyzeDriftSource } from "../../../scripts/design-system-drift-rules.mjs"

const path = "apps/packages/ui/src/components/ui/Notice.tsx"
const cssPath = "apps/packages/ui/src/assets/tailwind-shared.css"

function fixtureAuthority() {
  const shared = {
    semanticColors: new Set(["danger", "primary", "primary-foreground"]),
    cssVariables: new Set(["--color-danger", "--color-primary", "--radius"]),
    radiusClasses: new Set(["rounded", "rounded-none", "rounded-sm", "rounded-md", "rounded-lg", "rounded-full"]),
    radiusValues: new Set(["0", "0px", "2px", "6px", "8px", "9999px", "var(--radius)"]),
    definitionPaths: [cssPath]
  }
  return {
    surfaces: {
      "shared-ui": shared, webui: shared, extension: shared,
      "admin-ui": { ...shared, semanticColors: new Set(["background", "foreground", "destructive"]), cssVariables: new Set(["--background", "--foreground", "--radius"]) }
    },
    definitionScopes: [] as Array<{ path: string; start: number; end: number; owner: string; kind: string }>,
    diagnostics: []
  }
}

function analyze(source: string, filePath = path, authority = fixtureAuthority()) {
  return analyzeDriftSource({ path: filePath, source, authority })
}

describe("syntax-aware drift rules", () => {
  it("locates variants, layout utilities and inline named colors in their nearest owner", () => {
    const source = 'function Notice() {\n  return <div className="dark:hover:bg-red-500/20 sm:w-[44px]" style={{color: "red"}} />\n}'
    const result = analyze(source)
    expect(result.findings.map(f => [f.rule, f.value, f.classification, f.owner])).toEqual([
      ["palette-color", "dark:hover:bg-red-500/20", "token-review", "Notice"],
      ["layout-value", "sm:w-[44px]", "layout-review", "Notice"],
      ["literal-color", "red", "token-review", "Notice"]
    ])
    expect(result.findings.map(f => [f.line, f.column])).toEqual([[2, 26], [2, 51], [2, 80]])
    expect(result.diagnostics).toEqual([])
  })

  it("analyzes apply and CSS while excluding just individual token declarations", () => {
    const source = ':root { --color-danger: 224 88 109; } .notice { @apply bg-red-500 rounded-[13px]; color: red; }'
    const authority = fixtureAuthority()
    const definition = { path: cssPath, start: 8, end: 33, owner: ":root", kind: "css-variable" }
    authority.definitionScopes.push(definition)
    const result = analyze(source, cssPath, authority)
    expect(result.definitions).toEqual([definition])
    expect(result.findings.map(f => [f.rule, f.value, f.owner])).toEqual([
      ["palette-color", "bg-red-500", ".notice"], ["custom-radius", "rounded-[13px]", ".notice"],
      ["literal-color", "red", ".notice"]
    ])
    expect(result.findings.map(f => f.column)).toEqual([56, 67, 90])
  })

  it("recognizes semantic utilities, variables, allowed keywords and text sizes", () => {
    const result = analyze('<div className="bg-danger text-primary-foreground border-transparent text-[14px] bg-[var(--color-danger)] text-[color:var(--color-primary)] rounded-[8px] rounded-t-lg w-12" style={{color: "currentColor", backgroundColor: "inherit", borderRadius: "var(--radius)"}} />')
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it("uses separate admin vocabulary and preserves each product surface", () => {
    for (const [filePath, surface] of [[path, "shared-ui"], ["apps/tldw-frontend/app/page.tsx", "webui"], ["apps/extension/src/view.tsx", "extension"], ["admin-ui/app/page.tsx", "admin-ui"]]) {
      const result = analyze('<div class="bg-red-500" />', filePath)
      expect(result.findings[0]).toMatchObject({ surface, path: filePath, owner: null })
    }
    expect(analyze('<div className="bg-destructive bg-background" />', "admin-ui/app/page.tsx").findings).toEqual([])
  })

  it("finds palette neutrals, black/white, colors and gradient stops only at color prefixes", () => {
    const source = '<div className="bg-black text-white border-slate-200 ring-zinc-500 outline-red-500 divide-neutral-400 placeholder-gray-500 decoration-blue-600 fill-red-500 stroke-red-500 shadow-red-500 from-red-500 via-green-500 to-blue-500 red-500 font-black" />'
    const result = analyze(source)
    expect(result.findings).toHaveLength(14)
    expect(result.findings.every(f => f.rule === "palette-color")).toBe(true)
  })

  it("keeps arbitrary selectors, quoted spaces, opacity and important markers intact", () => {
    const source = `<div className="[&:nth-child(2)]:!bg-[#abc]/20 hover:rounded-[13px] -mt-[2px] [color:rebeccapurple] [background:url('https://red.test/#fff')] before:content-['bg-red-500 text-red-500'] [grid-template-columns:1fr_2fr]" />`
    expect(analyze(source).findings.map(f => [f.rule, f.value])).toEqual([
      ["literal-color", "[&:nth-child(2)]:!bg-[#abc]/20"], ["custom-radius", "hover:rounded-[13px]"],
      ["layout-value", "-mt-[2px]"], ["literal-color", "[color:rebeccapurple]"],
      ["layout-value", "[grid-template-columns:1fr_2fr]"]
    ])
  })

  it("never scans unrelated API strings, data objects, numeric props or prose", () => {
    expect(analyze('const payload = { color: "red", className: "bg-red-500", style: { width: 44 } }; const text = "rounded-[13px]"; send(payload); const View = () => <img width={44} alt="bg-red-500" data-color="red" />').findings).toEqual([])
  })

  it("resolves immutable local class maps and composition branches without scanning cva options as classes", () => {
    const source = `const styles = { base: "bg-red-500", sizes: { small: "w-[44px]" } } as const;
const badge = cva("text-white", { variants: { tone: { danger: "bg-danger", bad: "border-red-500" } }, defaultVariants: { tone: "bg-blue-500" } });
function Badge({active}) { return <div className={cn(styles.base, styles.sizes.small, active && "rounded-[13px]", {"ring-red-500": active}, badge({tone: "bad"}))} /> }`
    const result = analyze(source)
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500", "w-[44px]", "text-white", "border-red-500", "rounded-[13px]", "ring-red-500"])
    expect(result.diagnostics).toEqual([])
  })

  it("handles static conditionals, arrays and template fragments without joining interpolation gaps", () => {
    const source = 'function View({active, tone}) { return <div className={clsx(active ? "bg-red-500" : "text-white", ["w-[44px]"], `border-${tone}-500 bg-black`)} /> }'
    const result = analyze(source)
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500", "text-white", "w-[44px]", "bg-black"])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", path, line: 1})])
  })

  it("reports unresolved class/style/spread expressions and cycles as coverage", () => {
    const result = analyze('const cycle = cycle; const View = ({classes, styles}) => <div className={cn(classes, cycle)} style={{...styles, color: chooseColor()}} />')
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toHaveLength(4)
    expect(result.diagnostics.every(d => d.severity === "coverage" && d.line === 1 && d.column > 0)).toBe(true)
  })

  it("does not trust mutable local classes", () => {
    const result = analyze('let classes = "bg-red-500"; classes = getClasses(); const View = () => <div className={classes} />')
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("inspects named/functional colors in shorthand values but not URLs or quoted contents", () => {
    const source = `.notice { border: 1px solid rebeccapurple; background: url("https://red.test/#fff") no-repeat red; box-shadow: 0 1px 4px rgb(0 0 0 / .2); content: "red #fff"; font-family: red; background-image: url(data:image/svg+xml,red#fff); }`
    expect(analyze(source, cssPath).findings.map(f => [f.rule, f.value])).toEqual([
      ["literal-color", "rebeccapurple"], ["literal-color", "red"], ["literal-color", "rgb(0 0 0 / .2)"]
    ])
  })

  it("allows semantic variable references but inspects literal fallbacks and diagnoses unknown variables", () => {
    const source = '.notice { color: hsl(var(--color-danger)); background: var(--color-primary, red); border-color: var(--unregistered); outline-color: var(--color-danger, #fff); }'
    const result = analyze(source, cssPath)
    expect(result.findings.map(f => f.value)).toEqual(["red", "#fff"])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("checks radius shorthand per component without guessing unit conversions or variable values", () => {
    const source = '.notice { border-radius: 0 8px / 13px 6px; border-top-left-radius: var(--radius); border-bottom-right-radius: var(--other); }'
    const result = analyze(source, cssPath)
    expect(result.findings.map(f => [f.rule, f.value])).toEqual([["custom-radius", "13px"]])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("honors React numeric dimensions without reporting irrelevant numeric properties", () => {
    const result = analyze('const View = () => <div style={{width: 44, marginTop: -2, zIndex: 10, gridTemplateColumns: "1fr 2fr", borderRadius: 13, opacity: 0.5, lineHeight: 1.5, fontWeight: 400}} />')
    expect(result.findings.map(f => [f.rule, f.value])).toEqual([
      ["layout-value", "44px"], ["layout-value", "-2px"], ["layout-value", "10"], ["layout-value", "1fr 2fr"], ["custom-radius", "13px"]
    ])
  })

  it("excludes exact TypeScript definition ranges, not adjacent product or a whole theme file", () => {
    const filePath = "apps/packages/ui/src/themes/presets.ts"
    const source = 'export const theme = cva("bg-red-500");\nexport const View = cn("bg-[red]")'
    const authority = fixtureAuthority()
    const definition = {path: filePath, start: 0, end: 37, owner: "theme", kind: "theme"}
    authority.definitionScopes.push(definition)
    const result = analyze(source, filePath, authority)
    expect(result.definitions).toEqual([definition])
    expect(result.findings).toEqual([expect.objectContaining({rule: "literal-color", value: "bg-[red]", owner: "View", line: 2})])
  })

  it.each(["js", "jsx", "ts", "tsx", "mjs", "cjs"])("parses %s source and preserves escaped-string locations", extension => {
    const source = 'const View = cn("text-white\\n bg-red-500")'
    const result = analyze(source, `apps/packages/ui/src/View.${extension}`)
    expect(result.findings.map(f => [f.value, f.line, f.column])).toEqual([["text-white", 1, 18], ["bg-red-500", 1, 31]])
    expect(result.diagnostics).toEqual([])
  })

  it("resolves local style objects only when used as styles", () => {
    const result = analyze('const styles = {color: "red", width: 44} as const; const unused = {color: "blue"}; const View = () => <div style={styles} />')
    expect(result.findings.map(f => f.value)).toEqual(["red", "44px"])
  })

  it("returns error diagnostics for malformed JavaScript and CSS", () => {
    for (const [source, filePath] of [['const View = () => <div className="bg-red-500"', path], ['const View = () => <div className={"\\u{110000}"} />', path], [".bad { color: red;", cssPath]]) {
      expect(analyze(source, filePath).diagnostics).toContainEqual(expect.objectContaining({severity: "error", path: filePath, line: 1}))
    }
  })

  it("is deterministic and does not duplicate a class map referenced twice", () => {
    const source = 'const base = "bg-red-500"; const View = () => <><div className={base} /><div className={base} /></>'
    const first = analyze(source)
    expect(first.findings).toHaveLength(1)
    expect(analyze(source)).toEqual(first)
  })

  it("maps inline shorthand color offsets through JavaScript escapes", () => {
    const result = analyze('const View = () => <div style={{border: "1px\\t solid\\u0020red"}} />')
    expect(result.findings).toEqual([expect.objectContaining({value: "red", line: 1, column: 59, owner: "View"})])
  })

  it("analyzes standalone template expressions without fabricating classes across gaps", () => {
    const result = analyze('function View({active, prefix}) { return <div className={`${active ? "bg-red-500" : "text-white"} ${prefix}bg-black`} /> }')
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500", "text-white"])
    expect(result.diagnostics).toHaveLength(1)
  })

  it("analyzes independent inline style value branches", () => {
    const result = analyze('const View = ({active}) => <div style={{color: active ? "red" : "blue"}} />')
    expect(result.findings.map(f => f.value)).toEqual(["red", "blue"])
    expect(result.diagnostics).toEqual([])
  })

  it("allows semantic layout references and diagnoses arbitrary variable gaps", () => {
    const authority = fixtureAuthority()
    authority.surfaces["shared-ui"].cssVariables.add("--pane-width")
    const result = analyze('<div className="w-[var(--pane-width)] h-[var(--unknown)]" />', path, authority)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("does not claim a local map is immutable after property assignment", () => {
    const result = analyze('const styles = {base: "bg-red-500"}; styles.base = fetchClass(); const View = () => <div className={styles.base} />')
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("does not resolve a class map through a shadowed function parameter", () => {
    const result = analyze('const styles = {base: "bg-red-500"}; function View(styles) { return <div className={styles.base} /> }')
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("reports unresolved cva variants rather than silently dropping them", () => {
    const result = analyze('const View = cva("bg-danger", {variants: remoteVariants})')
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage"})])
  })

  it("reports coverage for dynamic local class-map keys instead of silently omitting styles", () => {
    const result = analyze('const styles = {error: "bg-red-500", ready: "bg-danger"}; const View = ({state}) => <div className={styles[state]} />')
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", code: "dynamic-styling"})])
  })

  it("allows recognized radius variable references even without an identical radius expression", () => {
    const authority = fixtureAuthority()
    authority.surfaces["shared-ui"].cssVariables.add("--corner")
    const result = analyze('.notice { border-radius: var(--corner); }', cssPath, authority)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it("does not label 4px off-scale when the documented radius uses rem and conversion is unknown", () => {
    const authority = fixtureAuthority()
    authority.surfaces["shared-ui"].radiusValues = new Set(["0", "0.25rem", "0.5rem", "9999px"])
    const result = analyze('.react-pdf { border-radius: 4px; }', cssPath, authority)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", code: "unresolved-radius", line: 1, column: 29})])
  })

  it("allows native system colors as platform semantics, including selectedItem", () => {
    const result = analyze('.react-pdf { background: selectedItem; color: CanvasText; border: 1px solid ButtonBorder; outline-color: AccentColor; }', cssPath)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it("preserves CSS offsets through parser-normalized comments without analyzing comment contents", () => {
    const source = '.x { border: 1px /* foo */ solid red; @apply bg-red-500 /* text-white */ rounded-[13px]; }'
    const result = analyze(source, cssPath)
    expect(result.findings.map(f => [f.value, f.column])).toEqual([["red", 34], ["bg-red-500", 46], ["rounded-[13px]", 74]])
  })

  it("finds intrinsic color literals in custom-property overrides outside exact definition scopes", () => {
    const source = ':root { --color-danger: #f00; }\n.notice { --color-danger: #f00; --tint: rgb(255 0 0); color: var(--color-danger); }'
    const authority = fixtureAuthority()
    authority.definitionScopes.push({path: cssPath, start: 8, end: 29, owner: ":root", kind: "css-variable"})
    const result = analyze(source, cssPath, authority)
    expect(result.findings.map(f => [f.rule, f.value, f.owner, f.line, f.column])).toEqual([
      ["literal-color", "#f00", ".notice", 2, 27],
      ["literal-color", "rgb(255 0 0)", ".notice", 2, 41]
    ])
    expect(result.diagnostics).toEqual([])
  })

  it("does not interpret named identifiers, quoted text or URLs in arbitrary custom properties as colors", () => {
    const result = analyze('.notice { --mode: red; --animation: blue; --label: "#f00 rgb(255 0 0)"; --image: url("https://red.test/#f00"); }', cssPath)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it("checks intrinsic colors in React custom-property overrides only when used as styles", () => {
    const result = analyze('const payload = {"--color-danger": "#f00"}; const View = () => <div style={{"--color-danger": "#f00", "--mode": "red"}} />')
    expect(result.findings).toEqual([expect.objectContaining({rule: "literal-color", value: "#f00", owner: "View"})])
    expect(result.diagnostics).toEqual([])
  })

  it("normalizes quoted CVA options and compound class keys without inspecting variant selector strings", () => {
    const source = `const View = cva('', {
  'variants': { tone: {bad: 'bg-red-500'} },
  'compoundVariants': [{tone: 'text-blue-500', 'class': 'text-white', "className": 'border-red-500'}]
})`
    const result = analyze(source)
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500", "text-white", "border-red-500"])
    expect(result.diagnostics).toEqual([])
  })

  it("follows static CVA options spreads into variants and compound classes", () => {
    const source = `const options = {'variants': {tone: {bad: 'bg-red-500'}}, compoundVariants: [{'className': 'text-white'}]};
const View = cva('', {...options})`
    const result = analyze(source)
    expect(result.findings.map(f => [f.value, f.owner])).toEqual([["bg-red-500", "options"], ["text-white", "options"]])
    expect(result.diagnostics).toEqual([])
  })

  it("reports coverage for unresolved CVA options spreads", () => {
    const result = analyze("const View = cva('', {...getOptions()})")
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", code: "dynamic-styling"})])
  })

  it.each([
    "rgb(255 0 0 / var(--moderation-hero-grid-opacity))",
    "rgba(255, 0, 0, var(--moderation-hero-grid-opacity))",
    "hsl(0 100% 50% / var(--moderation-hero-grid-opacity))",
    "color(srgb 1 0 0 / var(--moderation-hero-grid-opacity))"
  ])("finds literal color channels even when alpha is variable: %s", value => {
    const result = analyze(`.notice { color: ${value}; }`, cssPath)
    expect(result.findings).toEqual([expect.objectContaining({rule: "literal-color", value, owner: ".notice", line: 1, column: 18})])
  })

  it("finds variable-alpha literal channels in arbitrary utilities and inline styles", () => {
    const result = analyze('<div className="bg-[rgb(255_0_0_/_var(--alpha))]" style={{color: "rgb(255 0 0 / var(--alpha))"}} />')
    expect(result.findings.map(f => [f.rule, f.value])).toEqual([
      ["literal-color", "bg-[rgb(255_0_0_/_var(--alpha))]"], ["literal-color", "rgb(255 0 0 / var(--alpha))"]
    ])
  })

  it("keeps semantic color channel variables allowed with either static or variable alpha", () => {
    const authority = fixtureAuthority()
    authority.surfaces["shared-ui"].cssVariables.add("--alpha")
    const result = analyze('.notice { color: rgb(var(--color-danger) / var(--alpha)); background: hsl(var(--color-primary) / .5); }', cssPath, authority)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it.each([
    "...options, variants: {tone: {bad: 'bg-danger'}}",
    "variants: {tone: {bad: 'bg-red-500'}}, ...semanticOptions",
    "...options, compoundVariants: [], variants: {tone: {bad: 'bg-danger'}}"
  ])("honors effective CVA options after property overwrites: %s", body => {
    const source = `const options = {variants: {tone: {bad: 'bg-red-500'}}};
const semanticOptions = {variants: {tone: {bad: 'bg-danger'}}};
const badge = cva('', {${body}})`
    const result = analyze(source)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it("keeps a later spread's effective palette entry and original location", () => {
    const source = "const options = {variants: {tone: {bad: 'bg-red-500'}}};\nconst badge = cva('', {variants: {tone: {bad: 'bg-danger'}}, ...options})"
    const result = analyze(source)
    expect(result.findings).toEqual([expect.objectContaining({rule: "palette-color", value: "bg-red-500", owner: "options", line: 1, column: 42})])
    expect(result.diagnostics).toEqual([])
  })

  it("honors overwrites in nested option, variant and compound class spreads", () => {
    const source = `const bad = {bad: 'bg-red-500'};
const badCompound = {className: 'text-white'};
const original = {variants: {tone: bad}, compoundVariants: [badCompound]};
const replaced = {...original, 'variants': {tone: {...bad, bad: 'bg-danger'}}, compoundVariants: [{...badCompound, 'className': 'bg-danger'}]};
const badge = cva('', {...{...replaced}})`
    const result = analyze(source)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([])
  })

  it("does not report earlier CVA entries that an unresolved spread may overwrite", () => {
    const result = analyze("const options = {variants: {tone: {bad: 'bg-red-500'}}}; const badge = cva('', {...options, ...getOptions()})")
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", code: "dynamic-styling"})])
  })

  it("propagates nested unresolved spread uncertainty but preserves explicit later entries", () => {
    const result = analyze("const uncertain = {...getOptions(), variants: {tone: {bad: 'bg-danger'}}}; const badge = cva('', {compoundVariants: [{className: 'text-white'}], ...uncertain})")
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", code: "dynamic-styling"})])
  })

  it("analyzes explicit CVA entries after an unresolved spread", () => {
    const result = analyze("const badge = cva('', {...getOptions(), variants: {tone: {bad: 'bg-red-500'}}})")
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500"])
    expect(result.diagnostics).toEqual([expect.objectContaining({severity: "coverage", code: "dynamic-styling"})])
  })
})
