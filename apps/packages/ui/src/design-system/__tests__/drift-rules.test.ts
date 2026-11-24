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

  it.each(["clsx", "cn"])("uses effective conditional keys and omits statically falsy properties in %s", compose => {
    const source = [
      "const disabled = false",
      'const base = {"bg-red-500": true, "border-black": true}',
      'const View = ({active}) => <div className={COMPOSE({...base, ["bg-red-500"]: disabled, "bg-black": 0, "text-blue-500": "", "border-red-500": null, "ring-red-500": undefined, "text-white": active})} />'
    ].join("\n").replace("COMPOSE", compose)
    const result = analyze(source)
    expect(result.findings.map(f => f.value)).toEqual(["border-black", "text-white"])
    expect(result.diagnostics).toEqual([])
  })

  it("reports unknown conditional-map overwrites but keeps explicit later keys", () => {
    const source = 'const View = ({flags}) => <div className={clsx({"bg-red-500": true, ...{...flags, "text-white": true}})} />'
    const result = analyze(source)
    expect(result.findings.map(f => f.value)).toEqual(["text-white"])
    expect(result.diagnostics).toEqual([expect.objectContaining({
      code: "dynamic-styling", column: source.indexOf("...flags") + 4
    })])
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

  it("compares radius variable names case-sensitively while normalizing numbers and units", () => {
    const source = ".notice { border-radius: 8.00PX -0PX; border-radius: VAR(--radius); border-radius: INHERIT; border-radius: var(--RADIUS); }"
    const result = analyze(source, cssPath)
    expect(result.findings).toEqual([])
    expect(result.diagnostics).toEqual([
      expect.objectContaining({code: "unresolved-radius", severity: "coverage", column: source.indexOf("var(--RADIUS)") + 1})
    ])
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

  it("uses the TypeScript source position for findings and parse errors after a bare carriage return", () => {
    const source = 'const a = 0;\rconst View = () => <div className="text-white" />'
    expect(analyze(source).findings).toEqual([expect.objectContaining({value: "text-white", line: 2, column: 36})])
    expect(analyze("const a = 0;\rconst View =").diagnostics).toEqual([
      expect.objectContaining({code: "parse-error", severity: "error", line: 2, column: 13})
    ])
  })

  it("resolves local style objects only when used as styles", () => {
    const result = analyze('const styles = {color: "red", width: 44} as const; const unused = {color: "blue"}; const View = () => <div style={styles} />')
    expect(result.findings.map(f => f.value)).toEqual(["red", "44px"])
  })

  it("checks only the effective inline style keys after duplicate assignments and known spreads", () => {
    const source = 'const finalStyle = {color: "blue"};\nconst View = () => <div style={{color: "red", ...finalStyle, width: 44, width: "inherit"}} />'
    const result = analyze(source)
    expect(result.findings).toEqual([
      expect.objectContaining({value: "blue", owner: "finalStyle", line: 1, column: source.indexOf("blue") + 1})
    ])
    expect(result.diagnostics).toEqual([])
  })

  it("preserves only the inline styles that a nested unresolved spread cannot overwrite", () => {
    const source = 'const uncertain = {...getStyles(), color: "blue"};\nconst View = () => <div style={{color: "red", width: 44, ...uncertain, height: 52}} />'
    const result = analyze(source)
    expect(result.findings.map(f => [f.value, f.owner, f.line])).toEqual([
      ["blue", "uncertain", 1], ["52px", "View", 2]
    ])
    expect(result.diagnostics).toEqual([
      expect.objectContaining({code: "dynamic-styling", severity: "coverage", line: 1, column: source.indexOf("getStyles") + 1})
    ])
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

  it("decodes entities in quoted JSX attributes without cooking them as JavaScript strings", () => {
    const source = [
      '<div className="p-2&#32;bg-red-500&#x20;&#116;ext-white [&amp;>div]:bg-black" />',
      '<div className={"p-2&#32;bg-red-500"} />',
      '<div className="p-2\\u0020text-white" />',
      '<div className={"p-2\\u0020text-white"} />',
      '<div className="text-white&nbsp;bg-red-500" />',
      '<div className={"text-white\\u00a0bg-red-500"} />'
    ]
    const result = analyze(source.join(";\n"))
    expect(result.findings.map(f => [f.value, f.line, f.column])).toEqual([
      ["bg-red-500", 1, source[0].indexOf("bg-red-500") + 1],
      ["text-white", 1, source[0].indexOf("&#116;") + 1],
      ["[&>div]:bg-black", 1, source[0].indexOf("[&amp;") + 1],
      ["text-white", 4, source[3].indexOf("text-white") + 1]
    ])
    expect(result.diagnostics).toEqual([])
  })

  it("retains whitespace normalized by cn before the string reaches JSX", () => {
    const result = analyze('<div className={cn("text-white\\u00a0bg-red-500")} />')
    expect(result.findings.map(f => f.value)).toEqual(["text-white", "bg-red-500"])
  })

  it("analyzes standalone template expressions without fabricating classes across gaps", () => {
    const result = analyze('function View({active, prefix}) { return <div className={`${active ? "bg-red-500" : "text-white"} ${prefix}bg-black`} /> }')
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500", "text-white"])
    expect(result.diagnostics).toHaveLength(1)
  })

  it("keeps complete utilities after a template interpolation inside a bracket or a quote", () => {
    const source = [
      'const First = ({tone}) => <div className={`bg-[${tone}] text-white w-[44px]`} />',
      "const Second = () => <div className={`before:content-['${'bg-black'} text-blue-500'] bg-red-500`} />"
    ].join("\n")
    const result = analyze(source)
    expect(result.findings.map(f => [f.value, f.line, f.column])).toEqual([
      ["text-white", 1, source.split("\n")[0].indexOf("text-white") + 1],
      ["w-[44px]", 1, source.split("\n")[0].indexOf("w-[44px]") + 1],
      ["bg-red-500", 2, source.split("\n")[1].indexOf("bg-red-500") + 1]
    ])
    expect(result.diagnostics.map(d => [d.code, d.line])).toEqual([
      ["dynamic-styling", 1], ["dynamic-styling", 2]
    ])
  })

  it("analyzes only effective JSX styling props across static spreads and explicit attributes", () => {
    const source = [
      'const replacement = {className: "text-white", style: {color: "blue"}}',
      'const View = () => <><div className={cn("bg-red-500")} style={{color: "red"}} {...replacement} />',
      '<div {...{className: cn("bg-black"), style: {color: "green"}}} className="text-danger" style={{width: 44}} /></>'
    ].join("\n")
    const result = analyze(source)
    expect(result.findings.map(f => [f.value, f.owner, f.line])).toEqual([
      ["text-white", "replacement", 1], ["blue", "replacement", 1], ["44px", "View", 3]
    ])
    expect(result.diagnostics).toEqual([])
  })

  it("does not re-report an overwritten composition from a resolved JSX spread", () => {
    const source = 'const original = {className: cn("bg-red-500")}; const View = () => <div {...original} className="text-white" />'
    expect(analyze(source).findings.map(f => f.value)).toEqual(["text-white"])
  })

  it("reports uncertain JSX spread overwrites and retains proven later styling props", () => {
    const source = 'const View = ({props}) => <div className="text-white" style={{color: "red"}} {...{...props, className: "bg-red-500"}} />'
    const result = analyze(source)
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500"])
    expect(result.diagnostics).toEqual([expect.objectContaining({
      code: "dynamic-styling", column: source.indexOf("...props") + 4
    })])
  })

  it("matches CSS properties case-insensitively without folding custom variable names", () => {
    const authority = fixtureAuthority()
    authority.surfaces["shared-ui"].cssVariables.add("--Brand")
    const result = analyze('.notice { COLOR: red; Background-Color: var(--Brand); WIDTH: 44px; --Brand: #fff; color: var(--brand); }', cssPath, authority)
    expect(result.findings.map(f => [f.rule, f.value])).toEqual([
      ["literal-color", "red"], ["layout-value", "44px"], ["literal-color", "#fff"]
    ])
    expect(result.diagnostics).toEqual([expect.objectContaining({code: "unresolved-variable", column: 90})])
  })

  it("finds colors in image-hinted utilities without inspecting URLs or custom paint names", () => {
    const result = analyze('<div className="bg-[image:linear-gradient(red,blue)] bg-[image:url(https://red.test/#fff)] bg-[image:paint(red)]" />')
    expect(result.findings.map(f => [f.rule, f.value])).toEqual([
      ["literal-color", "bg-[image:linear-gradient(red,blue)]"]
    ])
    expect(result.diagnostics).toEqual([])
  })

  it("honors semantic text overrides only on text utilities", () => {
    const authority = fixtureAuthority()
    authority.surfaces["shared-ui"].semanticColors.add("blue-500")
    Object.assign(authority.surfaces["shared-ui"], {semanticTextColors: new Set(["red-500"])})
    const result = analyze('<div className="hover:text-red-500/50 bg-red-500 border-red-500 text-blue-500 bg-blue-500" />', path, authority)
    expect(result.findings.map(f => f.value)).toEqual(["bg-red-500", "border-red-500", "text-blue-500"])
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

  it.each([
    "let classes = getClasses(); ready();",
    "const {classes} of entries",
    "const classes in entries"
  ])("respects a loop initializer binding without masking the outer const: %s", initializer => {
    const source = [
      'const classes = "bg-red-500";',
      `for (${initializer}) {`,
      '  show(<div className={classes} />);',
      '}',
      'show(<div className={classes} />);'
    ]
    const result = analyze(source.join("\n"))
    expect(result.findings.map(f => [f.value, f.line])).toEqual([["bg-red-500", 1]])
    expect(result.diagnostics).toEqual([
      expect.objectContaining({code: "dynamic-styling", severity: "coverage", line: 3, column: source[2].indexOf("{classes}") + 2})
    ])
  })

  it.each([
    "get base() { return getClass() }",
    "set base(value) {}",
    "base() { return getClass() }"
  ])("stops resolving a class map when a matching member replaces its value: %s", member => {
    const usage = '<div className={cn(styles.base, styles.other)} />'
    const result = analyze(`const styles = {base: "bg-red-500", ${member}, other: "text-white", unrelated() {}};\n${usage}`)
    expect(result.findings.map(f => f.value)).toEqual(["text-white"])
    expect(result.diagnostics).toEqual([
      expect.objectContaining({code: "dynamic-styling", severity: "coverage", line: 2, column: usage.indexOf("styles.base") + 1})
    ])
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

  it("keeps HTML-only entities literal in quoted JSX attributes", () => {
    const source = '<div className="p-2&lbrack; bg-red-500" />'
    const result = analyze(source)
    expect(result.findings).toEqual([expect.objectContaining({ value: "bg-red-500", column: source.indexOf("bg-red-500") + 1 })])
  })

  it("retains both possible style values across a conditional spread", () => {
    const result = analyze('<div style={{width: 44, ...(active ? {color: "red"} : {color: "blue"})}} />')
    expect(result.findings.map(f => f.value)).toEqual(["44px", "red", "blue"])
    expect(result.diagnostics).toEqual([])
  })

  it("preserves an earlier style unless both branches or a later key overwrite it", () => {
    const result = analyze('<div style={{color: "red", width: 44, ...(active ? {color: "blue", width: 28} : {height: 32}), width: "auto"}} />')
    expect(result.findings.map(f => f.value)).toEqual(["red", "blue", "32px", "auto"])
    expect(result.diagnostics).toEqual([])
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
