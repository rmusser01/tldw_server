import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const frontendRoot = process.cwd()
const packagesUiRoot = path.resolve(frontendRoot, "../packages/ui/src")

const readSource = (absolutePath: string) => readFileSync(absolutePath, "utf8")

describe("Document workspace react-pdf CSS contract", () => {
  it("keeps react-pdf layer styles in shared app-level stylesheets instead of component imports", () => {
    const pdfDocumentSource = readSource(
      path.join(
        packagesUiRoot,
        "components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx"
      )
    )

    expect(pdfDocumentSource).not.toContain("react-pdf/dist/esm/Page/AnnotationLayer.css")
    expect(pdfDocumentSource).not.toContain("react-pdf/dist/esm/Page/TextLayer.css")
  })

  it("co-locates the shared react-pdf stylesheet with the web document-workspace async chunk", () => {
    // The web app must not load PDF layer CSS app-wide from _app (perf
    // remediation W4): it rides the document-workspace lazy chunk instead.
    const webAppSource = readSource(path.join(frontendRoot, "pages/_app.tsx"))
    expect(webAppSource).not.toContain('import "@/assets/react-pdf.css"')

    const webLazyEntrySource = readSource(
      path.join(frontendRoot, "routes/document-workspace.ts")
    )
    expect(webLazyEntrySource).toContain('import "@/assets/react-pdf.css"')
    expect(webLazyEntrySource).toContain("routes/option-document-workspace")

    const documentWorkspacePageSource = readSource(
      path.join(frontendRoot, "pages/document-workspace.tsx")
    )
    expect(documentWorkspacePageSource).toContain(
      'import("@web/routes/document-workspace")'
    )
  })

  it("keeps loading the shared react-pdf stylesheet from the extension app shells", () => {
    const optionsEntrySource = readSource(
      path.join(packagesUiRoot, "entries/options/main.tsx")
    )
    const sidepanelEntrySource = readSource(
      path.join(packagesUiRoot, "entries/sidepanel/main.tsx")
    )

    expect(optionsEntrySource).toContain('import "@/assets/react-pdf.css"')
    expect(sidepanelEntrySource).toContain('import "@/assets/react-pdf.css"')
  })

  it("ships a local react-pdf stylesheet with both text and annotation layer rules", () => {
    const stylesheetPath = path.join(packagesUiRoot, "assets/react-pdf.css")

    expect(existsSync(stylesheetPath)).toBe(true)

    const stylesheetSource = readSource(stylesheetPath)
    expect(stylesheetSource).toContain("--react-pdf-text-layer")
    expect(stylesheetSource).toContain("--react-pdf-annotation-layer")
  })
})
