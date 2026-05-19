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

  it("loads the shared react-pdf stylesheet from the web and extension app shells", () => {
    const webAppSource = readSource(path.join(frontendRoot, "pages/_app.tsx"))
    const optionsEntrySource = readSource(
      path.join(packagesUiRoot, "entries/options/main.tsx")
    )
    const sidepanelEntrySource = readSource(
      path.join(packagesUiRoot, "entries/sidepanel/main.tsx")
    )

    expect(webAppSource).toContain('import "@/assets/react-pdf.css"')
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
