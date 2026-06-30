import DOMPurify from "dompurify"
import { marked } from "marked"
import type {
  NoteStudioDocument,
  NoteStudioPayload,
  NoteStudioSectionPayload,
  NotesStudioHandwritingMode,
  NotesStudioPaperSize,
  NotesStudioTemplateType
} from "./notes-studio-types"

export type SingleNoteExportData = {
  id?: string | number | null
  title: string
  content: string
  keywords: string[]
}

export interface StudioPrintLabels {
  untitledNote: string
  printTitleSuffix: string
  exportedLabel: string
  templateLabel: string
  paperLabel: string
  diagramHeading: string
}

export type SingleNoteCopyMode = "content" | "markdown"
export type SingleNoteExportFormat = "md" | "json" | "print"

const normalizeKeywords = (keywords: string[]) =>
  keywords
    .map((keyword) => String(keyword || "").trim())
    .filter((keyword) => keyword.length > 0)

const escapeYamlString = (value: string) =>
  String(value || "")
    .replace(/\\/g, "\\\\")
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t")
    .replace(/"/g, '\\"')

export const buildSingleNoteMarkdown = (note: SingleNoteExportData): string => {
  const title = String(note.title || "").trim()
  const content = String(note.content || "")
  const keywords = normalizeKeywords(note.keywords || [])

  const heading = title ? `# ${title}\n\n` : ""
  if (keywords.length === 0) {
    return `${heading}${content}`.trimEnd()
  }

  const frontmatter = [
    "---",
    ...(title ? [`title: "${escapeYamlString(title)}"`] : []),
    "keywords:",
    ...keywords.map((keyword) => `  - "${escapeYamlString(keyword)}"`),
    "---",
    ""
  ].join("\n")

  return `${frontmatter}\n${heading}${content}`.trimEnd()
}

export const buildSingleNoteJson = (note: SingleNoteExportData): string =>
  JSON.stringify(
    {
      id: note.id ?? null,
      title: String(note.title || ""),
      content: String(note.content || ""),
      keywords: normalizeKeywords(note.keywords || [])
    },
    null,
    2
  )

export const buildSingleNoteCopyText = (
  note: SingleNoteExportData,
  mode: SingleNoteCopyMode
): string => {
  if (mode === "markdown") {
    return buildSingleNoteMarkdown(note)
  }
  return String(note.content || "")
}

const escapeHtml = (value: string) =>
  String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")

const renderPrintableMarkdownHtml = (markdown: string): string => {
  const rendered = marked.parse(markdown, {
    gfm: true,
    breaks: true
  })
  const html = typeof rendered === "string" ? rendered : String(markdown || "")
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true }
  })
}

export const SINGLE_NOTE_PRINT_STYLES = `
    body {
      margin: 0;
      font-family: "Helvetica Neue", Arial, sans-serif;
      color: #111827;
      background: #ffffff;
    }
    .print-shell {
      max-width: 860px;
      margin: 0 auto;
      padding: 24px;
    }
    .note-header {
      border-bottom: 1px solid #d1d5db;
      margin-bottom: 16px;
      padding-bottom: 12px;
    }
    .note-header h1 {
      margin: 0 0 8px 0;
      font-size: 28px;
      line-height: 1.2;
    }
    .meta-row {
      margin: 4px 0;
      color: #4b5563;
      font-size: 13px;
      line-height: 1.4;
    }
    .note-body {
      color: #111827;
      font-size: 15px;
      line-height: 1.65;
      word-break: break-word;
    }
    .note-body img {
      max-width: 100%;
      height: auto;
    }
    .note-body pre {
      overflow-x: auto;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      background: #f9fafb;
      padding: 10px;
    }
    .note-body blockquote {
      margin-left: 0;
      padding-left: 12px;
      border-left: 3px solid #d1d5db;
      color: #374151;
    }
    @media print {
      body {
        margin: 0;
      }
      .print-shell {
        max-width: none;
        padding: 12mm;
      }
      .note-header,
      .note-body pre,
      .note-body blockquote {
        break-inside: avoid;
        page-break-inside: avoid;
      }
      a {
        color: inherit;
        text-decoration: underline;
      }
    }
  `

export const buildSingleNotePrintableHtml = (
  note: SingleNoteExportData,
  options?: {
    generatedAtIso?: string
  }
): string => {
  const title = String(note.title || "").trim()
  const content = String(note.content || "")
  const keywords = normalizeKeywords(note.keywords || [])
  const printableTitle = title || "Untitled note"
  const generatedAtIso = String(options?.generatedAtIso || new Date().toISOString())

  const markdownSource = title
    ? `# ${title}\n\n${content}`
    : (content || "_(Empty note)_")
  const renderedContent = renderPrintableMarkdownHtml(markdownSource)
  const keywordsMarkup =
    keywords.length > 0
      ? `<p class="meta-row"><strong>Tags:</strong> ${escapeHtml(keywords.join(", "))}</p>`
      : ""

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(printableTitle)} - Printable Note</title>
  <style>
  ${SINGLE_NOTE_PRINT_STYLES}
  </style>
</head>
<body>
  <div class="print-shell">
    <header class="note-header">
      <h1>${escapeHtml(printableTitle)}</h1>
      <p class="meta-row"><strong>Exported:</strong> ${escapeHtml(generatedAtIso)}</p>
      ${keywordsMarkup}
    </header>
    <main class="note-body">
      ${renderedContent}
    </main>
  </div>
</body>
</html>`
}

const asStudioPayload = (value: unknown): NoteStudioPayload => {
  if (!value || typeof value !== "object") {
    return { layout: null, sections: [] }
  }
  return value as NoteStudioPayload
}

const normalizeStudioTemplateType = (value: unknown): NotesStudioTemplateType => {
  if (value === "grid" || value === "cornell") return value
  return "lined"
}

const normalizeStudioHandwritingMode = (value: unknown): NotesStudioHandwritingMode => {
  if (value === "off") return "off"
  return "accented"
}

const normalizeStudioSections = (value: unknown): NoteStudioSectionPayload[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => (entry && typeof entry === "object" ? (entry as NoteStudioSectionPayload) : null))
    .filter((entry): entry is NoteStudioSectionPayload => Boolean(entry?.id))
}

const normalizePaperSize = (value: unknown): NotesStudioPaperSize => {
  if (value === "US Letter" || value === "A5") return value
  return "A4"
}

const sanitizeSvgMarkup = (svg: string): string =>
  DOMPurify.sanitize(svg, {
    USE_PROFILES: { html: true, svg: true, svgFilters: true }
  })

const renderStudioSectionHtml = (
  section: NoteStudioSectionPayload,
  options: { handwritingMode: NotesStudioHandwritingMode }
): string => {
  const kind = String(section.kind || "notes")
  const sectionTitle = String(section.title || "")
  const accentHeadingClass = options.handwritingMode === "accented" ? "studio-handwriting-accent" : ""
  const cueItems = Array.isArray(section.items)
    ? section.items.map((item) => String(item || "")).filter((item) => item.length > 0)
    : []
  const contentText =
    typeof section.content === "string"
      ? section.content
      : kind !== "cue" && cueItems.length > 0
        ? cueItems.join("\n")
        : ""
  const cueMarkup =
    cueItems.length > 0
      ? `<ul class="studio-cue-list">${cueItems
          .map(
            (item) =>
              `<li class="studio-cue-item ${
                options.handwritingMode === "accented" ? "studio-handwriting-accent" : ""
              }">${escapeHtml(item)}</li>`
          )
          .join("")}</ul>`
      : ""
  const contentMarkup = contentText
    ? `<p class="studio-section-content ${
        options.handwritingMode === "accented" && kind === "prompt" ? "studio-handwriting-accent" : ""
      }">${escapeHtml(contentText).replace(/\n/g, "<br />")}</p>`
    : ""

  return `<article class="studio-section studio-section-${escapeHtml(kind)}">
    ${
      sectionTitle
        ? `<h3 class="studio-section-title ${accentHeadingClass}">${escapeHtml(sectionTitle)}</h3>`
        : ""
    }
    ${cueMarkup}
    ${contentMarkup}
  </article>`
}

const renderStudioDiagramHtml = (
  studioDocument: NoteStudioDocument,
  labels: StudioPrintLabels
): string => {
  const manifest = studioDocument.diagram_manifest_json
  const cachedSvg =
    manifest && typeof manifest === "object" && typeof (manifest as Record<string, unknown>).cached_svg === "string"
      ? String((manifest as Record<string, unknown>).cached_svg || "")
      : ""
  if (!cachedSvg.trim()) return ""

  return `<section class="studio-diagram-card">
    <h3 class="studio-section-title">${escapeHtml(labels.diagramHeading)}</h3>
    <div class="studio-diagram-svg">${sanitizeSvgMarkup(cachedSvg)}</div>
  </section>`
}

export const getDefaultStudioPaperSizeFromLocale = (locale?: string | null): NotesStudioPaperSize => {
  const normalized = String(locale || "")
    .trim()
    .toLowerCase()
    .replace(/_/g, "-")
  if (/(^|-)us($|-)/i.test(normalized)) {
    return "US Letter"
  }
  return "A4"
}

const STUDIO_PRINT_STYLES = `
  :root {
    --studio-paper-size: A4;
  }
  body {
    margin: 0;
    font-family: "Atkinson Hyperlegible", "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: #1f2937;
    background: #ffffff;
  }
  body[data-paper-size="US Letter"] { --studio-paper-size: Letter; }
  body[data-paper-size="A4"] { --studio-paper-size: A4; }
  body[data-paper-size="A5"] { --studio-paper-size: A5; }
  @page {
    size: var(--studio-paper-size);
    margin: 12mm;
  }
  .notes-studio-print-shell {
    max-width: 920px;
    margin: 0 auto;
    padding: 16px;
  }
  .notebook-font-fallback {
    font-family: "Atkinson Hyperlegible", "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  }
  .studio-header {
    border-bottom: 1px solid #d1d5db;
    margin-bottom: 14px;
    padding-bottom: 10px;
  }
  .studio-header h1 {
    margin: 0;
    font-size: 28px;
    line-height: 1.2;
  }
  .studio-meta {
    margin-top: 6px;
    color: #4b5563;
    font-size: 12px;
  }
  .studio-sheet {
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 14px;
  }
  .studio-template-lined {
    background-image: linear-gradient(to bottom, rgba(59, 130, 246, 0.16) 1px, transparent 1px);
    background-size: 100% 30px;
  }
  .studio-template-grid {
    background-image:
      linear-gradient(rgba(148, 163, 184, 0.22) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148, 163, 184, 0.22) 1px, transparent 1px);
    background-size: 24px 24px;
  }
  .studio-template-cornell {
    background: linear-gradient(to right, rgba(148, 163, 184, 0.12) 0, rgba(148, 163, 184, 0.12) 28%, transparent 28%);
  }
  .studio-cornell-layout .studio-sections {
    display: grid;
    grid-template-columns: minmax(160px, 0.9fr) minmax(0, 2fr);
    gap: 12px;
  }
  .studio-cornell-layout .studio-section-cue {
    grid-column: 1;
  }
  .studio-cornell-layout .studio-section-notes {
    grid-column: 2;
  }
  .studio-cornell-layout .studio-section-summary,
  .studio-cornell-layout .studio-section-prompt,
  .studio-cornell-layout .studio-section-generic {
    grid-column: 1 / span 2;
  }
  .studio-handwriting-accent {
    font-family: "Patrick Hand", "Comic Sans MS", "Bradley Hand", "Segoe Print", cursive;
    letter-spacing: 0.01em;
  }
  .studio-section {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.88);
    padding: 10px;
    margin: 0 0 10px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .studio-section-title {
    margin: 0 0 8px 0;
    font-size: 14px;
    line-height: 1.3;
  }
  .studio-cue-list {
    margin: 0;
    padding-left: 18px;
  }
  .studio-cue-item {
    margin: 0 0 6px 0;
    font-size: 13px;
  }
  .studio-section-content {
    margin: 0;
    font-size: 14px;
    line-height: 1.55;
    color: #111827;
    white-space: normal;
  }
  .studio-diagram-card {
    margin-top: 12px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #ffffff;
    padding: 10px;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .studio-diagram-svg svg {
    width: 100%;
    height: auto;
  }
  .studio-page-break {
    break-after: page;
    page-break-after: always;
    height: 0;
  }
`

export const buildStudioPrintableHtml = (
  note: SingleNoteExportData,
  studioDocument: NoteStudioDocument,
  options: {
    paperSize?: NotesStudioPaperSize
    generatedAtIso?: string
    labels: StudioPrintLabels
  }
): string => {
  const payload = asStudioPayload(studioDocument.payload_json)
  const templateType = normalizeStudioTemplateType(payload.layout?.template_type ?? studioDocument.template_type)
  const handwritingMode = normalizeStudioHandwritingMode(
    payload.layout?.handwriting_mode ?? studioDocument.handwriting_mode
  )
  const sections = normalizeStudioSections(payload.sections)
  const labels = options.labels
  const title = String(note.title || "").trim() || labels.untitledNote
  const paperSize = normalizePaperSize(options?.paperSize)
  const generatedAtIso = String(options?.generatedAtIso || new Date().toISOString())
  const sectionMarkup = sections
    .map((section) => renderStudioSectionHtml(section, { handwritingMode }))
    .join("")
  const diagramMarkup = renderStudioDiagramHtml(studioDocument, labels)

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)} - ${escapeHtml(labels.printTitleSuffix)}</title>
  <style>${STUDIO_PRINT_STYLES}</style>
</head>
<body data-paper-size="${escapeHtml(paperSize)}">
  <div class="notes-studio-print-shell notebook-font-fallback" data-paper-size="${escapeHtml(paperSize)}">
    <header class="studio-header">
      <h1 class="${handwritingMode === "accented" ? "studio-handwriting-accent" : ""}">${escapeHtml(title)}</h1>
      <p class="studio-meta"><strong>${escapeHtml(labels.exportedLabel)}:</strong> ${escapeHtml(generatedAtIso)}</p>
      <p class="studio-meta"><strong>${escapeHtml(labels.templateLabel)}:</strong> ${escapeHtml(templateType)} | <strong>${escapeHtml(labels.paperLabel)}:</strong> ${escapeHtml(paperSize)}</p>
    </header>
    <main class="studio-sheet studio-template-${escapeHtml(templateType)} ${
      templateType === "cornell" ? "studio-cornell-layout" : ""
    }">
      <section class="studio-sections">
        ${sectionMarkup}
      </section>
      ${diagramMarkup}
    </main>
    <div class="studio-page-break"></div>
  </div>
</body>
</html>`
}
