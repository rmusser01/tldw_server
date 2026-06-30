import { describe, expect, it } from "vitest"
import {
  buildSingleNoteCopyText,
  buildSingleNoteJson,
  buildSingleNoteMarkdown,
  buildSingleNotePrintableHtml,
  buildStudioPrintableHtml,
  getDefaultStudioPaperSizeFromLocale,
  SINGLE_NOTE_PRINT_STYLES
} from "../export-utils"

describe("notes export utils", () => {
  it("builds markdown with keyword frontmatter when keywords are present", () => {
    const markdown = buildSingleNoteMarkdown({
      id: 7,
      title: "Research Note",
      content: "Body text",
      keywords: ["research", "ml"]
    })

    expect(markdown).toContain("---")
    expect(markdown).toContain('keywords:\n  - "research"\n  - "ml"')
    expect(markdown).toContain("# Research Note")
    expect(markdown).toContain("Body text")
  })

  it("escapes backslashes and control characters in YAML frontmatter values", () => {
    const markdown = buildSingleNoteMarkdown({
      id: 8,
      title: 'Path "C:\\notes\\today"\nline',
      content: "Body",
      keywords: ['team\\alpha', 'line\nbreak']
    })

    expect(markdown).toContain('title: "Path \\"C:\\\\notes\\\\today\\"\\nline"')
    expect(markdown).toContain('  - "team\\\\alpha"')
    expect(markdown).toContain('  - "line\\nbreak"')
  })

  it("builds normalized single-note JSON payload", () => {
    const payload = buildSingleNoteJson({
      id: "n-1",
      title: "Json note",
      content: "Some content",
      keywords: ["alpha", " beta "]
    })
    expect(payload).toContain('"id": "n-1"')
    expect(payload).toContain('"title": "Json note"')
    expect(payload).toContain('"keywords": [\n    "alpha",\n    "beta"\n  ]')
  })

  it("supports content-only and markdown copy modes", () => {
    const note = {
      id: 9,
      title: "Copy note",
      content: "Copy content",
      keywords: ["tag"]
    }
    expect(buildSingleNoteCopyText(note, "content")).toBe("Copy content")
    expect(buildSingleNoteCopyText(note, "markdown")).toContain("# Copy note")
  })

  it("builds printable html with sanitized markdown content and tag metadata", () => {
    const html = buildSingleNotePrintableHtml(
      {
        id: 11,
        title: "Printable <Note>",
        content: "Paragraph with **bold** and <script>alert('x')</script>",
        keywords: ["research", "summary"]
      },
      {
        generatedAtIso: "2026-02-18T00:00:00.000Z"
      }
    )

    expect(html).toContain("Printable &lt;Note&gt;")
    expect(html).toContain("<strong>bold</strong>")
    expect(html).toContain("Tags:</strong> research, summary")
    expect(html).toContain("2026-02-18T00:00:00.000Z")
    expect(html).not.toContain("<script>")
  })

  it("ships a dedicated print stylesheet for note exports", () => {
    expect(SINGLE_NOTE_PRINT_STYLES).toMatchInlineSnapshot(`
      "
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
        "
    `)
  })

  it("builds Studio printable HTML with explicit paper size, template chrome, Cornell layout, and SVG diagrams", () => {
    const html = buildStudioPrintableHtml(
      {
        id: "studio-1",
        title: "Studio printable note",
        content: "Markdown companion body",
        keywords: ["study"]
      },
      {
        note_id: "studio-1",
        template_type: "cornell",
        handwriting_mode: "accented",
        source_note_id: "source-1",
        excerpt_hash: "sha256:excerpt",
        companion_content_hash: "sha256:companion",
        render_version: 1,
        created_at: "2026-03-28T10:00:00Z",
        last_modified: "2026-03-28T10:00:00Z",
        payload_json: {
          layout: {
            template_type: "cornell",
            handwriting_mode: "accented",
            render_version: 1
          },
          sections: [
            {
              id: "cue-1",
              kind: "cue",
              title: "Cue",
              items: ["Prompt"]
            },
            {
              id: "notes-1",
              kind: "notes",
              title: "Notes",
              content: "Main notes body"
            },
            {
              id: "prompt-1",
              kind: "prompt",
              title: "Try it yourself",
              content: "Sketch the notebook from memory."
            },
            {
              id: "summary-1",
              kind: "summary",
              title: "Summary",
              content: "Summary content"
            }
          ]
        },
        diagram_manifest_json: {
          diagram_type: "flowchart",
          source_section_ids: ["notes-1"],
          source_graph: "graph TD;A-->B;",
          cached_svg:
            '<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">Diagram</text></svg>',
          render_hash: "hash-1",
          generation_status: "ready"
        }
      },
      {
        paperSize: "A4",
        generatedAtIso: "2026-03-29T00:00:00.000Z",
        labels: {
          untitledNote: "Untitled note",
          printTitleSuffix: "Notes Studio Print",
          exportedLabel: "Exported",
          templateLabel: "Template",
          paperLabel: "Paper",
          diagramHeading: "Diagram"
        }
      }
    )

    expect(html).toContain('data-paper-size="A4"')
    expect(html).toContain("studio-template-cornell")
    expect(html).toContain("studio-cornell-layout")
    expect(html).toContain(".studio-cornell-layout .studio-sections")
    expect(html).toContain("grid-template-columns")
    expect(html).toContain("<svg")
    expect(html).toContain("notebook-font-fallback")
    expect(html).toContain("studio-section-content studio-handwriting-accent")
    expect(html).toContain("2026-03-29T00:00:00.000Z")
  })

  it("does not duplicate cue items as body paragraphs in Studio printable HTML", () => {
    const html = buildStudioPrintableHtml(
      {
        id: "studio-1",
        title: "Studio printable note",
        content: "Markdown companion body",
        keywords: ["study"]
      },
      {
        note_id: "studio-1",
        template_type: "cornell",
        handwriting_mode: "off",
        source_note_id: "source-1",
        excerpt_hash: "sha256:excerpt",
        companion_content_hash: "sha256:companion",
        render_version: 1,
        created_at: "2026-03-28T10:00:00Z",
        last_modified: "2026-03-28T10:00:00Z",
        payload_json: {
          layout: {
            template_type: "cornell",
            handwriting_mode: "off",
            render_version: 1
          },
          sections: [
            {
              id: "cue-1",
              kind: "cue",
              title: "Cue",
              items: ["Prompt"]
            }
          ]
        },
        diagram_manifest_json: null
      },
      {
        paperSize: "A4",
        generatedAtIso: "2026-03-29T00:00:00.000Z",
        labels: {
          untitledNote: "Untitled note",
          printTitleSuffix: "Notes Studio Print",
          exportedLabel: "Exported",
          templateLabel: "Template",
          paperLabel: "Paper",
          diagramHeading: "Diagram"
        }
      }
    )

    const cueSectionHtml = html.split('<article class="studio-section studio-section-cue">')[1] || ""

    expect(cueSectionHtml).toContain("studio-cue-item")
    expect(cueSectionHtml).not.toContain("studio-section-content")
  })

  it("returns locale-driven default Studio paper sizes", () => {
    expect(getDefaultStudioPaperSizeFromLocale("en-US")).toBe("US Letter")
    expect(getDefaultStudioPaperSizeFromLocale("es-US")).toBe("US Letter")
    expect(getDefaultStudioPaperSizeFromLocale("en-CA")).toBe("A4")
    expect(getDefaultStudioPaperSizeFromLocale("de-DE")).toBe("A4")
    expect(getDefaultStudioPaperSizeFromLocale("")).toBe("A4")
  })
})
