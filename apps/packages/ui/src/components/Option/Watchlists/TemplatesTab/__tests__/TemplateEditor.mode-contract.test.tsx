// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { message } from "antd"
import type { MessageType } from "antd/es/message/interface"
import { TemplateEditor } from "../TemplateEditor"
import type { WatchlistTemplate } from "@/types/watchlists"

const serviceMocks = vi.hoisted(() => ({
  createWatchlistTemplate: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  getWatchlistTemplate: vi.fn(),
  getWatchlistTemplateVersions: vi.fn(),
  validateWatchlistTemplate: vi.fn()
}))

const telemetryMock = vi.hoisted(() => ({
  trackWatchlistsPreventionTelemetry: vi.fn()
}))

const translationMock = vi.hoisted(() => ({
  t: (
    key: string,
    fallbackOrOptions?: string | { defaultValue?: string } | Record<string, unknown>,
    maybeOptions?: Record<string, unknown>
  ) => {
    if (typeof fallbackOrOptions === "string") {
      return interpolate(fallbackOrOptions, maybeOptions)
    }
    if (
      fallbackOrOptions &&
      typeof fallbackOrOptions === "object" &&
      typeof (fallbackOrOptions as { defaultValue?: string }).defaultValue === "string"
    ) {
      return interpolate(
        (fallbackOrOptions as { defaultValue: string }).defaultValue,
        maybeOptions
      )
    }
    return key
  }
}))

const interpolate = (template: string, values?: Record<string, unknown>) => {
  if (!values) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = values[token]
    return value == null ? "" : String(value)
  })
}

const createMessageType = (): MessageType => {
  const close = (() => undefined) as MessageType
  close.then = ((onfulfilled, onrejected) =>
    Promise.resolve(true).then(onfulfilled, onrejected)) as MessageType["then"]
  return close
}

const setViewport = (width: number) => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: width < 768,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

type MockTemplateCodeEditorHandle = {
  insertSnippet: (snippet: string) => void
  getValue: () => string
}

type MockTemplateCodeEditorProps = {
  value?: string
  onChange?: (value: string) => void
  validationErrors?: Array<{ message: string }>
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translationMock.t
  })
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistTemplate: (...args: unknown[]) => serviceMocks.createWatchlistTemplate(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  getWatchlistTemplate: (...args: unknown[]) => serviceMocks.getWatchlistTemplate(...args),
  getWatchlistTemplateVersions: (...args: unknown[]) =>
    serviceMocks.getWatchlistTemplateVersions(...args),
  validateWatchlistTemplate: (...args: unknown[]) => serviceMocks.validateWatchlistTemplate(...args)
}))

vi.mock("@/utils/watchlists-prevention-telemetry", () => ({
  trackWatchlistsPreventionTelemetry: (...args: unknown[]) =>
    telemetryMock.trackWatchlistsPreventionTelemetry(...args)
}))

vi.mock("../TemplatePreviewPane", () => ({
  TemplatePreviewPane: () => <div data-testid="template-preview-pane">Preview</div>
}))

vi.mock("../TemplateVariablesPanel", () => ({
  TemplateVariablesPanel: () => <div data-testid="template-variables-panel">Variables</div>
}))

vi.mock("../TemplateSnippetPalette", () => ({
  TemplateSnippetPalette: () => <div data-testid="template-snippet-palette">Snippets</div>
}))

vi.mock("../TemplateCodeEditor", async () => {
  const ReactModule = await import("react")
  const TemplateCodeEditor = ReactModule.forwardRef<
    MockTemplateCodeEditorHandle,
    MockTemplateCodeEditorProps
  >(({ value, onChange, validationErrors }, ref) => {
      const [internalValue, setInternalValue] = ReactModule.useState(String(value || ""))

      ReactModule.useEffect(() => {
        setInternalValue(String(value || ""))
      }, [value])

      ReactModule.useImperativeHandle(
        ref,
        () => ({
          insertSnippet: (snippet: string) => {
            const needsSpacer = internalValue.length > 0 && !internalValue.endsWith("\n")
            const nextValue = `${internalValue}${needsSpacer ? "\n\n" : ""}${snippet}`
            setInternalValue(nextValue)
            onChange?.(nextValue)
          },
          getValue: () => internalValue
        }),
        [internalValue, onChange]
      )

      return (
        <div>
          <textarea
            data-testid="template-code-editor"
            value={internalValue}
            onChange={(event) => {
              setInternalValue(event.target.value)
              onChange?.(event.target.value)
            }}
          />
          <div data-testid="template-code-editor-errors">
            {Array.isArray(validationErrors) ? validationErrors.length : 0}
          </div>
        </div>
      )
    })

  return {
    TemplateCodeEditor
  }
})

const buildTemplate = (overrides: Partial<WatchlistTemplate> = {}): WatchlistTemplate => ({
  name: "daily-brief",
  description: "Daily watchlist briefing",
  content: "# {{ title }}",
  format: "md",
  updated_at: "2026-02-23T00:00:00Z",
  version: 3,
  ...overrides
})

const openEditorTab = () => {
  fireEvent.click(screen.getByRole("tab", { name: "Editor" }))
}

describe("TemplateEditor authoring mode contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setViewport(1024)
    serviceMocks.createWatchlistTemplate.mockResolvedValue(undefined)
    serviceMocks.fetchWatchlistRuns.mockResolvedValue({
      items: [
        {
          id: 321,
          started_at: "2026-02-23T08:00:00Z"
        }
      ]
    })
    serviceMocks.getWatchlistTemplate.mockResolvedValue({
      name: "daily-brief",
      description: "Daily watchlist briefing",
      content: "# {{ title }}",
      format: "md",
      version: 3
    })
    serviceMocks.getWatchlistTemplateVersions.mockResolvedValue({
      items: [
        {
          version: 3,
          format: "md",
          description: "Current",
          updated_at: "2026-02-23T00:00:00Z",
          is_current: true
        },
        {
          version: 2,
          format: "md",
          description: "Previous",
          updated_at: "2026-02-22T00:00:00Z",
          is_current: false
        }
      ]
    })
    serviceMocks.validateWatchlistTemplate.mockResolvedValue({
      valid: true,
      errors: []
    })
    telemetryMock.trackWatchlistsPreventionTelemetry.mockResolvedValue(undefined)

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  it("uses full-width constrained modal chrome with reachable template actions", async () => {
    setViewport(420)

    render(<TemplateEditor open template={null} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    const modal = document.querySelector(".ant-modal") as HTMLElement | null
    expect(modal).not.toBeNull()
    expect(modal?.style.width).toBe("100vw")
    expect(modal?.style.maxWidth).toBe("100vw")
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument()
  })

  it("defaults to basic mode for create and hides advanced-only tabs/tools", async () => {
    render(<TemplateEditor open template={null} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    expect(
      screen.getByText("Basic mode is no-code: pick a recipe, edit text, and preview your output.")
    ).toBeInTheDocument()
    openEditorTab()
    expect(screen.getByTestId("template-recipe-builder")).toBeInTheDocument()
    expect(screen.queryByRole("tab", { name: "Variables & Snippets" })).not.toBeInTheDocument()
    expect(screen.queryByText("Version tools")).not.toBeInTheDocument()
  })

  it("defaults to advanced mode for edit and exposes docs plus version tools", async () => {
    render(<TemplateEditor open template={buildTemplate()} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(serviceMocks.getWatchlistTemplate).toHaveBeenCalledWith("daily-brief")
      expect(serviceMocks.getWatchlistTemplateVersions).toHaveBeenCalledWith("daily-brief")
    })

    expect(
      screen.getByText("Advanced mode adds Jinja2 snippets, variable docs, and version tools.")
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Variables & Snippets" })).toBeInTheDocument()
    expect(screen.getByText("Version tools")).toBeInTheDocument()
  })

  it("warns and reroutes to editor when switching from advanced docs to basic", async () => {
    const infoSpy = vi.spyOn(message, "info").mockImplementation(() => createMessageType())

    render(<TemplateEditor open template={buildTemplate()} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Version tools")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("tab", { name: "Variables & Snippets" }))
    expect(screen.getByTestId("template-variables-panel")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("template-editor-mode-basic"))

    await waitFor(() => {
      expect(infoSpy).toHaveBeenCalledWith(
        "Advanced template tools are hidden in Basic mode. Your content and version context are preserved."
      )
    })

    expect(screen.queryByRole("tab", { name: "Variables & Snippets" })).not.toBeInTheDocument()
    expect(screen.getByText("Version tools are available in Advanced mode.")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Editor" })).toHaveAttribute("aria-selected", "true")

    infoSpy.mockRestore()
  })

  it("preserves content and version context when toggling between modes", async () => {
    render(<TemplateEditor open template={buildTemplate()} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Currently loaded: v3. Saving restores this content as a new latest version.")).toBeInTheDocument()
    })

    openEditorTab()
    fireEvent.change(screen.getByTestId("template-code-editor"), {
      target: { value: "# Edited digest" }
    })

    await waitFor(() => {
      expect(
        screen.getByText("Current editor content differs from the loaded version.")
      ).toBeInTheDocument()
    })
    expect(
      screen
        .getByText("Current editor content differs from the loaded version.")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("template-editor-mode-basic"))
    fireEvent.click(screen.getByTestId("template-editor-mode-advanced"))

    await waitFor(() => {
      expect(screen.getByTestId("template-code-editor")).toHaveValue("# Edited digest")
    })

    expect(
      screen.getByText("Currently loaded: v3. Saving restores this content as a new latest version.")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Current editor content differs from the loaded version.")
    ).toBeInTheDocument()
    expect(
      screen
        .getByText("Current editor content differs from the loaded version.")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })

  it("renders visual repair warnings through the design-system Alert primitive", async () => {
    render(<TemplateEditor open template={null} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    openEditorTab()
    fireEvent.change(
      screen.getByPlaceholderText(
        "Start with plain text or Markdown. Advanced users can add Jinja2 tags later."
      ),
      {
        target: { value: "{% macro card(x) %}{{ x }}{% endmacro %}\n{{ card(title) }}" }
      }
    )
    fireEvent.click(screen.getByRole("tab", { name: "Visual" }))

    const repairTitle = await screen.findByText("Visual layout may be out of sync")

    expect(repairTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    expect(
      screen.getByText(
        "The template was edited in code mode and the visual block layout may need repair."
      )
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Repair layout" })).toBeInTheDocument()
    expect(screen.getByTestId("template-editor-repair-visual-layout")).toBeInTheDocument()
  })

  it("applies recipe defaults and autofills name/description in basic create mode", async () => {
    render(<TemplateEditor open template={null} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    openEditorTab()
    fireEvent.click(screen.getByTestId("template-recipe-apply"))

    const nameInput = screen.getByLabelText("Template Name")
    const descriptionInput = screen.getByLabelText("Description")
    const contentInput = screen.getByPlaceholderText(
      "Start with plain text or Markdown. Advanced users can add Jinja2 tags later."
    )
    const markdownRadio = screen.getByRole("radio", { name: "Markdown" })

    expect(nameInput).toHaveValue("briefing_md")
    expect(descriptionInput).toHaveValue("Daily markdown briefing template")
    expect(String((contentInput as HTMLTextAreaElement).value)).toContain("## Executive Summary")
    expect(markdownRadio).toBeChecked()
  }, 15_000)

  it("does not overwrite custom name and description when reapplying a recipe", async () => {
    render(<TemplateEditor open template={null} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    openEditorTab()
    fireEvent.change(screen.getByLabelText("Template Name"), {
      target: { value: "custom-template" }
    })
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Custom description" }
    })

    fireEvent.click(screen.getByTestId("template-recipe-apply"))

    expect(screen.getByLabelText("Template Name")).toHaveValue("custom-template")
    expect(screen.getByLabelText("Description")).toHaveValue("Custom description")
  })

  it("blocks save in basic mode when server-side validation fails", async () => {
    const errorSpy = vi.spyOn(message, "error").mockImplementation(() => createMessageType())
    const onClose = vi.fn()
    serviceMocks.validateWatchlistTemplate.mockResolvedValueOnce({
      valid: false,
      errors: [{ message: "Template syntax error" }]
    })

    render(<TemplateEditor open template={null} onClose={onClose} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    openEditorTab()
    fireEvent.click(screen.getByTestId("template-recipe-apply"))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(serviceMocks.validateWatchlistTemplate).toHaveBeenCalledWith(
        expect.any(String),
        "md"
      )
    })

    expect(serviceMocks.createWatchlistTemplate).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    expect(errorSpy).toHaveBeenCalledWith(
      "Could not save template. Fix syntax errors, then try again."
    )

    errorSpy.mockRestore()
  }, 15_000)

  it("saves basic-mode templates with recipe-generated payloads", async () => {
    const onClose = vi.fn()

    render(<TemplateEditor open template={null} onClose={onClose} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    openEditorTab()
    fireEvent.click(screen.getByTestId("template-recipe-apply"))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(serviceMocks.createWatchlistTemplate).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "briefing_md",
          description: "Daily markdown briefing template",
          format: "md",
          overwrite: false
        })
      )
    })

    expect(onClose).toHaveBeenCalledWith(true)
  })

  it("loads selected and latest template versions in advanced edit mode", async () => {
    const successSpy = vi.spyOn(message, "success").mockImplementation(() => createMessageType())
    serviceMocks.getWatchlistTemplate
      .mockResolvedValueOnce({
        name: "daily-brief",
        description: "Daily watchlist briefing",
        content: "# Current version",
        format: "md",
        version: 3
      })
      .mockResolvedValueOnce({
        name: "daily-brief",
        description: "Daily watchlist briefing",
        content: "# Historical version",
        format: "md",
        version: 2
      })
      .mockResolvedValueOnce({
        name: "daily-brief",
        description: "Daily watchlist briefing",
        content: "# Latest version",
        format: "md",
        version: 4
      })

    render(<TemplateEditor open template={buildTemplate()} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Currently loaded: v3. Saving restores this content as a new latest version.")).toBeInTheDocument()
    })

    openEditorTab()
    fireEvent.mouseDown(screen.getByText("Select a historical version"))
    fireEvent.click(await screen.findByText("v2"))
    fireEvent.click(screen.getByRole("button", { name: "Load version" }))

    await waitFor(() => {
      expect(serviceMocks.getWatchlistTemplate).toHaveBeenCalledWith("daily-brief", { version: 2 })
      expect(screen.getByTestId("template-code-editor")).toHaveValue("# Historical version")
      expect(
        screen.getByText("Currently loaded: v2. Saving restores this content as a new latest version.")
      ).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Load latest/i })).toBeEnabled()
    })
    fireEvent.click(screen.getByRole("button", { name: /Load latest/i }))

    await waitFor(() => {
      expect(serviceMocks.getWatchlistTemplate).toHaveBeenCalledWith("daily-brief", undefined)
      expect(screen.getByTestId("template-code-editor")).toHaveValue("# Latest version")
      expect(
        screen.getByText("Currently loaded: v4. Saving restores this content as a new latest version.")
      ).toBeInTheDocument()
    })

    expect(successSpy).toHaveBeenCalledWith("Loaded template version 2")
    expect(successSpy).toHaveBeenCalledWith("Loaded latest template version")
    successSpy.mockRestore()
  })

  it("blocks advanced-mode save on validation errors and surfaces editor markers", async () => {
    const errorSpy = vi.spyOn(message, "error").mockImplementation(() => createMessageType())
    const onClose = vi.fn()
    serviceMocks.validateWatchlistTemplate.mockResolvedValueOnce({
      valid: false,
      errors: [{ line: 1, column: 1, message: "Unexpected endfor" }]
    })

    render(<TemplateEditor open template={buildTemplate()} onClose={onClose} />)

    await waitFor(() => {
      expect(screen.getByText("Version tools")).toBeInTheDocument()
    })

    openEditorTab()
    fireEvent.change(screen.getByTestId("template-code-editor"), {
      target: { value: "{% for item in items %}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(serviceMocks.validateWatchlistTemplate).toHaveBeenCalledWith(
        "{% for item in items %}",
        "md"
      )
    })

    expect(serviceMocks.createWatchlistTemplate).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    expect(screen.getByTestId("template-code-editor-errors")).toHaveTextContent("1")
    expect(errorSpy).toHaveBeenCalledWith(
      "Could not save template. Fix syntax errors, then try again."
    )

    errorSpy.mockRestore()
  })

  it("emits authoring telemetry for start, mode change, recipe apply, and save", async () => {
    const onClose = vi.fn()

    render(<TemplateEditor open template={null} onClose={onClose} />)

    await waitFor(() => {
      expect(serviceMocks.fetchWatchlistRuns).toHaveBeenCalled()
    })

    openEditorTab()
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_authoring_started",
        surface: "template_editor",
        mode: "basic",
        context: "create"
      })
    )

    fireEvent.click(screen.getByTestId("template-recipe-apply"))
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_template_recipe_applied",
        surface: "template_editor",
        recipe: "briefing_md",
        mode: "basic"
      })
    )

    fireEvent.click(screen.getByTestId("template-editor-mode-advanced"))
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_authoring_mode_changed",
        surface: "template_editor",
        from_mode: "basic",
        to_mode: "advanced",
        context: "create"
      })
    )

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(serviceMocks.createWatchlistTemplate).toHaveBeenCalled()
      expect(onClose).toHaveBeenCalledWith(true)
    })

    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_authoring_saved",
        surface: "template_editor",
        mode: "advanced",
        context: "create"
      })
    )
  })
})
