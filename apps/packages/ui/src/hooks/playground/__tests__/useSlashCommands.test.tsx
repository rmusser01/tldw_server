// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { executeSlashCommand, useSlashCommands } from "../useSlashCommands"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?: string,
      values?: Record<string, unknown>
    ) => {
      const base = fallback || key
      if (!values) return base
      return Object.entries(values).reduce((acc, [name, value]) => {
        return acc.replaceAll(`{{${name}}}`, String(value))
      }, base)
    }
  })
}))

describe("playground useSlashCommands", () => {
  it("shows slash suggestions with concise descriptions and examples", () => {
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode: vi.fn(),
        webSearch: false,
        setWebSearch: vi.fn(),
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "",
        imageBackendLabel: "none",
        setOpenModelSettings: vi.fn(),
        currentMessage: "/gen"
      })
    )

    expect(result.current.showSlashMenu).toBe(true)
    const generateImage = result.current.filteredSlashCommands.find(
      (command) => command.command === "generate-image"
    )
    expect(generateImage).toBeDefined()
    expect(generateImage?.description).toContain("/generate-image:<provider>")
  })

  it("resolves a slash command without running it until it is executed (CC-N1)", () => {
    const setChatMode = vi.fn()
    const setWebSearch = vi.fn()
    const setOpenModelSettings = vi.fn()
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode,
        webSearch: false,
        setWebSearch,
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "flux",
        imageBackendLabel: "Flux",
        setOpenModelSettings,
        currentMessage: "/web"
      })
    )

    const intent = result.current.resolveSubmissionIntent("/web latest news")
    result.current.resolveSubmissionIntent("/model")

    expect(intent).toMatchObject({
      handled: true,
      isImageCommand: false,
      message: "latest news"
    })
    expect(setWebSearch).not.toHaveBeenCalled()
    expect(setOpenModelSettings).not.toHaveBeenCalled()

    act(() => {
      executeSlashCommand(intent)
    })
    expect(setWebSearch).toHaveBeenCalledTimes(1)
    expect(setWebSearch).toHaveBeenCalledWith(true)
  })

  it("leaves plain messages and unknown commands unhandled", () => {
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode: vi.fn(),
        webSearch: false,
        setWebSearch: vi.fn(),
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "",
        imageBackendLabel: "none",
        setOpenModelSettings: vi.fn(),
        currentMessage: ""
      })
    )

    const plain = result.current.resolveSubmissionIntent("hello")
    const unknown = result.current.resolveSubmissionIntent("/unknown thing")
    expect(plain).toMatchObject({ handled: false, message: "hello" })
    expect(plain.slashCommand).toBeUndefined()
    expect(unknown).toMatchObject({ handled: false, message: "/unknown thing" })
    expect(() => executeSlashCommand(unknown)).not.toThrow()
  })

  it("parses image commands", () => {
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode: vi.fn(),
        webSearch: false,
        setWebSearch: vi.fn(),
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "flux",
        imageBackendLabel: "Flux",
        setOpenModelSettings: vi.fn(),
        currentMessage: ""
      })
    )

    const imageIntent = result.current.resolveSubmissionIntent(
      "/generate-image:flux city skyline"
    )
    expect(imageIntent).toMatchObject({
      handled: true,
      isImageCommand: true,
      imageBackendOverride: "flux",
      message: "city skyline"
    })
  })
})
