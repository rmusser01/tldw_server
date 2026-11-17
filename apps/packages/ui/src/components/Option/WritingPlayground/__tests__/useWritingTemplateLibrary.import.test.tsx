import React from "react"
import { createInstance } from "i18next"
import optionEn from "@/assets/locale/en/option.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { act, cleanup, renderHook, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ConfigProvider, message } from "antd"
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { useWritingTemplateLibrary } from "../hooks/useWritingTemplateLibrary"
import { createWritingTemplate } from "@/services/writing-playground"

vi.mock("@/services/writing-playground", () => ({
  createWritingTemplate: vi.fn(),
  createWritingTheme: vi.fn(),
  deleteWritingTemplate: vi.fn(),
  deleteWritingTheme: vi.fn(),
  getWritingDefaults: vi.fn(),
  listWritingTemplates: vi.fn(),
  listWritingThemes: vi.fn(),
  updateWritingTemplate: vi.fn(),
  updateWritingTheme: vi.fn()
}))

const i18n = createInstance().use(ICUWithInterpolation)
beforeAll(async () => {
  await i18n.init({ lng: "en", fallbackLng: false, ns: ["option"], defaultNS: "option", resources: { en: { option: optionEn } }, interpolation: { escapeValue: false } })
  ConfigProvider.config({ theme: { token: { motion: false } } })
})
afterAll(() => { ConfigProvider.config({ theme: {} }) })

afterEach(async () => {
  await act(async () => { message.destroy() })
  await waitFor(() => expect(screen.queryAllByText(errorText)).toHaveLength(0))
  cleanup()
  vi.resetAllMocks()
})

const setup = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const hook = renderHook(() => useWritingTemplateLibrary({
    isOnline: false,
    hasWriting: false,
    hasTemplates: false,
    hasThemes: false,
    hasServerDefaultsCatalog: false,
    selectedTemplateName: null,
    selectedThemeName: null,
    handleTemplateChange: vi.fn(),
    handleThemeChange: vi.fn(),
    settingsDisabled: false,
    t: i18n.t.bind(i18n)
  }), { wrapper })
  return { ...hook, client }
}

const upload = (text: () => Promise<string>) => ({
  target: { files: [{ text }], value: "selected-template.json" }
}) as unknown as React.ChangeEvent<HTMLInputElement>

const errorText = /^Import failed: /

describe("Writing template import recovery", () => {
  it.each([
    ["malformed JSON", async () => '{"templates":', /JSON/],
    ["no templates", async () => '{}', /No templates found in file/],
    ["unreadable file", async () => { throw new Error("File read failed") }, /File read failed/],
    ["unknown file error", async () => { throw null }, /Check the JSON file and try again/]
  ] as const)("shows a visible rejection for %s without writing templates", async (_label, read, detail) => {
    const { result, unmount, client } = setup()
    const event = upload(read)
    try {
      await act(async () => {
        await expect(result.current.handleTemplateImport(event)).resolves.toBeUndefined()
      })
      const notice = await screen.findByText(errorText)
      expect(notice).toBeVisible()
      expect(notice).not.toHaveTextContent("{{detail}}")
      expect(notice.textContent).toMatch(detail)
      expect(createWritingTemplate).not.toHaveBeenCalled()
      expect(result.current.templateImporting).toBe(false)
      expect(event.target.value).toBe("")
    } finally {
      unmount()
      client.clear()
    }
  })

  it("accepts a corrected file after rejection and resets the picker", async () => {
    const { result, unmount, client } = setup()
    try {
      await act(async () => {
        await expect(result.current.handleTemplateImport(upload(async () => '{'))).resolves.toBeUndefined()
      })
      const event = upload(async () => '{"name":"C06 corrected","payload":{"sys_pre":"System"}}')
      await act(async () => {
        await result.current.handleTemplateImport(event)
      })
      expect(createWritingTemplate).toHaveBeenCalledExactlyOnceWith({
        name: "C06 corrected", payload: { sys_pre: "System" }, schema_version: 1, is_default: false
      })
      expect(result.current.templateImporting).toBe(false)
      expect(event.target.value).toBe("")
    } finally {
      unmount()
      client.clear()
    }
  })
})
