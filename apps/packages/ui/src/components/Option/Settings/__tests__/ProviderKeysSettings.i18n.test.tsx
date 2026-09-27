import { act, render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"

import commonEn from "@/assets/locale/en/common.json"
import settingsEn from "@/assets/locale/en/settings.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { ProviderKeysSettings } from "../ProviderKeysSettings"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { listUserProviderKeys: vi.fn() }
}))

const renderSettings = async () => {
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    ns: ["settings", "common"],
    defaultNS: "settings",
    resources: { en: { common: commonEn, settings: settingsEn } },
    interpolation: { escapeValue: false }
  })
  const view = render(
    <I18nextProvider i18n={i18n}>
      <ProviderKeysSettings />
    </I18nextProvider>
  )
  return { ...view, i18n }
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })
  return { promise, resolve, reject }
}

const byokDisabledError = () =>
  Object.assign(new Error("BYOK is disabled in this deployment"), {
    status: 403,
    details: { detail: "BYOK is disabled in this deployment" }
  })

describe("ProviderKeysSettings with English resources and ICU", () => {
  beforeEach(() => {
    vi.mocked(tldwClient.listUserProviderKeys).mockReset()
  })

  it("renders the scalar loading title until the key list settles empty", async () => {
    let resolve!: (value: { items: [] }) => void
    vi.mocked(tldwClient.listUserProviderKeys).mockReturnValue(
      new Promise((done) => {
        resolve = done
      })
    )

    await renderSettings()
    expect(screen.getByText("Loading…")).toBeInTheDocument()

    await act(async () => {
      resolve({ items: [] })
    })
    expect(screen.getByText(/No provider keys configured/)).toBeInTheDocument()
    expect(screen.queryByText("Loading…")).not.toBeInTheDocument()
  })

  it("renders configured key sources and hints after loading", async () => {
    vi.mocked(tldwClient.listUserProviderKeys).mockResolvedValue({
      items: [
        {
          provider: "openai",
          has_key: true,
          source: "user",
          key_hint: "1234",
          auth_source: "api_key",
          last_used_at: null
        }
      ]
    })

    await renderSettings()
    expect(await screen.findByText("openai")).toBeInTheDocument()
    expect(screen.getByText("User key")).toBeInTheDocument()
    expect(screen.getByText("...1234")).toBeInTheDocument()
    expect(screen.queryByText("Loading…")).not.toBeInTheDocument()
  })

  it("preserves deployment setup guidance for the confirmed BYOK-disabled response", async () => {
    vi.mocked(tldwClient.listUserProviderKeys).mockRejectedValue(
      Object.assign(new Error("BYOK is disabled in this deployment"), {
        status: 403,
        details: { detail: "BYOK is disabled in this deployment" }
      })
    )

    await renderSettings()
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Provider key management is not available"
    )
    expect(screen.getByText(/Set BYOK_ENCRYPTION_KEY/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add Provider Key" })
    ).not.toBeInTheDocument()
  })

  it.each([
    { status: 403 },
    Object.assign(new Error("Access denied"), {
      status: 403,
      details: { detail: "Access denied" }
    })
  ])(
    "does not misdiagnose a generic denial as disabled BYOK (%j)",
    async (error) => {
      vi.mocked(tldwClient.listUserProviderKeys).mockRejectedValue(error)

      await renderSettings()
      expect(await screen.findByRole("alert")).toHaveTextContent(
        /access.*denied/i
      )
      expect(screen.getByRole("alert")).toHaveTextContent(/administrator/i)
      expect(screen.queryByText(/BYOK_ENCRYPTION_KEY/)).not.toBeInTheDocument()
    }
  )

  it("shows load failure feedback for an unavailable service", async () => {
    vi.mocked(tldwClient.listUserProviderKeys).mockRejectedValue(
      new Error("network unavailable")
    )

    await renderSettings()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Failed to load provider keys"
    )
    expect(screen.queryByText(/BYOK_ENCRYPTION_KEY/)).not.toBeInTheDocument()
  })

  it.each([
    { error: { status: 403 }, expected: /Access to provider keys was denied/ },
    {
      error: new Error("network unavailable"),
      expected: /Failed to load provider keys/
    }
  ])(
    "replaces previous BYOK-disabled guidance with the latest load error ($expected)",
    async ({ error, expected }) => {
      const next = deferred<{ items: [] }>()
      vi.mocked(tldwClient.listUserProviderKeys)
        .mockRejectedValueOnce(byokDisabledError())
        .mockReturnValueOnce(next.promise)

      const { i18n } = await renderSettings()
      expect(
        await screen.findByText(/Set BYOK_ENCRYPTION_KEY/)
      ).toBeInTheDocument()

      await act(async () => {
        await i18n.changeLanguage("en-US")
      })
      expect(screen.queryByText(/BYOK_ENCRYPTION_KEY/)).not.toBeInTheDocument()
      expect(screen.getByText("Loading…")).toBeInTheDocument()

      await act(async () => {
        next.reject(error)
      })
      expect(screen.getByRole("alert")).toHaveTextContent(expected)
      expect(screen.queryByText(/BYOK_ENCRYPTION_KEY/)).not.toBeInTheDocument()
    }
  )

  it.each([
    { name: "BYOK-disabled", error: byokDisabledError() },
    { name: "permission", error: { status: 403 } },
    { name: "network", error: new Error("network unavailable") }
  ])(
    "ignores an older $name failure after a newer language-triggered load succeeds",
    async ({ error }) => {
      const older = deferred<{ items: [] }>()
      vi.mocked(tldwClient.listUserProviderKeys)
        .mockReturnValueOnce(older.promise)
        .mockResolvedValueOnce({ items: [] })

      const { i18n } = await renderSettings()
      await act(async () => {
        await i18n.changeLanguage("en-US")
      })
      expect(
        screen.getByText(/No provider keys configured/)
      ).toBeInTheDocument()

      await act(async () => {
        older.reject(error)
      })
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      expect(screen.queryByText(/BYOK_ENCRYPTION_KEY/)).not.toBeInTheDocument()
      expect(
        screen.getByText(/No provider keys configured/)
      ).toBeInTheDocument()
    }
  )

  it("keeps the newer request pending when an older request finishes", async () => {
    const older = deferred<{ items: [] }>()
    const newer = deferred<{ items: [] }>()
    vi.mocked(tldwClient.listUserProviderKeys)
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)

    const { i18n } = await renderSettings()
    await act(async () => {
      await i18n.changeLanguage("en-US")
    })
    await act(async () => {
      older.resolve({ items: [] })
    })
    expect(screen.getByText("Loading…")).toBeInTheDocument()

    await act(async () => {
      newer.resolve({ items: [] })
    })
    expect(screen.queryByText("Loading…")).not.toBeInTheDocument()
    expect(screen.getByText(/No provider keys configured/)).toBeInTheDocument()
  })

  it("ignores an older success after a newer request is denied", async () => {
    const older = deferred<{ items: [] }>()
    vi.mocked(tldwClient.listUserProviderKeys)
      .mockReturnValueOnce(older.promise)
      .mockRejectedValueOnce(byokDisabledError())

    const { i18n } = await renderSettings()
    await act(async () => {
      await i18n.changeLanguage("en-US")
    })
    expect(screen.getByText(/Set BYOK_ENCRYPTION_KEY/)).toBeInTheDocument()

    await act(async () => {
      older.resolve({ items: [] })
    })
    expect(screen.getByText(/Set BYOK_ENCRYPTION_KEY/)).toBeInTheDocument()
    expect(screen.queryByRole("table")).not.toBeInTheDocument()
  })
})
