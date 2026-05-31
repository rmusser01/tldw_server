import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <div data-testid="route-boundary" data-route-id={routeId} data-route-label={routeLabel}>
      {children}
    </div>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallback: string,
      values?: Record<string, string>
    ) => fallback.replace("{{featureName}}", values?.featureName ?? "")
  })
}))

describe("hosted audio route messaging", () => {
  afterEach(() => {
    vi.resetModules()
    vi.doUnmock("@/services/tldw/deployment-mode")
    vi.doUnmock("@/components/Option/Speech/SpeechPlaygroundPage")
    vi.doUnmock("@/components/Option/STT/SttPlaygroundPage")
  })

  it("shows self-hosting guidance instead of the TTS playground in hosted mode", async () => {
    const speechPlaygroundFactory = vi.fn(() => ({
      __esModule: true,
      default: () => <div data-testid="speech-playground">Speech</div>
    }))

    vi.doMock("@/services/tldw/deployment-mode", () => ({
      isHostedTldwDeployment: () => true
    }))
    vi.doMock("@/components/Option/Speech/SpeechPlaygroundPage", speechPlaygroundFactory)

    const { default: OptionTts } = await import("../option-tts")
    render(<OptionTts />)

    expect(screen.getByTestId("option-layout")).toBeVisible()
    expect(screen.getByRole("heading", {
      name: /audio features require a self-hosted tldw server/i
    })).toBeVisible()
    expect(screen.getByText(/TTS Playground/i)).toBeVisible()
    expect(screen.queryByTestId("speech-playground")).not.toBeInTheDocument()
    expect(speechPlaygroundFactory).not.toHaveBeenCalled()
  })

  it("shows self-hosting guidance instead of the STT playground in hosted mode", async () => {
    const sttPlaygroundFactory = vi.fn(() => ({
      __esModule: true,
      default: () => <div data-testid="stt-playground">STT</div>
    }))

    vi.doMock("@/services/tldw/deployment-mode", () => ({
      isHostedTldwDeployment: () => true
    }))
    vi.doMock("@/components/Option/STT/SttPlaygroundPage", sttPlaygroundFactory)

    const { default: OptionStt } = await import("../option-stt")
    render(<OptionStt />)

    expect(screen.getByTestId("option-layout")).toBeVisible()
    expect(screen.getByRole("heading", {
      name: /audio features require a self-hosted tldw server/i
    })).toBeVisible()
    expect(screen.getByText(/STT Playground/i)).toBeVisible()
    expect(screen.queryByTestId("stt-playground")).not.toBeInTheDocument()
    expect(sttPlaygroundFactory).not.toHaveBeenCalled()
  })

  it("marks the quickstart link as a safe new-tab link with accessible context", async () => {
    vi.doMock("@/services/tldw/deployment-mode", () => ({
      isHostedTldwDeployment: () => true
    }))

    const { default: OptionTts } = await import("../option-tts")
    render(<OptionTts />)

    const quickstartLink = screen.getByRole("link", {
      name: /open self-hosting quickstart.*opens in new tab/i
    })

    expect(quickstartLink).toHaveAttribute("target", "_blank")
    expect(quickstartLink).toHaveAttribute("rel", "noopener noreferrer")
  })
})
