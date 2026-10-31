import { render, screen } from "@testing-library/react"
import { I18nextProvider } from "react-i18next"
import { beforeAll, describe, expect, it } from "vitest"
import { SharedWorkspacePreview } from "../SharedResearchWorkspace/SharedWorkspacePreview"
import {
  createSharedWorkspaceTestI18n,
  preview
} from "./shared-research-workspace-test-utils"

let testI18n: Awaited<ReturnType<typeof createSharedWorkspaceTestI18n>>

describe("SharedWorkspacePreview origin links", () => {
  beforeAll(async () => {
    testI18n = await createSharedWorkspaceTestI18n()
  })

  it.each(["javascript:alert(1)", "data:text/html,unsafe"])(
    "renders an unsafe %s origin as non-clickable text",
    (originUrl) => {
      render(
        <I18nextProvider i18n={testI18n}>
          <SharedWorkspacePreview
            error={null}
            isMobile={false}
            loading={false}
            onClose={() => undefined}
            open
            preview={{
              ...preview,
              origin_host: "untrusted origin",
              origin_url: originUrl
            }}
          />
        </I18nextProvider>
      )

      expect(screen.getByText("untrusted origin").closest("a")).toBeNull()
    }
  )

  it.each([
    "https://example.test/source",
    "http://example.test/source",
    "mailto:research@example.test"
  ])("renders an allowed %s origin as a link", (originUrl) => {
    render(
      <I18nextProvider i18n={testI18n}>
        <SharedWorkspacePreview
          error={null}
          isMobile={false}
          loading={false}
          onClose={() => undefined}
          open
          preview={{
            ...preview,
            origin_host: "trusted origin",
            origin_url: originUrl
          }}
        />
      </I18nextProvider>
    )

    expect(screen.getByRole("link", { name: "trusted origin" })).toHaveAttribute(
      "href",
      originUrl
    )
  })
})
