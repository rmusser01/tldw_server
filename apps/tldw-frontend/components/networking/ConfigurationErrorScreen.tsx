import React from "react"
import type { NetworkingIssue } from "@web/lib/api-base"
import { SetupRequiredPanel } from "@tldw/ui/components/ui/state"

type ConfigurationErrorScreenProps = {
  issue: NetworkingIssue
}

export const ConfigurationErrorScreen = ({
  issue
}: ConfigurationErrorScreenProps) => {
  if (issue.kind === "loopback_api_not_browser_reachable") {
    const openSetup = () => {
      window.location.assign("/setup")
    }

    const openSettings = () => {
      window.location.assign("/settings")
    }

    return (
      <main
        data-testid="networking-config-error"
        className="flex min-h-screen items-center justify-center bg-bg px-4 py-10 text-text"
      >
        <SetupRequiredPanel
          title="WebUI networking configuration error"
          titleHeadingLevel={1}
          message={
            <>
              The configured API URL points to <code>{issue.apiOrigin}</code>,
              which is only reachable from the host machine. Set the WebUI API
              URL to a LAN-reachable address for the API host, or switch to
              quickstart mode so the browser uses the same-origin proxy.
            </>
          }
          diagnostics={[
            { label: "API origin", value: issue.apiOrigin, code: true },
            { label: "Page origin", value: issue.pageOrigin, code: true }
          ]}
          primaryAction={{ label: "Open setup", onClick: openSetup }}
          secondaryActions={[{ label: "Open Settings", onClick: openSettings }]}
          className="w-full max-w-2xl"
        />
      </main>
    )
  }

  return null
}
