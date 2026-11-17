import React from "react"
import { useTranslation } from "react-i18next"
import { RouteShell } from "./app-route"
import { DeferredOptionsRoute } from "./deferred-options-route"
import { useRouteDocumentTitle } from "./hooks/useRouteDocumentTitle"
import { optionStartupRoutes } from "./option-startup-routes"

export const OptionsRouteShell = () => {
  const { t } = useTranslation()
  // The tab title follows the route, as on the WebUI (XP-19).
  useRouteDocumentTitle(
    t("common:titles.options", { defaultValue: "tldw Assistant — Options" })
  )
  return (
    <RouteShell
      kind="options"
      routes={optionStartupRoutes}
      renderUnmatchedRoute={(props) => <DeferredOptionsRoute {...props} />}
    />
  )
}

export default OptionsRouteShell
