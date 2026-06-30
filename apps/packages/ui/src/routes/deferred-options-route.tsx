import React from "react"
import { Navigate, matchPath, useLocation } from "react-router-dom"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { platformConfig } from "@/config/platform"
import { isRouteEnabledForCapabilities } from "@/routes/route-capabilities"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"

import { RouteNotFoundState } from "./app-route"
import { isHostedVisibleOptionPath } from "./option-route-visibility"
import type { RouteDefinition } from "./route-registry"

type DeferredOptionsRouteProps = {
  attemptedRoute: string
  capabilities: ServerCapabilities
  capabilitiesLoading: boolean
  label: string
  description: string
}

const routeMatchesLocation = (route: RouteDefinition, pathname: string) =>
  Boolean(matchPath({ path: route.path, end: true }, pathname))

const loadOptionRoutesForPath = async (
  pathname: string
): Promise<RouteDefinition[]> => {
  if (pathname === "/settings" || pathname.startsWith("/settings/")) {
    const module = await import("./option-settings-route-registry")
    return module.optionSettingsRoutes
  }

  if (pathname === "/chat") {
    const module = await import("./option-chat-route-registry")
    return module.optionChatRoutes
  }

  if (pathname === "/media" || pathname === "/media-trash") {
    const module = await import("./option-media-view-route-registry")
    return module.optionMediaViewRoutes
  }

  if (
    pathname === "/media-multi" ||
    pathname === "/review" ||
    pathname === "/content-review"
  ) {
    const module = await import("./option-media-review-route-registry")
    return module.optionMediaReviewRoutes
  }

  const module = await import("./route-registry")
  return module.optionRoutes
}

export const DeferredOptionsRoute = ({
  attemptedRoute,
  capabilities,
  capabilitiesLoading,
  label,
  description
}: DeferredOptionsRouteProps) => {
  const location = useLocation()
  const [routes, setRoutes] = React.useState<RouteDefinition[] | null>(null)

  React.useEffect(() => {
    let active = true

    void loadOptionRoutesForPath(location.pathname)
      .then((module) => {
        if (!active) return
        setRoutes(module)
      })
      .catch(() => {
        if (!active) return
        setRoutes([])
      })

    return () => {
      active = false
    }
  }, [location.pathname])

  if (routes == null) {
    return <PageAssistLoader label={label} description={description} />
  }

  const visibleRoutes = routes.filter(
    (route) =>
      (!route.targets || route.targets.includes(platformConfig.target)) &&
      (!isHostedTldwDeployment() || isHostedVisibleOptionPath(route.path))
  )
  const matchedRoute = visibleRoutes.find((route) =>
    routeMatchesLocation(route, location.pathname)
  )

  if (!matchedRoute) {
    return <RouteNotFoundState routeLabel={attemptedRoute} kind="options" />
  }

  const routeEnabled =
    capabilitiesLoading ||
    isRouteEnabledForCapabilities(matchedRoute.path, capabilities)

  return routeEnabled
    ? matchedRoute.element
    : <Navigate to="/settings" replace />
}

export default DeferredOptionsRoute
