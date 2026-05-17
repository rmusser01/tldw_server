import React, { lazy } from "react"
import { ALL_TARGETS } from "@/config/platform"
import type { RouteDefinition } from "./route-registry"

const SidepanelChat = lazy(() => import("./sidepanel-chat"))
const SidepanelSettings = lazy(() => import("./sidepanel-settings"))
const SidepanelAgent = lazy(() => import("./sidepanel-agent"))
const SidepanelPersona = lazy(() => import("./sidepanel-persona"))
const SidepanelErrorBoundaryTest = lazy(
  () => import("./sidepanel-error-boundary-test")
)

const ERROR_BOUNDARY_TEST_ENABLED = process.env.NODE_ENV !== "production"

const errorBoundaryRoutes: RouteDefinition[] = ERROR_BOUNDARY_TEST_ENABLED
  ? [
      {
        kind: "sidepanel",
        path: "/error-boundary-test",
        element: <SidepanelErrorBoundaryTest />,
        targets: ALL_TARGETS
      }
    ]
  : []

export const sidepanelRoutes: RouteDefinition[] = [
  { kind: "sidepanel", path: "/", element: <SidepanelChat /> },
  {
    kind: "sidepanel",
    path: "/chat",
    element: <SidepanelChat />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/agent",
    element: <SidepanelAgent />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/persona",
    element: <SidepanelPersona />,
    targets: ALL_TARGETS
  },
  { kind: "sidepanel", path: "/settings", element: <SidepanelSettings /> },
  ...errorBoundaryRoutes
]
