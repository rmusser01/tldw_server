import { lazy } from "react"
import { ALL_TARGETS } from "@/config/platform"
import type { RouteDefinition } from "./route-registry"

const SidepanelChat = lazy(() => import("./sidepanel-chat"))
const SidepanelHomeResolver = lazy(() => import("./sidepanel-home-resolver"))
const SidepanelSettings = lazy(() => import("./sidepanel-settings"))
const SidepanelAgent = lazy(() => import("./sidepanel-agent"))
const SidepanelCompanion = lazy(() => import("./sidepanel-companion"))
const SidepanelCompanionConversation = lazy(
  () => import("./sidepanel-companion-conversation")
)
const SidepanelClipper = lazy(() => import("./sidepanel-clipper"))
const SidepanelPersona = lazy(() => import("./sidepanel-persona"))
const SidepanelFlashcards = lazy(() => import("./sidepanel-flashcards"))
const SidepanelErrorBoundaryTest = lazy(
  () => import("./sidepanel-error-boundary-test")
)

export const sidepanelRoutes: RouteDefinition[] = [
  { kind: "sidepanel", path: "/", element: <SidepanelHomeResolver /> },
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
    path: "/companion",
    element: <SidepanelCompanion />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/companion/conversation",
    element: <SidepanelCompanionConversation />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/clipper",
    element: <SidepanelClipper />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/persona",
    element: <SidepanelPersona />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/flashcards",
    element: <SidepanelFlashcards />,
    targets: ALL_TARGETS
  },
  { kind: "sidepanel", path: "/settings", element: <SidepanelSettings /> },
  {
    kind: "sidepanel",
    path: "/error-boundary-test",
    element: <SidepanelErrorBoundaryTest />,
    targets: ALL_TARGETS
  }
]
