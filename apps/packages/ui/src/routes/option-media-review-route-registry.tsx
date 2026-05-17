import { lazy } from "react"
import { Navigate } from "react-router-dom"

import type { RouteDefinition } from "./route-registry"

import OptionMediaMulti from "./option-media-multi"

const OptionContentReview = lazy(() => import("./option-content-review"))
const OptionMediaCollection = lazy(() => import("./option-media-collection"))

export const optionMediaReviewRoutes: RouteDefinition[] = [
  {
    kind: "options",
    path: "/review",
    element: <Navigate to="/media-multi" replace />
  },
  {
    kind: "options",
    path: "/media-multi",
    element: <OptionMediaMulti />,
  },
  {
    kind: "options",
    path: "/media-collections/:collectionId",
    element: <OptionMediaCollection />,
  },
  {
    kind: "options",
    path: "/content-review",
    element: <OptionContentReview />,
  }
]
