import { lazy } from "react"
import { Navigate } from "react-router-dom"

import type { RouteDefinition } from "./route-registry"
import { CLAIMS_REVIEW_PATH, MEDIA_COLLECTION_REVIEW_PATH } from "./route-paths"

import OptionMediaMulti from "./option-media-multi"

const OptionClaimsReview = lazy(() => import("./option-claims-review"))
const OptionContentReview = lazy(() => import("./option-content-review"))
const OptionMediaCollection = lazy(() => import("./option-media-collection"))

export const optionMediaReviewRoutes: RouteDefinition[] = [
  { kind: "options", path: CLAIMS_REVIEW_PATH, element: <OptionClaimsReview /> },
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
    path: MEDIA_COLLECTION_REVIEW_PATH,
    element: <OptionMediaCollection />,
  },
  {
    kind: "options",
    path: "/content-review",
    element: <OptionContentReview />,
  }
]
