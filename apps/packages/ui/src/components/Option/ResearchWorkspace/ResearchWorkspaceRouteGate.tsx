import React, { Suspense } from "react";
import { useLocation } from "react-router-dom";
import { parseSharedWorkspaceRoute } from "./shared-workspace-route-state";

const LocalResearchWorkspace = React.lazy(() =>
  import("./index").then((module) => ({ default: module.ResearchWorkspace })),
);
const SharedResearchWorkspace = React.lazy(
  () => import("./SharedResearchWorkspace"),
);

const RouteFallback: React.FC = () => (
  <div
    className="flex h-full min-h-0 w-full flex-1"
    data-testid="research-workspace-route-pending"
  />
);

export const ResearchWorkspaceRouteGate: React.FC = () => {
  const { search } = useLocation();
  const route = parseSharedWorkspaceRoute(search);

  if (route.kind === "local") {
    return (
      <Suspense fallback={<RouteFallback />}>
        <LocalResearchWorkspace />
      </Suspense>
    );
  }

  return (
    <Suspense fallback={<RouteFallback />}>
      <SharedResearchWorkspace
        key={route.kind === "shared-valid" ? route.shareId : "invalid"}
        shareId={route.kind === "shared-valid" ? route.shareId : undefined}
        invalidRoute={route.kind === "shared-invalid"}
      />
    </Suspense>
  );
};

export default ResearchWorkspaceRouteGate;
