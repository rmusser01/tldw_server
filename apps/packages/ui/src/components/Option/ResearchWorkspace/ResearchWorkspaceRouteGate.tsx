import React, { Suspense } from "react";
import { useLocation } from "react-router-dom";
import { parseSharedWorkspaceRoute } from "./shared-workspace-route-state";
import { getResearchWorkspaceDeepResearchReturnContext } from "./research-workspace-route-state";

const LocalResearchWorkspace = React.lazy(() =>
  import("./index").then((module) => ({ default: module.ResearchWorkspace })),
);
const RequestedResearchWorkspace = React.lazy(
  () => import("./RequestedResearchWorkspace"),
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
    const params = new URLSearchParams(search);
    const targets = params.getAll("source_workspace_id");
    const requested =
      targets.length === 1 &&
      targets[0].trim() &&
      targets[0].trim().length <= 128
        ? targets[0].trim()
        : null;
    return (
      <Suspense fallback={<RouteFallback />}>
        {targets.length &&
        !(
          targets.length === 1 &&
          getResearchWorkspaceDeepResearchReturnContext(params)
        ) ? (
          <RequestedResearchWorkspace workspaceId={requested} />
        ) : (
          <LocalResearchWorkspace />
        )}
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
