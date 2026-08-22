import React from "react";
import { useSharedResearchWorkspace } from "./useSharedResearchWorkspace";

type SharedResearchWorkspaceProps = {
  shareId?: number;
  invalidRoute?: boolean;
};

type PlaceholderProps = {
  heading: string;
  detail: string;
  shareId?: number;
  focus?: boolean;
};

const SharedPlaceholder: React.FC<PlaceholderProps> = ({
  heading,
  detail,
  shareId,
  focus = false,
}) => {
  const headingRef = React.useRef<HTMLHeadingElement>(null);

  React.useEffect(() => {
    if (focus) headingRef.current?.focus();
  }, [focus]);

  return (
    <main className="flex h-full min-h-0 w-full flex-1 items-center justify-center bg-bg p-6 text-center text-text">
      <div className="max-w-md space-y-3">
        <h1 ref={headingRef} tabIndex={-1} className="text-xl font-semibold">
          {heading}
        </h1>
        <p className="text-sm text-text-muted">{detail}</p>
        {shareId !== undefined && (
          <p className="text-sm text-text-subtle">Share {shareId}</p>
        )}
      </div>
    </main>
  );
};

const LoadedSharedWorkspace: React.FC<{ shareId: number }> = ({ shareId }) => {
  const { state } = useSharedResearchWorkspace(shareId);

  if (state.status === "loading") {
    return (
      <SharedPlaceholder
        heading="Loading shared workspace"
        detail="Shared workspace access is loading."
        shareId={shareId}
      />
    );
  }
  if (state.status === "not-found") {
    return (
      <SharedPlaceholder
        heading="Shared workspace not found"
        detail="This shared workspace is unavailable."
        shareId={shareId}
      />
    );
  }
  if (state.status !== "loaded" || !state.bootstrap) {
    return (
      <SharedPlaceholder
        heading="Shared workspace unavailable"
        detail="Shared workspace access is temporarily unavailable."
        shareId={shareId}
      />
    );
  }

  return (
    <SharedPlaceholder
      heading={state.bootstrap.workspace.name}
      detail="Shared workspace ready."
      shareId={shareId}
    />
  );
};

export const SharedResearchWorkspace: React.FC<
  SharedResearchWorkspaceProps
> = ({ shareId, invalidRoute = false }) => {
  if (invalidRoute || shareId === undefined) {
    return (
      <SharedPlaceholder
        heading="Shared workspace unavailable"
        detail="This shared workspace link is invalid."
        focus
      />
    );
  }

  return <LoadedSharedWorkspace shareId={shareId} />;
};

export default SharedResearchWorkspace;
