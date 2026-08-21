import React from "react";

type SharedResearchWorkspaceProps = {
  shareId?: number;
  invalidRoute?: boolean;
};

export const SharedResearchWorkspace: React.FC<
  SharedResearchWorkspaceProps
> = ({ shareId, invalidRoute = false }) => {
  const headingRef = React.useRef<HTMLHeadingElement>(null);

  React.useEffect(() => {
    if (invalidRoute) headingRef.current?.focus();
  }, [invalidRoute]);

  return (
    <main className="flex h-full min-h-0 w-full flex-1 items-center justify-center bg-bg p-6 text-center text-text">
      <div className="max-w-md space-y-3">
        <h1 ref={headingRef} tabIndex={-1} className="text-xl font-semibold">
          Shared workspace unavailable
        </h1>
        <p className="text-sm text-text-muted">
          {invalidRoute
            ? "This shared workspace link is invalid."
            : "Shared workspace access is being prepared."}
        </p>
        {shareId !== undefined && (
          <p className="text-sm text-text-subtle">Share {shareId}</p>
        )}
      </div>
    </main>
  );
};

export default SharedResearchWorkspace;
