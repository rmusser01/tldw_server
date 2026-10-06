import React, { Suspense } from "react";
import { Link, useLocation } from "react-router-dom";
import { FolderOpen, RefreshCw } from "lucide-react";
import { Button } from "@/components/Common/Button";
import { parseSharedWorkspaceRoute } from "./shared-workspace-route-state";
import { getResearchWorkspaceSearchFromLocation } from "./research-workspace-route-state";
import { useWorkspaceStore } from "@/store/workspace";
import { hydrateWorkspaceFromServer } from "@/store/workspace-api";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { resolveServicePromptScope } from "@/services/service-prompts";
import { servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error";
import { watchChatAccountChanges } from "@/services/chat-account-boundary";
import { hydrateCanonicalWorkspaceNote } from "./workspace-server-restore";

const LocalResearchWorkspace = React.lazy(() =>
  import("./index").then((module) => ({ default: module.ResearchWorkspace })),
);
const SharedResearchWorkspace = React.lazy(
  () => import("./SharedResearchWorkspace"),
);

const RouteFallback: React.FC = () => (
  <div
    role="status"
    className="flex h-full min-h-0 w-full flex-1 items-center justify-center text-sm text-text-muted"
    data-testid="research-workspace-route-pending"
  >Loading workspace</div>
);

const ActivationError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => (
  <div role="alert" className="p-6 text-sm text-text-muted">
    This workspace isn't available.
    {onRetry && (
      <Button variant="ghost" size="sm" iconOnly className="ml-2" onClick={onRetry}
        ariaLabel="Retry workspace" title="Retry workspace">
        <RefreshCw className="h-4 w-4" />
      </Button>
    )}
    <Link to="/workspaces"
      className="mt-3 flex min-h-11 w-fit max-w-full items-center gap-2 rounded-md border border-border px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus">
      <FolderOpen className="h-4 w-4 shrink-0" aria-hidden="true" />
      Open workspaces
    </Link>
  </div>
);

export const ActivatedLocalWorkspace: React.FC<{ workspaceId: string; webClip: boolean; children?: React.ReactNode }> = ({ workspaceId, webClip, children }) => {
  const hydrated = useWorkspaceStore(state => state.storeHydrated);
  const [status, setStatus] = React.useState<"pending" | "ready" | "failed">("pending");
  const [attempt, setAttempt] = React.useState(0);

  React.useEffect(() => {
    if (!hydrated) return;
    const controller = new AbortController();
    let mounted = true;
    const origin = useWorkspaceStore.getState().workspaceId;
    const stopWorkspace = useWorkspaceStore.subscribe((state, previous) => {
      // Record transitions, not just final identity: A -> B -> A is stale too.
      if (state.workspaceId !== previous.workspaceId) {
        controller.abort();
        if (mounted) setStatus("failed");
      }
    });
    const stopAccount = watchChatAccountChanges(invalidated => {
      if (!invalidated) return;
      controller.abort();
      if (mounted) setStatus("failed");
    });
    const current = () => mounted && !controller.signal.aborted &&
      useWorkspaceStore.getState().storeHydrated && useWorkspaceStore.getState().workspaceId === origin;
    const assertCurrent = () => {
      if (!current()) throw new Error("Workspace activation cancelled");
    };
    setStatus("pending");
    void (async () => {
      try {
        const store = useWorkspaceStore.getState();
        const localTarget = store.workspaceId === workspaceId ? store : store.workspaceSnapshots[workspaceId];
        if (webClip && localTarget && !localTarget.serverWorkspace && !localTarget.currentNote?.serverWorkspaceId) {
          stopWorkspace();
          if (store.workspaceId !== workspaceId) store.switchWorkspace(workspaceId);
          if (mounted) setStatus(useWorkspaceStore.getState().workspaceId === workspaceId ? "ready" : "failed");
          return;
        }
        const scope = await resolveServicePromptScope({ signal: controller.signal });
        assertCurrent();
        const options = { signal: controller.signal, requestScope: { config: scope.config, userId: scope.userId } };
        const staged = await hydrateWorkspaceFromServer(workspaceId, {
          requireComplete: true,
          fetch: async id => {
            const [metadata, sources, artifacts, notes] = await Promise.all([
              tldwClient.getWorkspace(id, options),
              tldwClient.getWorkspaceSources(id, options),
              tldwClient.getWorkspaceArtifacts(id, options),
              tldwClient.getWorkspaceNotes(id, options)
            ]);
            return { ...metadata, metadata, sources, artifacts, notes };
          }
        });
        assertCurrent();
        await hydrateCanonicalWorkspaceNote(staged, {
          scopeKey: scope.scopeKey, requestScope: options.requestScope
        }, controller.signal);
        assertCurrent();
        const verified = await resolveServicePromptScope({ signal: controller.signal });
        assertCurrent();
        if (verified.scopeKey !== scope.scopeKey || !servicePromptTargetsMatch(verified.config, scope.config) ||
            (verified.config.expectedSingleUserApiKeyScope ?? null) !== (scope.config.expectedSingleUserApiKeyScope ?? null) ||
            String(verified.userId) !== String(scope.userId)) throw new Error("Workspace account changed");
        stopWorkspace();
        // No await between the final fence and the atomic snapshot install.
        if (useWorkspaceStore.getState().installServerWorkspace(staged, {
          scopeKey: scope.scopeKey, expectedWorkspaceId: origin
        })) setStatus("ready");
        else setStatus("failed");
      } catch {
        controller.abort();
        if (mounted) setStatus("failed");
      }
    })();
    return () => {
      mounted = false;
      controller.abort();
      stopWorkspace();
      stopAccount();
    };
  }, [attempt, hydrated, workspaceId, webClip]);

  if (status === "failed") return <ActivationError onRetry={() => setAttempt(value => value + 1)} />;
  if (!hydrated || status !== "ready") return <RouteFallback />;
  return <Suspense fallback={<RouteFallback />}>{children ?? <LocalResearchWorkspace />}</Suspense>;
};

export const ResearchWorkspaceRouteGate: React.FC = () => {
  const location = useLocation();
  const search = getResearchWorkspaceSearchFromLocation({ search: location.search, hash: location.hash || "" });
  const route = parseSharedWorkspaceRoute(location.search);

  if (route.kind === "local") {
    const params = new URLSearchParams(search);
    if (params.has("workspace")) {
      const targets = params.getAll("workspace");
      const target = targets[0]?.trim();
      if (targets.length !== 1 || !target || target.includes("/")) return <ActivationError />;
      return <ActivatedLocalWorkspace key={`${location.key || ""}:${search}`} workspaceId={target}
        webClip={params.get("agent_task_handoff") === "web_clip"} />;
    }
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
