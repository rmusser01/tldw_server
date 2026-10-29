import React from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/Common/Button";
import { watchChatAccountChanges } from "@/services/chat-account-boundary";
import { useWorkspaceStore } from "@/store/workspace";
import { buildWorkspaceChatSessionKey } from "@/store/workspace-chat-session-key";
import { restoreMigratedResearchWorkspace } from "./workspace-server-restore";

const LocalWorkspace = React.lazy(() =>
  import("./index").then((module) => ({ default: module.ResearchWorkspace })),
);

type Result = {
  id: string | null;
  attempt: number;
  status: "ready" | "failed" | "local-work";
};

export default function RequestedResearchWorkspace({
  workspaceId,
}: {
  workspaceId: string | null;
}) {
  const { t } = useTranslation("playground");
  const [attempt, setAttempt] = React.useState(0);
  const [result, setResult] = React.useState<Result | null>(null);
  React.useEffect(
    () =>
      watchChatAccountChanges((invalidated) => {
        if (invalidated) setAttempt((value) => value + 1);
      }),
    [],
  );
  React.useEffect(() => {
    if (!workspaceId) return;
    let current = true;
    const controller = new AbortController();
    const localWorkError = new Error("Unverified local workspace work");
    const previousId = useWorkspaceStore.getState().workspaceId;
    const isCurrent = () =>
      current && useWorkspaceStore.getState().workspaceId === previousId;
    void restoreMigratedResearchWorkspace({
      workspaceId,
      signal: controller.signal,
      isCurrent,
      apply: (snapshot, chatSession) => {
        const state = useWorkspaceStore.getState();
        const cached =
          state.workspaceId === workspaceId
            ? state
            : state.workspaceSnapshots[workspaceId];
        const cachedChat = state.getWorkspaceChatSession(
          buildWorkspaceChatSessionKey(
            workspaceId,
            cached?.workspaceChatReferenceId,
          ),
        );
        if (
          cached?.currentNote.isDirty ||
          (cachedChat &&
            (cachedChat.historyId !== null ||
              cachedChat.messages.length ||
              cachedChat.history.length ||
              (cachedChat.serverChatId &&
                cachedChat.serverChatId !== chatSession?.serverChatId)))
        )
          throw localWorkError;
        state.saveCurrentWorkspace();
        state.restoreServerWorkspace(
          snapshot,
          chatSession ?? {
            messages: [],
            history: [],
            historyId: null,
            serverChatId: null,
          },
        );
        setResult({ id: workspaceId, attempt, status: "ready" });
      },
    })
      .then((restored) => {
        if (!restored && current)
          setResult({ id: workspaceId, attempt, status: "failed" });
      })
      .catch((error) => {
        if (current)
          setResult({
            id: workspaceId,
            attempt,
            status: error === localWorkError ? "local-work" : "failed",
          });
      });
    return () => {
      current = false;
      controller.abort();
    };
  }, [workspaceId, attempt]);

  const status =
    result?.id === workspaceId && result.attempt === attempt
      ? result.status
      : "loading";
  if (workspaceId && status === "ready") return <LocalWorkspace />;
  if (!workspaceId || status === "failed" || status === "local-work")
    return (
      <section
        role="alert"
        className="m-4 rounded-lg border border-border bg-surface p-4"
      >
        <h1 className="text-lg font-semibold">
          {t("requestedWorkspace.unavailable", "Workspace could not be opened")}
        </h1>
        <p>
          {status === "local-work"
            ? t(
                "requestedWorkspace.localWork",
                "Your local draft or saved Chat has been kept. Return to the account where it was created to save or export it, then retry.",
              )
            : t(
                "requestedWorkspace.recovery",
                "Your current workspace and unsaved work have been kept. Check your account and connection, then retry.",
              )}
        </p>
        {workspaceId && (
          <Button
            variant="primary"
            onClick={() => setAttempt((value) => value + 1)}
          >
            {t("requestedWorkspace.retry", "Retry")}
          </Button>
        )}
        {status === "local-work" && (
          <Link
            to="/settings/tldw"
            className="ml-3 inline-flex min-h-11 items-center rounded-md px-3 text-primary focus-visible:ring-2 focus-visible:ring-focus"
          >
            {t("requestedWorkspace.account", "Check account")}
          </Link>
        )}
        <Link
          to="/workspaces"
          className="ml-3 inline-flex min-h-11 items-center rounded-md px-3 text-primary focus-visible:ring-2 focus-visible:ring-focus"
        >
          {t("requestedWorkspace.back", "Back to Workspaces")}
        </Link>
      </section>
    );
  return (
    <div role="status" className="p-4">
      {t("requestedWorkspace.opening", "Opening workspace…")}
    </div>
  );
}
