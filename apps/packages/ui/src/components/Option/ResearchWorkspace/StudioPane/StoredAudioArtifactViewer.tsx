import React from "react";
import { Button } from "antd";
import { readStoredWorkspaceAudio } from "@/services/workspace-artifact-audio";
import { watchChatAccountChanges } from "@/services/chat-account-boundary";
import { getStoredWorkspaceAudioRef } from "@/store/workspace-artifact-codec";
import { useWorkspaceStore } from "@/store/workspace";
import type { GeneratedArtifact } from "@/types/workspace";

export default function StoredAudioArtifactViewer({
  artifact,
  workspaceId,
  isCurrent,
}: {
  artifact: GeneratedArtifact;
  workspaceId: string;
  isCurrent: () => boolean;
}) {
  const [attempt, retry] = React.useReducer((value) => value + 1, 0);
  const [state, setState] = React.useState<{
    url?: string;
    error?: string;
    retired?: boolean;
  }>({});
  React.useEffect(() => {
    let active = true;
    let url: string | undefined;
    let ref: ReturnType<typeof getStoredWorkspaceAudioRef> | undefined;
    try {
      ref = getStoredWorkspaceAudioRef(artifact);
    } catch {
      /* The reader presents the recoverable missing-file error. */
    }
    const controller = new AbortController();
    let stopAccount = () => {};
    let stopStore = () => {};
    const releaseUrl = () => {
      if (url) {
        URL.revokeObjectURL(url);
        url = undefined;
      }
    };
    const retire = () => {
      if (!active) return;
      active = false;
      controller.abort();
      stopAccount();
      stopStore();
      releaseUrl();
      setState({
        retired: true,
        error:
          "This output or account changed. Reopen it before playing audio.",
      });
    };
    const current = () => {
      const state = useWorkspaceStore.getState();
      if (
        !isCurrent() ||
        state.workspaceId !== workspaceId ||
        state.generatedArtifacts.find((item) => item.id === artifact.id) !==
          artifact
      )
        return false;
      try {
        const value = getStoredWorkspaceAudioRef(artifact);
        return value.fileId === ref?.fileId && value.format === ref?.format;
      } catch {
        return ref === undefined;
      }
    };
    stopAccount = watchChatAccountChanges((invalidated) => {
      if (invalidated) retire();
    });
    stopStore = useWorkspaceStore.subscribe(() => {
      if (!current()) retire();
    });
    setState({});
    void (async () => {
      try {
        const blob = await readStoredWorkspaceAudio(artifact, {
          workspaceId,
          isCurrent,
          signal: controller.signal,
        });
        if (!active) return;
        if (!current()) {
          retire();
          return;
        }
        url = URL.createObjectURL(blob);
        setState({ url });
      } catch {
        if (active && isCurrent())
          setState({
            error:
              "Saved audio could not be loaded. Retry, reopen this output in the current account, or generate its audio again.",
          });
        else if (active) retire();
      }
    })();
    return () => {
      active = false;
      controller.abort();
      stopAccount();
      stopStore();
      releaseUrl();
    };
  }, [artifact, workspaceId, isCurrent, attempt]);
  return (
    <div className="flex flex-col gap-4">
      {state.url && (
        <audio
          aria-label="Saved Audio Summary"
          controls
          className="w-full"
          src={state.url}
        />
      )}
      {state.error ? (
        <div role="alert" className="space-y-2 text-sm">
          <p>{state.error}</p>
          {!state.retired && (
            <Button onClick={retry}>Retry loading audio</Button>
          )}
        </div>
      ) : (
        !state.url && <p role="status">Loading saved audio…</p>
      )}
      {!state.retired && artifact.content && (
        <details>
          <summary className="cursor-pointer text-sm text-text-muted">
            View Script
          </summary>
          <div className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-surface2 p-2 text-sm">
            {artifact.content}
          </div>
        </details>
      )}
    </div>
  );
}
