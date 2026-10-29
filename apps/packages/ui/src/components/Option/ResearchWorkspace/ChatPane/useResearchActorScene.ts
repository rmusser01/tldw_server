import React from "react";
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner";
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection";
import {
  captureActorSceneSend,
  captureActorSceneSendForTarget,
} from "@/hooks/chat/actor-scene-source";
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition";
import { captureActorSettingsTarget } from "@/utils/actor-settings-target";
import {
  chatDraftAddressKey,
  type ChatDraftAddress,
} from "@/utils/chat-draft-address";
import { buildWorkspaceChatSessionKey } from "@/store/workspace-chat-session-key";
import { useWorkspaceStore } from "@/store/workspace";
import { useStoreMessageOption } from "@/store/option";
import { usePlaygroundSessionStore } from "@/store/playground-session";

/** Research retains a private Scene; its SDK chat IDs grant no native history authority. */
export const useResearchActorScene = (
  workspaceId: string | null,
  referenceId: string | null,
) => {
  const history = useHistorySelectionContext();
  const { isCurrent: accountIsCurrent, requestScope } = useChatDraftOwner(
    () => {},
  );
  const mounted = React.useRef(false);
  const draftRevision = useStoreMessageOption(
    (state) => state.chatDraftRevision,
  );
  const address = React.useMemo<ChatDraftAddress | null>(
    () =>
      workspaceId
        ? {
            kind: "fresh",
            id:
              "research:" +
              buildWorkspaceChatSessionKey(workspaceId, referenceId),
            scope: { type: "workspace", workspaceId },
            temporary: false,
          }
        : null,
    [workspaceId, referenceId],
  );
  React.useLayoutEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  React.useLayoutEffect(() => {
    if (history || !address || !accountIsCurrent()) return;
    const workspace = useWorkspaceStore.getState();
    if (
      workspace.workspaceId === workspaceId &&
      workspace.workspaceChatReferenceId === referenceId
    )
      useStoreMessageOption.getState().setChatDraftAddress(address);
  }, [history, address, accountIsCurrent, workspaceId, referenceId]);

  return React.useCallback(
    (commandIsCurrent: () => boolean = () => true) => {
      if (history) return captureActorSceneSend(history, commandIsCurrent);
      const draft = useStoreMessageOption.getState();
      const workspace = useWorkspaceStore.getState();
      const session = usePlaygroundSessionStore.getState();
      if (
        draft.chatDraftRevision !== draftRevision ||
        !address ||
        !draft.chatDraftAddress ||
        !requestScope ||
        !accountIsCurrent() ||
        chatDraftAddressKey(draft.chatDraftAddress) !==
          chatDraftAddressKey(address) ||
        workspace.workspaceId !== workspaceId ||
        workspace.workspaceChatReferenceId !== referenceId
      )
        return null;
      const current = () => {
        const now = useWorkspaceStore.getState();
        const active = useStoreMessageOption.getState();
        const latest = usePlaygroundSessionStore.getState();
        return (
          mounted.current &&
          accountIsCurrent() &&
          commandIsCurrent() &&
          now.workspaceId === workspaceId &&
          now.workspaceChatReferenceId === referenceId &&
          now.selectedSourceIds === workspace.selectedSourceIds &&
          now.selectedSourceFolderIds === workspace.selectedSourceFolderIds &&
          now.sourceFolderMemberships === workspace.sourceFolderMemberships &&
          now.sources === workspace.sources &&
          active.chatDraftAddress === draft.chatDraftAddress &&
          active.chatDraftRevision === draft.chatDraftRevision &&
          latest.restoreRevision === session.restoreRevision &&
          latest.sourceSelectionRevision === session.sourceSelectionRevision
        );
      };
      const source = captureChatDraftSource(current);
      const target = captureActorSettingsTarget({
        address: draft.chatDraftAddress,
        requestScope: requestScope,
        accountIsCurrent: accountIsCurrent,
        isCurrent: source.isCurrent,
      });
      return captureActorSceneSendForTarget(
        target,
        target?.account,
        source.isCurrent,
        current,
      );
    },
    [
      history,
      address,
      requestScope,
      accountIsCurrent,
      workspaceId,
      referenceId,
      draftRevision,
    ],
  );
};
