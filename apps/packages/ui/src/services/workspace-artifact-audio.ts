import { bgRequest } from "@/services/background-proxy";
import { watchChatAccountChanges } from "@/services/chat-account-boundary";
import {
  loadServicePromptSnapshot,
  type ServicePromptSnapshot,
} from "@/services/service-prompts";
import { readResponseHeader } from "@/services/tldw/domains/models-audio";
import { requestScopeFields } from "@/services/tldw/domains/service-prompts";
import type { AllowedPath } from "@/services/tldw/openapi-guard";
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error";
import {
  hasWorkspaceArtifactOrigin,
  getStoredWorkspaceAudioRef,
} from "@/store/workspace-artifact-codec";
import { useWorkspaceStore } from "@/store/workspace";
import type { GeneratedArtifact } from "@/types/workspace";

export { getStoredWorkspaceAudioRef } from "@/store/workspace-artifact-codec";

/** Read an already-owned file; cached IDs and arbitrary URLs cannot supply authority. */
export const readStoredWorkspaceAudio = async (
  artifact: GeneratedArtifact,
  options: {
    workspaceId: string;
    isCurrent: () => boolean;
    signal?: AbortSignal;
  },
): Promise<Blob> => {
  const ref = getStoredWorkspaceAudioRef(artifact);
  const deadline = performance.now() + 15000;
  const controller = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([controller.signal, options.signal])
    : controller.signal;
  let scope: ServicePromptSnapshot | undefined;
  const visible = () => {
    try {
      const state = useWorkspaceStore.getState();
      const current = getStoredWorkspaceAudioRef(artifact);
      return (
        options.isCurrent() &&
        state.workspaceId === options.workspaceId &&
        state.generatedArtifacts.find((item) => item.id === artifact.id) ===
          artifact &&
        current.fileId === ref.fileId &&
        current.format === ref.format
      );
    } catch {
      return false;
    }
  };
  const retire = () => controller.abort(createServicePromptScopeChangedError());
  const timer = setTimeout(
    () =>
      controller.abort(
        new DOMException(
          "Reading saved audio exceeded 15 seconds.",
          "TimeoutError",
        ),
      ),
    15000,
  );
  const stopAccount = watchChatAccountChanges((invalidated) => {
    if (invalidated) retire();
  });
  const stopStore = useWorkspaceStore.subscribe(() => {
    if (!visible()) retire();
  });
  try {
    signal.throwIfAborted();
    if (!visible()) throw createServicePromptScopeChangedError();
    scope = await loadServicePromptSnapshot([], { signal });
    const readSignal = AbortSignal.any([
      signal,
      scope.scopeSignal,
      scope.scopeInvalidatedSignal,
    ]);
    const assertCurrent = () => {
      readSignal.throwIfAborted();
      if (performance.now() >= deadline)
        throw new DOMException(
          "Reading saved audio exceeded 15 seconds.",
          "TimeoutError",
        );
      if (
        !visible() ||
        !hasWorkspaceArtifactOrigin(
          artifact,
          scope!.scopeKey,
          options.workspaceId,
        )
      )
        throw createServicePromptScopeChangedError();
    };
    assertCurrent();
    const response = await bgRequest<{
      ok: boolean;
      data: ArrayBuffer;
      headers?: unknown;
    }>({
      path: ("/api/v1/storage/files/" +
        ref.fileId +
        "/download") as AllowedPath,
      method: "GET",
      responseType: "arrayBuffer",
      returnResponse: true,
      ...requestScopeFields(scope.requestScope),
      abortSignal: readSignal,
      timeoutMs: Math.max(1, Math.ceil(deadline - performance.now())),
    });
    assertCurrent();
    const contentType = readResponseHeader(response?.headers, "Content-Type")
      ?.trim()
      .toLowerCase();
    if (
      !response?.ok ||
      !(response.data instanceof ArrayBuffer) ||
      !response.data.byteLength ||
      !contentType?.startsWith("audio/")
    ) {
      throw new Error(
        "Saved audio is unavailable. Reopen the output or generate its audio again.",
      );
    }
    return new Blob([response.data], { type: contentType });
  } finally {
    clearTimeout(timer);
    stopStore();
    stopAccount();
    scope?.release();
  }
};
