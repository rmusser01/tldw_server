import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  PrototypeCollaboratorSessionCreateInput,
  PrototypeWorkspaceDetail,
  PrototypePromotionCreateInput,
  PrototypePromotionRequest,
  PrototypeSessionJob,
  PrototypeWorkspace,
  PrototypeWorkspaceCreateInput,
  PrototypeWorkspaceSessionCreateInput
} from "@/types/prototype-workspace"
import {
  createPrototypeCollaboratorBranchSessionRequest,
  createPrototypeOwnerBranchSessionRequest,
  createPrototypePromotionRequestRequest,
  createPrototypeWorkspaceRequest,
  getPrototypeWorkspaceRequest
} from "@/services/tldw/domains/prototype-workspaces"

export const prototypeWorkspaceQueryKeys = {
  all: () => ["prototype-workspaces"] as const,
  workspaces: () => [...prototypeWorkspaceQueryKeys.all(), "workspaces"] as const,
  workspace: (prototypeWorkspaceId: string) =>
    [
      ...prototypeWorkspaceQueryKeys.workspaces(),
      "detail",
      String(prototypeWorkspaceId)
    ] as const,
  sessions: (prototypeWorkspaceId: string) =>
    [
      ...prototypeWorkspaceQueryKeys.all(),
      "sessions",
      String(prototypeWorkspaceId)
    ] as const,
  promotions: (prototypeWorkspaceId: string) =>
    [
      ...prototypeWorkspaceQueryKeys.all(),
      "promotions",
      String(prototypeWorkspaceId)
    ] as const
}

export const useCreatePrototypeWorkspace = () => {
  const queryClient = useQueryClient()
  return useMutation<PrototypeWorkspace, Error, PrototypeWorkspaceCreateInput>({
    mutationFn: createPrototypeWorkspaceRequest,
    onSuccess: async (workspace) => {
      queryClient.setQueryData(
        prototypeWorkspaceQueryKeys.workspace(workspace.id),
        workspace
      )
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.workspaces()
      })
    }
  })
}

export const usePrototypeWorkspace = (prototypeWorkspaceId: string | null | undefined) => {
  return useQuery<PrototypeWorkspaceDetail, Error>({
    queryKey: prototypeWorkspaceQueryKeys.workspace(
      prototypeWorkspaceId ?? "unknown"
    ),
    queryFn: () => getPrototypeWorkspaceRequest(String(prototypeWorkspaceId)),
    enabled: Boolean(prototypeWorkspaceId)
  })
}

export const useCreateOwnerBranchSession = (prototypeWorkspaceId: string) => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypeSessionJob,
    Error,
    PrototypeWorkspaceSessionCreateInput
  >({
    mutationFn: async (body) => {
      if (!prototypeWorkspaceId) {
        throw new Error("prototype_workspace_id is required")
      }
      return createPrototypeOwnerBranchSessionRequest(prototypeWorkspaceId, body)
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.sessions(prototypeWorkspaceId)
      })
    }
  })
}

export const useCreateCollaboratorBranchSession = () => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypeSessionJob,
    Error,
    PrototypeCollaboratorSessionCreateInput
  >({
    mutationFn: createPrototypeCollaboratorBranchSessionRequest,
    onSuccess: async (sessionJob) => {
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.sessions(
          sessionJob.prototype_workspace_id
        )
      })
    }
  })
}

export const useCreatePromotionRequest = () => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypePromotionRequest,
    Error,
    PrototypePromotionCreateInput
  >({
    mutationFn: createPrototypePromotionRequestRequest,
    onSuccess: async (_promotion, variables) => {
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.promotions(
          variables.prototype_workspace_id
        )
      })
    }
  })
}
