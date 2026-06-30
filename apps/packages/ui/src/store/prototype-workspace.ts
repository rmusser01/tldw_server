import { createWithEqualityFn } from "zustand/traditional"

type PrototypeWorkspaceStoreState = {
  activeWorkspaceId: string | null
  ownerSessionId: string | null
  collaboratorWorkspaceId: string | null
  collaboratorSessionId: string | null
  collaboratorSessionToken: string | null
  collaboratorShareToken: string | null
  sharedActorId: string | null
  lastPromotionRequestId: string | null
  setActiveWorkspaceId: (workspaceId: string | null) => void
  setOwnerSessionId: (sessionId: string | null) => void
  setCollaboratorEntry: (entry: {
    collaboratorWorkspaceId?: string | null
    collaboratorSessionId?: string | null
    collaboratorSessionToken?: string | null
    collaboratorShareToken?: string | null
    sharedActorId?: string | null
  }) => void
  setLastPromotionRequestId: (promotionRequestId: string | null) => void
  reset: () => void
}

const initialState = {
  activeWorkspaceId: null,
  ownerSessionId: null,
  collaboratorWorkspaceId: null,
  collaboratorSessionId: null,
  collaboratorSessionToken: null,
  collaboratorShareToken: null,
  sharedActorId: null,
  lastPromotionRequestId: null
}

export const usePrototypeWorkspaceStore =
  createWithEqualityFn<PrototypeWorkspaceStoreState>((set) => ({
    ...initialState,
    setActiveWorkspaceId: (activeWorkspaceId) => set({ activeWorkspaceId }),
    setOwnerSessionId: (ownerSessionId) => set({ ownerSessionId }),
    setCollaboratorEntry: (entry) =>
      set({
        collaboratorWorkspaceId:
          entry.collaboratorWorkspaceId !== undefined
            ? entry.collaboratorWorkspaceId
            : initialState.collaboratorWorkspaceId,
        collaboratorSessionId:
          entry.collaboratorSessionId !== undefined
            ? entry.collaboratorSessionId
            : initialState.collaboratorSessionId,
        collaboratorSessionToken:
          entry.collaboratorSessionToken !== undefined
            ? entry.collaboratorSessionToken
            : initialState.collaboratorSessionToken,
        collaboratorShareToken:
          entry.collaboratorShareToken !== undefined
            ? entry.collaboratorShareToken
            : initialState.collaboratorShareToken,
        sharedActorId:
          entry.sharedActorId !== undefined
            ? entry.sharedActorId
            : initialState.sharedActorId
      }),
    setLastPromotionRequestId: (lastPromotionRequestId) =>
      set({ lastPromotionRequestId }),
    reset: () => set(initialState)
  }))
