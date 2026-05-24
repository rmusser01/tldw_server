/**
 * Context for tracking shared workspace access state.
 *
 * When a workspace is opened via a share link (?shared=<shareId>),
 * this context provides the access level and restrictions to child
 * components (SourcesPane, ChatPane, StudioPane, WorkspaceHeader).
 */
import React, { createContext, useContext, useMemo } from "react"
import type { AccessLevel } from "@/types/sharing"

interface SharedWorkspaceState {
  /** Whether the current workspace is being accessed via a share */
  isShared: boolean
  /** The share record ID, if shared */
  shareId: number | null
  /** Owner's user ID */
  ownerUserId: number | null
  /** Effective access level */
  accessLevel: AccessLevel | null
  /** Whether cloning is allowed */
  allowClone: boolean
  /** Convenience: can the user add sources? */
  canAddSources: boolean
  /** Convenience: can the user edit/delete sources and workspace metadata? */
  canEdit: boolean
}

const defaultState: SharedWorkspaceState = {
  isShared: false,
  shareId: null,
  ownerUserId: null,
  accessLevel: null,
  allowClone: false,
  canAddSources: true,
  canEdit: true,
}

const SharedWorkspaceCtx = createContext<SharedWorkspaceState>(defaultState)

export function useSharedWorkspace(): SharedWorkspaceState {
  return useContext(SharedWorkspaceCtx)
}

interface SharedWorkspaceProviderProps {
  shareId: number | null
  ownerUserId: number | null
  accessLevel: AccessLevel | null
  allowClone: boolean
  children: React.ReactNode
}

export const SharedWorkspaceProvider: React.FC<SharedWorkspaceProviderProps> = ({
  shareId,
  ownerUserId,
  accessLevel,
  allowClone,
  children,
}) => {
  const value = useMemo<SharedWorkspaceState>(() => {
    if (!shareId || !accessLevel) {
      return defaultState
    }
    return {
      isShared: true,
      shareId,
      ownerUserId,
      accessLevel,
      allowClone,
      canAddSources: accessLevel === "view_chat_add" || accessLevel === "full_edit",
      canEdit: accessLevel === "full_edit",
    }
  }, [shareId, ownerUserId, accessLevel, allowClone])

  return (
    <SharedWorkspaceCtx.Provider value={value}>
      {children}
    </SharedWorkspaceCtx.Provider>
  )
}
