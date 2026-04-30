import React from "react"

export function PrototypeWorkspacePage() {
  const params = new URLSearchParams(window.location.search)
  const mode = params.get("prototype_session_token")
    ? "collaborator"
    : "owner"

  return <div data-testid="prototype-workspace-mode">{mode}</div>
}
