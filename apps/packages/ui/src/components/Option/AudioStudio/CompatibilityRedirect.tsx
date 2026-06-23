import React from "react"
import { Navigate } from "react-router-dom"

export const AUDIOBOOK_COMPATIBILITY_TARGET =
  "/audio-studio?workflow=narration"

export const CompatibilityRedirect: React.FC = () => (
  <Navigate to={AUDIOBOOK_COMPATIBILITY_TARGET} replace />
)
