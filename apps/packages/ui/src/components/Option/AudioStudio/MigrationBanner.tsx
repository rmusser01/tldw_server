import React from "react"
import { Alert } from "antd"

export const MigrationBanner: React.FC = () => (
  <Alert
    type="info"
    showIcon
    className="rounded-md"
    title="Audiobook projects can move into Audio Studio Narration"
    description="Legacy local-project migration checks are planned for TASK-2351; this route keeps the compatibility path available without changing local Audiobook data."
  />
)
