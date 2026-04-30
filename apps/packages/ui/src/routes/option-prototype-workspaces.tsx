import React from "react"

import OptionLayout from "~/components/Layouts/Layout"
import { PrototypeWorkspacePage } from "@/components/Option/PrototypeWorkspace"

const OptionPrototypeWorkspaces = () => {
  return (
    <OptionLayout>
      <div
        data-testid="prototype-workspace-route-shell"
        className="flex flex-1 min-h-0 overflow-hidden"
      >
        <PrototypeWorkspacePage />
      </div>
    </OptionLayout>
  )
}

export default OptionPrototypeWorkspaces
