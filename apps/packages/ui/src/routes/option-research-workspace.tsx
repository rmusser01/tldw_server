import OptionLayout from "~/components/Layouts/Layout"
import { ResearchWorkspaceRouteGate } from "@/components/Option/ResearchWorkspace/ResearchWorkspaceRouteGate"

const OptionResearchWorkspace = () => {
  return (
    <OptionLayout>
      <div className="flex h-full min-h-0 w-full flex-1 overflow-hidden">
        <ResearchWorkspaceRouteGate />
      </div>
    </OptionLayout>
  )
}

export default OptionResearchWorkspace
