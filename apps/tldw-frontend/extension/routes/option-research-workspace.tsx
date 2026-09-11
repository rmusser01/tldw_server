import OptionLayout from "@web/components/layout/WebLayout"
import { PageShell } from "@/components/Common/PageShell"
import { ResearchWorkspaceRouteGate } from "@/components/Option/ResearchWorkspace/ResearchWorkspaceRouteGate"

const OptionResearchWorkspace = () => {
  return (
    <OptionLayout>
      <PageShell className="flex h-full min-h-0 w-full flex-1 overflow-hidden" maxWidthClassName="max-w-full">
        <ResearchWorkspaceRouteGate />
      </PageShell>
    </OptionLayout>
  )
}

export default OptionResearchWorkspace
