import OptionLayout from "@web/components/layout/WebLayout"
import { PageShell } from "@/components/Common/PageShell"
import { WorkspacePlayground } from "@/components/Option/WorkspacePlayground"

const OptionWorkspacePlayground = () => {
  return (
    <OptionLayout>
      <PageShell className="flex h-full min-h-0 w-full flex-1 overflow-hidden" maxWidthClassName="max-w-full">
        <WorkspacePlayground />
      </PageShell>
    </OptionLayout>
  )
}

export default OptionWorkspacePlayground
