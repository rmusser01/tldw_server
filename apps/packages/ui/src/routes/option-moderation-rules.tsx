import OptionLayout from "~/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { ModerationPlayground } from "@/components/Option/ModerationPlayground"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionModerationRules = () => {
  return (
    <RouteErrorBoundary routeId="moderation-rules" routeLabel="Content Rules">
      <OptionLayout>
        <PageShell className="py-6" maxWidthClassName="max-w-7xl">
          <ModerationPlayground />
        </PageShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionModerationRules
