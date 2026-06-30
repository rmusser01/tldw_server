import OptionLayout from "~/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { KanbanPlayground } from "@/components/Option/KanbanPlayground"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionKanbanPlayground = () => {
  return (
    <RouteErrorBoundary routeId="kanban" routeLabel="Kanban">
      <OptionLayout>
        <PageShell className="py-6" maxWidthClassName="max-w-7xl">
          <KanbanPlayground />
        </PageShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionKanbanPlayground
