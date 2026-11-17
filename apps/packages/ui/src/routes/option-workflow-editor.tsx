/**
 * Workflow Editor Route
 *
 * Full-screen node-based workflow editor for creating
 * and editing visual workflows.
 */

import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { WorkflowEditor } from "@/components/WorkflowEditor"
import type { WorkflowInputValidator } from "@/types/workflow-editor"

const OptionWorkflowEditorRoute = ({ validateInputs }: { validateInputs?: WorkflowInputValidator }) => {
  return (
    <RouteErrorBoundary routeId="workflow-editor" routeLabel="Workflow Editor">
      <OptionLayout>
        <div className="h-[calc(100vh-64px)]">
          <WorkflowEditor className="h-full" validateInputs={validateInputs} />
        </div>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionWorkflowEditorRoute
