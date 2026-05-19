import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { SkillsWorkspace } from "~/components/Option/Skills/SkillsWorkspace"

const OptionSkillsRoute = () => {
  return (
    <RouteErrorBoundary routeId="skills" routeLabel="Skills">
      <OptionLayout>
        <SkillsWorkspace />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionSkillsRoute
