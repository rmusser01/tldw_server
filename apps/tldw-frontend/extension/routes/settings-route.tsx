import { lazy, type ComponentType, type ReactNode } from "react"
import { SettingsLayout } from "~/components/Layouts/SettingsOptionLayout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

type SettingsRouteProps = {
  children: ReactNode
}

export const SettingsRoute = ({ children }: SettingsRouteProps) => (
  <RouteErrorBoundary routeId="settings" routeLabel="Settings">
    <main>
      <SettingsLayout>{children}</SettingsLayout>
    </main>
  </RouteErrorBoundary>
)

type SettingsModule = {
  default?: ComponentType
  [key: string]: ComponentType | undefined
}

export function createSettingsRoute(
  loader: () => Promise<SettingsModule>,
  exportName: string = "default"
) {
  return lazy(async () => {
    const loadedModule = await loader()
    const Page =
      exportName === "default" ? loadedModule.default : loadedModule[exportName]

    if (!Page) {
      throw new Error(`Settings route missing export: ${exportName}`)
    }

    const Wrapped = () => (
      <SettingsRoute>
        <Page />
      </SettingsRoute>
    )

    return { default: Wrapped }
  })
}
