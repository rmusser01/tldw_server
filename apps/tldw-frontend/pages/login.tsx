import dynamic from "next/dynamic"

import { RouteRedirect } from "@web/components/navigation/RouteRedirect"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

const TldwSettings = dynamic(
  () => import("@/components/Option/Settings/tldw").then((m) => m.TldwSettings),
  { ssr: false }
)

const LoginPage = () => {
  const hostedMode = isHostedTldwDeployment()

  if (!hostedMode) {
    return (
      <RouteRedirect
        to="/settings/tldw"
        title="Login is managed in local settings"
        description="Self-host deployments configure server URL and authentication from the tldw settings page."
      />
    )
  }

  return (
    <div className="min-h-screen bg-bg">
      <div className="mx-auto w-full max-w-4xl px-4 py-10 sm:px-6 lg:px-8">
        <TldwSettings />
      </div>
    </div>
  )
}

export default LoginPage
