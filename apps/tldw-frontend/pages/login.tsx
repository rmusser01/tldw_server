import dynamic from "next/dynamic"
import { useRouter } from "next/router"
import { useEffect } from "react"

import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

const TldwSettings = dynamic(
  () => import("@/components/Option/Settings/tldw").then((m) => m.TldwSettings),
  { ssr: false }
)

const LoginPage = () => {
  const router = useRouter()
  const hostedMode = isHostedTldwDeployment()

  useEffect(() => {
    if (!hostedMode) {
      void router.replace("/settings/tldw")
    }
  }, [hostedMode, router])

  if (!hostedMode) {
    return null
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
