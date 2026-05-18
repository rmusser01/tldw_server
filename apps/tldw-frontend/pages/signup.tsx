import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function SignupPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Signup Is Not Part Of The OSS Web Surface"
      description="Hosted account creation now lives in the private hosted distribution. Self-host deployments keep account setup inside the local server configuration flow."
      plannedPath="/signup"
    />
  )
}
