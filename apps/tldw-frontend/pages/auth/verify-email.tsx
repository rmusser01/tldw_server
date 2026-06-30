import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function VerifyEmailPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Email Verification Is Not Active Here"
      description="Hosted verification routes live in the private hosted distribution. Self-host deployments handle account verification through their local auth configuration."
      plannedPath="/auth/verify-email"
    />
  )
}
