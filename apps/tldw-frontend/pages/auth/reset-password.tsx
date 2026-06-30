import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function ResetPasswordPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Password Reset Is Not Active Here"
      description="Hosted password recovery routes live in the private hosted distribution. Self-host deployments manage password recovery through local server configuration."
      plannedPath="/auth/reset-password"
    />
  )
}
