import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function MagicLinkPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Magic Link Sign-In Is Not Active Here"
      description="Hosted magic-link routes live in the private hosted distribution. Self-host deployments keep auth inside the local server and settings surface."
      plannedPath="/auth/magic-link"
    />
  )
}
