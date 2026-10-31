import { getProviderIconComponent } from "@/utils/provider-registry"

export const ProviderIcons = ({
  provider,
  className
}: {
  provider: string
  className?: string
}) => {
  const Icon = getProviderIconComponent(provider)
  // eslint-disable-next-line react-hooks/static-components -- TASK-12116: this lookup returns a module-level registry component; it does not create one (covered by the rerender identity test).
  return <Icon className={className} />
}
