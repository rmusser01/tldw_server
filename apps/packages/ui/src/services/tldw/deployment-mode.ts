export type TldwDeploymentMode = "self_host" | "hosted"

export const getTldwDeploymentMode = (): TldwDeploymentMode => {
  const raw = String(
    typeof process === "undefined"
      ? ""
      : process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || ""
  ).trim().toLowerCase()

  return raw === "hosted" ? "hosted" : "self_host"
}

export const isHostedTldwDeployment = (): boolean =>
  getTldwDeploymentMode() === "hosted"
