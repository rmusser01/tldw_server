import dynamic from "next/dynamic"

export default dynamic(
  () => import("@/components/Option/Admin/AdminOperationsOverviewPage"),
  { ssr: false }
)
