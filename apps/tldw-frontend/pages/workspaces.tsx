import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-workspaces"), {
  ssr: false
})
