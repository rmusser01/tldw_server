import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-prototype-workspaces"), {
  ssr: false
})
