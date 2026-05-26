import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-research-workspace"), {
  ssr: false
})
