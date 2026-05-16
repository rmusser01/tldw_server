import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-moderation-review"), {
  ssr: false
})
