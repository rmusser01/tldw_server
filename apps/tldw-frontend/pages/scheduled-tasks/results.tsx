import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-scheduled-tasks"), { ssr: false })
