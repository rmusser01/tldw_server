import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-admin-vllm"), { ssr: false })
