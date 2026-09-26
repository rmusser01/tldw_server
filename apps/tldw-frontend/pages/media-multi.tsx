import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const MediaAnalysis = dynamic(() => import("@/routes/option-media-multi"), { ssr: false })

export default withPageTitle("Media Analysis", MediaAnalysis)
