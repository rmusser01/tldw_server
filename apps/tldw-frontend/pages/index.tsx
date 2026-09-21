import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Home = dynamic(() => import("@/routes/option-index"), { ssr: false })

export default withPageTitle("Home", Home)
