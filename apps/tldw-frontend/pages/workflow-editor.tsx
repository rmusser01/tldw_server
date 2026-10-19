import dynamic from "next/dynamic"
import { validateWithAjv } from "@web/lib/ajv"

const WorkflowEditorRoute = dynamic(() => import("@/routes/option-workflow-editor"), { ssr: false })
const validateWorkflowInputs = (data: unknown, schema: Record<string, unknown>) => validateWithAjv(data, schema, { strict: true })
export default function WorkflowEditorPage() { return <WorkflowEditorRoute validateInputs={validateWorkflowInputs} /> }
