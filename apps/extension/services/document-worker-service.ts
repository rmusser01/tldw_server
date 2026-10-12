import { DocumentWorkerClient } from "../../packages/ui/src/services/document-editing/worker-client"
import NativeDocumentWorker from "../../packages/ui/src/services/document-editing/worker?worker"

let client: DocumentWorkerClient | undefined
export function getDocumentWorkerClient(): DocumentWorkerClient {
  return (client ??= new DocumentWorkerClient(() => new NativeDocumentWorker()))
}
