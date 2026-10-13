import { DocumentWorkerClient } from "./worker-client";

// This module is the host's single lane; the extension resolves its native factory.
let client: DocumentWorkerClient | undefined;
export function getDocumentWorkerClient(): DocumentWorkerClient {
  return (client ??= new DocumentWorkerClient());
}
