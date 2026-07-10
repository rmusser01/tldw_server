import { useRef } from "react";
import { generateID } from "@/db/dexie/helpers";
import { type UploadedFile } from "@/db/dexie/types";
import {
  cancelPreparedDocumentProcessing,
  normalizeDocumentPreflightResponse,
  withDefaultDocumentDecision,
} from "@/services/chat-document-processing";
import { tldwClient } from "@/services/tldw/TldwApiClient";

const toText = (value: unknown, fallback = ""): string =>
  typeof value === "string" && value.trim().length > 0 ? value : fallback;

export type UseFileUploadOptions = {
  maxContextFileSizeBytes: number;
  maxContextFileSizeLabel: string;
  notification: {
    error: (opts: { message: string; description: string }) => void;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  t: (...args: any[]) => unknown;
  uploadedFiles: UploadedFile[];
  setUploadedFiles: (files: UploadedFile[]) => void;
  contextFiles: UploadedFile[];
  setContextFiles: (files: UploadedFile[]) => void;
};

export const useFileUpload = ({
  maxContextFileSizeBytes,
  maxContextFileSizeLabel,
  notification,
  t,
  uploadedFiles,
  setUploadedFiles,
  contextFiles,
  setContextFiles,
}: UseFileUploadOptions) => {
  // Use refs to avoid stale closures when multiple uploads run concurrently.
  const uploadedFilesRef = useRef(uploadedFiles);
  uploadedFilesRef.current = uploadedFiles;
  const contextFilesRef = useRef(contextFiles);
  contextFilesRef.current = contextFiles;

  const commitUploadedFiles = (files: UploadedFile[]) => {
    const nextContextFiles = files.filter(
      (file) => file.processingMode !== "ingest_to_library",
    );
    uploadedFilesRef.current = files;
    contextFilesRef.current = nextContextFiles;
    setUploadedFiles(files);
    setContextFiles(nextContextFiles);
  };

  const markPreflightFailed = (fileId: string) => {
    const blockedReason = toText(
      t(
        "playground:documentProcessing.preflightFailed",
        "Document preflight failed. Try again or remove the file.",
      ),
      "Document preflight failed. Try again or remove the file.",
    );
    const nextFiles = uploadedFilesRef.current.map((uploadedFile) =>
      uploadedFile.id === fileId
        ? {
            ...uploadedFile,
            processingStatus: "blocked" as const,
            processingBlockedReason: blockedReason,
          }
        : uploadedFile,
    );
    commitUploadedFiles(nextFiles);
  };

  const preflightUploadedDocument = async (uploadedFile: UploadedFile) => {
    try {
      const response = await tldwClient.preflightDocumentUpload({
        files: [
          {
            client_id: uploadedFile.id,
            filename: uploadedFile.filename,
            mime_type: uploadedFile.type || null,
            size_bytes: uploadedFile.size,
          },
        ],
      });

      if (
        !uploadedFilesRef.current.some((file) => file.id === uploadedFile.id)
      ) {
        return;
      }

      commitUploadedFiles(
        normalizeDocumentPreflightResponse(
          response,
          uploadedFilesRef.current,
        ),
      );
    } catch (error) {
      console.error("Document preflight failed:", error);
      if (
        uploadedFilesRef.current.some((file) => file.id === uploadedFile.id)
      ) {
        markPreflightFailed(uploadedFile.id);
      }
    }
  };

  const handleFileUpload = async (file: File) => {
    try {
      const isImage = file.type.startsWith("image/");

      if (isImage) {
        return file;
      }

      if (file.size > maxContextFileSizeBytes) {
        notification.error({
          message: toText(
            t("upload.fileTooLargeTitle", "File Too Large"),
            "File Too Large",
          ),
          description: toText(
            t("upload.fileTooLargeDescription", {
              defaultValue: "File size must be less than {{size}}",
              size: maxContextFileSizeLabel,
            } as any),
            `File size must be less than ${maxContextFileSizeLabel}`,
          ),
        });
        return;
      }

      const fileId = generateID();

      const { processFileUpload } = await import("~/utils/file-processor");
      const source = await processFileUpload(file);

      const uploadedFile = withDefaultDocumentDecision({
        id: fileId,
        filename: file.name,
        type: file.type,
        content: source.content,
        size: file.size,
        uploadedAt: Date.now(),
        processed: false,
      });

      // Read current values from refs to avoid stale closure on concurrent uploads
      commitUploadedFiles([...uploadedFilesRef.current, uploadedFile]);
      void preflightUploadedDocument(uploadedFile);

      return file;
    } catch (error) {
      console.error("Error uploading file:", error);
      notification.error({
        message: toText(
          t("upload.uploadFailedTitle", "Upload Failed"),
          "Upload Failed",
        ),
        description: toText(
          t(
            "upload.uploadFailedDescription",
            "Failed to upload file. Please try again.",
          ),
          "Failed to upload file. Please try again.",
        ),
      });
      throw error;
    }
  };

  const removeUploadedFile = async (fileId: string) => {
    const removedFile = uploadedFilesRef.current.find((file) => file.id === fileId);
    if (
      removedFile &&
      (removedFile.ingestJobId != null ||
        removedFile.ingestBatchId ||
        removedFile.documentDraftId ||
        removedFile.processingResultRef?.kind === "draft")
    ) {
      await cancelPreparedDocumentProcessing([removedFile]).catch((error) => {
        console.error("Failed to cancel removed document processing:", error);
      });
    }
    commitUploadedFiles(uploadedFilesRef.current.filter((f) => f.id !== fileId));
  };

  const clearUploadedFiles = () => {
    commitUploadedFiles([]);
  };

  return {
    handleFileUpload,
    removeUploadedFile,
    clearUploadedFiles,
  };
};
