from __future__ import annotations

"""
Audio-specific batch helper for the /process-audios endpoint.

This module lifts the core "call audio library + merge results" logic out of
the prior `/process-audios` implementation while preserving behavior. The
HTTP layer (status codes, request/response models) remains in the endpoint
modules.
"""

import asyncio
import functools
from typing import Any

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import (
    process_audio_files,
)
from tldw_Server_API.app.core.Utils.Utils import logging as logger


def _dedupe_errors_preserve_order(errors: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for error in errors:
        if not error:
            continue
        error_text = str(error)
        if error_text in seen:
            continue
        seen.add(error_text)
        deduped.append(error_text)
    return deduped


async def run_audio_batch(
    all_inputs: list[str],
    *,
    form_data: Any,
    temp_dir: str,
    temp_path_to_original_name: dict[str, str],
    saved_files: list[dict[str, Any]],
    file_errors_raw: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute the audio processing library and merge results with file errors.

    This helper mirrors the logic that was previously embedded directly in
    the endpoint-local audio batch handler:

    - starts from any file-handling errors (upload/save failures),
    - calls `process_audio_files` in a thread executor,
    - adapts the library's per-item results into the public media item shape,
    - maps temp-path identifiers back to original filenames/URLs,
    - computes processed/error counts and top-level `errors`.

    It deliberately does *not* decide the final HTTP status code; callers
    should derive that from the returned `batch_result`.
    """
    loop = asyncio.get_running_loop()

    batch_result: dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": [],
    }

    # -------------------------------
    # 1) Normalize file error results
    # -------------------------------
    if file_errors_raw:
        batch_result["errors_count"] += len(file_errors_raw)
        for err in file_errors_raw:
            input_ref = (
                err.get("input_ref")
                or err.get("original_filename")
                or err.get("input")
                or "Unknown Upload"
            )
            error_message = err.get(
                "error", f"Failed to save uploaded file '{input_ref}'."
            )
            batch_result["errors"].append(error_message)
            batch_result["results"].append(
                {
                    "status": "Error",
                    "input_ref": input_ref,
                    "processing_source": "N/A - File Save Failed",
                    "media_type": "audio",
                    "error": error_message,
                    "metadata": {},
                    "content": "",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "warnings": None,
                    "db_id": None,
                    "db_message": "Processing only endpoint.",
                    "message": None,
                }
            )

    # Map temp file paths back to original filenames for uploads.
    for sf in saved_files:
        if sf.get("path") and sf.get("original_filename"):
            temp_path_to_original_name[str(sf["path"])] = sf["original_filename"]
        else:
            logger.warning(
                "Missing path or original_filename in saved_files_info item for audio: {}",
                sf,
            )

    # The caller is responsible for enforcing non-empty inputs when appropriate.
    uploaded_paths = [str(sf["path"]) for sf in saved_files if sf.get("path")]
    if not all_inputs:
        # No additional processing to perform. Ensure top-level errors list is
        # de-duplicated before returning.
        batch_result["errors"] = _dedupe_errors_preserve_order(batch_result["errors"])
        return batch_result

    # -------------------------------
    # 2) Invoke process_audio_files
    # -------------------------------
    audio_args = {
        "inputs": all_inputs,
        "transcription_model": form_data.transcription_model,
        "transcription_language": form_data.transcription_language,
        "hotwords": getattr(form_data, "hotwords", None),
        "perform_chunking": form_data.perform_chunking,
        "chunk_method": form_data.chunk_method if form_data.chunk_method else None,
        "max_chunk_size": form_data.chunk_size,
        "chunk_overlap": form_data.chunk_overlap,
        "use_adaptive_chunking": form_data.use_adaptive_chunking,
        "use_multi_level_chunking": form_data.use_multi_level_chunking,
        "chunk_language": form_data.chunk_language,
        "diarize": form_data.diarize,
        "vad_use": form_data.vad_use,
        "timestamp_option": form_data.timestamp_option,
        "perform_analysis": form_data.perform_analysis,
        "api_name": form_data.api_name if form_data.perform_analysis else None,
        "custom_prompt_input": form_data.custom_prompt,
        "system_prompt_input": form_data.system_prompt,
        "summarize_recursively": form_data.summarize_recursively,
        "use_cookies": form_data.use_cookies,
        "cookies": form_data.cookies,
        # `/process-audios` is processing-only; do not keep originals.
        "keep_original": False,
        "custom_title": form_data.title,
        "author": form_data.author,
        "temp_dir": temp_dir,
    }

    processing_output: Any = None
    try:
        logger.debug(
            "Calling process_audio_files for /process-audios with {} inputs.",
            len(all_inputs),
        )
        batch_func = functools.partial(process_audio_files, **audio_args)
        processing_output = await loop.run_in_executor(None, batch_func)
    except Exception as exec_err:
        # Mirror previous behavior on library execution failure: mark all
        # attempted inputs as errors while preserving any prior file errors.
        logger.error(
            "Error executing process_audio_files: {}", exec_err, exc_info=True
        )
        error_msg = "Audio processing failed"
        num_attempted = len(all_inputs)
        batch_result["errors_count"] += num_attempted
        batch_result["errors"].append(error_msg)

        error_results = []
        for input_src in all_inputs:
            original_ref = temp_path_to_original_name.get(
                str(input_src), str(input_src)
            )
            if input_src in uploaded_paths:
                for sf in saved_files:
                    if str(sf.get("path")) == input_src:
                        original_ref = sf.get("original_filename", input_src)
                        break
            error_results.append(
                {
                    "status": "Error",
                    "input_ref": original_ref,
                    "processing_source": input_src,
                    "media_type": "audio",
                    "error": error_msg,
                    "db_id": None,
                    "db_message": "Processing only endpoint.",
                    "metadata": {},
                    "content": "",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "warnings": None,
                    "message": "Processing execution failed.",
                }
            )
        batch_result["results"].extend(error_results)
        # Skip normal merge path; final counts are recomputed below.

    # -------------------------------
    # 3) Merge processing results
    # -------------------------------
    if (
        processing_output
        and isinstance(processing_output, dict)
        and "results" in processing_output
    ):
        # Update counts based on library's report.
        batch_result["processed_count"] += processing_output.get(
            "processed_count", 0
        )
        new_errors_count = processing_output.get("errors_count", 0)
        batch_result["errors_count"] += new_errors_count
        batch_result["errors"].extend(processing_output.get("errors", []))

        processed_items = processing_output.get("results", [])
        adapted_processed_items: list[dict[str, Any]] = []
        for item in processed_items:
            # Prefer mapping based on the library's processing_source (temp path)
            # but fall back to the original input_ref when no mapping exists.
            orig_input_ref = item.get("input_ref")
            processing_source = item.get("processing_source")
            identifier_for_lookup = str(processing_source or orig_input_ref or "")
            original_ref = temp_path_to_original_name.get(
                identifier_for_lookup,
                orig_input_ref or processing_source or "Unknown",
            )

            item["input_ref"] = original_ref
            # Preserve the library's processing_source if present; do not
            # overwrite it with the (possibly mapped) input_ref.
            if processing_source is None:
                item["processing_source"] = identifier_for_lookup or original_ref

            # Ensure DB and media fields are correctly populated.
            item["db_id"] = None
            item["db_message"] = "Processing only endpoint."
            item.setdefault("status", "Error")
            item.setdefault("input_ref", "Unknown")
            item.setdefault("processing_source", "Unknown")
            item.setdefault("media_type", "audio")
            item.setdefault("metadata", {})
            item.setdefault("content", None)
            item.setdefault("segments", None)
            item.setdefault("chunks", None)
            item.setdefault("analysis", None)
            item.setdefault("analysis_details", {})
            item.setdefault("error", None)
            item.setdefault("warnings", None)
            item.setdefault("message", None)
            adapted_processed_items.append(item)
        batch_result["results"].extend(adapted_processed_items)
    elif processing_output is None and not batch_result["results"]:
        # Legacy behavior: no additional action when there are neither
        # processing results nor prior file errors.
        pass
    elif processing_output is not None:
        # Handle unexpected output format from the library more gracefully.
        logger.error(
            "process_audio_files returned unexpected format: Type={}; Value={}",
            type(processing_output),
            processing_output,
        )
        error_msg = (
            "Audio processing library returned invalid data "
            "(unexpected output structure)."
        )
        num_attempted = len(all_inputs)
        batch_result["errors_count"] += num_attempted
        batch_result["errors"].append(error_msg)

        existing_refs = {res.get("input_ref") for res in batch_result["results"]}
        error_results = []
        for input_src in all_inputs:
            original_ref = temp_path_to_original_name.get(
                str(input_src), str(input_src)
            )
            if input_src in uploaded_paths:
                for sf in saved_files:
                    if str(sf.get("path")) == input_src:
                        original_ref = sf.get("original_filename", input_src)
                        break
            if original_ref not in existing_refs:
                error_results.append(
                    {
                        "status": "Error",
                        "input_ref": original_ref,
                        "processing_source": input_src,
                        "media_type": "audio",
                        "error": error_msg,
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                        "metadata": {},
                        "content": "",
                        "segments": None,
                        "chunks": None,
                        "analysis": None,
                        "analysis_details": {},
                        "warnings": None,
                        "message": "Invalid processing result.",
                    }
                )
        batch_result["results"].extend(error_results)

    # -------------------------------
    # 4) Finalize counts and errors
    # -------------------------------
    final_processed_count = sum(
        1
        for r in batch_result["results"]
        if str(r.get("status", "")).lower() in {"success", "warning"}
    )
    final_error_count = sum(
        1 for r in batch_result["results"] if str(r.get("status", "")).lower() == "error"
    )
    batch_result["processed_count"] = final_processed_count
    batch_result["errors_count"] = final_error_count

    batch_result["errors"] = _dedupe_errors_preserve_order(batch_result["errors"])

    return batch_result


__all__ = ["run_audio_batch"]
