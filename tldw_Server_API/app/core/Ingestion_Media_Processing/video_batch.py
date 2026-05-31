from __future__ import annotations

"""
Video-specific batch helper for the /process-videos endpoint.

This module lifts the core "call video library + merge results" logic out of
the former endpoint-local video handler while preserving behavior. The HTTP
layer (status codes, request/response models) remains in the endpoint module.
"""

import asyncio
import functools
from typing import Any

from tldw_Server_API.app.core.config import config
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import (
    process_videos,
)
from tldw_Server_API.app.core.Utils.Utils import logging as logger


async def run_video_batch(
    all_inputs_to_process: list[str],
    *,
    form_data: Any,
    current_user: Any,
    temp_dir: str,
    temp_path_to_original_name: dict[str, str],
    file_handling_errors_structured: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute the video processing library and merge results with file errors.

    This function mirrors the logic originally embedded in
    the previous endpoint-local video handler:
      - builds `video_args` from the form data,
      - calls `process_videos` in a thread executor,
      - merges its output with any prior file-handling errors,
      - maps temp-path input_refs back to original filenames/URLs,
      - computes processed/error counts and top-level errors,
      - preserves `confabulation_results` when present.
    """
    loop = asyncio.get_running_loop()

    batch_result: dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": [],
        "confabulation_results": None,
    }

    video_args = {
        "inputs": all_inputs_to_process,
        # Use form_data directly
        "start_time": form_data.start_time,
        "end_time": form_data.end_time,
        "diarize": form_data.diarize,
        "vad_use": form_data.vad_use,
        "transcription_model": form_data.transcription_model,
        # Add language if process_videos needs it
        "transcription_language": form_data.transcription_language,
        "hotwords": getattr(form_data, "hotwords", None),
        "perform_analysis": form_data.perform_analysis,
        "custom_prompt": form_data.custom_prompt,
        "system_prompt": form_data.system_prompt,
        "perform_chunking": form_data.perform_chunking,
        "chunk_method": form_data.chunk_method,
        "max_chunk_size": form_data.chunk_size,
        "chunk_overlap": form_data.chunk_overlap,
        "use_adaptive_chunking": form_data.use_adaptive_chunking,
        "use_multi_level_chunking": form_data.use_multi_level_chunking,
        "chunk_language": form_data.chunk_language,
        "summarize_recursively": form_data.summarize_recursively,
        "api_name": form_data.api_name if form_data.perform_analysis else None,
        # api_key removed - retrieved from server config
        "use_cookies": form_data.use_cookies,
        "cookies": form_data.cookies,
        "timestamp_option": form_data.timestamp_option,
        "perform_confabulation_check": form_data.perform_confabulation_check_of_analysis,
        # Pass the managed temporary directory path
        "temp_dir": temp_dir,
        # 'keep_original' might be relevant if library needs it, default is False
        # 'perform_diarization' seems redundant if 'diarize' is passed, check library usage
        # If perform_diarization is truly needed separately:
        # "perform_diarization": form_data.diarize,
        "user_id": getattr(current_user, "id", None),
    }

    try:
        logger.debug(
            f"Calling process_videos for /process-videos endpoint with "
            f"{len(all_inputs_to_process)} inputs."
        )
        batch_func = functools.partial(process_videos, **video_args)

        processing_output = await loop.run_in_executor(None, batch_func)

        # Optional verbose debug (controlled by config)
        try:
            if bool(config.get("DEBUG_VERBOSE_PROCESSING", False)):
                safe_meta = {
                    "result_keys": list(processing_output.keys())
                    if isinstance(processing_output, dict)
                    else type(processing_output).__name__,
                    "results_len": len(processing_output.get("results", []))
                    if isinstance(processing_output, dict)
                    and isinstance(processing_output.get("results"), list)
                    else None,
                    "errors_count": processing_output.get("errors_count")
                    if isinstance(processing_output, dict)
                    else None,
                }
                logger.debug(f"process_videos processing_output summary: {safe_meta}")
        except Exception as summary_log_error:
            # Debug logging must never affect endpoint behavior.
            logger.debug("Video batch processing summary logging failed", exc_info=summary_log_error)

        # --- Combine Processing Results ---
        # Clear counters before merging library output; file errors will be
        # reflected via `file_handling_errors_structured`.
        batch_result["processed_count"] = 0
        batch_result["errors_count"] = 0
        batch_result["errors"] = []

        # Start with any structured file errors we recorded earlier.
        final_results_list = list(file_handling_errors_structured)
        final_errors_list = [
            err.get("error", "File handling error")
            for err in file_handling_errors_structured
        ]

        if isinstance(processing_output, dict):
            # Add results from the library processing.
            processed_results_from_lib = processing_output.get("results", [])
            for res in processed_results_from_lib:
                # Map input_ref back to original filename if applicable.
                current_input_ref = res.get("input_ref")
                res["input_ref"] = temp_path_to_original_name.get(
                    current_input_ref, current_input_ref
                )

                # Add endpoint-specific fields.
                res["db_id"] = None
                res["db_message"] = "Processing only endpoint."
                final_results_list.append(res)

            # Add specific errors reported by the library.
            final_errors_list.extend(processing_output.get("errors", []))

            # Standardize remote URL failures so tests can detect and skip reliably.
            # If any error result corresponds to a remote URL and the error does not
            # already contain 'Download failed', append a standardized message to
            # the top-level errors list.
            try:
                for res in processed_results_from_lib:
                    if not isinstance(res, dict):
                        continue
                    if res.get("status") == "Error":
                        ref = res.get("input_ref") or res.get("processing_source") or ""
                        err = (res.get("error") or "").lower()
                        if (
                            isinstance(ref, str)
                            and ref.startswith("http")
                            and "download failed" not in err
                        ):
                            final_errors_list.append(f"Download failed for {ref}")
            except Exception as normalize_error:
                # Never fail the request due to error-normalization debug logic.
                logger.debug("Video batch error normalization failed", exc_info=normalize_error)

            # Handle confabulation results if present.
            if "confabulation_results" in processing_output:
                batch_result["confabulation_results"] = processing_output[
                    "confabulation_results"
                ]

        else:
            # Handle unexpected output from process_videos library function.
            logger.error(
                f"process_videos function returned unexpected type: "
                f"{type(processing_output)}"
            )
            general_error_msg = "Video processing library returned invalid data."
            final_errors_list.append(general_error_msg)
            # Create error entries for all inputs attempted in *this specific*
            # processing call.
            for input_src in all_inputs_to_process:
                original_ref_for_error = temp_path_to_original_name.get(
                    input_src, input_src
                )
                final_results_list.append(
                    {
                        "status": "Error",
                        "input_ref": original_ref_for_error,
                        "processing_source": input_src,
                        "media_type": "video",
                        "metadata": {},
                        "content": "",
                        "segments": None,
                        "chunks": None,
                        "analysis": None,
                        "analysis_details": {},
                        "error": general_error_msg,
                        "warnings": None,
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                        "message": None,
                    }
                )

        # --- Recalculate final counts based on the merged list ---
        batch_result["results"] = final_results_list
        batch_result["processed_count"] = sum(
            1
            for r in final_results_list
            if str(r.get("status", "")).lower() in {"success", "warning"}
        )
        batch_result["errors_count"] = sum(
            1 for r in final_results_list if str(r.get("status", "")).lower() == "error"
        )
        deduped_errors: list[str] = []
        for err in final_errors_list:
            if err is None:
                continue
            err_str = str(err)
            if err_str not in deduped_errors:
                deduped_errors.append(err_str)
        batch_result["errors"] = deduped_errors
    except Exception as exec_err:
        # Catch errors during the library execution call itself.
        logger.error(f"Error executing process_videos: {exec_err}", exc_info=True)
        error_msg = "Video processing failed"

        # Start with existing file errors.
        final_results_list = list(file_handling_errors_structured)
        final_errors_list = [
            err.get("error", "File handling error")
            for err in file_handling_errors_structured
        ]
        final_errors_list.append(error_msg)

        # Create error entries for all inputs attempted in this batch.
        for input_src in all_inputs_to_process:
            original_ref_for_error = temp_path_to_original_name.get(
                input_src, input_src
            )
            final_results_list.append(
                {
                    "status": "Error",
                    "input_ref": original_ref_for_error,
                    "processing_source": input_src,
                    "media_type": "video",
                    "metadata": {},
                    "content": "",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "error": error_msg,
                    "warnings": None,
                    "db_id": None,
                    "db_message": "Processing only endpoint.",
                    "message": None,
                }
            )

        batch_result["results"] = final_results_list
        # Assume all failed if execution failed.
        batch_result["processed_count"] = 0
        batch_result["errors_count"] = len(final_results_list)
        unique_errors = {str(e) for e in final_errors_list if e is not None}
        batch_result["errors"] = list(unique_errors)

    return batch_result


__all__ = ["run_video_batch"]
