# llamacpp.py
# Description: This file contains the API endpoints for managing Llama.cpp server operations in tldw_Server_API.
#
# Imports
from typing import Optional, Dict, Any
#
# Thid-party Libraries
from fastapi import APIRouter, HTTPException, Body, Depends
#
# Local Imports
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError, ServerError, InferenceError
#
########################################################################################################################
#
# Functions:

router = APIRouter()

#     LlamaCppConfig: Defines paths and default arguments for llama.cpp/server.
#
#     LlamaCpp_Handler:
#
#         Manages a single llama.cpp/server process (_active_server_process).
#
#         start_server(): This is your model swap function. If an existing server is running (managed by this handler), it calls stop_server() first, then starts a new server with the new model_filename and server_args.
#
#         stop_server(): Terminates the managed server process, handling process groups for robust cleanup.
#
#         inference(): Sends requests to the Llama.cpp server's OpenAI-compatible API (e.g., /v1/chat/completions).
#
#         list_models(): Scans the models_dir for .gguf files.
#
#         get_server_status(): Reports the current state of the managed server.
#
#         _cleanup_managed_server_sync(): Ensures server is stopped on application exit.
#
#         Optional logging of llama.cpp/server output to a file.
#
#     LLM_Inference_Manager Updates:
#
#         Initializes and provides access to LlamaCppHandler.
#
#         Delegates relevant calls (start_server, stop_server, run_inference, list_local_models) to the LlamaCppHandler.
#
#     API Endpoints: Provide HTTP interfaces to list models, start/swap the server with a specific model, stop it, get status, and run inference.

# Assuming 'llm_manager' is available, e.g., initialized in main.py and passed around or via Depends
# For simplicity, let's assume it's directly accessible here.
# from your_main_app_file import llm_manager_instance as llm_manager


# --- Llama.cpp Specific Endpoints ---
@router.post("/llamacpp/start_server", summary="Start or Swap Llama.cpp Server Model")
async def start_llamacpp_server_endpoint(
        model_filename: str = Body(..., embed=True,
                                   description="Filename of the GGUF model to load (e.g., 'mistral-7b-v0.1.Q4_K_M.gguf')"),
        server_args: Optional[Dict[str, Any]] = Body({}, embed=True,
                                                     description="Optional Llama.cpp server arguments (e.g., port, n_gpu_layers)")
):
    """
    Starts the Llama.cpp server with the specified model.
    If a server is already running, it will be stopped and restarted with the new model (model swap).
    """
    try:
        if not llm_manager.llamacpp:  # Or llm_manager.get_handler("llamacpp") and catch error
            raise HTTPException(status_code=400, detail="Llama.cpp backend is not enabled or configured.")
        result = await llm_manager.start_server(backend="llamacpp", model_name=model_filename, server_args=server_args)
        return result
    except (ModelNotFoundError, ServerError, InferenceError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        llm_manager.logger.error(f"Unexpected error starting Llama.cpp server: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.post("/llamacpp/stop_server", summary="Stop Llama.cpp Server")
async def stop_llamacpp_server_endpoint():
    try:
        if not llm_manager.llamacpp:
            raise HTTPException(status_code=400, detail="Llama.cpp backend is not enabled or configured.")
        result = await llm_manager.stop_server(backend="llamacpp")
        return {"message": result}
    except (ServerError, InferenceError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        llm_manager.logger.error(f"Unexpected error stopping Llama.cpp server: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/llamacpp/status", summary="Get Llama.cpp Server Status")
async def get_llamacpp_status_endpoint():
    try:
        if not llm_manager.llamacpp:
            raise HTTPException(status_code=400, detail="Llama.cpp backend is not enabled or configured.")
        # status = await llm_manager.llamacpp.get_server_status() # Direct access
        status = await llm_manager.get_server_status(backend="llamacpp")  # Via manager
        return status
    except Exception as e:
        llm_manager.logger.error(f"Unexpected error getting Llama.cpp server status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/llamacpp/models", summary="List available Llama.cpp models")
async def list_llamacpp_models_endpoint():
    try:
        if not llm_manager.llamacpp:
            raise HTTPException(status_code=400, detail="Llama.cpp backend is not enabled or configured.")
        models = await llm_manager.list_local_models(backend="llamacpp")
        return {"available_models": models}
    except Exception as e:
        llm_manager.logger.error(f"Unexpected error listing Llama.cpp models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.post("/llamacpp/inference", summary="Run inference with Llama.cpp")
async def run_llamacpp_inference_endpoint(
        payload: Dict[str, Any] = Body(..., description="OpenAI compatible payload (messages, temperature, etc.)")
):
    """
    Runs inference using the currently loaded Llama.cpp model.
    Payload should be OpenAI compatible (e.g., include 'messages' list).
    Example: {"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.7}
    """
    try:
        if not llm_manager.llamacpp:
            raise HTTPException(status_code=400, detail="Llama.cpp backend is not enabled or configured.")

        # The 'model_name_or_path' for manager.run_inference is for context,
        # LlamaCppHandler uses its internally known active model.
        # We can get it from status or just pass a placeholder.
        status = await llm_manager.get_server_status(backend="llamacpp")
        current_model = status.get("model", "unknown_active_model")

        result = await llm_manager.run_inference(
            backend="llamacpp",
            model_name_or_path=current_model,  # Contextual
            prompt=None,  # Assuming payload contains 'messages'
            **payload  # Pass the entire payload dict as kwargs
        )
        return result
    except (ServerError, InferenceError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        llm_manager.logger.error(f"Unexpected error during Llama.cpp inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

#
# End of llamacpp.py
##########################################################################################################################
