"""TASK-13950.25 PR E: the non-stream executor's closures and stage spans move into completion_pipeline, response_processor, persistence_service and tool_execution_service; the rest becomes completion_pipeline.run_non_stream_call (R10-R13, Appendix B).
"""

MANIFEST = {
    "base": "5b9e6b72fc",
    "file": "tldw_Server_API/app/core/Chat/chat_service.py",
    "func": "_execute_non_stream_call_impl",
    "pkg": "",
    "ctx": [["cst", "NonStreamCall"]],
    "ctx_import": ["from tldw_Server_API.app.core.Chat.call_state import NonStreamCall"],
    "facade": "cs",
    "facade_stdlib": ["time"],
    "task": "TASK-13950.25",
    # found by the strict crossing rule: N7 sets it, the rest tests it. Needs a NonStreamCall slot (a LOCALS entry).
    "rewrite": {"metrics_recorded": "cst.metrics_recorded", "bounded_error": "cst.bounded_error"},
    "lift": {
        "_record_provider_use": "completion_pipeline",
        "_record_dict_response_usage": "completion_pipeline",
        "_record_provider_use_after_cancel": "completion_pipeline",
        "_queued_processor": "completion_pipeline",
        "_invoke_non_stream_provider": "completion_pipeline",
        "_record_fallback_use_after_cancel": "completion_pipeline",
        "_refresh_primary_choice_state": "response_processor",
        "_apply_output_safety_to_current_choices": "response_processor",
    },
    "moves": [
        {"name": "prepare_non_stream_request", "module": "completion_pipeline", "span": [6030, 6058], "doc": "Structured-response preparation, the tool-autoexec guard and the prompt-cost guardrail."},
        {"name": "call_non_stream_provider", "module": "completion_pipeline", "span": [6076, 6183], "doc": "Call the provider through the request queue or a bounded worker."},
        {"name": "collect_non_stream_reply", "module": "response_processor", "span": [6378, 6416], "doc": "Normalise the reply and collect its choices."},
        {"name": "save_first_non_stream_reply", "module": "persistence_service", "span": [6458, 6526], "doc": "Validate a structured reply, then save the first reply or defer it."},
        {"name": "run_non_stream_tool_autoexec", "module": "tool_execution_service", "span": [6528, 6755], "doc": "Run local tool autoexec and the auto-continue turns."},
        {"name": "flush_deferred_non_stream_persistence", "module": "persistence_service", "span": [6757, 6792], "doc": "Validate the tail reply and flush the deferred persistence."},
        {"name": "encode_non_stream_payload", "module": "response_processor", "span": [6794, 6849], "doc": "Run-first metrics, the assistant name, the encoding and the tldw_* metadata."},
    ],
    "entry": {"name": "run_non_stream_call", "module": "completion_pipeline", "doc": "Run one non-streaming chat completion: queue, failover, moderation and persistence."},
    "late_imports": [{"module": "moderation_pipeline", "lines": [24, 30]}],
    "drop": [
        {"module": "completion_pipeline", "def": "ChatCompletionPipeline"},
        {"module": "completion_pipeline", "import": "dataclass"},
        {"module": "completion_pipeline", "import": "Awaitable"},
        {"module": "completion_pipeline", "import": "Callable"},
        {"module": "completion_pipeline", "all": ["ChatCompletionPipeline"], "all_add": ["run_non_stream_call"]},
        {"module": "chat_service", "import": "ChatCompletionPipeline"},
        {"module": "chat_service", "replace": {"in": "execute_non_stream_call", "old": "get_chat_completion_pipeline().execute_non_stream(**kwargs)", "new": "_execute_non_stream_call_impl(**kwargs)"}},
        {"module": "chat_service", "replace": {"in": "execute_streaming_call", "old": "get_chat_completion_pipeline().execute_streaming", "new": "create_chat_streaming_response"}},
        {"module": "chat_service", "def": "get_chat_completion_pipeline"},
    ],
}
