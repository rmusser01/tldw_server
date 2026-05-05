import pytest

from tldw_Server_API.app.core.VLLM_Management.command_builder import build_vllm_serve_argv


def test_command_builder_prefers_structured_fields_over_extra_args():
    argv = build_vllm_serve_argv(
        {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "port": 8002,
            "tensor_parallel_size": 2,
            "extra_args": ["--port", "9999", "--dtype", "float16"],
        }
    )

    assert argv[:2] == ["vllm", "serve"]
    assert "--port" in argv
    assert "8002" in argv
    assert "9999" not in argv
    assert "--tensor-parallel-size" in argv
    assert "2" in argv
    assert "--dtype" in argv
    assert "float16" in argv


def test_command_builder_rejects_dangerous_or_conflicting_flags():
    with pytest.raises(ValueError, match="not allowed"):
        build_vllm_serve_argv(
            {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "extra_args": ["; rm", "-rf", "/"],
            }
        )
