"""Full sliding-window cache is explicit configuration, never model-name policy."""

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppProfileCreateRequest,
    LlamaCppProfileUpdateRequest,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import LlamaCppProfile
from tldw_Server_API.app.core.Local_LLM.llamacpp_server_args import clean_server_args, server_arg_formatters

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, "", [], {}])
def test_full_cache_rejects_non_boolean_at_every_input_boundary(value):
    for schema, required in (
        (LlamaCppProfile, {"profile_id": "p", "name": "Any model"}),
        (LlamaCppProfileCreateRequest, {"name": "Any model"}),
        (LlamaCppProfileUpdateRequest, {}),
    ):
        with pytest.raises(ValidationError):
            schema(**required, server_args={"swa_full": value})
    with pytest.raises(ValueError):
        clean_server_args({"swa_full": value})


@pytest.mark.parametrize("value, expected", [(True, ["--swa-full"]), (False, [])])
def test_full_cache_flag_is_explicit_and_model_independent(value, expected):
    profile = LlamaCppProfile(profile_id="p", name="Arbitrary model", server_args={"swa_full": value})
    assert server_arg_formatters()["swa_full"](profile.server_args["swa_full"]) == expected
