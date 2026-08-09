"""Injectable dependencies for canonical extraction strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    validate_selector_rules: Callable[..., dict[str, Any]]
    extract_schema_fields: Callable[..., dict[str, Any]]
    perform_chat_api_call: Callable[..., Any]
    increment_counter: Callable[..., None]
    observe_histogram: Callable[..., None]
    log_counter: Callable[..., None]
    perf_counter: Callable[[], float]
    wall_time: Callable[[], float]
    sleep: Callable[[float], None]
    cancellation_checkpoint: Callable[[], None]


def _cancellation_checkpoint() -> None:
    """Provide a no-op cooperative cancellation seam until strategies move."""


def build_default_dependencies() -> ExtractionDependencies:
    """Build live extraction dependencies without eager application imports."""

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
    from tldw_Server_API.app.core.Web_Scraping.extraction.metrics import (
        default_increment_counter,
        default_log_counter,
        default_observe_histogram,
    )
    from tldw_Server_API.app.core.Web_Scraping.selectors import (
        extract_schema_fields,
        validate_selector_rules,
    )

    return ExtractionDependencies(
        validate_selector_rules=validate_selector_rules,
        extract_schema_fields=extract_schema_fields,
        perform_chat_api_call=perform_chat_api_call,
        increment_counter=default_increment_counter,
        observe_histogram=default_observe_histogram,
        log_counter=default_log_counter,
        perf_counter=time.perf_counter,
        wall_time=time.time,
        sleep=time.sleep,
        cancellation_checkpoint=_cancellation_checkpoint,
    )
