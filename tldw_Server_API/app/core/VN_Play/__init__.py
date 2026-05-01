"""VN Play runtime package."""

from tldw_Server_API.app.core.VN_Play.constants import (
    LINKED_CHAT_MODE_READ_ONLY_CONTEXT,
    MODE_FREEFORM,
    MODE_STORY,
    SESSION_STATUS_ACTIVE,
    TURN_STATUS_PENDING,
)
from tldw_Server_API.app.core.VN_Play.adapters import (
    ChatVNPlayTurnAdapter,
    DeterministicVNPlayAdapter,
    FreeformVNPlayAdapter,
    StoryVNPlayAdapter,
    VNPlayModelError,
)
from tldw_Server_API.app.core.VN_Play.models import (
    CharacterSafetyResult,
    GateResult,
    ResolvedAsset,
    SceneState,
    TurnResult,
    VisualDirectiveResolution,
)
from tldw_Server_API.app.core.VN_Play.parser import (
    DialogueLine,
    NormalizedTurnResult,
    TurnChoice,
    VNPlayParseError,
    coerce_turn_result,
    parse_model_turn,
)
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayError,
    VNPlayNotFoundError,
    VNPlayService,
    VNPlaySession,
    VNPlayTurnError,
    VNPlayTurnResponse,
)
from tldw_Server_API.app.core.VN_Play.state import derive_scene_state

__all__ = [
    "CharacterSafetyResult",
    "ChatVNPlayTurnAdapter",
    "DeterministicVNPlayTurnAdapter",
    "DeterministicVNPlayAdapter",
    "DialogueLine",
    "FreeformVNPlayAdapter",
    "GateResult",
    "LINKED_CHAT_MODE_READ_ONLY_CONTEXT",
    "MODE_FREEFORM",
    "MODE_STORY",
    "NormalizedTurnResult",
    "ResolvedAsset",
    "SESSION_STATUS_ACTIVE",
    "SceneState",
    "StoryVNPlayAdapter",
    "TURN_STATUS_PENDING",
    "TurnChoice",
    "TurnResult",
    "VNPlayConflictError",
    "VNPlayError",
    "VNPlayModelError",
    "VNPlayNotFoundError",
    "VNPlayParseError",
    "VNPlayService",
    "VNPlaySession",
    "VNPlayTurnError",
    "VNPlayTurnResponse",
    "VisualDirectiveResolution",
    "coerce_turn_result",
    "derive_scene_state",
    "parse_model_turn",
]
