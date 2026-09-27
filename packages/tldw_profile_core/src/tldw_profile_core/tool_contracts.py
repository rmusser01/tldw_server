from .enums import ToolOperation, ToolResultStatus
from .models import EvidenceSpan
from .payloads import FrozenModel


class ProfileToolResult(FrozenModel):
    operation: ToolOperation
    status: ToolResultStatus
    message: EvidenceSpan
