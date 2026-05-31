# VoiceAssistant/__init__.py
# Voice Assistant Core Module
#
# Provides wake-word triggered voice assistant capabilities including:
# - Voice command routing and execution
# - Intent parsing (keyword + LLM fallback)
# - Session context management
# - Integration with MCP tools and workflows
#
#######################################################################################################################
from .db_helpers import (
    cleanup_old_sessions,
    count_user_voice_sessions,
    delete_voice_command,
    delete_voice_session,
    get_active_voice_session_count,
    get_persona_live_voice_summary,
    get_user_voice_commands,
    get_user_voice_sessions,
    get_voice_analytics_summary_stats,
    get_voice_command,
    get_voice_command_counts,
    get_voice_command_usage_stats,
    get_voice_resolution_stats,
    get_voice_session,
    get_voice_top_commands,
    get_voice_usage_by_day,
    record_persona_live_voice_event,
    record_voice_command_event,
    save_voice_command,
    save_voice_session,
)
from .intent_parser import IntentParser, get_intent_parser
from .registry import VoiceCommandRegistry, get_voice_command_registry
from .router import VoiceCommandRouter, get_voice_command_router
from .schemas import (
    ActionType,
    ParsedIntent,
    VoiceCommand,
    VoiceIntent,
    VoiceSessionContext,
    VoiceSessionState,
)
from .session import VoiceSessionManager, get_voice_session_manager
from .workflow_handler import (
    VoiceWorkflowHandler,
    WorkflowProgressEvent,
    get_voice_workflow_handler,
)

__all__ = [
    # Schemas
    "ActionType",
    "VoiceCommand",
    "VoiceIntent",
    "ParsedIntent",
    "VoiceSessionContext",
    "VoiceSessionState",
    # Registry
    "VoiceCommandRegistry",
    "get_voice_command_registry",
    # Parser
    "IntentParser",
    "get_intent_parser",
    # Session
    "VoiceSessionManager",
    "get_voice_session_manager",
    # Router
    "VoiceCommandRouter",
    "get_voice_command_router",
    # Workflow handler
    "VoiceWorkflowHandler",
    "WorkflowProgressEvent",
    "get_voice_workflow_handler",
    # Database helpers
    "save_voice_command",
    "get_voice_command",
    "get_user_voice_commands",
    "delete_voice_command",
    "save_voice_session",
    "get_voice_session",
    "get_user_voice_sessions",
    "delete_voice_session",
    "cleanup_old_sessions",
    "count_user_voice_sessions",
    "record_persona_live_voice_event",
    "record_voice_command_event",
    "get_voice_command_usage_stats",
    "get_voice_resolution_stats",
    "get_voice_top_commands",
    "get_voice_usage_by_day",
    "get_voice_analytics_summary_stats",
    "get_persona_live_voice_summary",
    "get_active_voice_session_count",
    "get_voice_command_counts",
]
#
# End of VoiceAssistant/__init__.py
#######################################################################################################################
