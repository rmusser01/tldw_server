# VoiceAssistant/intent_parser.py
# Intent Parser - Parses voice input into actionable intents
#
# Uses a hybrid approach:
# 1. Keyword/prefix matching for known commands (fast)
# 2. Pattern matching for parameterized commands
# 3. LLM fallback for complex/ambiguous queries
#
#######################################################################################################################
import json
import re
import time
from typing import Any, Optional

from loguru import logger

from .registry import VoiceCommandRegistry, get_voice_command_registry
from .schemas import ActionType, ParsedIntent, VoiceCommand, VoiceIntent


class IntentParser:
    """
    Parses transcribed text into actionable intents.

    The parser uses a multi-stage approach:
    1. Keyword matching - Fast prefix/exact matching for registered phrases
    2. Pattern matching - Regex patterns for structured commands
    3. LLM fallback - For complex queries that don't match known patterns
    """

    # Confidence thresholds
    KEYWORD_MATCH_THRESHOLD = 0.5
    LLM_FALLBACK_THRESHOLD = 0.3

    def __init__(
        self,
        registry: Optional[VoiceCommandRegistry] = None,
        llm_enabled: bool = True,
    ):
        """
        Initialize the intent parser.

        Args:
            registry: Voice command registry (uses singleton if not provided)
            llm_enabled: Whether to use LLM fallback for unmatched intents
        """
        self.registry = registry or get_voice_command_registry()
        self.llm_enabled = llm_enabled

        # Pattern extractors for common entity types
        self._entity_patterns = {
            "query": [
                r"(?:search|find|look up|look for)\s+(?:for\s+)?(.+)",
                r"(?:what is|what's|who is|who's|tell me about)\s+(.+)",
            ],
            "note_content": [
                r"(?:note|remember|take a note)\s+(?:that\s+)?(.+)",
                r"(?:create|make)\s+(?:a\s+)?note\s+(?:about\s+|saying\s+)?(.+)",
            ],
            "confirmation": [
                r"^(yes|yeah|yep|sure|ok|okay|confirm|go ahead|do it)$",
                r"^(no|nope|cancel|stop|never mind|abort)$",
            ],
        }

    async def parse(
        self,
        text: str,
        user_id: int = 0,
        context: Optional[dict[str, Any]] = None,
        persona_id: Optional[str] = None,
    ) -> ParsedIntent:
        """
        Parse transcribed text into an intent.

        Args:
            text: The transcribed text to parse
            user_id: User ID for user-specific commands
            context: Optional conversation context

        Returns:
            ParsedIntent with matched intent and metadata
        """
        start_time = time.time()
        text = text.strip()

        if not text:
            return ParsedIntent(
                intent=VoiceIntent(
                    action_type=ActionType.CUSTOM,
                    action_config={"action": "empty_input"},
                    raw_text="",
                ),
                match_method="empty",
                processing_time_ms=0.0,
            )

        # Stage 1: Check for confirmation responses if awaiting confirmation
        if context and context.get("awaiting_confirmation"):
            confirm_result = self._check_confirmation(text)
            if confirm_result is not None:
                return ParsedIntent(
                    intent=VoiceIntent(
                        action_type=ActionType.CUSTOM,
                        action_config={"action": "confirmation", "confirmed": confirm_result},
                        raw_text=text,
                        confidence=1.0,
                    ),
                    match_method="confirmation",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

        # Stage 2: Keyword/prefix matching
        keyword_result = await self._keyword_match(text, user_id, persona_id=persona_id)
        if keyword_result and keyword_result.intent.confidence >= self.KEYWORD_MATCH_THRESHOLD:
            keyword_result.processing_time_ms = (time.time() - start_time) * 1000
            return keyword_result

        # Stage 3: Pattern matching for entity extraction
        pattern_result = await self._pattern_match(text, user_id)
        if pattern_result and pattern_result.intent.confidence >= self.KEYWORD_MATCH_THRESHOLD:
            pattern_result.processing_time_ms = (time.time() - start_time) * 1000
            return pattern_result

        # Stage 4: LLM fallback for complex queries
        if self.llm_enabled:
            llm_result = await self._llm_parse(text, user_id, context)
            if llm_result:
                llm_result.processing_time_ms = (time.time() - start_time) * 1000
                return llm_result

        # Default: Treat as general chat
        return ParsedIntent(
            intent=VoiceIntent(
                action_type=ActionType.LLM_CHAT,
                action_config={"message": text},
                raw_text=text,
                confidence=0.5,
            ),
            match_method="default",
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    async def parse_registered_command(
        self,
        text: str,
        user_id: int = 0,
        *,
        context: Optional[dict[str, Any]] = None,
        persona_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> Optional[ParsedIntent]:
        """Match only against registered command phrases without generic fallbacks."""
        start_time = time.time()
        text = text.strip()
        if not text:
            return None

        if context and context.get("awaiting_confirmation"):
            confirm_result = self._check_confirmation(text)
            if confirm_result is not None:
                return ParsedIntent(
                    intent=VoiceIntent(
                        action_type=ActionType.CUSTOM,
                        action_config={"action": "confirmation", "confirmed": confirm_result},
                        raw_text=text,
                        confidence=1.0,
                    ),
                    match_method="confirmation",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

        keyword_result = await self._keyword_match(
            text,
            user_id,
            persona_id=persona_id,
            include_disabled=include_disabled,
        )
        if keyword_result and keyword_result.intent.confidence >= self.KEYWORD_MATCH_THRESHOLD:
            keyword_result.processing_time_ms = (time.time() - start_time) * 1000
            return keyword_result
        return None

    def _check_confirmation(self, text: str) -> Optional[bool]:
        """Check if text is a confirmation/denial response."""
        text_lower = text.lower().strip()

        # Positive confirmations
        if re.match(r"^(yes|yeah|yep|sure|ok|okay|confirm|go ahead|do it|affirmative|correct)$", text_lower):
            return True

        # Negative responses
        if re.match(r"^(no|nope|cancel|stop|never mind|abort|negative|don't|do not)$", text_lower):
            return False

        return None

    async def _keyword_match(
        self,
        text: str,
        user_id: int,
        *,
        persona_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> Optional[ParsedIntent]:
        """Match text against registered command phrases."""
        matches = self.registry.find_matching_commands(
            text,
            user_id,
            persona_id=persona_id,
            include_disabled=include_disabled,
        )

        if not matches:
            return None

        # Get the best match
        best_command, matched_phrase, score = matches[0]

        # Extract any additional parameters from the text
        entities = self._extract_entities(text, matched_phrase, best_command)

        intent = VoiceIntent(
            command_id=best_command.id,
            action_type=best_command.action_type,
            action_config={**best_command.action_config, **entities},
            entities=entities,
            confidence=score,
            requires_confirmation=best_command.requires_confirmation,
            raw_text=text,
        )

        # Collect alternatives if there are other close matches
        alternatives = []
        for cmd, phrase, alt_score in matches[1:4]:  # Top 3 alternatives
            if alt_score >= self.LLM_FALLBACK_THRESHOLD:
                alt_entities = self._extract_entities(text, phrase, cmd)
                alternatives.append(VoiceIntent(
                    command_id=cmd.id,
                    action_type=cmd.action_type,
                    action_config={**cmd.action_config, **alt_entities},
                    entities=alt_entities,
                    confidence=alt_score,
                    requires_confirmation=cmd.requires_confirmation,
                    raw_text=text,
                ))

        return ParsedIntent(
            intent=intent,
            matched_phrase=matched_phrase,
            match_reason=self._match_reason_for_phrase(text, matched_phrase),
            match_method="keyword",
            alternatives=alternatives,
        )

    def _match_reason_for_phrase(self, text: str, matched_phrase: str) -> str:
        text_lower = text.lower().strip()
        phrase_lower = matched_phrase.lower().strip()
        if "{" in matched_phrase and "}" in matched_phrase:
            return "phrase_pattern"
        if text_lower == phrase_lower:
            return "phrase_exact"
        return "phrase_prefix"

    def _compile_slot_phrase_pattern(self, phrase: str) -> tuple[Optional[re.Pattern[str]], list[str]]:
        slot_names = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", phrase)
        if not slot_names:
            return None, []
        pattern = "^" + re.sub(
            r"\\\{([a-zA-Z_][a-zA-Z0-9_]*)\\\}",
            r"(?P<\1>.+?)",
            re.escape(phrase),
        ) + "$"
        pattern = re.sub(r"\\ ", r"\\s+", pattern)
        return re.compile(pattern, re.IGNORECASE), slot_names

    def _extract_entities(
        self,
        text: str,
        matched_phrase: str,
        command: VoiceCommand,
    ) -> dict[str, Any]:
        """Extract entities from text based on command configuration."""
        entities = {}
        slot_pattern, slot_names = self._compile_slot_phrase_pattern(matched_phrase)
        if slot_pattern is not None:
            match = slot_pattern.match(text.strip())
            if match:
                for slot_name in slot_names:
                    value = str(match.group(slot_name) or "").strip()
                    if value:
                        entities[slot_name] = value
                if entities:
                    return entities

        text_lower = text.lower()
        phrase_lower = matched_phrase.lower()

        # Extract the remainder after the matched phrase
        if text_lower.startswith(phrase_lower):
            remainder = text[len(matched_phrase):].strip()
            if remainder:
                # Determine entity type based on action config
                if command.action_config.get("extract_query"):
                    entities["query"] = remainder
                elif command.action_config.get("extract_content"):
                    entities["content"] = remainder
                else:
                    entities["argument"] = remainder

        return entities

    async def _pattern_match(
        self,
        text: str,
        user_id: int,
    ) -> Optional[ParsedIntent]:
        """Match text against entity extraction patterns."""
        text_lower = text.lower()

        # Check for search queries
        for pattern in self._entity_patterns["query"]:
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                return ParsedIntent(
                    intent=VoiceIntent(
                        action_type=ActionType.MCP_TOOL,
                        action_config={
                            "tool_name": "media.search",
                            "query": query,
                        },
                        entities={"query": query},
                        confidence=0.8,
                        raw_text=text,
                    ),
                    match_method="pattern",
                )

        # Check for note creation
        for pattern in self._entity_patterns["note_content"]:
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                return ParsedIntent(
                    intent=VoiceIntent(
                        action_type=ActionType.MCP_TOOL,
                        action_config={
                            "tool_name": "notes.create",
                            "content": content,
                        },
                        entities={"content": content},
                        confidence=0.8,
                        raw_text=text,
                    ),
                    match_method="pattern",
                )

        return None

    async def _llm_parse(
        self,
        text: str,
        user_id: int,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[ParsedIntent]:
        """Use LLM to parse complex or ambiguous intents."""
        try:
            # Import here to avoid circular imports and allow graceful degradation
            from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls_Unified import chat_api_call_async

            # Build prompt for intent classification
            available_commands = self.registry.get_all_commands(user_id)
            command_descriptions = "\n".join([
                f"- {cmd.name}: {cmd.description or 'No description'} (triggers: {', '.join(cmd.phrases[:3])})"
                for cmd in available_commands[:10]  # Limit to top 10
            ])

            system_prompt = f"""You are an intent parser for a voice assistant. Classify the user's intent and extract any relevant entities.

Available commands:
{command_descriptions}

If the user's request matches a command, respond with JSON:
{{"intent": "command", "command_name": "<name>", "entities": {{"key": "value"}}, "confidence": 0.0-1.0}}

If it's a general question or chat, respond with JSON:
{{"intent": "chat", "message": "<user message>", "confidence": 0.0-1.0}}

Only respond with valid JSON, no other text."""

            # Add context if available
            messages = []
            if context and context.get("conversation_history"):
                for turn in context["conversation_history"][-3:]:  # Last 3 turns
                    messages.append({
                        "role": turn["role"],
                        "content": turn["content"],
                    })

            messages.append({"role": "user", "content": text})

            # Call LLM
            response = await chat_api_call_async(
                input_data=text,
                custom_prompt=system_prompt,
                api_endpoint="openai",  # Use configured default
                api_key=None,  # Use configured key
                temp=0.1,
                max_tokens=200,
                llm_surface="voice.assistant",
            )

            if not response:
                return None

            # Parse LLM response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                result = json.loads(json_match.group()) if json_match else json.loads(response)

                confidence = float(result.get("confidence", 0.5))

                if result.get("intent") == "command":
                    # Find the matching command
                    cmd_name = result.get("command_name", "").lower()
                    matching_cmd = None
                    for cmd in available_commands:
                        if cmd.name.lower() == cmd_name:
                            matching_cmd = cmd
                            break

                    if matching_cmd:
                        entities = result.get("entities", {})
                        return ParsedIntent(
                            intent=VoiceIntent(
                                command_id=matching_cmd.id,
                                action_type=matching_cmd.action_type,
                                action_config={**matching_cmd.action_config, **entities},
                                entities=entities,
                                confidence=confidence,
                                requires_confirmation=matching_cmd.requires_confirmation,
                                raw_text=text,
                            ),
                            match_method="llm",
                        )

                # Default to chat intent
                return ParsedIntent(
                    intent=VoiceIntent(
                        action_type=ActionType.LLM_CHAT,
                        action_config={"message": text},
                        confidence=confidence,
                        raw_text=text,
                    ),
                    match_method="llm",
                )

            except json.JSONDecodeError:
                logger.debug(f"Failed to parse LLM response as JSON: {response[:100]}")
                return None

        except ImportError:
            logger.warning("LLM API not available for intent parsing")
            return None
        except Exception as e:
            logger.warning(f"LLM intent parsing failed: {e}")
            return None


# Singleton instance
_parser_instance: Optional[IntentParser] = None


def get_intent_parser() -> IntentParser:
    """Get the singleton intent parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = IntentParser()
    return _parser_instance


#
# End of VoiceAssistant/intent_parser.py
#######################################################################################################################
