"""
Quizzes Module for Unified MCP

CRUD operations for quizzes, questions, and attempts stored in ChaChaNotes DB.
Supports quiz creation, question management, attempt tracking, and AI generation.
"""

import asyncio
import json
from typing import Any, Optional

from loguru import logger

from ....config import load_and_log_configs
from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from ....DB_Management.media_db.api import managed_media_database
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb

_QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)


class QuizzesModule(BaseModule):
    """Quiz management module for MCP"""

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Quizzes module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Quizzes module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        try:
            _ = CharactersRAGDB
            checks["driver_available"] = True
        except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS:
            checks["driver_available"] = False
        try:
            from pathlib import Path
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                base = Path(get_project_root())
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS:
                base = Path(__file__).resolve().parents[5]
            free_gb = get_free_disk_space_gb(base)
            checks["disk_space"] = free_gb > 1
        except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS:
            checks["disk_space"] = False
        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            # Quiz CRUD
            create_tool_definition(
                name="quizzes.list",
                description="List quizzes with optional filters.",
                parameters={
                    "properties": {
                        "q": {"type": "string", "maxLength": 500, "description": "Search query for name/description"},
                        "media_id": {"type": "integer", "description": "Filter by source media ID"},
                        "workspace_tag": {"type": "string", "maxLength": 64, "description": "Filter by workspace tag"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="quizzes.get",
                description="Get a quiz by ID.",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer", "description": "Quiz ID"},
                    },
                    "required": ["quiz_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="quizzes.create",
                description="Create a new quiz.",
                parameters={
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 256},
                        "description": {"type": "string", "maxLength": 2000},
                        "workspace_tag": {"type": "string", "maxLength": 64},
                        "media_id": {"type": "integer", "description": "Optional source media ID"},
                        "time_limit_seconds": {"type": "integer", "minimum": 0},
                        "passing_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="quizzes.update",
                description="Update quiz metadata.",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer"},
                        "updates": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "maxLength": 256},
                                "description": {"type": "string", "maxLength": 2000},
                                "workspace_tag": {"type": "string", "maxLength": 64},
                                "media_id": {"type": "integer"},
                                "time_limit_seconds": {"type": "integer", "minimum": 0},
                                "passing_score": {"type": "integer", "minimum": 0, "maximum": 100},
                            },
                        },
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["quiz_id", "updates"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="quizzes.delete",
                description="Delete a quiz (soft or hard delete).",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer"},
                        "hard_delete": {"type": "boolean", "default": False},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["quiz_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Questions
            create_tool_definition(
                name="quizzes.questions.list",
                description="List questions for a quiz.",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer"},
                        "q": {"type": "string", "maxLength": 500, "description": "Search query"},
                        "include_answers": {"type": "boolean", "default": False, "description": "Include correct answers"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "required": ["quiz_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="quizzes.questions.create",
                description="Add a question to a quiz.",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer"},
                        "question_type": {"type": "string", "enum": ["multiple_choice", "multi_select", "true_false", "fill_blank"]},
                        "question_text": {"type": "string", "minLength": 1, "maxLength": 5000},
                        "options": {"type": "array", "items": {"type": "string"}, "description": "Choices for multiple_choice"},
                        "correct_answer": {
                            "oneOf": [{"type": "integer"}, {"type": "string"}, {"type": "boolean"}, {"type": "array", "items": {"type": "integer"}}],
                            "description": "Index (0-based for MC), text, or boolean for true/false",
                        },
                        "explanation": {"type": "string", "maxLength": 2000},
                        "points": {"type": "integer", "minimum": 1, "default": 1},
                        "order_index": {"type": "integer", "minimum": 0, "default": 0},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["quiz_id", "question_type", "question_text", "correct_answer"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="quizzes.questions.update",
                description="Update a question.",
                parameters={
                    "properties": {
                        "question_id": {"type": "integer"},
                        "updates": {
                            "type": "object",
                            "properties": {
                                "question_type": {"type": "string", "enum": ["multiple_choice", "multi_select", "true_false", "fill_blank"]},
                                "question_text": {"type": "string", "maxLength": 5000},
                                "options": {"type": "array", "items": {"type": "string"}},
                                "correct_answer": {"oneOf": [{"type": "integer"}, {"type": "string"}, {"type": "boolean"}, {"type": "array", "items": {"type": "integer"}}]},
                                "explanation": {"type": "string", "maxLength": 2000},
                                "points": {"type": "integer", "minimum": 1},
                                "order_index": {"type": "integer", "minimum": 0},
                                "tags": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["question_id", "updates"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="quizzes.questions.delete",
                description="Delete a question.",
                parameters={
                    "properties": {
                        "question_id": {"type": "integer"},
                        "hard_delete": {"type": "boolean", "default": False},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["question_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Attempts
            create_tool_definition(
                name="quizzes.attempts.start",
                description="Start a new quiz attempt.",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer"},
                    },
                    "required": ["quiz_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="quizzes.attempts.submit",
                description="Submit answers for a quiz attempt.",
                parameters={
                    "properties": {
                        "attempt_id": {"type": "integer"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question_id": {"type": "integer"},
                                    "user_answer": {"oneOf": [{"type": "integer"}, {"type": "string"}, {"type": "array", "items": {"type": "integer"}}]},
                                    "time_spent_ms": {"type": "integer", "minimum": 0},
                                },
                                "required": ["question_id", "user_answer"],
                            },
                        },
                    },
                    "required": ["attempt_id", "answers"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="quizzes.attempts.list",
                description="List quiz attempts.",
                parameters={
                    "properties": {
                        "quiz_id": {"type": "integer", "description": "Filter by quiz ID"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="quizzes.attempts.get",
                description="Get attempt details with results.",
                parameters={
                    "properties": {
                        "attempt_id": {"type": "integer"},
                        "include_questions": {"type": "boolean", "default": False},
                        "include_answers": {"type": "boolean", "default": False},
                    },
                    "required": ["attempt_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            # Generation
            create_tool_definition(
                name="quizzes.generate",
                description="AI-generate a quiz from media content.",
                parameters={
                    "properties": {
                        "media_id": {"type": "integer", "description": "Source media ID"},
                        "name": {"type": "string", "maxLength": 256, "description": "Quiz name"},
                        "num_questions": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                        "question_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["multiple_choice", "multi_select", "true_false", "fill_blank"]},
                            "default": ["multiple_choice"],
                        },
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"], "default": "medium"},
                        "focus_topics": {"type": "array", "items": {"type": "string"}, "description": "Topics to focus on"},
                        "provider": {"type": "string", "description": "LLM provider override"},
                        "model": {"type": "string", "description": "LLM model to use for generation"},
                    },
                    "required": ["media_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "quizzes.list":
            q = arguments.get("q")
            if q is not None and (not isinstance(q, str) or len(q) > 500):
                raise ValueError("q must be a string <= 500 chars")
            limit = int(arguments.get("limit", 50))
            offset = int(arguments.get("offset", 0))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
            if offset < 0:
                raise ValueError("offset must be >= 0")
        elif tool_name == "quizzes.get":
            quiz_id = arguments.get("quiz_id")
            if not isinstance(quiz_id, int) or quiz_id < 1:
                raise ValueError("quiz_id must be a positive integer")
        elif tool_name == "quizzes.create":
            self._validate_quiz_payload(arguments, require_name=True)
        elif tool_name == "quizzes.update":
            quiz_id = arguments.get("quiz_id")
            if not isinstance(quiz_id, int) or quiz_id < 1:
                raise ValueError("quiz_id must be a positive integer")
            updates = arguments.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("updates must be a non-empty object")
        elif tool_name == "quizzes.delete":
            quiz_id = arguments.get("quiz_id")
            if not isinstance(quiz_id, int) or quiz_id < 1:
                raise ValueError("quiz_id must be a positive integer")
        elif tool_name == "quizzes.questions.list":
            quiz_id = arguments.get("quiz_id")
            if not isinstance(quiz_id, int) or quiz_id < 1:
                raise ValueError("quiz_id must be a positive integer")
            limit = int(arguments.get("limit", 50))
            if limit < 1 or limit > 200:
                raise ValueError("limit must be 1..200")
        elif tool_name == "quizzes.questions.create":
            quiz_id = arguments.get("quiz_id")
            if not isinstance(quiz_id, int) or quiz_id < 1:
                raise ValueError("quiz_id must be a positive integer")
            self._validate_question_payload(arguments, require_core_fields=True)
        elif tool_name == "quizzes.questions.update":
            qid = arguments.get("question_id")
            if not isinstance(qid, int) or qid < 1:
                raise ValueError("question_id must be a positive integer")
            updates = arguments.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("updates must be a non-empty object")
        elif tool_name == "quizzes.questions.delete":
            qid = arguments.get("question_id")
            if not isinstance(qid, int) or qid < 1:
                raise ValueError("question_id must be a positive integer")
        elif tool_name == "quizzes.attempts.start":
            quiz_id = arguments.get("quiz_id")
            if not isinstance(quiz_id, int) or quiz_id < 1:
                raise ValueError("quiz_id must be a positive integer")
        elif tool_name == "quizzes.attempts.submit":
            attempt_id = arguments.get("attempt_id")
            if not isinstance(attempt_id, int) or attempt_id < 1:
                raise ValueError("attempt_id must be a positive integer")
            answers = arguments.get("answers")
            if not isinstance(answers, list):
                raise ValueError("answers must be a list")
            for ans in answers:
                if not isinstance(ans, dict):
                    raise ValueError("each answer must be an object")
                if "question_id" not in ans or "user_answer" not in ans:
                    raise ValueError("each answer must have question_id and user_answer")
        elif tool_name == "quizzes.attempts.list":
            limit = int(arguments.get("limit", 50))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
        elif tool_name == "quizzes.attempts.get":
            attempt_id = arguments.get("attempt_id")
            if not isinstance(attempt_id, int) or attempt_id < 1:
                raise ValueError("attempt_id must be a positive integer")
        elif tool_name == "quizzes.generate":
            media_id = arguments.get("media_id")
            if not isinstance(media_id, int) or media_id < 1:
                raise ValueError("media_id must be a positive integer")
            num_q = arguments.get("num_questions", 10)
            if not isinstance(num_q, int) or num_q < 1 or num_q > 50:
                raise ValueError("num_questions must be 1..50")
            provider = arguments.get("provider")
            if provider is not None and (not isinstance(provider, str) or not provider.strip()):
                raise ValueError("provider must be a non-empty string")

    def _validate_quiz_payload(self, quiz: dict[str, Any], *, require_name: bool = False) -> None:
        if "name" in quiz:
            name = quiz.get("name")
            if not isinstance(name, str) or not (1 <= len(name.strip()) <= 256):
                raise ValueError("name must be 1..256 chars")
        elif require_name:
            raise ValueError("name must be 1..256 chars")

        description = quiz.get("description")
        if description is not None and (not isinstance(description, str) or len(description) > 2000):
            raise ValueError("description must be <= 2000 chars")

        workspace_tag = quiz.get("workspace_tag")
        if workspace_tag is not None and (not isinstance(workspace_tag, str) or len(workspace_tag) > 64):
            raise ValueError("workspace_tag must be <= 64 chars")

        time_limit_seconds = quiz.get("time_limit_seconds")
        if time_limit_seconds is not None and (
            isinstance(time_limit_seconds, bool)
            or not isinstance(time_limit_seconds, int)
            or time_limit_seconds < 0
        ):
            raise ValueError("time_limit_seconds must be a non-negative integer")

        passing_score = quiz.get("passing_score")
        if passing_score is not None and (not isinstance(passing_score, int) or passing_score < 0 or passing_score > 100):
            raise ValueError("passing_score must be 0..100")

    def _validate_question_payload(self, question: dict[str, Any], *, require_core_fields: bool = False) -> None:
        qtype = question.get("question_type")
        valid_types = {"multiple_choice", "multi_select", "true_false", "fill_blank"}
        if "question_type" in question:
            if qtype not in valid_types:
                raise ValueError("question_type must be multiple_choice, multi_select, true_false, or fill_blank")
        elif require_core_fields:
            raise ValueError("question_type must be multiple_choice, multi_select, true_false, or fill_blank")

        if "question_text" in question:
            qtext = question.get("question_text")
            if not isinstance(qtext, str) or not (1 <= len(qtext.strip()) <= 5000):
                raise ValueError("question_text must be 1..5000 chars")
        elif require_core_fields:
            raise ValueError("question_text must be 1..5000 chars")

        if qtype == "multiple_choice":
            opts = question.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                raise ValueError("multiple_choice requires at least 2 options")
            ans = question.get("correct_answer")
            if not isinstance(ans, int) or ans < 0 or ans >= len(opts):
                raise ValueError("correct_answer must be valid option index")
        elif qtype == "multi_select":
            opts = question.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                raise ValueError("multi_select requires at least 2 options")
            ans = question.get("correct_answer")
            if not isinstance(ans, list) or len(ans) == 0:
                raise ValueError("correct_answer must be a non-empty index list for multi_select questions")
            if not all(isinstance(entry, int) and 0 <= entry < len(opts) for entry in ans):
                raise ValueError("correct_answer entries must be valid option indices for multi_select")
        elif qtype == "true_false":
            ans = question.get("correct_answer")
            if isinstance(ans, str):
                if ans.lower() not in {"true", "false"}:
                    raise ValueError("correct_answer must be 'true' or 'false'")
            elif isinstance(ans, bool):
                pass
            else:
                raise ValueError("correct_answer must be true/false for true_false questions")
        elif qtype == "fill_blank":
            ans = question.get("correct_answer")
            if not isinstance(ans, str) or not ans.strip():
                raise ValueError("correct_answer must be a non-empty string for fill_blank")

        points = question.get("points")
        if points is not None and (not isinstance(points, int) or points < 1):
            raise ValueError("points must be a positive integer")

        order_index = question.get("order_index")
        if order_index is not None and (not isinstance(order_index, int) or order_index < 0):
            raise ValueError("order_index must be a non-negative integer")

    def _cleanup_generated_quiz(self, db: CharactersRAGDB, quiz_id: int, *, reason: str) -> bool:
        try:
            deleted = db.delete_quiz(quiz_id, hard_delete=True)
        except Exception as exc:
            logger.error(
                f"Exception during cleanup of generated quiz {quiz_id} after {reason}: {exc}",
                exc_info=True,
            )
            return False
        if not deleted:
            logger.error(f"Failed to clean up generated quiz {quiz_id} after {reason}")
            return False
        return True

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except (TypeError, ValueError) as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve

        if tool_name == "quizzes.list":
            return await self._list_quizzes(args, context)
        if tool_name == "quizzes.get":
            return await self._get_quiz(args, context)
        if tool_name == "quizzes.create":
            return await self._create_quiz(args, context)
        if tool_name == "quizzes.update":
            return await self._update_quiz(args, context)
        if tool_name == "quizzes.delete":
            return await self._delete_quiz(args, context)
        if tool_name == "quizzes.questions.list":
            return await self._list_questions(args, context)
        if tool_name == "quizzes.questions.create":
            return await self._create_question(args, context)
        if tool_name == "quizzes.questions.update":
            return await self._update_question(args, context)
        if tool_name == "quizzes.questions.delete":
            return await self._delete_question(args, context)
        if tool_name == "quizzes.attempts.start":
            return await self._start_attempt(args, context)
        if tool_name == "quizzes.attempts.submit":
            return await self._submit_attempt(args, context)
        if tool_name == "quizzes.attempts.list":
            return await self._list_attempts(args, context)
        if tool_name == "quizzes.attempts.get":
            return await self._get_attempt(args, context)
        if tool_name == "quizzes.generate":
            return await self._generate_quiz(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _open_db(self, context: Any) -> CharactersRAGDB:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Quizzes access")
        chacha_path = context.db_paths.get("chacha")
        if not chacha_path:
            raise ValueError("ChaChaNotes DB path not available in context")
        return CharactersRAGDB(db_path=chacha_path, client_id=f"mcp_quizzes_{self.config.name}")

    def _get_client_id(self, context: Any) -> str:
        try:
            return context.client_id or "mcp_quizzes"
        except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS:
            return "mcp_quizzes"

    # Quiz CRUD

    async def _list_quizzes(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        q = args.get("q")
        media_id = args.get("media_id")
        workspace_tag = args.get("workspace_tag")
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        return await asyncio.to_thread(
            self._list_quizzes_sync, context, q, media_id, workspace_tag, limit, offset
        )

    def _list_quizzes_sync(
        self,
        context: Any,
        q: Optional[str],
        media_id: Optional[int],
        workspace_tag: Optional[str],
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            result = db.list_quizzes(
                q=q,
                media_id=media_id,
                workspace_tag=workspace_tag,
                limit=limit,
                offset=offset,
            )
            items = result.get("items", [])
            count = result.get("count", 0)
            has_more = offset + len(items) < count
            return {
                "quizzes": items,
                "total": count,
                "has_more": has_more,
                "next_offset": offset + len(items) if has_more else None,
            }
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _get_quiz(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        quiz_id = args.get("quiz_id")
        return await asyncio.to_thread(self._get_quiz_sync, context, quiz_id)

    def _get_quiz_sync(self, context: Any, quiz_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            quiz = db.get_quiz(quiz_id)
            if not quiz:
                raise ValueError(f"Quiz not found: {quiz_id}")
            return {"quiz": quiz}
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _create_quiz(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_quiz_sync, context, args)

    def _create_quiz_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            self._validate_quiz_payload(args, require_name=True)
            quiz_id = db.create_quiz(
                name=args.get("name"),
                description=args.get("description"),
                workspace_tag=args.get("workspace_tag"),
                media_id=args.get("media_id"),
                time_limit_seconds=args.get("time_limit_seconds"),
                passing_score=args.get("passing_score"),
                client_id=self._get_client_id(context),
            )
            quiz = db.get_quiz(quiz_id)
            return {"quiz_id": quiz_id, "success": True, "quiz": quiz}
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _update_quiz(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_quiz_sync, context, args)

    def _update_quiz_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            quiz_id = args.get("quiz_id")
            updates = dict(args.get("updates", {}))
            expected_version = args.get("expected_version")
            existing = db.get_quiz(quiz_id, include_deleted=False)
            if not existing:
                raise ValueError(f"Quiz not found or version conflict: {quiz_id}")
            merged = dict(existing)
            merged.update(updates)
            self._validate_quiz_payload(merged)
            if expected_version is not None:
                updates["expected_version"] = expected_version
            success = db.update_quiz(
                quiz_id=quiz_id,
                updates=updates,
                client_id=self._get_client_id(context),
            )
            if not success:
                raise ValueError(f"Quiz not found or version conflict: {quiz_id}")
            return {"quiz_id": quiz_id, "success": True, "updated_fields": list(updates.keys())}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _delete_quiz(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._delete_quiz_sync, context, args)

    def _delete_quiz_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            quiz_id = args.get("quiz_id")
            hard_delete = bool(args.get("hard_delete", False))
            expected_version = args.get("expected_version")
            success = db.delete_quiz(
                quiz_id=quiz_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
            if not success:
                raise ValueError(f"Quiz not found: {quiz_id}")
            return {
                "quiz_id": quiz_id,
                "action": "permanently_deleted" if hard_delete else "soft_deleted",
                "success": True,
            }
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Questions

    async def _list_questions(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_questions_sync, context, args)

    def _list_questions_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            quiz_id = args.get("quiz_id")
            q = args.get("q")
            include_answers = bool(args.get("include_answers", False))
            limit = int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            result = db.list_questions(
                quiz_id=quiz_id,
                q=q,
                include_answers=include_answers,
                limit=limit,
                offset=offset,
            )
            items = result.get("items", [])
            count = result.get("count", 0)
            has_more = offset + len(items) < count
            return {
                "questions": items,
                "total": count,
                "has_more": has_more,
                "next_offset": offset + len(items) if has_more else None,
            }
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _create_question(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_question_sync, context, args)

    def _create_question_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            self._validate_question_payload(args, require_core_fields=True)
            question_id = db.create_question(
                quiz_id=args.get("quiz_id"),
                question_type=args.get("question_type"),
                question_text=args.get("question_text"),
                correct_answer=args.get("correct_answer"),
                options=args.get("options"),
                explanation=args.get("explanation"),
                points=args.get("points", 1),
                order_index=args.get("order_index", 0),
                tags=args.get("tags"),
                client_id=self._get_client_id(context),
            )
            question = db.get_question(question_id, include_deleted=False)
            return {"question_id": question_id, "success": True, "question": question}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _update_question(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_question_sync, context, args)

    def _update_question_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            question_id = args.get("question_id")
            updates = dict(args.get("updates", {}))
            expected_version = args.get("expected_version")
            existing = db.get_question(question_id, include_deleted=False)
            if not existing:
                raise ValueError(f"Question not found or version conflict: {question_id}")
            merged = dict(existing)
            merged.update(updates)
            self._validate_question_payload(merged)
            if expected_version is not None:
                updates["expected_version"] = expected_version
            success = db.update_question(
                question_id=question_id,
                updates=updates,
                client_id=self._get_client_id(context),
            )
            if not success:
                raise ValueError(f"Question not found or version conflict: {question_id}")
            return {"question_id": question_id, "success": True, "updated_fields": list(updates.keys())}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _delete_question(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._delete_question_sync, context, args)

    def _delete_question_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            question_id = args.get("question_id")
            hard_delete = bool(args.get("hard_delete", False))
            expected_version = args.get("expected_version")
            success = db.delete_question(
                question_id=question_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
            if not success:
                raise ValueError(f"Question not found: {question_id}")
            return {
                "question_id": question_id,
                "action": "permanently_deleted" if hard_delete else "soft_deleted",
                "success": True,
            }
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Attempts

    async def _start_attempt(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._start_attempt_sync, context, args)

    def _start_attempt_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            quiz_id = args.get("quiz_id")
            attempt = db.start_attempt(
                quiz_id=quiz_id,
                client_id=self._get_client_id(context),
            )
            return {"attempt": attempt, "success": True}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _submit_attempt(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._submit_attempt_sync, context, args)

    def _submit_attempt_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            attempt_id = args.get("attempt_id")
            answers = args.get("answers", [])
            result = db.submit_attempt(
                attempt_id=attempt_id,
                answers=answers,
            )
            return {"result": result, "success": True}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _list_attempts(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_attempts_sync, context, args)

    def _list_attempts_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            quiz_id = args.get("quiz_id")
            limit = int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            result = db.list_attempts(
                quiz_id=quiz_id,
                limit=limit,
                offset=offset,
            )
            items = result.get("items", [])
            count = result.get("count", 0)
            has_more = offset + len(items) < count
            return {
                "attempts": items,
                "total": count,
                "has_more": has_more,
                "next_offset": offset + len(items) if has_more else None,
            }
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _get_attempt(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_attempt_sync, context, args)

    def _get_attempt_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            attempt_id = args.get("attempt_id")
            include_questions = bool(args.get("include_questions", False))
            include_answers = bool(args.get("include_answers", False))
            attempt = db.get_attempt(
                attempt_id=attempt_id,
                include_questions=include_questions,
                include_answers=include_answers,
            )
            if not attempt:
                raise ValueError(f"Attempt not found: {attempt_id}")
            return {"attempt": attempt}
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Generation

    async def _generate_quiz(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """AI-generate a quiz from media content."""
        media_id = args.get("media_id")
        name = args.get("name") or f"Quiz from Media {media_id}"
        num_questions = int(args.get("num_questions", 10))
        question_types = args.get("question_types") or ["multiple_choice"]
        difficulty = args.get("difficulty", "medium")
        focus_topics = args.get("focus_topics")
        provider, model = self._resolve_llm_settings(args)

        # Get media content
        media_path = context.db_paths.get("media") if context and hasattr(context, "db_paths") else None
        if not media_path:
            raise ValueError("Media DB path not available")

        media_content = await asyncio.to_thread(
            self._get_media_content, media_path, media_id
        )

        if not media_content:
            raise ValueError(f"Media not found or no content: {media_id}")

        # Build prompt for LLM
        prompt = self._build_generation_prompt(
            content=media_content,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            focus_topics=focus_topics,
        )

        # Call LLM to generate questions
        try:
            response_text = await self._call_llm(prompt, provider=provider, model=model)
            questions_data = self._parse_generated_questions(response_text)
        except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Quiz generation failed: {e}")
            raise ValueError(f"Failed to generate quiz: {e}") from e

        # Create quiz and questions
        return await asyncio.to_thread(
            self._create_generated_quiz_sync,
            context,
            name,
            media_id,
            questions_data,
        )

    def _get_media_content(self, media_path: str, media_id: int) -> Optional[str]:
        """Get media content for quiz generation."""
        try:
            with managed_media_database(
                "mcp_quizzes_gen",
                db_path=media_path,
                initialize=False,
            ) as db:
                media = db.get_media_by_id(media_id)
                if not media:
                    return None
                return media.get("content") or media.get("transcript") or media.get("summary")
        except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get media content: {e}")
            return None

    def _build_generation_prompt(
        self,
        content: str,
        num_questions: int,
        question_types: list[str],
        difficulty: str,
        focus_topics: Optional[list[str]],
    ) -> str:
        types_str = ", ".join(question_types)
        topics_str = f"\nFocus on these topics: {', '.join(focus_topics)}" if focus_topics else ""

        return f"""Based on the following content, generate {num_questions} quiz questions.

Question types to include: {types_str}
Difficulty level: {difficulty}{topics_str}

Content:
{content[:10000]}

Return the questions as a JSON array with this structure:
[
  {{
    "question_type": "multiple_choice|true_false|fill_blank",
    "question_text": "The question text",
    "options": ["A", "B", "C", "D"],  // only for multiple_choice
    "correct_answer": 0,  // index for MC, "true"/"false" for TF, text for fill_blank
    "explanation": "Why this is the correct answer",
    "points": 1
  }}
]

Return ONLY the JSON array, no other text."""

    def _parse_generated_questions(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into question data."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated questions: {e}")
            raise ValueError("Failed to parse generated questions from LLM response") from e

    def _resolve_llm_settings(self, args: dict[str, Any]) -> tuple[str, Optional[str]]:
        """Resolve provider/model for quiz generation from args or config defaults."""
        provider = args.get("provider")
        model = args.get("model")
        try:
            settings = load_and_log_configs() or {}
        except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS:
            settings = {}
        if not provider:
            provider = str(settings.get("default_api") or "").strip() or None
        if not provider:
            provider = str(settings.get("RAG_DEFAULT_LLM_PROVIDER") or "").strip() or None
        if not provider:
            provider = "openai"
        if not model:
            model = str(settings.get("RAG_DEFAULT_LLM_MODEL") or "").strip() or None
        return provider, model

    async def _call_llm(self, prompt: str, *, provider: str, model: Optional[str]) -> str:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        system_prompt = (
            "You are a quiz generation assistant. Generate questions in valid JSON format."
        )
        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            provider=provider,
            model=model,
            messages=messages,
            system_message=system_prompt,
        )
        return self._extract_llm_text(response)

    def _extract_llm_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "choices" in response and isinstance(response["choices"], list) and response["choices"]:
                choice = response["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, list):
                            return "".join(
                                str(part.get("text", "")) for part in content if isinstance(part, dict)
                            )
                        if content is not None:
                            return str(content)
                    if "text" in choice:
                        return str(choice.get("text"))
            if "content" in response:
                content = response.get("content")
                if isinstance(content, list):
                    return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
                return str(content)
            if "output_text" in response:
                return str(response.get("output_text"))
        return str(response)

    def _create_generated_quiz_sync(
        self,
        context: Any,
        name: str,
        media_id: int,
        questions_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create quiz and questions from generated data."""
        db = self._open_db(context)
        try:
            client_id = self._get_client_id(context)
            valid_questions: list[dict[str, Any]] = []
            for i, q in enumerate(questions_data):
                try:
                    self._validate_question_payload(q, require_core_fields=True)
                    valid_questions.append(q)
                except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Failed to validate generated question {i}: {e}")

            if not valid_questions:
                raise ValueError("Failed to generate quiz: no valid questions were created")

            quiz_id = db.create_quiz(
                name=name,
                description=f"AI-generated quiz from media {media_id}",
                media_id=media_id,
                client_id=client_id,
            )

            created_questions = []
            try:
                for i, q in enumerate(valid_questions):
                    try:
                        qid = db.create_question(
                            quiz_id=quiz_id,
                            question_type=q.get("question_type", "multiple_choice"),
                            question_text=q.get("question_text"),
                            correct_answer=q.get("correct_answer"),
                            options=q.get("options"),
                            explanation=q.get("explanation"),
                            points=q.get("points", 1),
                            order_index=i,
                            client_id=client_id,
                        )
                        created_questions.append(qid)
                    except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Failed to create question {i}: {e}")
            except Exception:
                self._cleanup_generated_quiz(
                    db,
                    quiz_id,
                    reason="unexpected question persistence failure",
                )
                raise

            if not created_questions:
                self._cleanup_generated_quiz(
                    db,
                    quiz_id,
                    reason="question persistence failure",
                )
                raise ValueError("Failed to generate quiz: no valid questions were created")

            quiz = db.get_quiz(quiz_id)
            return {
                "quiz_id": quiz_id,
                "quiz": quiz,
                "questions_created": len(created_questions),
                "success": True,
            }
        finally:
            try:
                db.close_all_connections()
            except _QUIZZES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")
