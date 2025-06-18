# rag.py - RAG Endpoint
# Description: FastAPI endpoint for Retrieval-Augmented Generation (RAG)
#
# Imports
import json
import logging
import threading
import time
from typing import Optional, Dict, Any, List
from uuid import uuid4
from pathlib import Path
import asyncio
#
# Third-Party Imports
from fastapi import APIRouter, HTTPException, Body, status, Depends
from fastapi.responses import StreamingResponse
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.rag_schemas import (
    RetrievalAgentRequest,
    RetrievalAgentResponse,
    Message,
    MessageRole, SearchApiResponse, SearchApiResultItem, SearchApiRequest, AgentModeEnum, Citation,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
from tldw_Server_API.app.core.config import settings, RAG_SERVICE_CONFIG
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig
#
#######################################################################################################################
#
# Functions:

router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval Agent"],
)

# Thread-safe cache for user-specific RAG services with TTL
class UserRAGServiceCache:
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour TTL
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
    
    def get(self, user_id: int) -> Optional[RAGService]:
        with self._lock:
            if user_id not in self._cache:
                return None
            
            entry = self._cache[user_id]
            if time.time() - entry['timestamp'] > self._ttl:
                # Expired - clean up
                if 'service' in entry:
                    try:
                        entry['service'].cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up RAG service for user {user_id}: {e}")
                del self._cache[user_id]
                return None
            
            return entry['service']
    
    def set(self, user_id: int, service: RAGService) -> None:
        with self._lock:
            # Clean up old service if exists
            if user_id in self._cache:
                old_entry = self._cache[user_id]
                if 'service' in old_entry:
                    try:
                        old_entry['service'].cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up old RAG service for user {user_id}: {e}")
            
            self._cache[user_id] = {
                'service': service,
                'timestamp': time.time()
            }
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries. Returns number of entries cleaned."""
        with self._lock:
            expired_users = []
            current_time = time.time()
            
            for user_id, entry in self._cache.items():
                if current_time - entry['timestamp'] > self._ttl:
                    expired_users.append(user_id)
                    if 'service' in entry:
                        try:
                            entry['service'].cleanup()
                        except Exception as e:
                            logger.warning(f"Error cleaning up RAG service for user {user_id}: {e}")
            
            for user_id in expired_users:
                del self._cache[user_id]
            
            return len(expired_users)
    
    def clear(self) -> None:
        """Clear all cached services."""
        with self._lock:
            for entry in self._cache.values():
                if 'service' in entry:
                    try:
                        entry['service'].cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up RAG service during cache clear: {e}")
            self._cache.clear()

_user_rag_services = UserRAGServiceCache()

# Background task to cleanup expired services
def cleanup_expired_rag_services():
    """Background task to cleanup expired RAG services."""
    try:
        cleaned_count = _user_rag_services.cleanup_expired()
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired RAG services")
    except Exception as e:
        logger.error(f"Error during RAG service cleanup: {e}")


async def get_rag_service_for_user(
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> RAGService:
    """
    Get or create a RAG service instance for the current user.
    
    This creates a user-specific RAG service with proper database paths
    and configuration for multi-user support.
    """
    user_id = current_user.id
    
    # Check if we have a cached service for this user
    cached_service = _user_rag_services.get(user_id)
    if cached_service:
        return cached_service
    
    # Get user-specific paths
    user_dir = Path(settings.get("USER_DB_BASE_DIR")) / str(user_id)
    media_db_path = user_dir / "user_media_library.sqlite"
    chacha_db_path = user_dir / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"
    chroma_path = user_dir / "chroma"
    
    # Create RAG config from settings
    config = RAGConfig()
    
    # Apply settings from RAG_SERVICE_CONFIG
    for key, value in RAG_SERVICE_CONFIG.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                # Handle nested config objects
                config_attr = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(config_attr, nested_key):
                        setattr(config_attr, nested_key, nested_value)
            else:
                setattr(config, key, value)
    
    # Create RAG service for this user
    rag_service = RAGService(
        config=config,
        media_db_path=media_db_path,
        chachanotes_db_path=chacha_db_path,
        chroma_path=chroma_path,
        # LLM handler will be set up based on user's API keys
        llm_handler=None  # Will be configured based on request
    )
    
    # Initialize the service
    await rag_service.initialize()
    
    # Cache for future requests with TTL
    _user_rag_services.set(user_id, rag_service)
    
    logging.info(f"Created RAG service for user {user_id}")
    return rag_service


@router.post(
    "/search",  # Endpoint path
    response_model=SearchApiResponse,
    summary="Perform a comprehensive search",
    description=(
            "Search across various data sources using configurable parameters. "
            "Supports basic, advanced, and custom search modes with options for hybrid, "
            "semantic, and full-text search, filtering, and RAG-specific configurations."
    ),
    status_code=status.HTTP_200_OK,
)
async def perform_search(
        request_body: SearchApiRequest = Body(...),
        rag_service: RAGService = Depends(get_rag_service_for_user),
        current_user: User = Depends(get_request_user)
):
    """
    Processes a search request based on the provided querystring and settings.

    - **`querystring`**: The main search query.
    - **`search_mode`**: "basic", "advanced", or "custom".
    - **`search_settings`**: Optional nested object to define detailed search parameters,
      which can override top-level parameters or defaults for "basic"/"advanced" modes.
    - Top-level parameters (e.g., `limit`, `filters`, `use_hybrid_search`) can also be used directly,
      especially when `search_settings` is not provided or for quick overrides.
    """
    logging.info(f"User {current_user.id}: Received search request for querystring: '{request_body.querystring[:100]}...'")
    logging.debug(f"Full search request payload: {request_body.model_dump_json(indent=2)}")

    try:
        # --- 1. Determine effective search parameters ---
        effective_search_params = request_body
        if request_body.search_settings:
            logging.info("Using provided 'search_settings' to configure search.")
            # Merge search_settings with top-level parameters
            if request_body.search_settings.limit is not None:
                effective_search_params.limit = request_body.search_settings.limit
            if request_body.search_settings.offset is not None:
                effective_search_params.offset = request_body.search_settings.offset
            if request_body.search_settings.filters is not None:
                effective_search_params.filters = request_body.search_settings.filters

        # --- 2. Determine which data sources to search ---
        sources = []
        if hasattr(effective_search_params, 'search_databases') and effective_search_params.search_databases:
            # Map database names to DataSource enums
            source_mapping = {
                "media_db": DataSource.MEDIA_DB,
                "notes": DataSource.NOTES,
                "chat_history": DataSource.CHAT_HISTORY,
                "character_cards": DataSource.CHARACTER_CARDS
            }
            for db_name in effective_search_params.search_databases:
                if db_name.lower() in source_mapping:
                    sources.append(source_mapping[db_name.lower()])
        else:
            # Default to searching media and notes
            sources = [DataSource.MEDIA_DB, DataSource.NOTES]

        # --- 3. Prepare filters from request ---
        filters = {}
        if effective_search_params.filters and effective_search_params.filters.root:
            filters = effective_search_params.filters.root
        
        # Add date range filters if provided
        if hasattr(effective_search_params, 'date_range_start') and effective_search_params.date_range_start:
            filters["date_start"] = effective_search_params.date_range_start
        if hasattr(effective_search_params, 'date_range_end') and effective_search_params.date_range_end:
            filters["date_end"] = effective_search_params.date_range_end

        # --- 4. Perform the actual search using RAG service ---
        # Convert DataSource enums to strings for the RAG service (expects uppercase names)
        source_strings = [source.name for source in sources] if sources else None
        
        search_results = await rag_service.search(
            query=effective_search_params.querystring,
            sources=source_strings,
            filters=filters,
            fts_top_k=effective_search_params.limit,
            vector_top_k=effective_search_params.limit if effective_search_params.use_semantic_search else 0,
            hybrid_alpha=0.5 if effective_search_params.use_hybrid_search else 0.0
        )

        # --- 5. Convert search results to API response format ---
        api_results = []
        for doc in search_results[:effective_search_params.limit]:
            api_results.append(
                SearchApiResultItem(
                    id=doc["id"],
                    score=doc.get("score", 0.0),
                    title=doc["metadata"].get("title", "Untitled"),
                    snippet=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    metadata={
                        "source": doc["source"],
                        **doc["metadata"]
                    }
                )
            )

        # Apply offset if specified
        if effective_search_params.offset:
            api_results = api_results[effective_search_params.offset:]

        # --- 6. Construct and return response ---
        response = SearchApiResponse(
            querystring_echo=request_body.querystring,
            results=api_results[:effective_search_params.limit],
            total_results=len(api_results),
            debug_info={
                "user_id": current_user.id,
                "search_mode_used": effective_search_params.search_mode.value,
                "limit_applied": effective_search_params.limit,
                "offset_applied": effective_search_params.offset,
                "filters_provided": bool(filters),
                "hybrid_search_flag": effective_search_params.use_hybrid_search,
                "sources_searched": [s.name for s in sources],
                "rag_config_provided": bool(effective_search_params.rag_generation_config)
            }
        )

        return response

    except Exception as e:
        logging.error(f"Error performing search for user {current_user.id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while performing the search: {str(e)}"
        )


@router.post(
    "/agent",
    response_model=RetrievalAgentResponse, # Can also be Any or StreamingResponse for flexibility
    summary="RAG-powered Conversational Agent",
    description=(
        "Engage with an intelligent agent for information retrieval, analysis, and research. "
        "This endpoint supports standard RAG (Retrieval-Augmented Generation) as well as "
        "a more advanced 'research' mode with multi-step reasoning and tool use. "
        "It can maintain conversational context."
    ),
    status_code=status.HTTP_200_OK,
)
async def run_retrieval_agent(
    request_body: RetrievalAgentRequest = Body(...),
    rag_service: RAGService = Depends(get_rag_service_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Processes user messages for RAG or research tasks.

    Key parameters:
    - **`message`**: The current user message.
    - **`conversation_id`**: To continue an existing conversation.
    - **`mode`**: Choose between "rag" and "research" modes.
    - **`search_settings`**: Customize how context is retrieved.
    - **`rag_generation_config` / `research_generation_config`**: Fine-tune LLM generation.
    - **`rag_tools` / `research_tools`**: Grant specific capabilities to the agent.
    """
    logging.info(f"User {current_user.id}: Received request for /retrieval/agent with mode: {request_body.mode}")

    # --- 1. Determine current input message and conversation history ---
    current_user_message_obj: Optional[Message] = None
    conversation_history: List[Dict[str, str]] = []

    if request_body.message:
        current_user_message_obj = request_body.message
        logging.info(f"Processing single 'message' field for conversation_id: {request_body.conversation_id}")
        
        # Load conversation history if conversation_id is provided
        if request_body.conversation_id:
            try:
                # Load messages from the conversation using ChaChaNotes DB
                messages = chacha_db.get_messages_for_conversation(
                    request_body.conversation_id, 
                    limit=50,  # Limit to last 50 messages for context
                    order_by_timestamp="ASC"  # Chronological order
                )
                
                # Convert to conversation history format
                for msg in messages:
                    # Skip the current message if it's already in the conversation
                    if msg.get('content') != current_user_message_obj.content:
                        conversation_history.append({
                            "role": msg.get('sender', 'user').lower(),
                            "content": msg.get('content', '')
                        })
                
                logger.info(f"Loaded {len(conversation_history)} messages from conversation {request_body.conversation_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load conversation history for {request_body.conversation_id}: {e}")
                # Continue without history rather than failing the request
                conversation_history = []
            
    elif request_body.messages:  # Deprecated path
        if not request_body.messages:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty 'messages' list provided.")
        current_user_message_obj = request_body.messages[-1]
        # Convert messages to conversation history format
        for msg in request_body.messages[:-1]:
            conversation_history.append({
                "role": msg.role.value,
                "content": msg.content
            })
        logging.warning(f"Processing deprecated 'messages' field. Conversation_id: {request_body.conversation_id}")
    else:
        logging.error("Critical: No message input found despite Pydantic validation.")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Message input is missing.")

    if not current_user_message_obj or not current_user_message_obj.content:
        logging.warning("Received message object with no content.")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Message content cannot be empty.")

    try:
        # --- 2. Prepare search parameters ---
        sources = []
        if request_body.search_settings and hasattr(request_body.search_settings, 'search_databases') and request_body.search_settings.search_databases:
            # Map database names to DataSource enums
            source_mapping = {
                "media_db": DataSource.MEDIA_DB,
                "notes": DataSource.NOTES,
                "chat_history": DataSource.CHAT_HISTORY,
                "character_cards": DataSource.CHARACTER_CARDS
            }
            for db_name in request_body.search_settings.search_databases:
                if db_name.lower() in source_mapping:
                    sources.append(source_mapping[db_name.lower()])
        else:
            # Default sources based on mode
            if request_body.mode == AgentModeEnum.RESEARCH:
                sources = [DataSource.MEDIA_DB, DataSource.NOTES]
            else:  # RAG mode
                sources = [DataSource.MEDIA_DB, DataSource.NOTES, DataSource.CHAT_HISTORY]

        # Prepare filters
        filters = {}
        if request_body.search_settings and request_body.search_settings.filters:
            filters = request_body.search_settings.filters.root

        # --- 3. Configure generation settings ---
        generation_config = request_body.rag_generation_config
        if request_body.mode == AgentModeEnum.RESEARCH and request_body.research_generation_config:
            generation_config = request_body.research_generation_config

        # Check for streaming request
        is_streaming_requested = generation_config and generation_config.stream

        # --- 4. Execute RAG pipeline ---
        if is_streaming_requested:
            logging.info(f"User {current_user.id}: Streaming response requested.")
            
            # Streaming implementation
            async def stream_generator():
                try:
                    # Stream header
                    yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(request_body.conversation_id or uuid4())})}\n\n"
                    
                    # Call RAG service with streaming
                    # Convert DataSource enums to strings for the RAG service
                    source_strings = [source.name for source in sources] if sources else None
                    
                    async for chunk in rag_service.generate_answer_stream(
                        query=current_user_message_obj.content,
                        sources=source_strings,
                        filters=filters,
                        conversation_history=conversation_history,
                        mode=request_body.mode.value,
                        generation_config={
                            "model": generation_config.model if generation_config else None,
                            "temperature": generation_config.temperature if generation_config else 0.7,
                            "max_tokens": generation_config.max_tokens_to_sample if generation_config else 2000,
                            "system_prompt": getattr(generation_config, 'system_prompt', None) if generation_config else None
                        }
                    ):
                        if chunk.get("type") == "content":
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        elif chunk.get("type") == "citation":
                            yield f"data: {json.dumps({'type': 'citation', 'citation': chunk['citation']})}\n\n"
                    
                    # Stream end
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    
                except Exception as e:
                    logging.error(f"Streaming error for user {current_user.id}: {str(e)}")
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        else:
            logging.info(f"User {current_user.id}: Non-streaming response requested.")
            
            # Non-streaming generation
            # Convert DataSource enums to strings for the RAG service
            source_strings = [source.name for source in sources] if sources else None
            
            result = await rag_service.generate_answer(
                query=current_user_message_obj.content,
                sources=source_strings,
                filters=filters,
                conversation_history=conversation_history,
                mode=request_body.mode.value,
                generation_config={
                    "model": generation_config.model if generation_config else None,
                    "temperature": generation_config.temperature if generation_config else 0.7,
                    "max_tokens": generation_config.max_tokens_to_sample if generation_config else 2000,
                    "system_prompt": getattr(generation_config, 'system_prompt', None) if generation_config else None,
                    "api_provider": request_body.api_config.get('api_provider') if request_body.api_config else None,
                    "api_key": request_body.api_config.get('api_key') if request_body.api_config else None
                }
            )

            # --- 5. Construct and Return Response ---
            response_conv_id = request_body.conversation_id if request_body.conversation_id else uuid4()

            agent_final_reply = Message(
                role=MessageRole.ASSISTANT,
                content=result.get("answer", "I couldn't generate a response.")
            )

            # Convert sources to citations
            final_citations = []
            if "sources" in result:
                for source in result["sources"]:
                    final_citations.append(
                        Citation(
                            source_name=source.get("title", "Unknown Source"),
                            document_id=source.get("id"),
                            content=source.get("snippet", ""),
                            link=source.get("link")
                        )
                    )

            return RetrievalAgentResponse(
                conversation_id=response_conv_id,
                response_message=agent_final_reply,
                citations=final_citations if final_citations else None,
                debug_info={
                    "user_id": current_user.id,
                    "mode": request_body.mode.value,
                    "message_content_preview": current_user_message_obj.content[:30] + "..." if len(current_user_message_obj.content) > 30 else current_user_message_obj.content,
                    "search_mode": request_body.search_mode.value,
                    "streaming_requested": is_streaming_requested,
                    "sources_used": [s.name for s in sources],
                    "context_size": result.get("context_size", 0)
                }
            )

    except Exception as e:
        logging.error(f"Error in RAG agent for user {current_user.id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )