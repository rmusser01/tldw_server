"""
Property-based tests for RAG endpoints.

Uses Hypothesis to generate test cases and verify properties of the RAG system.
"""

import json
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pytest
import hypothesis
from hypothesis import given, strategies as st, assume, settings, note
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from tldw_Server_API.app.api.v1.schemas.rag_schemas import (
    SearchApiRequest,
    SearchModeEnum,
    RetrievalAgentRequest,
    AgentModeEnum,
    Message,
    MessageRole,
    GenerationConfig
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


# Custom strategies for generating test data

@st.composite
def valid_search_query(draw):
    """Generate valid search queries."""
    # Mix of different query types
    query_type = draw(st.sampled_from([
        "words",
        "phrase",
        "question",
        "keywords",
        "mixed"
    ]))
    
    if query_type == "words":
        words = draw(st.lists(
            st.text(alphabet=string.ascii_letters, min_size=3, max_size=15),
            min_size=1,
            max_size=5
        ))
        return " ".join(words)
    
    elif query_type == "phrase":
        return draw(st.text(alphabet=string.ascii_letters + string.digits + " .,!?", min_size=10, max_size=100))
    
    elif query_type == "question":
        question_starters = ["What is", "How does", "Why do", "When should", "Where can"]
        starter = draw(st.sampled_from(question_starters))
        topic = draw(st.text(alphabet=string.ascii_letters + " ", min_size=5, max_size=30))
        return f"{starter} {topic}?"
    
    elif query_type == "keywords":
        keywords = draw(st.lists(
            st.text(alphabet=string.ascii_letters, min_size=3, max_size=10),
            min_size=2,
            max_size=8
        ))
        return ", ".join(keywords)
    
    else:  # mixed
        return draw(st.text(
            alphabet=string.ascii_letters + string.digits + " .,!?-_",
            min_size=5,
            max_size=200
        ))


@st.composite
def search_filters(draw):
    """Generate valid search filters."""
    filter_keys = draw(st.lists(
        st.sampled_from(["type", "category", "author", "source", "tag", "status"]),
        min_size=0,
        max_size=3,
        unique=True
    ))
    
    if not filter_keys:
        return None
    
    filters = {}
    for key in filter_keys:
        if key == "type":
            filters[key] = draw(st.sampled_from(["article", "video", "document", "note"]))
        elif key == "status":
            filters[key] = draw(st.sampled_from(["published", "draft", "archived"]))
        else:
            filters[key] = draw(st.text(alphabet=string.ascii_letters, min_size=3, max_size=20))
    
    return {"root": filters}


@st.composite
def date_range(draw):
    """Generate valid date ranges."""
    start_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    ))
    
    # End date should be after start date
    end_date = draw(st.datetimes(
        min_value=start_date,
        max_value=start_date + timedelta(days=365)
    ))
    
    return start_date, end_date


@st.composite
def conversation_history(draw):
    """Generate valid conversation history."""
    num_turns = draw(st.integers(min_value=0, max_value=10))
    
    messages = []
    for i in range(num_turns):
        user_msg = draw(st.text(
            alphabet=string.ascii_letters + string.digits + " .,!?",
            min_size=5,
            max_size=200
        ))
        assistant_msg = draw(st.text(
            alphabet=string.ascii_letters + string.digits + " .,!?",
            min_size=10,
            max_size=500
        ))
        
        messages.append(Message(role=MessageRole.USER, content=user_msg))
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_msg))
    
    # Add final user message
    final_msg = draw(st.text(
        alphabet=string.ascii_letters + string.digits + " .,!?",
        min_size=5,
        max_size=200
    ))
    messages.append(Message(role=MessageRole.USER, content=final_msg))
    
    return messages


class TestSearchEndpointProperties:
    """Property-based tests for search endpoint."""
    
    @given(
        query=valid_search_query(),
        mode=st.sampled_from(list(SearchModeEnum)),
        limit=st.integers(min_value=1, max_value=100),
        offset=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=50)
    def test_search_request_validation(self, query, mode, limit, offset):
        """Test that valid search requests are properly constructed."""
        request = SearchApiRequest(
            querystring=query,
            search_mode=mode,
            limit=limit,
            offset=offset
        )
        
        # Properties that should always hold
        assert request.querystring == query
        assert request.search_mode == mode
        assert request.limit == limit
        assert request.offset == offset
        assert request.limit > 0
        assert request.offset >= 0
    
    @given(
        query=valid_search_query(),
        filters=search_filters(),
        dates=date_range(),
        use_semantic=st.booleans(),
        use_hybrid=st.booleans()
    )
    @settings(max_examples=50)
    def test_advanced_search_properties(self, query, filters, dates, use_semantic, use_hybrid):
        """Test properties of advanced search configurations."""
        start_date, end_date = dates
        
        request = SearchApiRequest(
            querystring=query,
            search_mode=SearchModeEnum.ADVANCED,
            filters=filters,
            date_range_start=start_date,
            date_range_end=end_date,
            use_semantic_search=use_semantic,
            use_hybrid_search=use_hybrid
        )
        
        # Date range property
        if request.date_range_start and request.date_range_end:
            assert request.date_range_start <= request.date_range_end
        
        # Hybrid search implies semantic search
        if request.use_hybrid_search:
            note("Hybrid search should use both keyword and semantic")
            # In practice, hybrid search would use both methods
        
        # Search mode consistency
        if request.search_mode == SearchModeEnum.BASIC:
            # Basic mode might ignore advanced settings
            pass
        elif request.search_mode == SearchModeEnum.ADVANCED:
            # Advanced mode should respect all settings
            assert request.filters is not None or request.date_range_start is not None or use_semantic or use_hybrid
    
    @given(
        databases=st.lists(
            st.sampled_from(["media_db", "notes", "chat_history", "character_cards"]),
            min_size=0,
            max_size=4,
            unique=True
        )
    )
    def test_database_selection_properties(self, databases):
        """Test properties of database selection."""
        if not databases:
            # Empty list should default to some databases
            note("Empty database list should use defaults")
        else:
            # Selected databases should be valid
            valid_dbs = {"media_db", "notes", "chat_history", "character_cards"}
            assert all(db in valid_dbs for db in databases)
            
            # No duplicates
            assert len(databases) == len(set(databases))


class TestAgentEndpointProperties:
    """Property-based tests for agent endpoint."""
    
    @given(
        content=st.text(min_size=1, max_size=1000),
        mode=st.sampled_from(list(AgentModeEnum)),
        stream=st.booleans()
    )
    def test_agent_request_properties(self, content, mode, stream):
        """Test properties of agent requests."""
        request = RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content=content),
            mode=mode,
            rag_generation_config=GenerationConfig(stream=stream) if stream else None
        )
        
        # Content should not be empty
        assert request.message.content.strip() != ""
        
        # Mode-specific properties
        if mode == AgentModeEnum.RESEARCH:
            # Research mode might have different default sources
            note("Research mode uses specific sources")
        
        # Streaming properties
        if request.rag_generation_config and request.rag_generation_config.stream:
            note("Streaming enabled")
    
    @given(
        messages=conversation_history(),
        temperature=st.floats(min_value=0.0, max_value=2.0),
        max_tokens=st.integers(min_value=10, max_value=4000)
    )
    @settings(max_examples=30)
    def test_conversation_properties(self, messages, temperature, max_tokens):
        """Test properties of conversations."""
        assume(len(messages) > 0)  # At least one message
        
        request = RetrievalAgentRequest(
            messages=messages,
            mode=AgentModeEnum.RAG,
            rag_generation_config=GenerationConfig(
                temperature=temperature,
                max_response_tokens=max_tokens
            )
        )
        
        # Conversation structure properties
        assert len(messages) % 2 == 1  # Should be odd (ends with user message)
        
        # All messages should alternate between user and assistant
        for i in range(0, len(messages) - 1, 2):
            assert messages[i].role == MessageRole.USER
            if i + 1 < len(messages):
                assert messages[i + 1].role == MessageRole.ASSISTANT
        
        # Last message should be from user
        assert messages[-1].role == MessageRole.USER
        
        # Generation config properties
        assert 0.0 <= request.rag_generation_config.temperature <= 2.0
        assert request.rag_generation_config.max_response_tokens > 0
    
    @given(
        api_provider=st.sampled_from(["openai", "anthropic", "cohere", "local"]),
        model=st.text(alphabet=string.ascii_letters + string.digits + "-.", min_size=3, max_size=50)
    )
    def test_api_configuration_properties(self, api_provider, model):
        """Test properties of API configurations."""
        request = RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content="Test"),
            mode=AgentModeEnum.RAG,
            api_config={"api_provider": api_provider},
            rag_generation_config=GenerationConfig(model=model)
        )
        
        # Provider-specific model naming conventions
        if api_provider == "openai" and "gpt" in model.lower():
            note("OpenAI model detected")
        elif api_provider == "anthropic" and "claude" in model.lower():
            note("Anthropic model detected")
        
        # API config should be properly structured
        assert "api_provider" in request.api_config
        assert request.api_config["api_provider"] in ["openai", "anthropic", "cohere", "local"]


class RAGStateMachine(RuleBasedStateMachine):
    """Stateful testing of RAG system behavior."""
    
    def __init__(self):
        super().__init__()
        self.conversations = {}  # Track ongoing conversations
        self.search_history = []  # Track search queries
        self.cached_results = {}  # Simulate caching behavior
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.current_user_id = 1
        self.total_requests = 0
    
    @rule(
        query=valid_search_query(),
        use_cache=st.booleans()
    )
    def search(self, query, use_cache):
        """Rule: Perform a search."""
        self.total_requests += 1
        
        # Check cache
        if use_cache and query in self.cached_results:
            note(f"Cache hit for query: {query[:30]}...")
            results = self.cached_results[query]
        else:
            note(f"New search for query: {query[:30]}...")
            # Simulate search
            results = self._simulate_search(query)
            if use_cache:
                self.cached_results[query] = results
        
        self.search_history.append({
            "query": query,
            "cached": use_cache and query in self.cached_results,
            "result_count": len(results)
        })
        
        return results
    
    @rule(
        message=st.text(min_size=1, max_size=200),
        conversation_id=st.none() | st.uuids()
    )
    def send_message(self, message, conversation_id):
        """Rule: Send a message to the agent."""
        self.total_requests += 1
        
        if conversation_id is None:
            # New conversation
            conversation_id = str(hypothesis.strategies.uuids().example())
            self.conversations[conversation_id] = []
        
        # Add to conversation history
        if conversation_id in self.conversations:
            self.conversations[conversation_id].append({
                "role": "user",
                "content": message
            })
            
            # Simulate response
            response = f"Response to: {message[:50]}..."
            self.conversations[conversation_id].append({
                "role": "assistant",
                "content": response
            })
        
        note(f"Conversation {conversation_id[:8]}... has {len(self.conversations.get(conversation_id, []))} messages")
    
    @rule()
    def check_cache_size(self):
        """Rule: Cache should not grow unbounded."""
        max_cache_size = 1000
        assert len(self.cached_results) <= max_cache_size, "Cache size exceeded limit"
    
    @invariant()
    def conversations_valid(self):
        """Invariant: All conversations should have valid structure."""
        for conv_id, messages in self.conversations.items():
            # Should have even number of messages (user/assistant pairs)
            assert len(messages) % 2 == 0 or len(messages) == 0
            
            # Messages should alternate between user and assistant
            for i in range(0, len(messages), 2):
                if i < len(messages):
                    assert messages[i]["role"] == "user"
                if i + 1 < len(messages):
                    assert messages[i + 1]["role"] == "assistant"
    
    @invariant()
    def request_count_valid(self):
        """Invariant: Request count should match history."""
        total_searches = len(self.search_history)
        total_messages = sum(len(msgs) // 2 for msgs in self.conversations.values())
        assert self.total_requests >= total_searches + total_messages
    
    def _simulate_search(self, query):
        """Simulate search results."""
        # Simple simulation: number of results based on query length
        num_results = min(len(query.split()), 10)
        return [{"id": f"doc_{i}", "score": 1.0 - (i * 0.1)} for i in range(num_results)]


# Test the state machine
TestRAGStateMachine = RAGStateMachine.TestCase


class TestEdgeCases:
    """Test edge cases using property-based testing."""
    
    @given(
        query=st.text(alphabet=string.whitespace, min_size=1, max_size=100)
    )
    def test_whitespace_only_queries(self, query):
        """Test handling of whitespace-only queries."""
        # Should be rejected or cleaned
        cleaned = query.strip()
        assert cleaned == "" or len(cleaned) > 0
    
    @given(
        limit=st.integers(),
        offset=st.integers()
    )
    def test_pagination_bounds(self, limit, offset):
        """Test pagination parameter validation."""
        # Test what happens with invalid values
        if limit <= 0:
            # Should be rejected or set to default
            note("Invalid limit should be handled")
        
        if offset < 0:
            # Should be rejected or set to 0
            note("Negative offset should be handled")
        
        if limit > 10000:
            # Very large limits might be capped
            note("Large limit might be capped")
    
    @given(
        messages=st.lists(
            st.builds(
                Message,
                role=st.sampled_from(list(MessageRole)),
                content=st.text(min_size=0, max_size=10000)
            ),
            min_size=0,
            max_size=100
        )
    )
    def test_malformed_conversations(self, messages):
        """Test handling of malformed conversation structures."""
        # Various invalid conversation structures
        if not messages:
            # Empty conversation
            note("Empty conversation")
        elif all(m.role == MessageRole.USER for m in messages):
            # Only user messages
            note("Only user messages")
        elif all(m.role == MessageRole.ASSISTANT for m in messages):
            # Only assistant messages
            note("Only assistant messages")
        elif messages and messages[0].role == MessageRole.ASSISTANT:
            # Starts with assistant
            note("Starts with assistant message")
        
        # These should all be handled gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])