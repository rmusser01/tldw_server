"""
Example of how to integrate the RAG service into the TUI application.

This shows how the RAG service can be used within the existing
tldw_chatbook TUI event handlers and UI components.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import asyncio

from textual.app import ComposeResult
from textual.widgets import Static, Input, Button
from textual.containers import Container, Vertical
from loguru import logger

from .integration import RAGService


class RAGSearchWidget(Container):
    """
    Example widget for RAG search in the TUI.
    
    This could be integrated into the existing chat or search tabs.
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rag_service = rag_service
    
    def compose(self) -> ComposeResult:
        """Compose the RAG search interface."""
        with Vertical():
            yield Input(
                placeholder="Ask a question about your media, chats, or notes...",
                id="rag-query-input"
            )
            yield Button("Search with RAG", id="rag-search-button")
            yield Static("", id="rag-results")


class RAGEventHandlers:
    """
    Example event handlers for RAG functionality in the TUI.
    
    These methods would be integrated into the main app or relevant event handler classes.
    """
    
    def __init__(self, app, rag_service: RAGService):
        self.app = app
        self.rag_service = rag_service
    
    async def handle_rag_search(self, query: str) -> None:
        """Handle RAG search request from the TUI."""
        try:
            # Update UI to show loading state
            results_widget = self.app.query_one("#rag-results", Static)
            results_widget.update("🔍 Searching...")
            
            # Get user preferences for sources
            search_media = self.app.config.get("rag_search_media", True)
            search_chat = self.app.config.get("rag_search_chat", True)
            search_notes = self.app.config.get("rag_search_notes", True)
            
            sources = []
            if search_media:
                sources.append("MEDIA_DB")
            if search_chat:
                sources.append("CHAT_HISTORY")
            if search_notes:
                sources.append("NOTES")
            
            # Perform RAG search
            result = await self.rag_service.generate_answer(
                query=query,
                sources=sources,
                filters=self._get_search_filters()
            )
            
            # Format and display results
            formatted_result = self._format_rag_result(result)
            results_widget.update(formatted_result)
            
            # Log for debugging
            logger.info(f"RAG search completed: {len(result['sources'])} sources used")
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            results_widget.update(f"❌ Error: {str(e)}")
    
    async def handle_rag_search_streaming(self, query: str) -> None:
        """Handle streaming RAG response for better UX."""
        try:
            results_widget = self.app.query_one("#rag-results", Static)
            results_widget.update("🔍 Searching and generating response...")
            
            # This would integrate with the streaming generator
            # For now, we'll simulate streaming
            result = await self.rag_service.generate_answer(
                query=query,
                stream=True
            )
            
            # In real implementation, this would update progressively
            results_widget.update(result["answer"])
            
        except Exception as e:
            logger.error(f"Streaming RAG error: {e}")
            results_widget.update(f"❌ Error: {str(e)}")
    
    def _get_search_filters(self) -> Dict[str, Any]:
        """Get search filters based on current UI state."""
        filters = {}
        
        # Example: Get active character for chat search
        if hasattr(self.app, 'current_chat_active_character_data'):
            char_data = self.app.current_chat_active_character_data
            if char_data:
                filters["character_id"] = char_data.get("id")
        
        # Example: Date range filter
        # Could be from a date picker widget
        # filters["date_range"] = "last_7_days"
        
        return filters
    
    def _format_rag_result(self, result: Dict[str, Any]) -> str:
        """Format RAG result for display in TUI."""
        lines = []
        
        # Add the answer
        lines.append("📝 Answer:")
        lines.append("")
        lines.append(result["answer"])
        lines.append("")
        
        # Add sources
        if result["sources"]:
            lines.append("📚 Sources:")
            for i, source in enumerate(result["sources"][:5], 1):
                source_emoji = {
                    "MEDIA_DB": "🎬",
                    "CHAT_HISTORY": "💬",
                    "NOTES": "📝",
                    "CHARACTER_CARDS": "👤"
                }.get(source["source"], "📄")
                
                lines.append(f"{i}. {source_emoji} {source['title']} (score: {source['score']:.2f})")
        
        # Add metadata
        lines.append("")
        lines.append(f"⏱️ Generated in {result['metadata']['elapsed_time']:.2f}s")
        
        return "\n".join(lines)


async def integrate_rag_into_chat_tab(app, chat_tab_widget):
    """
    Example of how to integrate RAG into the existing chat tab.
    
    This would be called during app initialization.
    """
    # Initialize RAG service
    rag_service = RAGService(
        config_path=app.config_path,
        media_db_path=app.media_db_path,
        chachanotes_db_path=app.chachanotes_db_path,
        llm_handler=app.llm_handler  # Assuming the app has an LLM handler
    )
    
    # Initialize the service
    await rag_service.initialize()
    
    # Store reference in app
    app.rag_service = rag_service
    
    # Add RAG toggle to chat settings
    # This would be in the actual chat UI composition
    
    # Add event handler
    @app.on(Button.Pressed, "#rag-mode-toggle")
    async def toggle_rag_mode(event):
        """Toggle RAG mode in chat."""
        app.chat_rag_mode = not getattr(app, 'chat_rag_mode', False)
        
        if app.chat_rag_mode:
            event.button.label = "RAG Mode: ON"
            app.notify("RAG mode enabled - responses will use your media, chats, and notes")
        else:
            event.button.label = "RAG Mode: OFF"
            app.notify("RAG mode disabled - using standard chat")
    
    # Modify the send message handler to use RAG when enabled
    original_send_handler = app.handle_send_message
    
    async def enhanced_send_handler(message: str):
        """Enhanced send handler with RAG support."""
        if getattr(app, 'chat_rag_mode', False) and app.rag_service:
            # Use RAG to enhance the response
            rag_result = await app.rag_service.generate_answer(
                query=message,
                sources=["MEDIA_DB", "CHAT_HISTORY", "NOTES"]
            )
            
            # Add context to the message for the LLM
            enhanced_prompt = f"""
Based on the following context from the user's data:

{rag_result['context_preview']}

User question: {message}

Please provide a helpful response using the context above.
"""
            
            # Call original handler with enhanced prompt
            return await original_send_handler(enhanced_prompt)
        else:
            # Normal chat mode
            return await original_send_handler(message)
    
    app.handle_send_message = enhanced_send_handler
    
    logger.info("RAG service integrated into chat tab")


class RAGQuickSearchCommand:
    """
    Example of a quick search command for the TUI.
    
    This could be triggered by a keyboard shortcut.
    """
    
    def __init__(self, app, rag_service: RAGService):
        self.app = app
        self.rag_service = rag_service
    
    async def execute(self, query: str) -> None:
        """Execute quick RAG search and show results in a modal."""
        try:
            # Show loading modal
            self.app.push_screen("loading", "Searching...")
            
            # Perform search
            results = await self.rag_service.search(
                query=query,
                sources=None,  # Search all sources
                filters={}
            )
            
            # Show results in a modal or sidebar
            self.app.pop_screen()  # Remove loading
            self.app.push_screen(
                "rag_results",
                self._create_results_screen(query, results)
            )
            
        except Exception as e:
            logger.error(f"Quick search error: {e}")
            self.app.pop_screen()
            self.app.notify(f"Search failed: {str(e)}", severity="error")
    
    def _create_results_screen(self, query: str, results: list):
        """Create a screen showing search results."""
        # This would create an actual Textual Screen
        # For now, just a placeholder
        return f"Results for '{query}': {len(results)} documents found"


# Example configuration for config.toml
RAG_CONFIG_EXAMPLE = """
# Add this to your config.toml file

[rag]
# Performance settings
batch_size = 32
num_workers = 4  # Reduced for single-user TUI
use_gpu = false  # Set to true if you have GPU
log_level = "INFO"
log_performance_metrics = true

[rag.retriever]
fts_top_k = 10        # Number of results from full-text search
vector_top_k = 10     # Number of results from vector search
hybrid_alpha = 0.7    # Weight for vector search (0=FTS only, 1=vector only)
chunk_size = 512      # Size of text chunks for processing
chunk_overlap = 128   # Overlap between chunks

[rag.processor]
enable_reranking = true              # Use FlashRank for better results
reranker_top_k = 5                   # Keep top N after reranking
deduplication_threshold = 0.85       # Similarity threshold for dedup
max_context_length = 4096            # Max tokens for context
combination_method = "weighted"      # How to combine scores

[rag.generator]
default_temperature = 0.7
max_tokens = 1024
enable_streaming = true              # Stream responses in TUI
system_prompt_template = '''You are a helpful AI assistant with access to the user's personal knowledge base.
Use the provided context to give accurate, relevant answers.
If the context doesn't contain the information needed, say so clearly.

Context:
{context}

Question: {question}
Answer:'''

[rag.cache]
enable_cache = true      # Cache for better performance
cache_ttl = 3600        # 1 hour
max_cache_size = 1000   # Maximum cached items
cache_search_results = true
cache_embedding_results = true

[rag.chroma]
persist_directory = "~/.tldw_chatbook/chroma"  # Where to store vectors
embedding_model = "all-MiniLM-L6-v2"          # Fast, good quality
distance_metric = "cosine"
"""