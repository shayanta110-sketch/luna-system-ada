import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ada.memory.hybrid_store import HybridStore

logger = logging.getLogger(__name__)


class AddToMemoryInput(BaseModel):
    """Input schema for add_to_memory tool."""
    content: str = Field(description="Content of the conversation exchange to store")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata to attach to the memory entry")


class SearchMemoryInput(BaseModel):
    """Input schema for search_memory tool."""
    query: str = Field(description="Search query to find relevant memory entries")
    limit: int = Field(default=5, description="Maximum number of results to return")


class AddToMemoryTool(BaseTool):
    """Tool to add conversation exchanges to hybrid memory storage."""
    name: str = "add_to_memory"
    description: str = (
        "Store a conversation exchange or important information in hybrid memory. "
        "Use this to save user messages, agent responses, or any notable interaction "
        "that should be retrievable later."
    )
    args_schema: type[BaseModel] = AddToMemoryInput
    
    def __init__(self, hybrid_store: HybridStore, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_store = hybrid_store
    
    def _run(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool to add content to memory."""
        try:
            timestamp = datetime.now().isoformat()
            if metadata is None:
                metadata = {}
            metadata["timestamp"] = timestamp
            
            memory_id = self.hybrid_store.add(
                content=content,
                metadata=metadata
            )
            return json.dumps({
                "status": "success",
                "message": "Memory entry added successfully",
                "memory_id": memory_id,
                "timestamp": timestamp
            })
        except Exception as e:
            logger.exception("Failed to add to memory")
            return json.dumps({
                "status": "error",
                "message": f"Failed to add to memory: {str(e)}"
            })
    
    async def _arun(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Async execution."""
        return self._run(content, metadata)


class SearchMemoryTool(BaseTool):
    """Tool to search hybrid memory for relevant information."""
    name: str = "search_memory"
    description: str = (
        "Search through stored memory entries using semantic and keyword search. "
        "Retrieves the most relevant past conversations or facts based on the query."
    )
    args_schema: type[BaseModel] = SearchMemoryInput
    
    def __init__(self, hybrid_store: HybridStore, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_store = hybrid_store
    
    def _run(self, query: str, limit: int = 5) -> str:
        """Execute the tool to search memory."""
        try:
            results = self.hybrid_store.search(query=query, limit=limit)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score")
                })
            return json.dumps({
                "status": "success",
                "query": query,
                "limit": limit,
                "result_count": len(formatted_results),
                "results": formatted_results
            }, indent=2)
        except Exception as e:
            logger.exception("Failed to search memory")
            return json.dumps({
                "status": "error",
                "message": f"Failed to search memory: {str(e)}"
            })
    
    async def _arun(self, query: str, limit: int = 5) -> str:
        """Async execution."""
        return self._run(query, limit)


class GetMemoryStatsTool(BaseTool):
    """Tool to retrieve statistics about the hybrid memory store."""
    name: str = "get_memory_stats"
    description: str = (
        "Retrieve storage statistics about the hybrid memory system, including "
        "total entries count, storage size, and other performance metrics."
    )
    
    def __init__(self, hybrid_store: HybridStore, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_store = hybrid_store
    
    def _run(self, _: str = "") -> str:
        """Execute the tool to get memory statistics."""
        try:
            stats = self.hybrid_store.get_stats()
            return json.dumps({
                "status": "success",
                "stats": stats
            }, indent=2)
        except Exception as e:
            logger.exception("Failed to get memory statistics")
            return json.dumps({
                "status": "error",
                "message": f"Failed to get memory statistics: {str(e)}"
            })
    
    async def _arun(self, _: str = "") -> str:
        """Async execution."""
        return self._run(_)


def create_memory_tools(hybrid_store: HybridStore) -> List[BaseTool]:
    """Factory function to create all memory-related tools.
    
    Args:
        hybrid_store: An instance of HybridStore to back the tools.
        
    Returns:
        List of LangChain tools (add_to_memory, search_memory, get_memory_stats).
    """
    return [
        AddToMemoryTool(hybrid_store=hybrid_store),
        SearchMemoryTool(hybrid_store=hybrid_store),
        GetMemoryStatsTool(hybrid_store=hybrid_store)
    ]
