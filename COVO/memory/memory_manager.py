"""Coordinates database operations, extraction, retrieval, and conflict resolution."""

from typing import List, Dict, Any, Optional
from memory.database import (
    init_db,
    add_memory,
    get_all_memories,
    get_memories_by_category,
    search_memories,
    update_memory,
    delete_memory,
    clear_all_memories,
    find_duplicate_or_similar
)
from memory.memory_extractor import MemoryExtractor
from memory.memory_retriever import MemoryRetriever
from llm.ollama_client import OllamaClient

class MemoryManager:
    def __init__(self, client: Optional[OllamaClient] = None):
        init_db()
        self.extractor = MemoryExtractor(client=client)
        self.retriever = MemoryRetriever()

    def process_user_input(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyze user input, extract facts, resolve conflicts, and save new memories.
        Returns the list of newly added or updated memories.
        """
        extracted_facts = self.extractor.extract(text)
        saved_memories = []

        for fact in extracted_facts:
            cat = fact["category"]
            content = fact["content"]
            imp = fact.get("importance", 0.8)

            # Check if this memory is updating an existing one of same category (e.g. name or primary goal)
            existing_match = find_duplicate_or_similar(content)
            if existing_match:
                # If exact or very similar, refresh importance or last used
                update_memory(
                    memory_id=existing_match["id"],
                    content=content,
                    importance=max(existing_match["importance"], imp)
                )
                saved_memories.append({"id": existing_match["id"], **fact, "status": "updated"})
                continue

            # Check if category is single-slot like 'name'
            if "name is" in content.lower():
                existing_all = get_all_memories()
                name_mem = next((m for m in existing_all if "name is" in m["content"].lower()), None)
                if name_mem:
                    update_memory(name_mem["id"], content=content, importance=0.95)
                    saved_memories.append({"id": name_mem["id"], **fact, "status": "updated"})
                    continue

            # Otherwise, insert as a new memory
            new_id = add_memory(category=cat, content=content, importance=imp)
            saved_memories.append({"id": new_id, **fact, "status": "added"})

        return saved_memories

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the current user query."""
        return self.retriever.retrieve(query)

    def format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for LLM prompt injection."""
        return self.retriever.format_for_prompt(memories)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all memories."""
        return get_all_memories()

    def delete(self, memory_id: int) -> bool:
        """Delete specific memory."""
        return delete_memory(memory_id)

    def clear(self) -> bool:
        """Clear all memories."""
        return clear_all_memories()

    def add_manual(self, category: str, content: str, importance: float = 0.8) -> int:
        """Manually inject a memory (useful for testing & admin UI)."""
        return add_memory(category=category, content=content, importance=importance)
