"""Retrieves relevant memories based on user queries and importance."""

from typing import List, Dict, Any
from memory.database import get_all_memories, touch_memory
from config import MAX_RETRIEVED_MEMORIES

# Stopwords to avoid false positive matches on common filler words
COMMON_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "can", "could", "should", "would", "will", "i", "my", "me",
    "what", "how", "why", "when", "where", "who", "which", "this", "that", "it",
    "you", "your", "we", "they", "them", "about", "with", "from", "by"
}

class MemoryRetriever:
    def __init__(self, max_results: int = MAX_RETRIEVED_MEMORIES):
        self.max_results = max_results

    def score_memory(self, memory: Dict[str, Any], query_tokens: set) -> float:
        """Calculate a relevance score combining keyword overlap and importance."""
        content = memory["content"].lower()
        category = memory["category"].lower()
        importance = float(memory.get("importance", 0.5))

        # Split content into alphanumeric tokens
        mem_tokens = {w for w in content.split() if w not in COMMON_STOPWORDS}
        mem_tokens.add(category)

        # Keyword overlap
        overlap = query_tokens & mem_tokens
        keyword_score = len(overlap) / max(1, len(query_tokens))

        # Core identity facts (name, primary career/goal) get baseline presence
        baseline_score = 0.0
        if category in {"personal", "goal"} and importance >= 0.9:
            baseline_score = 0.3

        final_score = (keyword_score * 0.6) + (importance * 0.3) + baseline_score
        return final_score

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve the top most relevant memories for the current input."""
        all_memories = get_all_memories()
        if not all_memories:
            return []

        # If query is short or general, return top general/important memories
        query_words = {w.lower().strip(",.?!") for w in query.split() if len(w) > 2}
        meaningful_query_tokens = {w for w in query_words if w not in COMMON_STOPWORDS}

        scored_memories = []
        for mem in all_memories:
            score = self.score_memory(mem, meaningful_query_tokens)
            scored_memories.append((score, mem))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Pick top candidates
        top_candidates = [mem for score, mem in scored_memories[:self.max_results]]

        # Update last_used timestamp for retrieved memories
        for mem in top_candidates:
            touch_memory(mem["id"])

        return top_candidates

    def format_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format a list of memories as a clean Markdown bullet list for the prompt."""
        if not memories:
            return ""
        
        lines = ["USER MEMORY:"]
        for mem in memories:
            lines.append(f"- [{mem['category'].title()}] {mem['content']}")
        return "\n".join(lines)
