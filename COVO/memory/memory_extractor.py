"""Memory extraction engine: filters trivial chat and extracts long-term user facts."""

import re
import json
from typing import Dict, Any, List, Optional
from llm.ollama_client import OllamaClient
from llm.prompts import MEMORY_EXTRACTION_SYSTEM_PROMPT

TRIVIAL_STARTS = {
    "hi", "hello", "hey", "hola", "thanks", "thank you", "thx", "ok", "okay",
    "cool", "nice", "great", "awesome", "bye", "goodbye", "see you", "tell me a joke",
    "help", "yes", "no", "sure", "yep", "nope", "alright"
}

FIRST_PERSON_PATTERNS = [
    r"\bi\b", r"\bmy\b", r"\bme\b", r"\bmine\b", r"\bmyself\b",
    r"\bi'm\b", r"\bi've\b", r"\bi'll\b", r"\bi'd\b"
]

class MemoryExtractor:
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def is_candidate(self, text: str) -> bool:
        """Heuristic pre-filter: Returns False if the message is clearly not personal info."""
        cleaned = text.strip().lower()
        if not cleaned:
            return False

        # Trivial single-phrase checks
        if cleaned in TRIVIAL_STARTS:
            return False
            
        words = cleaned.split()
        if len(words) <= 2 and words[0] in TRIVIAL_STARTS:
            return False

        # If user asks generic question without first-person references
        # e.g., "What is Python?", "Tell me a joke", "How does transformers work?"
        has_first_person = any(re.search(pat, cleaned) for pat in FIRST_PERSON_PATTERNS)
        if not has_first_person:
            # If user didn't mention themselves, it's rarely a personal memory
            return False

        return True

    def extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """Fast pattern-based extractor for high-confidence explicit statements."""
        memories = []
        clean = text.strip()

        # Name patterns: "My name is Kishore", "I am Kishore,"
        name_match = re.search(r"\b(?:my name is|call me)\s+([A-Za-z]+)", clean, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).capitalize()
            if name.lower() not in {"an", "a", "the", "not", "just", "ready", "learning", "trying"}:
                memories.append({
                    "category": "personal",
                    "content": f"User's name is {name}",
                    "importance": 0.95
                })

        # Goal patterns: "I want to become an AI Engineer", "My goal is..."
        goal_match = re.search(
            r"\b(?:want to become|dream is to become|goal is to become|aiming to become|aspire to be)\s+(?:an?\s+)?([^,.]+)",
            clean,
            re.IGNORECASE
        )
        if goal_match:
            goal_target = goal_match.group(1).strip()
            memories.append({
                "category": "goal",
                "content": f"User wants to become {goal_target}",
                "importance": 0.95
            })

        # Education / Exam patterns: "I am preparing for GATE DA 2027"
        prep_match = re.search(
            r"\b(?:preparing for|studying for|clearing)\s+([^,.]+)",
            clean,
            re.IGNORECASE
        )
        if prep_match:
            exam = prep_match.group(1).strip()
            memories.append({
                "category": "education",
                "content": f"User is preparing for {exam}",
                "importance": 0.9
            })

        # Preference patterns: "I prefer Python", "I prefer concise explanations"
        pref_match = re.search(
            r"\b(?:prefer|preference is)\s+([^,.]+)",
            clean,
            re.IGNORECASE
        )
        if pref_match:
            pref = pref_match.group(1).strip()
            memories.append({
                "category": "preference",
                "content": f"User prefers {pref}",
                "importance": 0.85
            })

        # Project patterns: "I am building a personal AI assistant"
        proj_match = re.search(
            r"\b(?:building|working on|developing)\s+(?:a\s+|an\s+|the\s+)?([^,.]+)",
            clean,
            re.IGNORECASE
        )
        if proj_match:
            proj = proj_match.group(1).strip()
            memories.append({
                "category": "project",
                "content": f"User is working on {proj}",
                "importance": 0.85
            })

        return memories

    def extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Ask LLM to extract structured memories in JSON format."""
        prompt = f"""{MEMORY_EXTRACTION_SYSTEM_PROMPT}

User statement: "{text}"
Response:"""
        try:
            raw_response = self.client.extract(prompt)
            # Find JSON block
            json_str = raw_response.strip()
            json_str = re.sub(r"^```(?:json)?", "", json_str, flags=re.MULTILINE)
            json_str = re.sub(r"```$", "", json_str, flags=re.MULTILINE).strip()

            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}")
            if start_idx != -1 and end_idx != -1:
                data = json.loads(json_str[start_idx:end_idx + 1])
                extracted = data.get("memories", [])
                
                # Sanitize extracted records (prevent self-hallucinations like 'qwen')
                cleaned = []
                for item in extracted:
                    content = item.get("content", "").strip()
                    cat = item.get("category", "personal").strip().lower()
                    imp = float(item.get("importance", 0.8))
                    if not content or "qwen" in content.lower() or "assistant" in content.lower():
                        continue
                    cleaned.append({
                        "category": cat,
                        "content": content,
                        "importance": min(1.0, max(0.1, imp))
                    })
                return cleaned
        except Exception:
            pass
        return []

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Complete hybrid extraction pipeline: pre-filter -> rules + LLM extraction."""
        if not self.is_candidate(text):
            return []

        # 1. Rule-based extraction
        rule_memories = self.extract_with_rules(text)
        
        # 2. LLM extraction
        llm_memories = self.extract_with_llm(text)

        # Merge results, prioritizing rules for high-confidence exact facts
        merged = []
        seen_contents = set()

        for mem in rule_memories + llm_memories:
            c_lower = mem["content"].lower()
            if c_lower not in seen_contents:
                seen_contents.add(c_lower)
                merged.append(mem)

        return merged
