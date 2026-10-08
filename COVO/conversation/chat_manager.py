"""Short-term conversation history management."""

from datetime import datetime
from typing import List, Dict
from config import MAX_HISTORY_TURNS

class ChatManager:
    def __init__(self, max_history_turns: int = MAX_HISTORY_TURNS):
        self.max_history_turns = max_history_turns
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to conversation history."""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to conversation history."""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_recent_turns(self, max_turns: int = None) -> List[Dict[str, str]]:
        """Return the most recent messages (up to max_turns * 2)."""
        turns = max_turns if max_turns is not None else self.max_history_turns
        max_messages = turns * 2
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages

    def format_history_for_prompt(self, max_turns: int = None) -> str:
        """Format recent dialogue turns for prompt injection."""
        recent = self.get_recent_turns(max_turns)
        if not recent:
            return ""

        lines = ["RECENT CONVERSATION:"]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear short-term history."""
        self.messages.clear()
