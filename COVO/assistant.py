"""Main pipeline coordinating LLM, Memory, and Conversation History."""

from typing import Generator, Dict, Any, List, Tuple
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from llm.ollama_client import OllamaClient
from llm.prompts import CHAT_SYSTEM_PROMPT, PERSONALIZED_PROMPT_TEMPLATE
from memory.memory_manager import MemoryManager
from conversation.chat_manager import ChatManager

class PersonalAIAssistant:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.llm_client = OllamaClient(model_name=model_name, temperature=temperature)
        self.memory_manager = MemoryManager(client=self.llm_client)
        self.chat_manager = ChatManager()

    def build_prompt(self, user_message: str, retrieved_memories: List[Dict[str, Any]]) -> str:
        """Construct the personalized prompt containing system prompt, memories, and history."""
        memory_context = self.memory_manager.format_memories(retrieved_memories)
        history_context = self.chat_manager.format_history_for_prompt()

        return PERSONALIZED_PROMPT_TEMPLATE.format(
            system_prompt=CHAT_SYSTEM_PROMPT,
            memory_context=memory_context,
            history_context=history_context,
            user_message=user_message
        ).strip()

    def chat_step(self, user_message: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Non-streaming chat step:
        Returns: (new_memories, retrieved_memories, assistant_response)
        """
        # 1. Analyze and extract long-term memory
        new_memories = self.memory_manager.process_user_input(user_message)

        # 2. Retrieve relevant memories for the prompt
        retrieved_memories = self.memory_manager.retrieve_context(user_message)

        # 3. Construct personalized prompt
        prompt = self.build_prompt(user_message, retrieved_memories)

        # 4. Generate response
        response = self.llm_client.generate(prompt)

        # 5. Record short-term conversation turn
        self.chat_manager.add_user_message(user_message)
        self.chat_manager.add_assistant_message(response)

        return new_memories, retrieved_memories, response

    def chat_step_stream(
        self,
        user_message: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Generator[str, None, None]]:
        """
        Streaming chat step:
        Returns: (new_memories, retrieved_memories, stream_generator)
        """
        # 1. Analyze and extract long-term memory
        new_memories = self.memory_manager.process_user_input(user_message)

        # 2. Retrieve relevant memories
        retrieved_memories = self.memory_manager.retrieve_context(user_message)

        # 3. Construct personalized prompt
        prompt = self.build_prompt(user_message, retrieved_memories)

        # 4. Stream generator
        stream_gen = self.llm_client.stream(prompt)

        # Record user message in short-term history immediately
        self.chat_manager.add_user_message(user_message)

        return new_memories, retrieved_memories, stream_gen

    def finalize_stream_response(self, response_text: str) -> None:
        """Call this once streaming finishes to record assistant turn into history."""
        self.chat_manager.add_assistant_message(response_text)
