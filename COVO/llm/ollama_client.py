"""Ollama integration using LangChain ChatOllama."""

from typing import Generator, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config import DEFAULT_MODEL, OLLAMA_BASE_URL, DEFAULT_TEMPERATURE, EXTRACTION_TEMPERATURE

class OllamaClient:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self._llm = self._create_llm(temperature=temperature)
        self._extraction_llm = self._create_llm(temperature=EXTRACTION_TEMPERATURE)

    def _create_llm(self, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        """Synchronously generate a response for a prompt."""
        response = self._llm.invoke(prompt)
        return response.content

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream chunks of response for real-time token display."""
        for chunk in self._llm.stream(prompt):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    def extract(self, prompt: str) -> str:
        """Call the model with low temperature for extraction tasks."""
        response = self._extraction_llm.invoke(prompt)
        return response.content

    def check_connection(self) -> bool:
        """Test if Ollama and the model are reachable."""
        try:
            res = self._llm.invoke("Hi")
            return bool(res and res.content)
        except Exception:
            return False
