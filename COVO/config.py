import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "memory.db"

# LLM Configurations
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat & Memory Settings
DEFAULT_TEMPERATURE = 0.7
EXTRACTION_TEMPERATURE = 0.0
MAX_HISTORY_TURNS = 6  # Keep last 6 exchanges (12 messages) in short-term context
DEFAULT_IMPORTANCE = 0.8
MAX_RETRIEVED_MEMORIES = 5
