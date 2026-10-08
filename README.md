# 🧠 Personal Memory AI Assistant

A local, private AI assistant that learns about you over time, stores long-term memories in SQLite, and retrieves relevant context to personalize responses using **Ollama (`qwen2.5:0.5b`)**, **LangChain**, and **Streamlit**.

---

## 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    │                  │
                    │ User enters msg  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Conversation     │
                    │ Manager          │
                    └────────┬─────────┘
                             ↓
                 ┌──────────────────────┐
                 │ Memory Analyzer      │
                 │                      │
                 │ Important?           │
                 │ What to remember?    │
                 └──────────┬───────────┘
                            │
                   ┌────────┴────────┐
                   ↓                 ↓
                 YES                NO
                   ↓                 ↓
             SQLite Memory       Continue
                   │                 │
                   └────────┬────────┘
                            ↓
                 Retrieve relevant
                     memories
                            ↓
                 ┌──────────────────┐
                 │ Prompt Builder   │
                 │                  │
                 │ Memory + Chat    │
                 │ + Current input  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Ollama + Qwen    │
                 │                  │
                 │ qwen2.5:0.5b     │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Personalized     │
                 │ Response (Stream)│
                 └────────┬─────────┘
                          ↓
                    Streamlit UI
```

---

## 📁 Project Structure

```text
COVO/
│
├── app.py                    # Streamlit web UI with live streaming & memory management
├── assistant.py              # Central pipeline coordinator
├── config.py                 # Configuration settings (model, paths, temperatures)
│
├── llm/
│   ├── __init__.py
│   ├── ollama_client.py      # Ollama/LangChain ChatOllama wrapper (generate & stream)
│   └── prompts.py            # System prompts & memory extraction templates
│
├── memory/
│   ├── __init__.py
│   ├── database.py           # SQLite database layer (schema, CRUD, duplicate check)
│   ├── memory_extractor.py   # Hybrid rule + LLM extractor (filters out trivial chat)
│   ├── memory_retriever.py   # Relevance scoring & prompt injection
│   └── memory_manager.py     # High-level coordinator & conflict resolver
│
├── conversation/
│   ├── __init__.py
│   └── chat_manager.py       # Short-term session chat history
│
├── data/
│   └── memory.db             # Persistent SQLite database
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Ollama is installed and running with `qwen2.5:0.5b`:
```bash
ollama run qwen2.5:0.5b
```

### 2. Activate Environment & Run UI
```bash
# From workspace root
streamlit run app.py
```

---

## 🧪 Pipeline Features

1. **Short-Term Memory**: Conversation history maintained across recent turns.
2. **Selective Long-Term Memory**: Only meaningful personal details (name, goals, preferences, skills, education) are persisted. Greetings ("hi", "thanks") and generic questions are ignored.
3. **Persistent SQLite Database**: Stores memories with `category`, `content`, `importance`, and timestamps.
4. **Context Retrieval**: Scores memories against the user query to inject only the most relevant background.
5. **Real-time Streaming**: Full streaming token generation via Streamlit.
