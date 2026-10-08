"""Streamlit Web UI for Personal Memory AI Assistant."""

import streamlit as st
from assistant import PersonalAIAssistant
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from memory.database import get_all_memories, delete_memory, clear_all_memories, add_memory

st.set_page_config(
    page_title="Personal Memory AI Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session State
if "assistant" not in st.session_state:
    st.session_state.assistant = PersonalAIAssistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_detected_memories" not in st.session_state:
    st.session_state.last_detected_memories = []

assistant: PersonalAIAssistant = st.session_state.assistant

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🧠 Long-Term Memory")
    st.caption("Memories automatically extracted and persisted in SQLite")

    # Connection Status
    with st.expander("⚙️ Assistant Settings", expanded=False):
        st.write(f"**Model:** `{assistant.llm_client.model_name}`")
        temp = st.slider("Temperature", min_value=0.0, max_value=1.5, value=DEFAULT_TEMPERATURE, step=0.1)
        assistant.llm_client.temperature = temp
        assistant.llm_client._llm.temperature = temp
        
        if st.button("🧹 Clear Chat History", use_container_width=True):
            assistant.chat_manager.clear()
            st.session_state.messages = []
            st.session_state.last_detected_memories = []
            st.rerun()

    # Manual Memory Addition
    with st.expander("➕ Add Memory Manually", expanded=False):
        new_cat = st.selectbox(
            "Category",
            ["personal", "goal", "education", "preference", "project", "skill"]
        )
        new_content = st.text_input("Memory Fact", placeholder="e.g. User prefers concise code")
        new_importance = st.slider("Importance", 0.1, 1.0, 0.85, 0.05)
        if st.button("Save Memory", use_container_width=True):
            if new_content.strip():
                assistant.memory_manager.add_manual(new_cat, new_content.strip(), new_importance)
                st.success("Memory saved!")
                st.rerun()

    # Display Stored Memories
    memories = assistant.memory_manager.get_all()
    st.subheader(f"Stored Memories ({len(memories)})")

    if not memories:
        st.info("No long-term memories saved yet. Chat naturally to see them automatically detected!")
    else:
        # Category Filter
        categories = sorted(list({m["category"] for m in memories}))
        selected_category = st.selectbox("Filter Category", ["All"] + categories)

        filtered_memories = [
            m for m in memories
            if selected_category == "All" or m["category"] == selected_category
        ]

        # Category Emoji map
        emoji_map = {
            "personal": "👤",
            "goal": "🎯",
            "education": "🎓",
            "preference": "⚙️",
            "project": "💻",
            "skill": "🐍",
            "career": "💼"
        }

        for mem in filtered_memories:
            cat_icon = emoji_map.get(mem["category"].lower(), "📌")
            with st.container(border=True):
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"**{cat_icon} {mem['category'].upper()}**")
                    st.write(mem["content"])
                    st.caption(f"Importance: {mem['importance']:.2f}")
                with col_del:
                    if st.button("🗑️", key=f"del_{mem['id']}", help="Delete memory"):
                        assistant.memory_manager.delete(mem["id"])
                        st.rerun()

        if st.button("⚠️ Clear All Memories", use_container_width=True):
            assistant.memory_manager.clear()
            st.success("All memories wiped clean.")
            st.rerun()

# ----------------- MAIN CHAT AREA -----------------
st.title("🧠 Personal Memory AI Assistant")
st.markdown("A local conversational assistant that learns about you over time and personalizes answers.")

# Display detected memory notification if any
if st.session_state.last_detected_memories:
    for new_mem in st.session_state.last_detected_memories:
        status = new_mem.get("status", "added")
        verb = "Remembered" if status == "added" else "Updated memory"
        st.toast(f"💡 {verb}: [{new_mem['category']}] {new_mem['content']}")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieved_memories" in msg and msg["retrieved_memories"]:
            with st.expander(f"🔍 Context: {len(msg['retrieved_memories'])} memories referenced", expanded=False):
                for rm in msg["retrieved_memories"]:
                    st.markdown(f"- **[{rm['category'].title()}]** {rm['content']}")

# Chat Input & Response Generation
if user_input := st.chat_input("Type your message..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Assistant Response with streaming
    with st.chat_message("assistant"):
        new_mems, retrieved_mems, stream_gen = assistant.chat_step_stream(user_input)
        
        # Save any newly extracted memory notification
        st.session_state.last_detected_memories = new_mems

        # Stream response
        full_response = st.write_stream(stream_gen)
        assistant.finalize_stream_response(full_response)

        # Show memories used in context
        if retrieved_mems:
            with st.expander(f"🔍 Context: {len(retrieved_mems)} memories referenced", expanded=False):
                for rm in retrieved_mems:
                    st.markdown(f"- **[{rm['category'].title()}]** {rm['content']}")

    # Save assistant message to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "retrieved_memories": retrieved_mems
    })
    
    # Rerun to update sidebar memories dynamically
    st.rerun()
