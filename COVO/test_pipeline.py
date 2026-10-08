"""End-to-end automated test for the Personal Memory AI Assistant pipeline."""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure COVO directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory.database import init_db, clear_all_memories, get_all_memories
from assistant import PersonalAIAssistant

def run_tests():
    print("========================================")
    print("🤖 STARTING END-TO-END PIPELINE TESTS")
    print("========================================")

    # 1. Initialize Assistant and clear test DB
    assistant = PersonalAIAssistant()
    assistant.memory_manager.clear()
    assistant.chat_manager.clear()
    print("✅ 1. Database initialized and cleaned.")

    # 2. Test Trivial Messages (Should NOT store anything)
    trivial_messages = ["Hello!", "How's the weather today?", "Thanks a lot!"]
    for msg in trivial_messages:
        new_mems, _, _ = assistant.chat_step(msg)
        assert len(new_mems) == 0, f"Expected 0 memories for '{msg}', got {new_mems}"
    
    mems = assistant.memory_manager.get_all()
    assert len(mems) == 0, f"Expected 0 stored memories, found {len(mems)}"
    print("✅ 2. Filter test passed: Trivial messages correctly ignored.")

    # 3. Test Memory Extraction
    profile_msg1 = "My name is Kishore and I want to become an AI Engineer."
    new_mems1, _, resp1 = assistant.chat_step(profile_msg1)
    print(f"\nUser: {profile_msg1}")
    print(f"Extracted: {new_mems1}")
    print(f"Assistant: {resp1[:100]}...\n")
    assert len(new_mems1) > 0, "Expected memories to be extracted from profile statement!"

    profile_msg2 = "I am preparing for GATE DA 2027 and I prefer Python."
    new_mems2, _, resp2 = assistant.chat_step(profile_msg2)
    print(f"User: {profile_msg2}")
    print(f"Extracted: {new_mems2}")
    print(f"Assistant: {resp2[:100]}...\n")
    assert len(new_mems2) > 0, "Expected memories from GATE / Python statement!"

    # 4. Verify SQLite Database State
    all_stored = assistant.memory_manager.get_all()
    print("✅ 4. Current SQLite Database Contents:")
    for m in all_stored:
        print(f"   [{m['id']}] [{m['category']}] {m['content']} (importance={m['importance']})")
    assert len(all_stored) >= 2, f"Expected at least 2 memories, found {len(all_stored)}"

    # 5. Test Retrieval and Personalization
    career_query = "What roadmap should I follow for my career?"
    _, retrieved, resp3 = assistant.chat_step(career_query)
    print(f"\nUser Query: {career_query}")
    print(f"Retrieved Memories used in prompt:")
    for r in retrieved:
        print(f"   - [{r['category']}] {r['content']}")
    print(f"\nPersonalized Assistant Response:\n{resp3}")

    assert len(retrieved) > 0, "Expected retrieved memories for career query!"
    print("\n✅ 5. Retrieval test passed: Relevant memories successfully injected.")

    # 6. Test Streaming Generation
    stream_query = "Summarize what you know about me in one sentence."
    new_mems_s, retr_s, stream_gen = assistant.chat_step_stream(stream_query)
    streamed_text = ""
    for chunk in stream_gen:
        streamed_text += chunk
    assistant.finalize_stream_response(streamed_text)
    print(f"\nUser Query (Streaming): {stream_query}")
    print(f"Streamed Response: {streamed_text}")
    assert len(streamed_text) > 0, "Streaming produced no text!"
    print("✅ 6. Streaming generation test passed.")

    print("\n========================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
