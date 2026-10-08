"""SQLite database layer for persistent long-term memory."""

import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DB_PATH

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create and return a database connection with dict-like row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize the memories table if it doesn't already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.8,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Index on category and importance for faster lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_imp ON memories(importance)")
        conn.commit()

def add_memory(
    category: str,
    content: str,
    importance: float = 0.8,
    db_path: Path = DB_PATH
) -> int:
    """Insert a new memory record and return the generated ID."""
    clean_category = category.strip().lower()
    clean_content = content.strip()
    now = datetime.now().isoformat()
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (category, content, importance, created_at, last_used)
            VALUES (?, ?, ?, ?, ?)
        """, (clean_category, clean_content, importance, now, now))
        conn.commit()
        return cursor.lastrowid

def get_all_memories(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all stored memories sorted by importance descending and recency."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, category, content, importance, created_at, last_used
            FROM memories
            ORDER BY importance DESC, created_at DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_memories_by_category(category: str, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve memories belonging to a specific category."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, category, content, importance, created_at, last_used
            FROM memories
            WHERE category = ?
            ORDER BY importance DESC, created_at DESC
        """, (category.lower().strip(),))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def search_memories(keyword: str, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Search memories containing keyword in content or category."""
    like_term = f"%{keyword.strip()}%"
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, category, content, importance, created_at, last_used
            FROM memories
            WHERE content LIKE ? OR category LIKE ?
            ORDER BY importance DESC
        """, (like_term, like_term))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def update_memory(
    memory_id: int,
    content: Optional[str] = None,
    importance: Optional[float] = None,
    category: Optional[str] = None,
    db_path: Path = DB_PATH
) -> bool:
    """Update fields of an existing memory."""
    updates = []
    params = []
    
    if content is not None:
        updates.append("content = ?")
        params.append(content.strip())
    if importance is not None:
        updates.append("importance = ?")
        params.append(importance)
    if category is not None:
        updates.append("category = ?")
        params.append(category.strip().lower())
        
    if not updates:
        return False
        
    updates.append("last_used = ?")
    params.append(datetime.now().isoformat())
    params.append(memory_id)
    
    query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0

def touch_memory(memory_id: int, db_path: Path = DB_PATH) -> None:
    """Update the last_used timestamp of a memory when it gets retrieved."""
    now = datetime.now().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE memories SET last_used = ? WHERE id = ?", (now, memory_id))
        conn.commit()

def delete_memory(memory_id: int, db_path: Path = DB_PATH) -> bool:
    """Delete a memory by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return cursor.rowcount > 0

def clear_all_memories(db_path: Path = DB_PATH) -> bool:
    """Clear all records from the memories table."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        conn.commit()
        return True

def find_duplicate_or_similar(content: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Check if an identical or nearly identical memory already exists."""
    clean_target = content.strip().lower()
    memories = get_all_memories(db_path)
    for mem in memories:
        existing = mem["content"].strip().lower()
        if clean_target == existing:
            return mem
        # check high token overlap
        words_target = set(clean_target.split())
        words_existing = set(existing.split())
        if words_target and words_existing:
            overlap = len(words_target & words_existing) / max(len(words_target), len(words_existing))
            if overlap >= 0.85:
                return mem
    return None
