import sqlite3
import os
import json
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "northgpt.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        phone TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Chat Sessions Table (like ChatGPT / Claude sidebar history)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    
    # 3. Chat Messages Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources_json TEXT,
        pins_json TEXT,
        guides_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
    )
    """)
    
    # 4. Guide Bookings & Inquiries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inquiries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_name TEXT,
        user_phone TEXT,
        guide_id TEXT,
        guide_name TEXT,
        destination TEXT,
        dates TEXT,
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

# ── User Auth Functions ───────────────────────────────────────────
def register_user(name, email, password, phone=""):
    conn = get_db()
    cursor = conn.cursor()
    email_clean = email.strip().lower()
    
    cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
    if cursor.fetchone():
        conn.close()
        return None, "An account with this email address already exists."
    
    pw_hash = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO users (name, email, password_hash, phone) VALUES (?, ?, ?, ?)",
        (name.strip(), email_clean, pw_hash, phone.strip())
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": user_id, "name": name.strip(), "email": email_clean}, None

def authenticate_user(email, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None, "Invalid email or password."
    if not check_password_hash(user["password_hash"], password):
        return None, "Invalid email or password."
        
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "phone": user["phone"]
    }, None

def get_user_by_id(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# ── Chat Session & Message Management ──────────────────────────────
def create_chat_session(user_id, title="New Mountain Journey"):
    session_id = str(uuid.uuid4())
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)",
        (session_id, user_id, title)
    )
    conn.commit()
    conn.close()
    return {"id": session_id, "title": title}

def get_user_chat_sessions(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_chat_session(session_id, user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
    session = cursor.fetchone()
    if not session:
        conn.close()
        return None
        
    cursor.execute(
        "SELECT role, content, sources_json, pins_json, guides_json, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    messages = []
    for m in cursor.fetchall():
        item = {
            "role": m["role"],
            "content": m["content"],
            "sources": json.loads(m["sources_json"]) if m["sources_json"] else [],
            "pins": json.loads(m["pins_json"]) if m["pins_json"] else [],
            "guides": json.loads(m["guides_json"]) if m["guides_json"] else [],
            "created_at": m["created_at"]
        }
        messages.append(item)
    conn.close()
    
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "messages": messages
    }

def add_chat_message(session_id, role, content, sources=None, pins=None, guides=None):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT INTO chat_messages (session_id, role, content, sources_json, pins_json, guides_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            json.dumps(sources or []),
            json.dumps(pins or []),
            json.dumps(guides or [])
        )
    )
    cursor.execute(
        "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (session_id,)
    )
    conn.commit()
    conn.close()

def update_session_title(session_id, user_id, title):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
        (title.strip()[:60], session_id, user_id)
    )
    conn.commit()
    conn.close()

def delete_chat_session(session_id, user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
    conn.commit()
    conn.close()

# Initialize DB on load
init_db()
