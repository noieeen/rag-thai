import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ragthai.db"

def get_connection():
    """Get a new database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Create tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_document(filename: str, content: str):
    """Insert a new document into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, content) VALUES (?, ?)",
        (filename, content)
    )
    conn.commit()
    conn.close()

def get_document_by_id(doc_id: int):
    """Retrieve a document by its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM documents WHERE id = ?",
        (doc_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_documents():
    """List all documents."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]