from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from graph.builder import build_graph # ✅ Correctly importing your LangGraph builder
from main import run_index_manager 
import threading

import sqlite3
import os
from pathlib import Path
import uuid

import logging
from logging.handlers import RotatingFileHandler
import os

# 1. Define where the logs will live (same folder as your database)
APP_DIR = os.path.expanduser("~/.doclamar")
os.makedirs(APP_DIR, exist_ok=True)
LOG_FILE = os.path.join(APP_DIR, "doclamar.log")

# 2. Configure the logging system
logging.basicConfig(
    level=logging.INFO, # Change to logging.DEBUG if you want crazy detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # Auto-rotates the file when it hits 5MB, keeps 2 backups max
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=2),
        # Also print to the terminal so you can still see it in Dev Mode
        logging.StreamHandler() 
    ]
)

# 3. Create the logger object you will use throughout your code
logger = logging.getLogger("doclamar")

# --- 1. SETUP SAFE PATH FOR DESKTOP DATABASE ---
# This creates a folder at /Users/aryanmullick/.doclamar/
HOME_DIR = str(Path.home())
APP_DIR = os.path.join(HOME_DIR, ".doclamar")
DB_PATH = os.path.join(APP_DIR, "chats.db")

os.makedirs(APP_DIR, exist_ok=True)

# --- 2. CREATE TABLES ---
def init_db():
    logger.info(f"Initializing database at: {DB_PATH}")
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # --- NEW: AUTO-MIGRATION LOGIC ---
        # 2. Check if the 'username' column exists. If not, add it!
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'username' not in columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN username TEXT DEFAULT 'guest'")
            logger.info("Database upgraded: Added multi-user support.")
        # ---------------------------------

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        conn.commit()
        logger.info("Database tables verified/created.")
    except Exception as e:
        logger.error(f"Database init failed: {e}", exc_info=True)
    finally:
        conn.close()
# Run this the moment the server boots up
init_db()

app = FastAPI(title="DocLamar Agent API")

# Allow requests from your Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected data from React
class QueryRequest(BaseModel):
    query: str
    directory: str
    username: str = "guest" # <-- NEW
    session_id: str

# Initialize the AI graph once when the server starts
ai_graph = build_graph()

@app.post("/chat")
async def process_chat(request: QueryRequest):
    print(f"\n🚀 [1/3] Received new query: '{request.query}'")
    print(f"📂 [2/3] Scanning directory: {request.directory}")
    print("🧠 [3/3] AI is thinking... (Please wait 30-60 seconds)")
    
    try:
        # 1. Setup the initial state required by your LangGraph nodes
        initial_state = {
            "query": request.query,
            "root_path": request.directory,
            "top_k": 5,
            "routing_plan": None,
            "candidate_documents": None,
            "parsed_chunks": None,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "final_answer": None,
            "source_files": None,
            "citations": None,
            "search_stats": None,
            "evaluation": None,
            "error": None,
            "retry_count": 0,
            "node_timings": {},
        }

        result = ai_graph.invoke(initial_state)
        answer = result.get("final_answer") or "I couldn't generate an answer for that."
        
        # --- SAVE TO DB ---
        # We use a hash of the directory path as a session_id so 
        # chats in the same folder group together, or just a new UUID.
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO sessions (id, title, username) 
        VALUES (?, ?, ?)
    ''', (request.session_id, f"Folder: {os.path.basename(request.directory)}", request.username))
        
        cursor.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', 
                    (request.session_id, 'user', request.query))
        cursor.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', 
                    (request.session_id, 'ai', answer))
        conn.commit()
        conn.close()

        raw_citations = result.get("citations") or []

        formatted_citations = [
            {"file": c.get("file_name", "Unknown"),"path": c.get("file_path", ""), # <--- ADD THIS LINE,
            "snippet": c.get("preview", "")}
            for c in raw_citations
        ]

        # 4. Return exactly what your App.jsx expects (data.response)
        return {
            "response": answer,
            "citations": formatted_citations,
            "session_id": request.session_id
        }

    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

import uuid
from chat.document_chat import load_document, chat as doc_chat

# 1. Store active file sessions in memory
active_sessions = {}

class ChatLoadRequest(BaseModel):
    file_path: str

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    username: str = "guest" # <-- NEW

# 2. Endpoint to load and index a single document
@app.post("/chat/load")
async def load_single_doc(request: ChatLoadRequest):
    try:
        logger.info(f"📄 Loading specific document: {request.file_path}")
        session = load_document(request.file_path)
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = session
        return {"session_id": session_id, "file_name": session.file_name}
    except Exception as e:
        logger.error(f"❌ Error loading doc: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 3. Endpoint to chat with the loaded document
@app.post("/chat/message")
async def doc_message(request: ChatMessageRequest):
    session = active_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    logger.info(f"💬 Chatting with {session.file_name}...")
    result = doc_chat(session, request.message)

    answer = result["answer"]

    # --- SAVE TO DB ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ensure session exists (using the file name as the title)
    cursor.execute('''
        INSERT OR IGNORE INTO sessions (id, title, username) 
        VALUES (?, ?, ?)
    ''', (request.session_id, f"Doc: {session.file_name}", request.username))
    
    cursor.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', 
                  (request.session_id, 'user', request.message))
    cursor.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', 
                  (request.session_id, 'ai', answer))
    
    conn.commit()
    conn.close()
    
    # Format citations to match your frontend
    formatted_citations = [
        {
            "file": session.file_name, 
            "path": session.file_path, 
            "snippet": c.get("preview", "")
        }
        for c in result.get("citations", [])
    ]
    
    return {"response": result["answer"], "citations": formatted_citations}

@app.get("/history")
def get_all_sessions(username: str = "guest"): # <-- Add parameter here
    """Returns a list of all past chats for the specific user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    
    # Filter by username!
    cursor.execute('SELECT * FROM sessions WHERE username = ? ORDER BY created_at DESC', (username,))
    
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"sessions": sessions}

@app.get("/history/{session_id}")
def get_session_messages(session_id: str):
    """Returns all messages for a specific chat."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC', (session_id,))
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"messages": messages}

@app.delete("/history/{session_id}")
def delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Delete messages first (foreign key constraint)
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        # Delete the session
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        conn.commit()
        logger.info(f"🗑️ Successfully deleted chat session: {session_id}")
        return {"status": "success"}
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Failed to delete session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/health")
def health_check():
    return {"status": "ready"}


@app.post("/index/start")
async def trigger_indexing(request: QueryRequest):
    """
    Triggers the run_index_manager logic for a specific directory.
    We run it in a background thread so the UI doesn't freeze.
    """
    logger.info(f"🛠️ Starting manual indexing for: {request.directory}")
    
    def background_indexing():
        try:
            # This calls your actual main.py logic
            run_index_manager(request.directory) 
            logger.info(f"✅ Indexing completed successfully for {request.directory}")
        except Exception as e:
            logger.error(f"❌ Manual indexing failed: {e}", exc_info=True)

    # Start the thread
    thread = threading.Thread(target=background_indexing)
    thread.start()
    
    return {"status": "indexing_started", "message": "The engine is now indexing your files."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000,log_config=None)