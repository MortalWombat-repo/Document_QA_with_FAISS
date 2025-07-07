from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from core import (
    import_google_api,
    embedding_function,
    query_document,
    query_document_hr,
    load_chunks,
    load_faiss_index
)
from pathlib import Path
from typing import Optional
import uvicorn
import logging
import shutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
class QueryRequest(BaseModel):
    user_query: str
    user_language: str
    top_k: int = 1
    user_id: Optional[str] = None
    session_id: Optional[str] = None

# Global variables
client = None
gemini_embedding_function = None

# Cache for user/session specific data with timestamps
user_session_cache = {}

def cleanup_old_sessions(timeout=3600):  # 1 hour timeout
    sessions_path = Path("/app/sessions")
    if sessions_path.exists():
        for session_folder in sessions_path.iterdir():
            if session_folder.is_dir():
                mtime = session_folder.stat().st_mtime
                if time.time() - mtime > timeout:
                    logger.info(f"Cleaning up old session folder: {session_folder}")
                    shutil.rmtree(session_folder)

def get_user_session_path(user_id: Optional[str] = None, session_id: Optional[str] = None) -> Path:
    if user_id == "global" and session_id == "manual":
        return Path("/app/files") / user_id / session_id
    elif user_id:
        return Path("/app/users") / user_id
    elif session_id:
        return Path("/app/sessions") / session_id
    raise HTTPException(status_code=400, detail="Either user_id or session_id must be provided")

def load_user_session_data(user_id: Optional[str] = None, session_id: Optional[str] = None):
    cache_key = f"{user_id or ''}_{session_id or ''}"
    logger.info(f"Loading data for user_id: {user_id}, session_id: {session_id}")

    if cache_key in user_session_cache:
        logger.info(f"Using cached data for {cache_key}")
        return user_session_cache[cache_key]

    user_path = get_user_session_path(user_id, session_id)

    chunks_path = user_path / "vector_store" / "chunks.json"
    logger.info(f"Checking for chunks at: {chunks_path}")
    chunks = load_chunks(str(chunks_path)) if chunks_path.exists() else None

    faiss_path = user_path / "vector_store" / "index.faiss"
    logger.info(f"Checking for FAISS index at: {faiss_path}")
    faiss_index = load_faiss_index(str(faiss_path)) if faiss_path.exists() else None

    if chunks is None or faiss_index is None:
        logger.error(f"No data found at {user_path}/vector_store")
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {'user_id: ' + user_id if user_id else 'session_id: ' + session_id}"
        )

    user_session_cache[cache_key] = {
        'chunks': chunks,
        'faiss_index': faiss_index
    }
    logger.info(f"Cached data for {cache_key}")
    return user_session_cache[cache_key]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, gemini_embedding_function

    logger.info("Initializing Google API client and embedding function")
    client = import_google_api()
    gemini_embedding_function = embedding_function(client)
    cleanup_old_sessions()
    yield
    logger.info("Cleaning up resources")
    client = None
    gemini_embedding_function = None
    user_session_cache.clear()
    cleanup_old_sessions()

app = FastAPI(title="Document Query Microservice", lifespan=lifespan)

@app.post("/ask")
async def query_endpoint(request: QueryRequest):
    try:
        if not all([client, gemini_embedding_function]):
            logger.error("Resources not initialized")
            raise HTTPException(status_code=500, detail="Resources not initialized")

        if not (request.user_id or request.session_id):
            logger.error("Either user_id or session_id is required")
            raise HTTPException(status_code=400, detail="Either user_id or session_id is required")

        user_data = load_user_session_data(request.user_id, request.session_id)
        chunks = user_data['chunks']
        faiss_index = user_data['faiss_index']

        lang = request.user_language.lower()
        logger.info(f"Processing query in language: {lang}")

        user_query = request.user_query.lower()
        if lang in ["hr", "croatian", "hrvatski"]:
            response = query_document_hr(
                user_query=user_query,
                embed_fn=gemini_embedding_function,
                chunks=chunks,
                faiss_index=faiss_index,
                client=client,
                user_language=request.user_language,
                top_k=request.top_k
            )
        else:
            response = query_document(
                user_query=user_query,
                embed_fn=gemini_embedding_function,
                chunks=chunks,
                faiss_index=faiss_index,
                client=client,
                user_language=request.user_language,
                top_k=request.top_k
            )

        logger.info(f"Query response generated successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cached_sessions": len(user_session_cache)
    }

if __name__ == "__main__":
    uvicorn.run("ask:app", host="0.0.0.0", port=8002, reload=True)
