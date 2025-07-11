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
import uvicorn
from typing import Optional
import logging
import shutil
import time
from redis_utils import redis_cache

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

# Global variables for resources
client = None
gemini_embedding_function = None

# Cache for user/session specific data with timestamps
user_session_cache = {}

def cleanup_old_sessions(timeout=3600):  # 1 hour timeout
    """Remove session folders older than timeout seconds"""
    sessions_path = Path("/app/sessions")
    if sessions_path.exists():
        for session_folder in sessions_path.iterdir():
            if session_folder.is_dir():
                mtime = session_folder.stat().st_mtime
                if time.time() - mtime > timeout:
                    logger.info(f"Cleaning up old session folder: {session_folder}")
                    shutil.rmtree(session_folder)

def get_user_session_path(user_id: Optional[str] = None, session_id: Optional[str] = None) -> Path:
    """Get the path for data"""
    if user_id == "global" and session_id == "manual":
        return Path("/app/files") / user_id / session_id
    elif user_id:
        return Path("/app/users") / user_id
    elif session_id:
        return Path("/app/sessions") / session_id
    raise HTTPException(status_code=400, detail="Either user_id or session_id must be provided")

# Modify the load_user_session_data function:
def load_user_session_data(user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Load chunks and FAISS index with Redis caching."""
    cache_key = ["user_session", user_id or "", session_id or ""]
    cached_data = redis_cache.get(cache_key)
    if cached_data:
        logger.info(f"Using cached data for user_id: {user_id}, session_id: {session_id}")
        return cached_data
    
    user_path = get_user_session_path(user_id, session_id)

    # Load chunks
    chunks_path = user_path / "vector_store" / "chunks.json"
    logger.info(f"Checking for chunks at: {chunks_path}")
    chunks = load_chunks(str(chunks_path)) if chunks_path.exists() else None

    # Load FAISS index
    faiss_path = user_path / "vector_store" / "index.faiss"
    logger.info(f"Checking for FAISS index at: {faiss_path}")
    faiss_index = load_faiss_index(str(faiss_path)) if faiss_path.exists() else None

    if chunks is None or faiss_index is None:
        logger.error(f"No data found at {user_path}/vector_store")
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {'user_id: ' + user_id if user_id else 'session_id: ' + session_id}"
        )

    # Cache the loaded data for 30 minutes
    data_to_cache = {
        'chunks': chunks,
        'faiss_index': faiss_index
    }
    redis_cache.set(cache_key, data_to_cache, ttl=1800)
    
    return data_to_cache

# Lifespan event to initialize resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, gemini_embedding_function

    logger.info("Initializing Google API client and embedding function")
    client = import_google_api()
    gemini_embedding_function = embedding_function(client)

    # Clean up old sessions at startup
    cleanup_old_sessions()

    yield

    # Cleanup
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

        # Require either user_id or session_id
        if not (request.user_id or request.session_id):
            logger.error("Either user_id or session_id is required")
            raise HTTPException(
                status_code=400,
                detail="Either user_id or session_id is required"
            )

        # Load data
        user_data = load_user_session_data(request.user_id, request.session_id)
        chunks = user_data['chunks']
        faiss_index = user_data['faiss_index']

        lang = request.user_language.lower()
        logger.info(f"Processing query in language: {lang}")

        if lang in ["en", "english"]:
            user_query = request.user_query
            response = query_document(
                user_query=user_query,
                embed_fn=gemini_embedding_function,
                chunks=chunks,
                faiss_index=faiss_index,
                client=client,
                user_language=request.user_language,
                top_k=request.top_k
            )
        elif lang in ["hr", "croatian", "hrvatski"]:
            user_query = request.user_query.lower()
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
            user_query = request.user_query.lower()
            response = query_document(
                user_query=user_query,
                embed_fn=gemini_embedding_function,
                chunks=chunks,
                faiss_index=faiss_index,
                client=client,
                user_language=request.user_language,
                top_k=request.top_k
            )

        logger.info("Query response generated successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cached_sessions": len(user_session_cache)
    }

if __name__ == "__main__":
    uvicorn.run("ask:app", host="0.0.0.0", port=8002, reload=True)
