from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from core import (
<<<<<<< HEAD
    extract_texts_from_folder, 
    chunking, 
    import_google_api, 
    embedding_function, 
    build_or_update_faiss_index, 
    query_document,
    query_document_hr, 
    load_chunks,
    save_chunks, 
=======
    import_google_api,
    embedding_function,
    query_document,
    query_document_hr,
    load_chunks,
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    load_faiss_index
)
from pathlib import Path
import uvicorn
<<<<<<< HEAD

# request model
=======
from typing import Optional
import logging
import shutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
>>>>>>> f32d650 (Proper project root: move contents to repo root)
class QueryRequest(BaseModel):
    user_query: str
    user_language: str
    top_k: int = 1
<<<<<<< HEAD
=======
    user_id: Optional[str] = None
    session_id: Optional[str] = None
>>>>>>> f32d650 (Proper project root: move contents to repo root)

# Global variables for resources
client = None
gemini_embedding_function = None
<<<<<<< HEAD
chunks = None
faiss_index = None
folder_path = Path("files").resolve()
=======

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

def load_user_session_data(user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Load chunks and FAISS index"""
    cache_key = f"{user_id or ''}_{session_id or ''}"
    logger.info(f"Loading data for user_id: {user_id}, session_id: {session_id}")

    if cache_key in user_session_cache:
        logger.info(f"Using cached data for {cache_key}")
        return user_session_cache[cache_key]

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

    # Cache the loaded data
    user_session_cache[cache_key] = {
        'chunks': chunks,
        'faiss_index': faiss_index
    }
    logger.info(f"Cached data for {cache_key}")

    return user_session_cache[cache_key]
>>>>>>> f32d650 (Proper project root: move contents to repo root)

# Lifespan event to initialize resources
@asynccontextmanager
async def lifespan(app: FastAPI):
<<<<<<< HEAD
    global client, gemini_embedding_function, chunks, faiss_index
    
    client = import_google_api()
    
    gemini_embedding_function = embedding_function(client)
    
    chunks = load_chunks()
    if not chunks:
        contract_text = extract_texts_from_folder(folder_path)
        if not contract_text:
            raise RuntimeError("No text extracted from PDFs")
        chunks = chunking(contract_text)
        save_chunks(chunks)
    
    faiss_index = load_faiss_index()
    if faiss_index is None:
        faiss_index = build_or_update_faiss_index(chunks, gemini_embedding_function)
        if faiss_index is None:
            raise RuntimeError("Failed to create or load FAISS index")
    yield
    
    client = None
    gemini_embedding_function = None
    chunks = None
    faiss_index = None
=======
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
>>>>>>> f32d650 (Proper project root: move contents to repo root)

app = FastAPI(title="Document Query Microservice", lifespan=lifespan)

@app.post("/ask")
async def query_endpoint(request: QueryRequest):
    try:
<<<<<<< HEAD
        if not all([client, gemini_embedding_function, chunks, faiss_index]):
            raise HTTPException(status_code=500, detail="Resources not initialized")

        lang = request.user_language.lower()

        
=======
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

>>>>>>> f32d650 (Proper project root: move contents to repo root)
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
<<<<<<< HEAD

        elif lang in ["hr", "croatian", "hrvatski"]:
            
=======
        elif lang in ["hr", "croatian", "hrvatski"]:
>>>>>>> f32d650 (Proper project root: move contents to repo root)
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
<<<<<<< HEAD
            
=======
>>>>>>> f32d650 (Proper project root: move contents to repo root)
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

<<<<<<< HEAD
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("ask:app", host="0.0.0.0", port=8002, reload=True)
=======
        logger.info(f"Query response generated successfully")
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
>>>>>>> f32d650 (Proper project root: move contents to repo root)
