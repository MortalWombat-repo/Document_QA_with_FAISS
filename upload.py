from pathlib import Path
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from uuid import uuid4
import shutil
import uvicorn

from redis_utils import redis_cache
from core import (
    extract_text_from_pdf,
    chunking,
    import_google_api,
    embedding_function,
    build_or_update_faiss_index,
    save_chunks
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/upload")
async def upload_documents(
    user_id: Optional[str] = Form(None),
    session: Optional[bool] = Form(False),
    manual_paths: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    if session:
        session_id = str(uuid4())
        user_folder = Path("/app/sessions") / session_id
    elif user_id:
        session_id = "manual"
        user_folder = Path("/app/users") / user_id if user_id != "global" else Path("/app/files") / user_id / "manual"
    else:
        raise HTTPException(status_code=400, detail="Either user_id or session=true must be provided")

    logger.info(f"Creating folder: {user_folder}")
    if user_folder.exists():
        shutil.rmtree(user_folder)
    user_folder.mkdir(parents=True, exist_ok=True)

    valid_paths = []

    # Save uploaded files
    if files:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            file_path = user_folder / file.filename
            logger.info(f"Saving file: {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            valid_paths.append(file_path)

    # Handle manual paths
    if manual_paths:
        paths = [Path(p.strip()) for p in manual_paths.split(",")]
        for path in paths:
            if not path.exists() or path.suffix.lower() != ".pdf":
                raise HTTPException(status_code=400, detail=f"Invalid file path: {path}")
        valid_paths.extend(paths)

    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files provided.")

    try:
        # Extract and process
        extracted_text = ""
        for pdf_path in valid_paths:
            logger.info(f"Extracting text from file: {pdf_path}")
            extracted_text += extract_text_from_pdf(str(pdf_path)) + "\n"

        if not extracted_text.strip():
            logger.error("No text extracted from PDFs")
            raise HTTPException(status_code=400, detail="No text found in uploaded PDFs.")

        logger.info(f"Chunking text, length: {len(extracted_text)}")
        chunks = chunking(extracted_text)
        logger.info(f"Created {len(chunks)} chunks.")

        client = import_google_api()
        gemini_embed_fn = embedding_function(client)

        vector_store_path = user_folder / "vector_store"
        chunks_path = vector_store_path / "chunks.json"
        faiss_index_path = vector_store_path / "index.faiss"
        hash_path = vector_store_path / "file_hashes.json"

        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(chunks)} chunks to: {chunks_path}")
        save_chunks(chunks, filename=str(chunks_path))

        logger.info(f"Building FAISS index at: {faiss_index_path}")
        build_or_update_faiss_index(
            chunks,
            gemini_embed_fn,
            batch_size=10,
            index_path=str(faiss_index_path),
            hash_path=str(hash_path)
        )

        # Invalidate related Redis cache
        if session:
            redis_cache.delete("app_session", "Session", None, session_id)
            redis_cache.delete("user_session", None, session_id)
        else:
            redis_cache.delete("app_session", "User", user_id, None)
            redis_cache.delete("user_session", user_id, None)

        redis_cache.delete("chunks", str(chunks_path))
        redis_cache.delete("faiss_index", str(faiss_index_path))

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return JSONResponse({
        "message": f"{len(valid_paths)} file(s) uploaded and/or processed.",
        "user_id": user_id,
        "session_id": session_id,
        "file_paths": [str(p) for p in valid_paths],
        "session_mode": session,
        "cache_invalidated": True
    })

if __name__ == "__main__":
    uvicorn.run("upload:app", host="0.0.0.0", port=8001, reload=True)