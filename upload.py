from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from uuid import uuid4
import shutil
from pathlib import Path
import uvicorn
import os

from core import (
    extract_texts_from_folder,
    chunking,
    import_google_api,
    embedding_function,
    build_or_update_faiss_index
)

app = FastAPI()

UPLOAD_BASE = Path("files")
UPLOAD_BASE.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_documents(
    user_id: str = Form(...),
    session: Optional[bool] = Form(False),
    manual_paths: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    session_id = str(uuid4()) if session else "global"
    user_folder = UPLOAD_BASE / user_id / session_id
    user_folder.mkdir(parents=True, exist_ok=True)

    valid_paths = []

    # Save uploaded files into user_folder
    if files:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            file_path = user_folder / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            valid_paths.append(file_path)

    # Use manual_paths directly, no copying
    if manual_paths:
        paths = [Path(p.strip()) for p in manual_paths.split(",")]
        for path in paths:
            if not path.exists() or not path.suffix.lower() == ".pdf":
                raise HTTPException(status_code=400, detail=f"Invalid file path: {path}")
        valid_paths.extend(paths)

    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files provided.")

    try:
        # Extract text from all valid pdf paths
        extracted_text = ""
        for pdf_path in valid_paths:
            extracted_text += extract_texts_from_folder(pdf_path.parent)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text found in uploaded PDFs.")

        chunks = chunking(extracted_text)

        client = import_google_api()
        gemini_embed_fn = embedding_function(client)

        faiss_index_path = user_folder / "vector_store/index.faiss"
        hash_path = user_folder / "vector_store/file_hashes.json"

        build_or_update_faiss_index(
            chunks,
            gemini_embed_fn,
            batch_size=10,
            index_path=str(faiss_index_path),
            hash_path=str(hash_path)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return JSONResponse({
        "message": f"{len(valid_paths)} file(s) uploaded and/or processed.",
        "user_id": user_id,
        "session_id": session_id,
        "file_paths": [str(p) for p in valid_paths],
        "session_mode": session
    })

if __name__ == "__main__":
    uvicorn.run("upload:app", host="0.0.0.0", port=8001, reload=True)