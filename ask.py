from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from core import (
    extract_texts_from_folder, 
    chunking, 
    import_google_api, 
    embedding_function, 
    build_or_update_faiss_index, 
    query_document,
    query_document_hr, 
    load_chunks,
    save_chunks, 
    load_faiss_index
)
from pathlib import Path
import uvicorn

# request model
class QueryRequest(BaseModel):
    user_query: str
    user_language: str
    top_k: int = 1

# Global variables for resources
client = None
gemini_embedding_function = None
chunks = None
faiss_index = None
folder_path = Path("files").resolve()

# Lifespan event to initialize resources
@asynccontextmanager
async def lifespan(app: FastAPI):
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

app = FastAPI(title="Document Query Microservice", lifespan=lifespan)

@app.post("/ask")
async def query_endpoint(request: QueryRequest):
    try:
        if not all([client, gemini_embedding_function, chunks, faiss_index]):
            raise HTTPException(status_code=500, detail="Resources not initialized")

        lang = request.user_language.lower()

        
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

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("ask:app", host="0.0.0.0", port=8002, reload=True)