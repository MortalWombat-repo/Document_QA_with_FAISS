import json
from pathlib import Path
import logging
from core import (
    extract_text_from_pdf,
    chunking,
    import_google_api,
    embedding_function,
    build_or_update_faiss_index,
    save_chunks,
    hash_chunk,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store(pdf_folder="files"):
    """Create vector store from all PDF files in the specified folder."""
    logger.info(f"Starting vector store creation from PDFs in {pdf_folder}")

    vector_store_path = Path("vector_store")
    index_path = vector_store_path / "index.faiss"
    hash_path = vector_store_path / "file_hashes.json"
    vector_store_path.mkdir(exist_ok=True)
    logger.info(f"Ensured vector_store directory exists at {vector_store_path}")

    # Initialize embedding function
    try:
        client = import_google_api()
        embed_fn = embedding_function(client)
    except Exception as e:
        logger.error(f"Failed to initialize Google API client: {str(e)}")
        raise

    pdf_folder_path = Path(pdf_folder)
    pdf_files = [f for f in pdf_folder_path.glob("*.pdf") if f.is_file()]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return
    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Load existing hashes if available
    existing_hashes = set()
    if hash_path.exists():
        try:
            with open(hash_path, "r", encoding="utf-8") as f:
                existing_hashes = set(json.load(f))
        except json.JSONDecodeError:
            logger.warning("Hash file is empty or corrupted, starting fresh.")

    # Extract and filter new chunks
    all_chunks = []
    new_chunks = []
    new_hashes = []

    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}")
        try:
            text = extract_text_from_pdf(str(pdf_file))
            if not text:
                logger.warning(f"No text extracted from {pdf_file}")
                continue

            chunks = chunking(text)
            all_chunks.extend(chunks)

            for chunk in chunks:
                h = hash_chunk(chunk)
                if h not in existing_hashes:
                    new_chunks.append(chunk)
                    new_hashes.append(h)

            logger.info(f"Extracted {len(chunks)} chunks from {pdf_file}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            continue

    if not new_chunks:
        logger.info("No new chunks to embed. Vector store is already up to date.")
        return

    try:
        index = build_or_update_faiss_index(
            chunks=new_chunks,
            gemini_embedding_function=embed_fn,
            index_path=str(index_path),
            hash_path=str(hash_path)
        )
        logger.info(f"Updated FAISS index with {index.ntotal} vectors")

        # Save all chunks to JSON (includes all, not just new ones)
        save_chunks(all_chunks, filename="vector_store/chunks.json")
        logger.info("Vector store creation completed successfully")
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_store()
