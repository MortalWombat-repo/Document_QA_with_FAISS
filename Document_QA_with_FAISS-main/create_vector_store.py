import os
from pathlib import Path
import logging
from core import extract_text_from_pdf, chunking, import_google_api, embedding_function, build_or_update_faiss_index, save_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store(pdf_folder="files"):
    """Create vector store from all PDF files in the specified folder."""
    logger.info(f"Starting vector store creation from PDFs in {pdf_folder}")

    # Create vector_store directory if it doesn't exist
    vector_store_path = Path("vector_store")
    vector_store_path.mkdir(exist_ok=True)
    logger.info(f"Ensured vector_store directory exists at {vector_store_path}")

    # Initialize Google API client and embedding function
    try:
        client = import_google_api()
        embed_fn = embedding_function(client)
    except Exception as e:
        logger.error(f"Failed to initialize Google API client: {str(e)}")
        raise

    # Get all PDF files from the folder
    pdf_folder_path = Path(pdf_folder)
    pdf_files = [f for f in pdf_folder_path.glob("*.pdf") if f.is_file()]

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF file
    all_chunks = []
    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}")

        # Extract text and create chunks
        try:
            text = extract_text_from_pdf(str(pdf_file))
            if not text:
                logger.warning(f"No text extracted from {pdf_file}")
                continue

            chunks = chunking(text)
            all_chunks.extend(chunks)
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_file}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            continue

    if not all_chunks:
        logger.error("No chunks generated from any PDF files")
        return

    # Build FAISS index and save chunks
    try:
        index = build_or_update_faiss_index(
            chunks=all_chunks,
            gemini_embedding_function=embed_fn,
            index_path="vector_store/index.faiss",
            hash_path="vector_store/file_hashes.json"
        )
        logger.info(f"Created FAISS index with {index.ntotal} vectors")

        # Save chunks to JSON file
        save_chunks(all_chunks, filename="vector_store/chunks.json")

        logger.info("Vector store creation completed successfully")
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_store()
