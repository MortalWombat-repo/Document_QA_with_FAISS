import faiss
import fitz
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from google.api_core import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.genai import types
from tqdm import tqdm
import time
import hashlib
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF file."""
    logger.info(f"Extracting text from {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text
    except FileNotFoundError:
        logger.error(f"PDF not found: {pdf_path}")
        return ""
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return ""

def chunking(contract_text: str) -> list[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    logger.info(f"Chunking text, length: {len(contract_text)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(contract_text)
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks

def import_google_api():
    """Initialize Google API client."""
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        raise ValueError("GOOGLE_API_KEY not found")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    logger.info("Google API client initialized")
    return client

def embedding_function(client):
    """Create a Gemini embedding function."""
    class GeminiEmbeddingFunction:
        def __init__(self, client):
            self.client = client
            self._retry = retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503})
            self.document_mode = True  # Set to False for queries

        def __call__(self, input: list[str]) -> list[list[float]]:
            embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
            try:
                response = self._retry(self.client.models.embed_content)(
                    model="models/text-embedding-004",
                    contents=input,
                    config=types.EmbedContentConfig(task_type=embedding_task),
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                logger.error(f"Error embedding content: {str(e)}")
                raise

    return GeminiEmbeddingFunction(client)

def hash_chunk(text: str) -> str:
    """Generate SHA-256 hash for a chunk."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def build_or_update_faiss_index(chunks, gemini_embedding_function, batch_size=10, index_path="vector_store/index.faiss", hash_path="vector_store/file_hashes.json"):
    """Build a new FAISS index in the specified path, ignoring existing indexes."""
    logger.info(f"Building new FAISS index at {index_path}")
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(hash_path).parent.mkdir(parents=True, exist_ok=True)

    # Always create a new index
    all_embeddings = []
    new_hashes = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i:i+batch_size]
        try:
            embeddings = gemini_embedding_function(batch)
            all_embeddings.extend(embeddings)
            new_hashes.extend([hash_chunk(chunk) for chunk in batch])
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
            raise

    if not all_embeddings:
        logger.error("No embeddings generated")
        raise ValueError("No embeddings generated")

    embedding_matrix = np.array(all_embeddings).astype("float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    faiss.write_index(index, index_path)

    with open(hash_path, 'w', encoding='utf-8') as f:
        json.dump(new_hashes, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved FAISS index with {index.ntotal} vectors to {index_path} and hashes to {hash_path}")
    return index

def get_language_name(language_code: str) -> str:
    """Map language code to name."""
    language_map = {
        "EN": "English",
        "HR": "Croatian"
    }
    return language_map.get(language_code.upper(), "English")

def query_document(user_query, embed_fn, chunks, faiss_index, client, user_language, top_k=1):
    """Query the document in English."""
    logger.info(f"Querying in {user_language.upper()}: {user_query}")
    embed_fn.document_mode = False

    # Embed the query
    query_embedding = embed_fn([user_query])[0]
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

    # Search the FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    # Get top matching chunks as passages
    all_passages = [chunks[i] for i in indices[0] if i < len(chunks)]

    query_oneline = user_query.replace("\n", " ")
    prompt = f"""
    You are a professional and informative assistant responding in the **same language as the user's query**, which is {get_language_name(user_language.upper())}, based on the reference passage provided below.

    Your role:
    You assist users in understanding and interpreting **formal documents**, particularly **contracts, agreements, or legal communications**. Your explanations must maintain a **clear, formal tone** suitable for professional contexts such as business, law, or compliance.

    Your communication style:
    - Maintain a **formal, objective, and concise** tone—avoid conversational language.
    - Begin your response with: **"Here is a summary of the relevant contractual information..."**
    - Present insights in **structured paragraphs**, using bullet points when listing terms, clauses, obligations, or risks.

    URL Handling:
    - If a URL is present in the text, extract and **display it at the beginning**, separated by a newline.
    - The format will resemble: `"URL is [url to display]"`
    - If multiple URLs exist, **prioritize the one related to a party or person** involved in the document.

    Content Instructions:
    - Respond with at least **100 words**.
    - Use **precise, professional legal language** where appropriate.
    - Clarify contractual terms (e.g., indemnity, force majeure, jurisdiction).
    - If the passage is irrelevant or lacks information, respond using **general knowledge about standard contracts**.
    - Provide **context** behind clauses, their typical implications, responsibility, timelines, or penalties are discussed, even if minor.

    QUESTION in {get_language_name(user_language.upper())}: {query_oneline}
    """

    for passage in all_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"\nPASSAGE: {passage_oneline}"

    try:
        answer = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        logger.info("Query response generated successfully")
        return answer.text
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise

def query_document_hr(user_query, embed_fn, chunks, faiss_index, client, user_language, top_k=1):
    """Query the document in Croatian, translating to English first."""
    logger.info(f"Querying in {user_language.upper()}: {user_query}")
    embed_fn.document_mode = False

    translate_to_lang = f"Translate {user_query} to English"
    try:
        user_query_en = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=translate_to_lang
        ).text
    except Exception as e:
        logger.error(f"Error translating query to English: {str(e)}")
        raise

    # Embed the query
    query_embedding = embed_fn([user_query_en])[0]
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

    # Search the FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    # Get top matching chunks as passages
    all_passages = [chunks[i] for i in indices[0] if i < len(chunks)]

    query_oneline = user_query_en.replace("\n", " ")
    prompt = f"""
    Ti si profesionalan i informativan asistent koji odgovara na **Hrvatskom jeziku**.
    Izvorni odlomci su na engleskom jeziku. Prije obrade, prevedi odlomke na Hrvatski kako bi se osigurala semantička usklađenost s upitom. ODLOMAK je na kraju i obavezno ga prevedi na hrvatski prije nego kreneš s ičim drugim.
    Prijevod koristi kao kontekst i NE stavljaj ga u rezultat.

    Tvoja uloga:
    Pomažeš korisnicima u razumijevanju i tumačenju **službenih dokumenata**, posebno **ugovora, sporazuma ili pravne korespondencije**. Tvoja objašnjenja moraju imati **jasan, formalan ton** prikladan za profesionalne kontekste kao što su poslovanje, pravo ili usklađenost.

    Stil komunikacije:
    - Održavaj **formalan, objektivan i sažet** ton — izbjegavaj razgovorni jezik.
    - Počni odgovor s: **"Ovo je sažetak relevantnih ugovornih informacija..."**
    - Predstavi uvide u **strukturiranim odlomcima**, koristeći nabrajanja kod navođenja uvjeta, klauzula, obveza ili rizika.

    Rukovanje URL-ovima:
    - Ako je URL prisutan u tekstu, izdvoji ga i **prikaži na početku**, odvojen novim redom.
    - Format će izgledati ovako: `"URL je [url za prikaz]"`
    - Ako postoji više URL-ova, **prioritet daj onome koji se odnosi na stranku ili osobu** uključenu u dokument.

    Upute za sadržaj:
    - Odgovori s najmanje **100 riječi**.
    - Koristi **precizan, profesionalan pravni jezik** gdje je to prikladno.
    - Pojasni ugovorne pojmove (npr. odšteta, viša sila, nadležnost).
    - Ako prevedeni odlomci nisu dovoljni ili nisu relevantni, odgovori koristeći **opće znanje o standardnim ugovorima** na jeziku upita.
    - Pruži **kontekst** klauzula, njihovu tipičnu primjenu, odgovornosti, rokove ili kazne, čak i ako su manje važni.

    PITANJE na jeziku Engleskom: {query_oneline}, koje treba prevesti na Hrvatski.
    """

    for passage in all_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"\nODLOMCI: {passage_oneline}"

    try:
        answer = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        logger.info("Query response generated successfully")
        return answer.text
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise

def save_chunks(chunks, filename="vector_store/chunks.json"):
    """Save chunks to a JSON file."""
    logger.info(f"Saving {len(chunks)} chunks to {filename}")
    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved {len(chunks)} chunks to {filename}")
    except Exception as e:
        logger.error(f"Failed to save chunks to {filename}: {str(e)}")
        raise

def load_chunks(filename="vector_store/chunks.json"):
    """Load chunks from a JSON file."""
    logger.info(f"Loading chunks from {filename}")
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {filename}")
            return chunks
        logger.warning(f"Chunks file not found: {filename}")
        return None
    except Exception as e:
        logger.error(f"Error loading chunks from {filename}: {str(e)}")
        return None

def load_faiss_index(index_path="vector_store/index.faiss"):
    """Load FAISS index from a file."""
    logger.info(f"Loading FAISS index from {index_path}")
    try:
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors from {index_path}")
            return index
        logger.warning(f"FAISS index file not found: {index_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading FAISS index from {index_path}: {str(e)}")
        return None
