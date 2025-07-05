import faiss
import fitz
import numpy as np
from google import genai
from IPython.display import Markdown
from IPython.display import display
from dotenv import load_dotenv
import os
from google.api_core import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.genai import types
from tqdm import tqdm
import time
import hashlib
import json
import glob
from pathlib import Path

folder_path = Path("files").resolve()

def extract_texts_from_folder(folder_path):
    all_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    all_text += page.get_text()
                doc.close()
                all_text += "\n"  # Optional: separate files by a newline
            except FileNotFoundError:
                print(f"Error: {pdf_path} not found.")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
    return all_text

def chunking(contract_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(contract_text)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def import_google_api():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    for m in client.models.list():
        if "embedContent" in m.supported_actions:
            print(m.name)

    return client

def embedding_function(client):
    class GeminiEmbeddingFunction:
        def __init__(self, client):
            self.client = client
            self._retry = retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503})
            self.document_mode = True  # Set to False if you're embedding queries

        def __call__(self, input: list[str]) -> list[list[float]]:
            embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
            response = self._retry(self.client.models.embed_content)(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(task_type=embedding_task),
            )
            return [e.values for e in response.embeddings]

    return GeminiEmbeddingFunction(client)

def hash_chunk(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def build_or_update_faiss_index(chunks, gemini_embedding_function, batch_size=10, index_path="vector_store/index.faiss", hash_path="vector_store/file_hashes.json"):
    hash_path = "vector_store/file_hashes.json"
    faiss_files = glob.glob("*.faiss")
    
    if faiss_files:
        index_path = faiss_files[0]
        index = faiss.read_index(index_path)
        print(f"Loaded existing FAISS index '{index_path}' with {index.ntotal} vectors.")
    else:
        index_path = "vector_store/index.faiss"
        index = None
        print("No existing .faiss file found. Creating new FAISS index.")

    existing_hashes = set()
    if os.path.exists(hash_path):
        with open(hash_path, 'r') as f:
            existing_hashes = set(json.load(f))

    all_embeddings = []
    new_hashes = []
    new_chunks = []

    for chunk in chunks:
        chunk_hash = hash_chunk(chunk)
        if chunk_hash not in existing_hashes:
            new_chunks.append(chunk)
            new_hashes.append(chunk_hash)

    print(f"Found {len(new_chunks)} new chunks to embed.")

    if len(new_chunks) == 0:
        print("All chunks already embedded. Skipping update.")
        return index

    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding new chunks"):
        batch = new_chunks[i:i+batch_size]
        embeddings = gemini_embedding_function(batch)
        all_embeddings.extend(embeddings)
        time.sleep(0.5)

    if all_embeddings:
        embedding_matrix = np.array(all_embeddings).astype("float32")
        dimension = embedding_matrix.shape[1]

        if index is None:
            index = faiss.IndexFlatL2(dimension)
        else:
            if index.d != dimension:
                raise ValueError(f"Dimension mismatch: index has {index.d}, new embeddings have {dimension}")

        index.add(embedding_matrix)
        faiss.write_index(index, index_path)

        updated_hashes = existing_hashes.union(new_hashes)
        with open(hash_path, 'w') as f:
            json.dump(list(updated_hashes), f)

        print(f"Updated FAISS index. Total vectors: {index.ntotal}")
    
    return index

def get_language_name(language_code):
    language_map = {
        "EN": "English",
        "HR": "Croatian"
    }
    return language_map.get(language_code.upper(), "English")

def query_document(user_query, embed_fn, chunks, faiss_index, client, user_language, top_k=1):
    print(user_language.upper())
    embed_fn.document_mode = False

    # Embed the query
    query_embedding = embed_fn([user_query])[0]
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

    # Search the FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    # Get top matching chunks as passages
    all_passages = [chunks[i] for i in indices[0] if i < len(chunks)]

    query_oneline = user_query.replace("\n", " ")
    print(query_oneline)

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

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return answer.text

def query_document_hr(user_query, embed_fn, chunks, faiss_index, client, user_language, top_k=1):
    print(user_language.upper())
    embed_fn.document_mode = False

    translate_to_lang = f"Translate {user_query} to English"

    user_query = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=translate_to_lang
    )

    user_query = user_query.text

    # Embed the query
    query_embedding = embed_fn([user_query])[0]
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

    # Search the FAISS index
    distances, indices = faiss_index.search(query_vector, top_k)

    # Get top matching chunks as passages
    all_passages = [chunks[i] for i in indices[0] if i < len(chunks)]

    query_oneline = user_query.replace("\n", " ")
    print(query_oneline)

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

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return answer.text

def save_chunks(chunks, filename="vector_store/chunks.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_chunks(filename="vector_store/chunks.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_faiss_index(index_path="vector_store/index.faiss"):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return None

def main():
    user_query = "Fakture moraju sadržavati koje informacije?"
    user_language = "hrvatski"
    
    # Initialize Google API
    client = import_google_api()
    
    # Create embedding function
    gemini_embedding_function = embedding_function(client)
    
    # Extract and chunk the contract text
    contract_text = extract_texts_from_folder(folder_path)
    if not contract_text:
        return
    chunks = chunking(contract_text)
    
    # Build or update FAISS index
    faiss_index = build_or_update_faiss_index(chunks, gemini_embedding_function)
    if faiss_index is None:
        print("Failed to create or load FAISS index.")
        return
    
    # Query the document
    response = query_document_hr(user_query, gemini_embedding_function, chunks, faiss_index, client, user_language)
    #display(Markdown(response))
    print(response)

if __name__ == "__main__":
    main()