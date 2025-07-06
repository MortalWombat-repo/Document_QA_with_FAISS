from dotenv import load_dotenv
import streamlit as st
import os
import json
import faiss
from pathlib import Path
<<<<<<< HEAD
#import fasttext

from core import (
    extract_texts_from_folder,
=======
import requests
from core import (
    extract_text_from_pdf,
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    chunking,
    import_google_api,
    embedding_function,
    hash_chunk,
    build_or_update_faiss_index,
    get_language_name,
    query_document,
    query_document_hr,
    save_chunks,
    load_chunks,
    load_faiss_index
)

# Streamlit Config
st.set_page_config(page_title="Contract chatbot", layout="wide")

from lingua import Language, LanguageDetectorBuilder
@st.cache_resource
def load_language_detector():
    detector = LanguageDetectorBuilder.from_languages(
<<<<<<< HEAD
    Language.CROATIAN,
    Language.ENGLISH
=======
        Language.CROATIAN,
        Language.ENGLISH
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    ).build()
    return detector

def detect_language_lingua(text):
    detector = load_language_detector()
    lang = detector.detect_language_of(text)
    return lang.name if lang else "Unknown"

import nltk
from nltk.corpus import stopwords

<<<<<<< HEAD
nltk.download('stopwords')
=======
try:
    nltk.download('stopwords', quiet=True)
except:
    pass
>>>>>>> f32d650 (Proper project root: move contents to repo root)

# Cache the stop words
@st.cache_resource
def load_stopwords():
<<<<<<< HEAD
    english_stopwords = set(stopwords.words('english'))
    croatian_stopwords = "croatian_stopwords.txt"
    croatian_stopwords = set()

=======
    try:
        english_stopwords = set(stopwords.words('english'))
    except:
        english_stopwords = set()

    croatian_stopwords = set()
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    combined_stopwords = (
        english_stopwords
        .union(croatian_stopwords)
    )
<<<<<<< HEAD

    combined_stopwords.update([
        "said", "also", "one", "would", "could", "us", "get", "like"
    ])
    print('croatian' in stopwords.fileids())

=======
    combined_stopwords.update([
        "said", "also", "one", "would", "could", "us", "get", "like"
    ])
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    return combined_stopwords

# Load the stop words
combined_stopwords = load_stopwords()

<<<<<<< HEAD
# -----------------------------------
# Importing Google API key, embedding function, collection, and retry
# -----------------------------------
client = import_google_api()
gemini_embed_fn = embedding_function(client)

# Chunking logic
chunks = load_chunks()
if chunks is None:
    # Extract text from PDF
    contract_text = extract_texts_from_folder("files")
    if not contract_text:
        print("No contract text extracted from any file.")
        chunks = []
    else:
        # chunk the extracted text
        chunks = chunking(contract_text)
        # Save the chunks
        save_chunks(chunks)

# Load FAISS index if exists
faiss_index = load_faiss_index()
if faiss_index is None:
    faiss_index = build_or_update_faiss_index(chunks, gemini_embed_fn)
else:
    print(f"Using cached FAISS index with {faiss_index.ntotal} vectors.")

if faiss_index is None:
    print("Failed to create or load FAISS index.")

# -----------------------------------
# Streamlit UI
# -----------------------------------


#def detect_language_fasttext(text):
    #prediction = ft_lang_model.predict(text.replace('\n', ' ').strip())
    #lang_code = prediction[0][0].replace('__label__', '')
    #return lang_code


def main():
    #Title page
=======
def get_data_path(mode, user_id=None, session_id=None):
    """Get data path based on mode"""
    if mode == "Global":
        return Path("./vector_store")  # Outside Docker container, same as Dockerfile level
    elif mode == "User":
        return Path("/app/users") / user_id
    elif mode == "Session":
        return Path("/app/sessions") / session_id
    raise ValueError("Invalid mode or missing user_id/session_id")

def get_user_session_data(mode, user_id=None, session_id=None):
    """Load data from appropriate vector store based on mode"""
    data_path = get_data_path(mode, user_id, session_id)
    chunks_path = data_path / "chunks.json"
    faiss_path = data_path / "index.faiss"
    hash_path = data_path / "file_hashes.json"

    if chunks_path.exists() and faiss_path.exists() and hash_path.exists():
        chunks = load_chunks(str(chunks_path))
        faiss_index = load_faiss_index(str(faiss_path))
        return chunks, faiss_index
    return None, None

def extract_texts_from_folder(mode, user_id=None, session_id=None):
    """Extract text from all PDFs in the specified folder"""
    folder_path = get_data_path(mode, user_id, session_id)
    if not folder_path.exists():
        return ""

    contract_text = ""
    for pdf_file in folder_path.glob("*.pdf"):
        contract_text += extract_text_from_pdf(str(pdf_file)) + "\n"
    return contract_text

def main():
    # Title page
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    col1, col2 = st.columns([1, 15])

    with col1:
        st.write("\n")
        st.image("img/icon.svg", width=40)

    with col2:
        st.title("Contract chatbot")

    st.markdown("Query the database of contracts.")

<<<<<<< HEAD
    # Sidebar
    with st.sidebar:
        st.markdown("## About")
        st.markdown("")
        st.markdown("A Streamlit app for querying contract information.")

    # Tabs
    explain_tab = st.tabs(["Explain entry"])[0]

    with explain_tab:
        with st.form(key="query_form"):
            user_query = st.text_input("💬 Query the database:", placeholder="e.g., The invoices must include what information?")
            submit_button = st.form_submit_button("🔎 Search the database")

        if submit_button and user_query:
            with st.spinner("Searching for the article..."):
                user_language = detect_language_lingua(user_query)
                if user_language == "ENGLISH":
                    article = query_document(user_query, gemini_embed_fn, chunks, faiss_index, client, user_language, top_k=1)
                elif user_language == "CROATIAN":
                    article = query_document_hr(user_query, gemini_embed_fn, chunks, faiss_index, client, user_language, top_k=1)

            st.write(f"Detected language: {user_language}")
            st.subheader("📖 Answer")
            st.markdown(article)

            # Save into session_state
            #st.session_state['article'] = article
            #st.session_state['user_query'] = user_query

        # Always display article if it exists in session state
        #if 'article' in st.session_state:
            #st.subheader("📖 Answer")
            #st.markdown(st.session_state['article'])

if __name__ == "__main__":
    main()
=======
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## Configuration")

        # Mode selection
        mode_type = st.radio(
            "Select storage mode:",
            ["Global", "User", "Session"],
            key="mode_type_selection"
        )

        user_id = None
        session_id = None

        if mode_type == "User":
            user_id = st.text_input("User ID:", key="user_id_input")
        elif mode_type == "Session":
            session_id = st.text_input("Session ID:", key="session_id_input")

        st.markdown("---")
        st.markdown("## About")
        st.markdown("A Streamlit app for querying contract information.")
        st.markdown(f"**Current mode:** {mode_type}")
        if user_id:
            st.markdown(f"**User ID:** {user_id}")
        elif session_id:
            st.markdown(f"**Session ID:** {session_id}")

    # Validate input for User/Session modes
    if mode_type in ["User", "Session"] and not (user_id or session_id):
        st.warning(f"Please enter {'User ID' if mode_type == 'User' else 'Session ID'}.")
        return

    client = import_google_api()
    gemini_embed_fn = embedding_function(client)

    # Load or create data
    chunks, faiss_index = get_user_session_data(mode_type, user_id, session_id)
    if chunks is None or faiss_index is None:
        contract_text = extract_texts_from_folder(mode_type, user_id, session_id)
        if contract_text:
            chunks = chunking(contract_text)
            vector_store_path = get_data_path(mode_type, user_id, session_id)
            chunks_path = vector_store_path / "chunks.json"
            faiss_path = vector_store_path / "index.faiss"
            hash_path = vector_store_path / "file_hashes.json"

            vector_store_path.mkdir(parents=True, exist_ok=True)
            save_chunks(chunks, str(chunks_path))
            faiss_index = build_or_update_faiss_index(
                chunks, gemini_embed_fn,
                index_path=str(faiss_path),
                hash_path=str(hash_path)
            )
        else:
            chunks = []
            faiss_index = None
            st.error(f"No data found for {mode_type}{' ID: ' + (user_id or session_id) if mode_type in ['User', 'Session'] else ''}")
            st.info("Please upload documents first using the upload API.")
            return

    # Main interface
    explain_tab = st.tabs(["Query Documents"])[0]

    with explain_tab:
        with st.form(key="query_form"):
            user_query = st.text_input(
                "💬 Query the database:",
                placeholder="e.g., The invoices must include what information?"
            )
            submit_button = st.form_submit_button("🔎 Search the database")

        if submit_button and user_query:
            if not chunks or not faiss_index:
                st.error("No data available for querying. Please check your configuration.")
                return

            with st.spinner("Searching for the answer..."):
                user_language = detect_language_lingua(user_query)

                try:
                    if user_language == "ENGLISH":
                        article = query_document(
                            user_query, gemini_embed_fn, chunks, faiss_index,
                            client, user_language, top_k=1
                        )
                    elif user_language == "CROATIAN":
                        article = query_document_hr(
                            user_query, gemini_embed_fn, chunks, faiss_index,
                            client, user_language, top_k=1
                        )
                    else:
                        article = query_document(
                            user_query, gemini_embed_fn, chunks, faiss_index,
                            client, user_language, top_k=1
                        )

                    st.success(f"Detected language: {user_language}")
                    st.subheader("📖 Answer")
                    st.markdown(article)

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

        # Display current data info
        if chunks and faiss_index:
            st.markdown("---")
            st.markdown("### Current Data Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", len(chunks))
            with col2:
                st.metric("FAISS Index Vectors", faiss_index.ntotal)

if __name__ == "__main__":
    main()
>>>>>>> f32d650 (Proper project root: move contents to repo root)
