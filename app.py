from dotenv import load_dotenv
import streamlit as st
import os
import json
import faiss
from pathlib import Path
import requests
from core import (
    extract_text_from_pdf,
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
        Language.CROATIAN,
        Language.ENGLISH
    ).build()
    return detector

def detect_language_lingua(text):
    detector = load_language_detector()
    lang = detector.detect_language_of(text)
    return lang.name if lang else "Unknown"

import nltk
from nltk.corpus import stopwords

try:
    nltk.download('stopwords', quiet=True)
except:
    pass

@st.cache_resource
def load_stopwords():
    try:
        english_stopwords = set(stopwords.words('english'))
    except:
        english_stopwords = set()

    croatian_stopwords = set()
    combined_stopwords = english_stopwords.union(croatian_stopwords)
    combined_stopwords.update(["said", "also", "one", "would", "could", "us", "get", "like"])
    return combined_stopwords

combined_stopwords = load_stopwords()

def get_data_path(mode, user_id=None, session_id=None):
    if mode == "Global":
        return Path("./vector_store")
    elif mode == "User":
        return Path("/app/users") / user_id
    elif mode == "Session":
        return Path("/app/sessions") / session_id
    raise ValueError("Invalid mode or missing user_id/session_id")

def get_user_session_data(mode, user_id=None, session_id=None):
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
    folder_path = get_data_path(mode, user_id, session_id)
    if not folder_path.exists():
        return ""

    contract_text = ""
    for pdf_file in folder_path.glob("*.pdf"):
        contract_text += extract_text_from_pdf(str(pdf_file)) + "\n"
    return contract_text

def main():
    col1, col2 = st.columns([1, 15])
    with col1:
        st.write("\n")
        st.image("img/icon.svg", width=40)
    with col2:
        st.title("Contract chatbot")

    st.markdown("Query the database of contracts.")

    with st.sidebar:
        st.markdown("## Configuration")
        mode_type = st.radio("Select storage mode:", ["Global", "User", "Session"], key="mode_type_selection")

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

    if mode_type in ["User", "Session"] and not (user_id or session_id):
        st.warning(f"Please enter {'User ID' if mode_type == 'User' else 'Session ID'}.")
        return

    client = import_google_api()
    gemini_embed_fn = embedding_function(client)

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
