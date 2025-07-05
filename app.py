from dotenv import load_dotenv
import streamlit as st
import os
import json
import faiss
from pathlib import Path
#import fasttext

from core import (
    extract_texts_from_folder,
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

nltk.download('stopwords')

# Cache the stop words
@st.cache_resource
def load_stopwords():
    english_stopwords = set(stopwords.words('english'))
    croatian_stopwords = "croatian_stopwords.txt"
    croatian_stopwords = set()

    combined_stopwords = (
        english_stopwords
        .union(croatian_stopwords)
    )

    combined_stopwords.update([
        "said", "also", "one", "would", "could", "us", "get", "like"
    ])
    print('croatian' in stopwords.fileids())

    return combined_stopwords

# Load the stop words
combined_stopwords = load_stopwords()

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
    col1, col2 = st.columns([1, 15])

    with col1:
        st.write("\n")
        st.image("img/icon.svg", width=40)

    with col2:
        st.title("Contract chatbot")

    st.markdown("Query the database of contracts.")

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