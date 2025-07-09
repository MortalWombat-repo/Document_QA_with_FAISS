# Document_QA_with_FAISS

# Table of Contents
1. [Overview](#overview)
2. [Stack](#stack)
   - [Features](#features)
   - [Technologies](#technologies)
3. [Usage](#usage)
   - [System Requirements](#system-requirements)
   - [Cloning the Project](#cloning-the-project)
   - [Supplying Your Own API Key](#supplying-your-own-api-key)
4. [Docker Container Creation](#docker-container-creation)
5. [REST API](#rest-api)
   - [Uploading Documents](#uploading-documents)
   - [Querying the Database](#querying-the-database)
6. [Different Modes](#different-modes)
   - [Local Mode](#local-mode)
   - [User-based Mode](#user-based-mode)
   - [Session-based Mode](#session-based-mode)
7. [Streamlit App](#streamlit-app)
8. [Video Demonstration](#video-demonstration)
   - [Building the Container](#building-the-container)
   - [Populating the Vector Database](#populating-the-vector-database)
   - [Querying via POST Request](#querying-via-post-request)
   - [Streamlit Demo](#streamlit-demo)
9. [Possible Improvements](#possible-improvements)


## Overview
This is a **Retrieval Augmented Generation (RAG)** system for querying information from documents. The focus is on contractual information but it can be of any type generally. <br>
It uses **FAISS (Facebook AI Similarity Search)** as its vector store and uses **Google Gemini API** for inference. <br>
The indexed data is chunked for better throughoutput and hashed so no duplicates are accepted. <br>
The system is served in a **Streamlit** frontend, and has 3 modes of use:

* **Local:**  
  This mode allows you to query documents directly on your local machine without any user authentication or session tracking. Ideal for single-user use or quick testing of      the vector store.
  
* **User-based:**  
  In this mode, the system tracks individual users, allowing personalized document queries and maintaining user-specific data across sessions.

* **Session-based:**  
  This mode enables temporary sessions where queries and data are isolated per session, useful for short-lived or anonymous interactions without persistent user accounts.

## Stack (Technologies used and features supported)

### Features

- **PDF Text Extraction**: Extracts text from PDFs using PyMuPDF.
- **Semantic Search**: Queries contracts using natural language with FAISS-based vector search.
- **Multilingual Support**: Handles English and Croatian queries with automatic language detection.
- **Flexible Storage Modes**: Supports Global, User, and Session-based data storage.
- **Web Interface**: Streamlit app for easy querying and data status visualization.
- **API Services**: FastAPI endpoints for uploading PDFs and querying documents.
- **Robust Error Handling**: Comprehensive error management and logging.
- **Session Management**: Automatic cleanup of temporary session data.
- **Professional Responses**: Formal, structured answers with context for contract-related queries.

### Technologies

- **Python**: Core programming language.
- **Streamlit**: Web interface framework.
- **FastAPI**: Asynchronous API framework.
- **FAISS**: Vector storage and search.
- **PyMuPDF**: PDF text extraction.
- **LangChain**: Text chunking.
- **Google Generative AI**: Text embedding and content generation.
- **Lingua**: Language detection.
- **NLTK**: Stopwords processing. -- In progress, first there is a need for automatic labeling of metadata when chunking
- **Uvicorn**: ASGI server for FastAPI.
- **JSON/FAISS**: Data storage formats.
- **Bash**: Deployment script

## Usage
Important!

This project was designed for a Linux-like environment, you may use either Linux or WSL (Windows subsystem for Linux) on Windows.  
[Link for WSL tutorial](https://www.howtogeek.com/744328/how-to-install-the-windows-subsystem-for-linux-on-windows-11/)  

You will also need Docker.  
[Docker download link](https://www.docker.com/)

### Cloning the project
1. cd into your desired folder and download the project
```
git clone https://github.com/MortalWombat-repo/Document_QA_with_FAISS.git
```
2. cd into the folder
```
cd Document_QA_with_FAISS
```

### Supplying your own API key
This project uses Google Gemini API for inference.
To use this project you should supply your own API key.

You can create your own API key [AT THIS LINK](https://aistudio.google.com/app/apikey)  
You should add it to the .env file.
```
nano .env
```

### Docker container creation
1. To build a Docker container run:
```
docker build -t my-app .
```
2. To run a container with all of the exposed ports run:
```
docker run -it -p 8000:8000 -p 8001:8001 -p 8002:8002 my-app
```
   Do NOT omit the -p flag, and do not change the ports unless you also change them in the .py files that uvicorn serves.

### REST API
The REST API is implemented with FastAPI and a production ready server uvicorn.

For session and user-based modes, you should utilize the functionality from the `upload.py` file that borrows from `core.py`, by sending a POST request with the file to be vectorized to a listening server, either by a direct CURL request from the terminal, or by a script that mimics curl-like behavior in testing folder.
After the vectorization you can either query via POST request or from a Streamlit frontend.

1. cd into testing
```
cd testing
```
   
2. run either test_upload_manual.py or test_upload_session.py
```
python test_upload_manual.py
```
   
```
python test_upload_session.py
```
   
   Alternatively run it through curl
   for user/manual
```bash
curl -X POST http://localhost:8001/upload \
-F "files=@YOUR_FILES_ABSOLUTE_PATH" \
-F "user_id=123" \
-F "session=false"
```
   or for session based
```
curl -X POST http://localhost:8001/upload \
-F "files=@YOUR_FILES_ABSOLUTE_PATH \
-F "session=true"
```
3. run a query from ask.py
```
python test_ask.py
```
  or run from curl
```
curl -X POST http://localhost:8002/ask \
-H "Content-Type: application/json" \
-d '{
"user_query": "The invoices must contain which information?",
"user_language": "en",
"top_k": 1,
"user_id": "123",
"session_id": "manual"
}'
```

### Different modes
Next steps may vary.  
You will have different options based on the mode you opt for.

* **Local:**  
  The use case is as follows: First add your .pdf files in the files folder, and next run `python create_vector_store.py` to populate with vecor_store files. After that       either run `streamlit run app.py` and choose local option or build a docker container and use it like that.

* **User-based:**  
  For this mode there will be a need for building your own docker container with the use of a custom REST API for populating the vector store.
  Populate the database like was demonstrated in REST API section and copy the username, you will need it for authentication with Streamlit.

* **Session-based:**  
  This mode also follows the same logic as user-based option.
  You need to record your session id for authentication in Streamlit.

### Streamlit App
You have a fully functional Streamlit frontend when ran in Docker or outside of it as explained earlier.
To access Streamlit on your localhost, either look for a link in the terminal output when first running your Docker container or enter into your browser:
```
http://localhost:8000
```
Unfortunately, even though it is an active issue that is being worked on, there is still no https support in streamlit.
One might try running nginx as a reverse proxy to remedy that, but that is complex and not recommended for this setup.

## Video Demonstration

### Building the container
After cloning the repository, cd-ing into the project, and generating your own API KEY as described, build the container.

[![docker](https://i.imgur.com/fDSCDmN.gif)](https://i.imgur.com/fDSCDmN.gif)

### Populating the vector database
You can either run `create_vector_store.py` or populate it yourself using POST requests, either from prepared scripts in the `testing` folder or manually through curl as mentioned.
We will demonstrate both methods.

#### Creating the vector database from the files folder locally
run `python create_vector_store.py`

[![create_vector_store](https://i.imgur.com/0QJxheF.gif)](https://i.imgur.com/0QJxheF.gif)

#### Creating the vector database using a POST request in a Docker container
using a POST request and demonstrating the created files in Docker.
We will need to keep a running container for this process.

[![create_vector_store_post_request](https://i.imgur.com/HkqHnyY.gif)](https://i.imgur.com/HkqHnyY.gif)

### Querying the database through a POST request

[![post_request_query](https://i.imgur.com/MpAB60x.gif)](https://i.imgur.com/MpAB60x.gif)

### Streamlit demo
As mentioned there are three modes you can choose from.  
Global/Local mode takes data in from the vector_store folder that you populate with `create_vector_store.py`script.  
For User and Session based, you will need to authorize in Streamlit by copying user_id or session_id.

[streamlit demo video](https://i.imgur.com/EEQLaKa.gif)

The video is too large to embed.

## Possible improvements

* adding metadata when chunking and labeling each chunk
* using said matadata to represent chunks as a whole document for easier keyword visualization and whole document operations such as summarizing and paraphrasing.
* labeler of key terms and ranking of documents by importance






