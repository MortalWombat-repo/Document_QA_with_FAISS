# Document_QA_with_FAISS

## Overview
This is a Retreival Augmented Generation(RAG) system for querying information from documents. The focus is on contractual information but it can be of any type generally. <br>
It uses FAISS(Facebook AI Similarity Search) as its vector store and uses Google Gemini API for inference. <br>
The indexed data is chunked for better throughoutput and hashed so no duplicates are accepted. <br>
It is served in a Streamlit frontend and has 3 modes of use:
* **Local:**  
  This mode allows you to query documents directly on your local machine without any user authentication or session tracking. Ideal for single-user use or quick testing of      the vector store.
  
* **User-based:**  
  In this mode, the system tracks individual users, allowing personalized document queries and maintaining user-specific data across sessions.

* **Session-based:**  
  This mode enables temporary sessions where queries and data are isolated per session, useful for short-lived or anonymous interactions without persistent user accounts.

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

### Different modes
Next steps may vary.  
You will have different options based on the mode you opt for.

* **Local:**  
  The use case is as follows: First add your .pdf files in the files folder, and next run `python create_vector_store.py` to populate with vecor_store files. After that       either run `streamlit run app.py` and choose local option or build a docker container and have many options.

* **User-based:**  
  For this mode there will be a need for building your own docker container with the use of a custom REST API for populating the vector store.
  Docker and REST API usage explained below.

* **Session-based:**  
  This mode also follows the same logic as user-based option.
  Docker and REST API usage bellow.

### Docker container creation
1. To build a Docker container run:
   ```
   docker build -t my-app .
   ```
2. To run a container with all of the exposed ports run:
   ```
   docker run -it -p 8000:8000 -p 8001:8001 -p 8002:8002 my-app
   ```
   Do NOT ommit the -p flag, and do not change the ports unless you also change them in the .py files that uvicorn serves.

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
   ```
   curl -X POST http://localhost:8001/upload \
  -F "files=@YOUR_FILES_ABSOLUTE_PATH" \
  -F "user_id=123" \
  -F "session=false"
   ```
   or for session based
   ```
   curl -X POST http://localhost:8001/upload \
  -F "files=YOUR_FILES_ABSOLUTE_PATH \
  -F "session=true"
   ```
3. run a query from ask.py
   ```
   python ask.py
   ```

### Streamlit app
You have a fully functional Streamlit frontend when ran in Docker or outside of it as explained earlier.
To access Streamlit on your localhost, either look for a link in the terminal output when first running your Docker container or enter into your browser:
   ```
   http://localhost:8000
   ```
Unfortunately, even though it is an active issue that is being worked on, there is still no https support in streamlit.
One might try running nginx as a reverse proxy to remedy that but it is a tiring endeavour, which I do not recommend.



  
