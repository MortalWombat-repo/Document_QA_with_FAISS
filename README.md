# Document_QA_with_FAISS

## Overview
This is a Retreival Augmented Generation(RAG) system for querying information from documents. The focus is on contractual information but it can be of any type generally. <br>
It uses FAISS(Facebook AI Similarity Search) as its vector store and uses Google Gemini API for inference. <br>
The indexed data is chunked for better throughoutput and hashed so no duplicates are accepted. <br>
It is served in a Streamlit frontend and has 3 modes of use:
* **Local:**  
  This mode allows you to query documents directly on your local machine without any user authentication or session tracking. Ideal for single-user use or quick testing of      the vector store.
  
  The use case is as follows: First add your .pdf files in the files folder, and next run `python create_vector_store.py` to populate with vecor_store files. After that just    run

* **User-based:**  
  In this mode, the system tracks individual users, allowing personalized document queries and maintaining user-specific data across sessions.

* **Session-based:**  
  This mode enables temporary sessions where queries and data are isolated per session, useful for short-lived or anonymous interactions without persistent user accounts.

## Usage
Important!

This project was designed for a Linux-like environment, you may use either Linux or WSL (Windows subsystem for Linux) on Windows. <br>
[Link for WSL tutorial](https://www.howtogeek.com/744328/how-to-install-the-windows-subsystem-for-linux-on-windows-11/) <br>
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
To build a Docker container

  
