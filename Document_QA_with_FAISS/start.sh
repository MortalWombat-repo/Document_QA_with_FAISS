#!/bin/bash
uvicorn upload:app --host 0.0.0.0 --port 8001 &
uvicorn ask:app --host 0.0.0.0 --port 8002 &
streamlit run app.py --server.port=8000 --server.enableCORS=false