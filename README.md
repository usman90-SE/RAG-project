A Retreival-Augmented Generation (RAG) api build with FasApi, chromaDB, and Groq
Upload a PDF document and ask a question, and the system retrieves relevant context from 
Your document, which you upload, generates accurate answers using an LLM


Tech stack:
FastAPI API framework
ChromaDB vector database for semantic search
Groq  for answer generation 
PyPDF + LangChain text splitters, document ingestion, and chunking


setup
1. Clone the repo
2. Install dependencies: pip install fastapi uvicorn chromadb pypdf
langchain-text-splitters groq python-dotenv sentence-transformer
3. Create a .env file and add: api key
4. Run the server: uvicorn RAG_file:app --reload


Endpoints:
POST/ingest: upload a PDF file, split it into chunks, and store it in 
chromaDB
POST/ask: ask a question, retrieves relevant chunks, and returns answers
POST/search: search the database and return the top 2 most relevant document chunks
