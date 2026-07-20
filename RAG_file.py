from fastapi import FastAPI, HTTPException, UploadFile, File
import chromadb
import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

client_db=chromadb.PersistentClient(path=("./chroma"))
collection= client_db.get_or_create_collection("knowledge")

app=FastAPI()

@app.post("/ingest")
def pdf_reader(file:UploadFile= File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="please upload pdf file.")
    
    content= file.file.read()
    reader= PdfReader(io.BytesIO(content))

    full_text=""
    for page in reader.pages:
        full_text+=page.extract_text()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap=10
    )
    chunks= splitter.split_text(full_text)
    ids=[f"{file.filename} _chunk{i}"for i in range (len(chunks))]
    collection.upsert(documents=chunks, ids=ids)

    return{"message":f"{len(chunks)}chunks ingested form{file.filename}"}

load_dotenv()
client_grog= Groq(api_key=os.getenv("GROQ_API_KEY"))


class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query:Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="first need to enter question")
    

    result= collection.query(
        query_texts=[query.question],
        n_results=2,
        include=["documents", "distances"]
    )

    best_distance= result["distances"][0][0]
    if best_distance>1.0:
        return{"answer":"i dont have answer of this question in my database"}
    

    context= " ".join(result["documents"][0])


    prompt =f"""
     you are the strict technical assistant. you must answer to the user question if the 
     only provided context in the database. if the context does not contain the answer
     of the question then EXACTLY say : "i am sorry, but i do not have information related
     to this question in my database
     
     context:{context}
     Question:{query.question}
"""
    


    response= client_grog.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user", "content":prompt}]
    )
    return{"answer": response.choices[0].message.content}




@app.post("/search")
def search_database(query:Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="please enter question to check the distance")
    


    result= collection.query(
        query_texts=[query.question],
        n_results=2,
        include=["documents", "distance"]
    )

    return{
        "user_querstion": query.question,
        "best_match": result["distance"][0][0],
        "distance_match": result["distance"][0][0]
    }