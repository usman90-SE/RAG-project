from fastapi import FastAPI, HTTPException, UploadFile, File
import chromadb
import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()
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


client_groq= Groq(api_key=os.getenv("GROQ_API_KEY"))


class UserQuery(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: UserQuery):
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
    


    response= client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user", "content":prompt}]
    )
    return{"answer": response.choices[0].message.content}



@app.post("/search")
def search_database(query: UserQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="please enter question to check the distance")
    


    result= collection.query(
        query_texts=[query.question],
        n_results=2,
        include=["documents", "distances"]
    )

    return{
        "user_querstion": query.question,
        "best_match": result["documents"][0][0],
        "distance_match": result["distances"][0][0]
    }





def expand_query(question: str):
    prompt = f"""You are an expert search engine query optimizer for a vector database.
Generate 3 highly optimized, distinct search queries to maximize document recall.

STRICT RULES:
1. No conversational filler. No "What is", "Can you show", "How to".
2. Do not formulate as questions. Use dense noun phrases and exact keywords.
3. Return ONLY the 3 queries, one per line, no numbering.

Base term: {question}"""

    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    variations = response.choices[0].message.content.strip().split("\n")
    return [question] + variations[:3]


def search_with_expansion(question: str):
    queries = expand_query(question)
    all_docs = []
    for q in queries:
        results = collection.query(
            query_texts=[q],
            n_results=2,
            include=["documents"]
        )
        all_docs.extend(results["documents"][0])

    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)

    return unique_docs[:3]


@app.post("/ask-v2")
def ask_v2(query: UserQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    context_docs = search_with_expansion(query.question)

    if not context_docs:
        return {"answer": "I don't have relevant information about this in my database."}

    context = " ".join(context_docs)

    prompt = f"""You are a strict technical assistant. Answer the question using only the context below.
If the context does not contain the answer, say exactly: "I am sorry, I do not have that information in my database."

Context: {context}
Question: {query.question}"""

    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response.choices[0].message.content}