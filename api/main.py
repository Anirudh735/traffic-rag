import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# --- Set up FastAPI app ---
app = FastAPI(
    title="Traffic Violation RAG Assistant",
    description="Query traffic violation records using natural language",
    version="1.0.0"
)

# --- Connect to ChromaDB (same as query.py) ---
client = PersistentClient(path="./chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(
    name="traffic_violations",
    embedding_function=embedding_fn
)

# --- Set up Groq LLM ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# --- Pydantic models define the shape of incoming requests ---
# FastAPI uses these to auto-validate JSON input and auto-generate docs

class QueryRequest(BaseModel):
    question: str
    n_results: int = 4  # default to 4 if not provided

class IngestRequest(BaseModel):
    plate: str
    camera: str
    violation: str
    timestamp: str
    fine_amount: int
    status: str
    speed_kmph: int | None = None


# --- /query endpoint ---
@app.post("/query")
def query_violations(request: QueryRequest):
    results = collection.query(
        query_texts=[request.question],
        n_results=request.n_results
    )
    retrieved_docs = results["documents"][0]

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No matching records found")

    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    system_prompt = (
        "You are a traffic violation assistant. Answer the user's question "
        "using ONLY the violation records provided below. If the records don't "
        "contain enough information to answer, say so clearly. Be concise and factual."
    )
    user_prompt = f"""Violation records:
{context}

Question: {request.question}

Answer based only on the records above:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "question": request.question,
        "answer": response.content,
        "retrieved_records": retrieved_docs
    }


# --- /ingest endpoint ---
@app.post("/ingest")
def ingest_violation(request: IngestRequest):
    # Build the same descriptive sentence format used in ingest.py
    speed_info = f" at {request.speed_kmph} km/h" if request.speed_kmph else ""
    document = (
        f"Vehicle {request.plate} was caught {request.violation}{speed_info} "
        f"by {request.camera} on {request.timestamp}. "
        f"Fine: Rs.{request.fine_amount}. Status: {request.status}."
    )

    # Generate a new unique ID based on current collection size
    new_id = f"violation_{collection.count()}"

    collection.add(
        documents=[document],
        ids=[new_id],
        metadatas=[{
            "plate": request.plate,
            "camera": request.camera,
            "violation": request.violation,
            "timestamp": request.timestamp,
            "fine": str(request.fine_amount),
            "status": request.status,
        }]
    )

    return {
        "message": "Violation record added successfully",
        "id": new_id,
        "document": document
    }


# --- Health check endpoint (good practice for any real API) ---
@app.get("/")
def health_check():
    return {
        "status": "running",
        "total_records": collection.count()
    }
    