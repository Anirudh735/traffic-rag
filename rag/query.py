import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# --- Step 1: Load environment variables (your Groq API key) ---
load_dotenv()  # reads the .env file and loads GROQ_API_KEY into the environment

# --- Step 2: Reconnect to your existing ChromaDB ---
# We're NOT re-embedding — just opening the database we already built
client = PersistentClient(path="./chroma_db")

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="traffic_violations",
    embedding_function=embedding_fn
)

# --- Step 3: Set up the Groq LLM connection ---
# temperature=0 makes answers more factual/consistent, less "creative"
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# --- Step 4: The core RAG function ---
def ask_question(question, n_results=4):
    # Retrieve the most relevant violation records
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    retrieved_docs = results["documents"][0]

    # Combine retrieved records into one context block
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    # Build the prompt: system instructions + the actual question+context
    system_prompt = (
        "You are a traffic violation assistant. Answer the user's question "
        "using ONLY the violation records provided below. If the records don't "
        "contain enough information to answer, say so clearly. Be concise and factual."
    )

    user_prompt = f"""Violation records:
{context}

Question: {question}

Answer based only on the records above:"""

    # Send to Groq LLaMA 3
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content, retrieved_docs


# --- Step 5: Test it with a few questions ---
if __name__ == "__main__":
    test_questions = [
        "How many red light violations happened at Whitefield Main Road?",
        "What's the fine amount for speeding violations?",
        "Are there any contested violations from camera 6?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('='*60)
        answer, docs = ask_question(q)
        print(f"\nAnswer: {answer}")