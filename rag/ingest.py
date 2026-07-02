import json
import os
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Step 1: Load your 250 violation records from the JSON file ---
with open("data/violations.json", "r") as f:
    violations = json.load(f)

print(f"Loaded {len(violations)} violations from JSON")


# --- Step 2: Convert each record into a human-readable sentence ---
def record_to_text(v):
    speed_info = f" at {v['speed_kmph']} km/h" if v.get("speed_kmph") else ""
    return (
        f"Vehicle {v['plate']} was caught {v['violation']}{speed_info} "
        f"by {v['camera']} on {v['timestamp']}. "
        f"Fine: Rs.{v['fine_amount']}. Status: {v['status']}."
    )

documents = [record_to_text(v) for v in violations]
ids = [f"violation_{i}" for i in range(len(violations))]
metadatas = [
    {
        "plate":      v["plate"],
        "camera":     v["camera"],
        "violation":  v["violation"],
        "timestamp":  v["timestamp"],
        "fine":       str(v["fine_amount"]),
        "status":     v["status"],
    }
    for v in violations
]

print("Sample document text:")
print(documents[0])
print()


# --- Step 3: Set up ChromaDB with a local embedding model ---
client = PersistentClient(path="./chroma_db")

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="traffic_violations",
    embedding_function=embedding_fn
)

print("Embedding and storing records in ChromaDB...")
print("(First run downloads the model ~80MB — takes 1-2 min)")


# --- Step 4: Add everything to ChromaDB ---
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"\nDone! {collection.count()} records stored in ChromaDB")
print("ChromaDB is saved to ./chroma_db folder — no need to re-run this")


# --- Step 5: Quick sanity test ---
print("\nRunning a quick test query...")
results = collection.query(
    query_texts=["red light violation near MG Road"],
    n_results=3
)

print("Top 3 results for 'red light violation near MG Road':")
for i, doc in enumerate(results["documents"][0]):
    print(f"\n  Result {i+1}: {doc}")