# Traffic Violation RAG Assistant

A production-style AI backend that lets users query traffic violation records 
using natural language. Built with LangChain, ChromaDB, Groq LLaMA 3, FastAPI, 
and Docker.

## What it does

Instead of writing database queries, you ask plain English questions:

- *"Which vehicles were caught speeding above 100 km/h?"*
- *"Show all red-light violations from Silk Board camera"*
- *"Are there any contested violations with fines above Rs.1500?"*

The system retrieves the most relevant records from a vector database using 
semantic search, then passes them to LLaMA 3 to generate a grounded, 
accurate answer — without hallucinating data that isn't there.

## Architecture

User Question
│
▼
Sentence Transformer (all-MiniLM-L6-v2)
│  converts question to vector
▼
ChromaDB Vector Store
│  finds top-4 semantically similar violation records
▼
Prompt Builder
│  question + retrieved records → single prompt
▼
Groq LLaMA 3.3 70B
│  reads context, generates grounded answer
▼
FastAPI Response (JSON)

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 (runs locally) |
| Vector Database | ChromaDB (persistent) |
| LLM | Groq LLaMA 3.3 70B |
| LLM Framework | LangChain |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |

## Project Structure

traffic-rag/
├── data/
│   ├── generate_data.py     # generates 250 synthetic violation records
│   └── violations.json      # generated dataset
├── rag/
│   ├── ingest.py            # embeds records into ChromaDB
│   └── query.py             # RAG chain (retrieval + LLM)
├── api/
│   └── main.py              # FastAPI service (/query, /ingest, /)
├── chroma_db/               # persistent vector store (auto-created)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                     # GROQ_API_KEY (not committed)

## Getting Started

### Option 1 — Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/Anirudh735/traffic-rag
cd traffic-rag

# 2. Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Generate data and build ChromaDB
py data/generate_data.py
py rag/ingest.py

# 4. Start the API
docker compose up --build
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

### Option 2 — Local

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

py data/generate_data.py
py rag/ingest.py
py -m uvicorn api.main:app --reload
```

## API Endpoints

### `POST /query`
Query the violation database in natural language.

**Request:**
```json
{
  "question": "Which vehicles were caught speeding above 100 km/h?",
  "n_results": 4
}
```

**Response:**
```json
{
  "question": "Which vehicles were caught speeding above 100 km/h?",
  "answer": "Based on the records, 3 vehicles were caught speeding above 100 km/h: KA-09-M-8774 at 103 km/h, KA-14-R-7775 at 107 km/h, and KA-66-A-4720 at 109 km/h.",
  "retrieved_records": ["..."]
}
```

### `POST /ingest`
Add a new violation record to the vector store in real time.

**Request:**
```json
{
  "plate": "KA-01-AB-1234",
  "camera": "CAM-001 MG Road",
  "violation": "red light jumping",
  "timestamp": "2026-07-02 09:30",
  "fine_amount": 1000,
  "status": "pending",
  "speed_kmph": null
}
```

### `GET /`
Health check — returns status and total record count.

## Performance

- Retrieval: top-4 semantic search across 250 records in under 100ms  
- End-to-end response: under 2 seconds (Groq low-latency inference)  
- Hallucination guardrail: LLM explicitly instructed to refuse 
  answers not supported by retrieved context

## Known Limitations

- RAG is optimised for semantic search, not aggregation queries 
  (e.g. "count all speeding violations" may be inaccurate as it only 
  sees the top-4 retrieved records, not the full dataset)
- Location matching is semantic, not geographic — "MG Road" and 
  "Whitefield" are treated as text strings, not coordinates

## Author

**Anirudh P** — [LinkedIn](https://www.linkedin.com/in/anirudhp21) 
| [Portfolio](https://portfolio-topaz-pi-5nsvj2h6xr.vercel.app/)