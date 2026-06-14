# ResearchPaperQandA
 
A Graph RAG application for analyzing research papers. Upload PDFs, automatically build a knowledge graph of entities and relationships, and query across papers using a hybrid retrieval agent that combines graph traversal with vector search.
 
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Neo4j](https://img.shields.io/badge/Neo4j-latest-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
 
## How It Works
 
The application has two phases:
 
**Ingestion** — uploaded PDFs are loaded, references and appendix sections are stripped, and the remaining content is split into chunks. Each chunk is processed in parallel through two pipelines: an LLM extracts entities (papers, authors, methods, datasets, tasks, metrics) and relationships into a Neo4j knowledge graph, while embeddings are stored in a Chroma vector database.
 
**Querying** — an agent powered by Mistral decides which retrieval tool to use per question. Relational questions ("which methods were evaluated on dataset X across all papers?") route to the knowledge graph via Cypher queries. Detail questions ("how does the paper define contrastive loss?") route to vector search over the raw text. The agent can use both tools for complex questions that need structural and textual context.
 
## Project Structure
 
```
graph_rag/
├── app.py                  # Streamlit frontend
├── main.py                 # Ingestion and query entry points
├── config.py               # LLM, embeddings, and Neo4j setup
├── agent.py                # Agent creation with graph + vector tools
├── ingestion/
│   ├── __init__.py
│   ├── loader.py           # PDF loading, back matter removal, chunking
│   ├── graph_builder.py    # Entity/relationship extraction → Neo4j
│   └── vector_store.py     # Chunk embeddings → Chroma
├── tools/
│   ├── __init__.py
│   ├── graph_tool.py       # Knowledge graph query tool
│   └── retriever_tool.py   # Vector search tool
├── docker-compose.yml      # Neo4j container
├── .env                    # API keys (not committed)
├── requirements.txt
└── data/
    └── papers/             # Uploaded PDFs (not committed)
```
 
## Tech Stack
 
| Component | Technology |
|---|---|
| LLM | Mistral (via Mistral API) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Knowledge Graph | Neo4j |
| Vector Store | Chroma |
| Orchestration | LangChain + LangGraph |
| Frontend | Streamlit |
| Containerization | Docker |
 
## Setup
 
### Prerequisites
 
- Python 3.10+
- Docker and Docker Compose
- Mistral API key
### 1. Clone the repository
 
```bash
git clone https://github.com/yourusername/research-comp.git
cd research-comp
```
 
### 2. Create a virtual environment
 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Configure environment variables
 
```bash
cp .env.example .env
```
 
Edit `.env` and add your API key:
 
```
MISTRAL_API_KEY=your-mistral-api-key
```
 
### 5. Start Neo4j
 
```bash
docker compose up -d
```
 
This starts a Neo4j instance with APOC plugin enabled. The browser UI is available at `http://localhost:7474`.
 
### 6. Run the application
 
```bash
streamlit run app.py
```
 
## Usage
 
1. Upload one or more research paper PDFs using the file uploader
2. Click the **Ingest** button to process the papers
3. Once ingestion completes, the chat interface appears
4. Ask questions about your papers — the agent automatically chooses between graph and vector retrieval
### Example Questions
 
**Relational (uses knowledge graph):**
- "Which methods are used across both papers?"
- "What datasets were evaluated in this paper?"
- "Which authors are connected to method X?"
**Detail (uses vector search):**
- "How does the paper define contrastive loss?"
- "What were the main findings?"
- "Explain the methodology used in the study"
**Cross-paper (uses both):**
- "Compare the approaches used in paper A vs paper B"
- "What methods from paper A were also evaluated in paper B?"
## Architecture
 
The ingestion pipeline forks each chunk into two parallel paths:
 
- **Graph path** — an LLM extracts structured entities and relationships using a predefined schema, which are loaded into Neo4j as nodes and edges. Shared entities across papers (e.g., the same method appearing in multiple papers) naturally merge into single nodes, enabling cross-paper queries.
- **Vector path** — chunks are embedded with all-MiniLM-L6-v2 and stored in Chroma for semantic similarity search.
At query time, the agent has access to both retrieval tools and autonomously decides which to invoke based on the question type. The `GraphCypherQAChain` converts natural language to Cypher queries for graph traversal, while the vector retriever returns semantically similar chunks for detail-oriented questions.
 
## Knowledge Graph Schema
 
**Entity types:** Paper, Author, Method, Dataset, Task, Metric
 
**Relationship types:** AUTHORED_BY, USES_METHOD, EVALUATED_ON, ACHIEVES_RESULT, CITES, EXTENDS
 
## Useful Commands
 
```bash
# Start Neo4j
docker compose up -d
 
# Stop Neo4j (data persists)
docker compose down
 
# Stop Neo4j and delete all data
docker compose down -v
 
# Clear Neo4j graph
docker exec graph_rag_neo4j cypher-shell -u neo4j -p your_password "MATCH (n) DETACH DELETE n"
 
# Clear Chroma vector store
rm -rf ./chroma_db
 
# View Neo4j browser
open http://localhost:7474
```
 
## Configuration
 
Key parameters can be adjusted in the respective files:
 
- **Chunk size/overlap** — `ingestion/loader.py` (default: 1500 chars, 300 overlap)
- **LLM model** — `config.py` (default: mistral-small-latest)
- **Embedding model** — `config.py` (default: all-MiniLM-L6-v2)
- **Extraction schema** — `ingestion/graph_builder.py` (entity and relationship types)
- **Neo4j credentials** — `docker-compose.yml` and `config.py`
## Known Limitations
 
- Entity deduplication is name-based — "GPT-4" and "GPT4" create separate nodes
- The `source_file` property on nodes tracks only the last paper processed, so shared entities may lose provenance when a paper is deleted
- HuggingFace free-tier models may produce inconsistent extraction results; Mistral API is recommended
- Back matter removal uses keyword matching, which may occasionally cut content on the same page as the references heading
