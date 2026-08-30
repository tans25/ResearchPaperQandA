from dotenv import load_dotenv
load_dotenv()

import sys
from ingestion.loader import load_pdfs
from ingestion.graph_builder import build_graph
from ingestion.vector_store import build_vector_store
from agent import create_rag_agent


def ingest(uploaded_files):
    try:
        splits = load_pdfs(uploaded_files)
        build_graph(splits)
        build_vector_store(splits)
        print("Ingestion complete")
        return {"success": True, "message": "Ingestion complete"}
    except Exception as e:
        print(f"Something went wrong: {e}")
        return {"success": False, "message": str(e)}


def query():
    agent = create_rag_agent()
    while True:
        q = input("\nAsk a question (or 'quit' to exit): ")
        if q.lower() == "quit":
            break 
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print("\n", result["messages"][-1].content)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        query()     