from dotenv import load_dotenv
load_dotenv()

import sys
from ingestion.loader import load_pdfs
from ingestion.graph_builder import build_graph
from ingestion.vector_store import build_vector_store
from agent import create_rag_agent


def ingest(pdf_dir):
    try:
        print("here", pdf_dir)
        splits = load_pdfs(pdf_dir)
        build_graph(splits)
        build_vector_store(splits)
        print("Ingestion complete")
        return True
    except Exception as e:
        print(f"Something went wrong: {e}")
        return False 


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