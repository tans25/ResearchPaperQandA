from langchain_community.vectorstores import Chroma
from ingestion.config import embeddings

def build_vector_store(splits):
    vector_store = Chroma.from_documents(
        embedding=embeddings,
        documents=splits, 
        persist_directory="./chroma_db"
    )
    print("Vector store built successfully")
    return vector_store 

def load_vector_store():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

def delete_paper_from_vector_store(filename):
    vector_store = load_vector_store()
    results = vector_store.get()
    ids_to_delete = [
        id_ for id_, meta in zip(results["ids"], results["metadatas"])
        if meta.get("source", "").endswith(filename)
    ]
    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} chunks for {filename}")
        return True