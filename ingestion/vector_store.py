from langchain_community.vectorstores import Chroma
from ingestion.config import embeddings
import streamlit as st 
from langchain_neo4j import Neo4jVector
from ingestion.config import graph

def get_neo4j_creds():
    return {
        "url":st.secrets["NEO4J_URL"],
        "username": st.secrets["NEO4J_USER"],
        "password":st.secrets["NEO4J_PASSWORD"],
        "database": st.secrets["NEO4J_DATABASE"]
    }

def build_vector_store(splits):
    creds = get_neo4j_creds()
    vector_store = Neo4jVector.from_documents(
        embedding=embeddings,
        documents=splits, 
        url = creds["url"],
        username = creds["username"],
        password = creds["password"],
    )
    print("Vector store built successfully")
    return vector_store 

def load_vector_store():
    creds = get_neo4j_creds()
    return Neo4jVector.from_existing_index(
        embedding = embeddings,
        index_name="vector",
        url=creds["url"],
        username=creds["username"],
        password=creds["password"],
    )

def delete_paper_from_vector_store(filename):
    # vector_store = load_vector_store()
    # results = vector_store.get()
    # ids_to_delete = [
    #     id_ for id_, meta in zip(results["ids"], results["metadatas"])
    #     if meta.get("source", "").endswith(filename)
    # ]
    # if ids_to_delete:
    #     vector_store.delete(ids=ids_to_delete)
    #     print(f"Deleted {len(ids_to_delete)} chunks for {filename}")
    #     return True
    graph.query(
        """
        MATCH (n:Chunk)
        WHERE n.source ENDS WITH $filename
        AND n.embedding IS NOT NULL
        DETACH DELETE n
        """,
        {"filename": filename},
    )
    print(f"Deleted vector chunks for {filename}")
    return True 