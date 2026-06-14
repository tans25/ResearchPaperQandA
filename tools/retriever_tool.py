from langchain_core.tools.retriever import create_retriever_tool
from ingestion.vector_store import load_vector_store

def get_retriever_tool():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="research_paper_research",
        description="Search the research paper for relevant information. "
        "Use this tool when you need to answer questions about the paper.",
    ) 
    return retriever_tool