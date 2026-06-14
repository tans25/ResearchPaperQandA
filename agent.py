from langchain.agents import create_agent 
from tools.graph_tool import query_knowledge_graph
from tools.retriever_tool import get_retriever_tool
from ingestion.config import llm


def create_rag_agent(ingested_files):
    retriever_tool = get_retriever_tool()
    paper_list = "\n".join(f"- {f}" for f in ingested_files)

    agent = create_agent(
        model=llm, 
        tools=[retriever_tool, query_knowledge_graph],
        system_prompt="You are a research assistant for question-answering tasks based on the following research papers:\n"
            f"{paper_list}\n\n"
            "You have two tools:\n"
            "1. query_knowledge_graph - for relationship questions across papers\n"
            "2. paper_text_search - for detailed text questions\n\n"
            "When asked about multiple papers, use both tools to gather "
            "information from all relevant papers before answering."
            "If you don't know the answer, say that you don't know. "
    )
    return agent 
