from langchain.tools import tool 
from langchain_neo4j import GraphCypherQAChain
from ingestion.config import llm, graph 

cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

@tool
def query_knowledge_graph(question):
    """Query the research paper knowledge graph for relationships
    between papers, authors, methods, datasets, and results.
    Use this for questions like 'which methods were used on dataset X'
    or 'what papers did author Y write'."""
    result = cypher_chain.invoke({"query": question})
    return result["result"]