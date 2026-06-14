from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_mistralai import ChatMistralAI
from langchain_neo4j import Neo4jGraph
import os

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-72B-Instruct",
#     task="text-generation",
#     max_new_tokens=512, 
#     temperature=0.7
# )
# llm = ChatHuggingFace(llm=llm_endpoint)
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7, 
    max_tokens=512,
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)
