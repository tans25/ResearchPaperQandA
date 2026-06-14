from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field 
from ingestion.config import llm, graph 
import json 
import os 

class Entity(BaseModel):
    name: str = Field(description="The entity name")
    type: str = Field(description="Either a Paper, Author, Method, Dataset, Task, Metric")

class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type like USES_METHOD, EVALUTATED_ON")

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


# structured_llm = llm.with_structured_output(ExtractionResult)
# extraction_llm_endpoint = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.1,  # lower temp for more consistent JSON
# )
# extraction_llm = ChatHuggingFace(llm=extraction_llm_endpoint)
extraction_prompt = """Extract all entities and relationships from the following research paper text. Respond ONLY with valid JSON, no other text. 
Entity types: Paper, Author, Method, Dataset, Task, Metric
Relationship types: AUTHORED_BY, USES_METHOD, EVALUATED_ON, ACHIEVES_RESULT, CITES, EXTENDS
Return format:
{{
    "entities": [
        {{"name": "entity name", "type": "entity type"}}
    ],
    "relationships": [
        {{"source": "source name", "target": "target name", "type": "relationship type"}}
    ]
}}
Text:
{chunk_text}
"""
def parse_extraction(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def build_graph(splits):
    for chunk in splits:
        source_file = os.path.basename(chunk.metadata.get("source", "unknown"))
        result = llm.invoke(extraction_prompt.format(chunk_text=chunk.page_content))
        try:
            result = parse_extraction(result.content)
            print("Extraction result: ", result)
        except json.JSONDecodeError:
            continue 

        for entity in result.get("entities", []):
            graph.query(
                "MERGE (e:{type} {{name: $name}}) SET e.source_file = $file".format(type=entity["type"]),
                {"name": entity["name"], "file": source_file}
            )
        
        for rel in result.get("relationships", []):
            graph.query(
                """
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[:{type}]->(b)
                """.format(type=rel["type"]),
                {"source": rel["source"], "target": rel["target"]}
            )
    print("Graph built successfully.")