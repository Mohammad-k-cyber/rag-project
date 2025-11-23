"""
Complete workflow combining everything:
Prompts → Templates → Chains → RAG → Output Parsing → Web API
"""

from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

class ContentAnalysis(BaseModel):
    title: str = Field(description="Content title")
    summary: str = Field(description="Brief summary")
    key_points: list = Field(description="Key points")
    difficulty: str = Field(description="Difficulty level")

json_parser = JsonOutputParser(pydantic_object=ContentAnalysis)

def analyze_content(content: str) -> dict:
    """Complete workflow to analyze content"""
    
    print("=== COMPLETE WORKFLOW ===\n")
    
    # Step 1: Template with structure
    print("Step 1: Creating prompt template...")
    template = """
    Analyze this content and provide structured output:
    {format_instructions}
    
    Content: {content}
    """
    prompt = PromptTemplate.from_template(template)
    
    # Step 2: Format the prompt
    print("Step 2: Formatting prompt...")
    formatted = prompt.format(
        content=content,
        format_instructions=json_parser.get_format_instructions()
    )
    
    # Step 3: Call LLM
    print("Step 3: Calling LLM...")
    response = model.invoke(formatted)
    
    # Step 4: Parse JSON output
    print("Step 4: Parsing output...")
    try:
        result = json.loads(response.content)
        return result
    except:
        return {"error": "Failed to parse response"}

# Test the workflow
sample_content = """
Machine Learning is a subset of Artificial Intelligence that focuses on 
enabling computers to learn from data without being explicitly programmed. 
It uses algorithms and statistical models to identify patterns and make predictions.
"""

result = analyze_content(sample_content)
print("\nFinal Result:")
print(json.dumps(result, indent=2))