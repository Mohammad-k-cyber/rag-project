from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

# Define output structure
class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="Positive, Negative, or Neutral")
    score: int = Field(description="Score from 1-10")
    summary: str = Field(description="One sentence summary")

json_parser = JsonOutputParser(pydantic_object=ReviewAnalysis)

# Create chain with JSON output
template = """
Analyze this review and respond in JSON format:
{format_instructions}

Review: {review}
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | model | json_parser

result = chain.invoke({
    "review": "The product works great but arrived late",
    "format_instructions": json_parser.get_format_instructions()
})

print("Parsed JSON Output:")
print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']}")
print(f"Summary: {result['summary']}")