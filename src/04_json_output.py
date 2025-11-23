from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.prompts import PromptTemplate
import json

params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}

model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    params=params,
    credentials={"url": "https://us-south.ml.cloud.ibm.com"},
    project_id="skills-network"
)

# JSON Output Example
print("=== JSON OUTPUT ===")
template = """
Analyze this review and respond in JSON format with these fields:
- sentiment (Positive/Negative/Neutral)
- score (1-10)
- summary (one sentence)

Review: {review}

JSON Response:
"""

prompt = PromptTemplate.from_template(template)
formatted = prompt.format(review="The product works great but arrived late")

response = model.generate(formatted)
result_text = response['results'][0]['generated_text']

print(f"Response: {result_text}")

# Try to parse JSON if model returns it
try:
    json_str = result_text[result_text.find('{'):result_text.rfind('}')+1]
    parsed = json.loads(json_str)
    print(f"\nParsed JSON: {json.dumps(parsed, indent=2)}")
except:
    print("(Could not parse JSON from response)")