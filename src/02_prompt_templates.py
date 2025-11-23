from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.prompts import PromptTemplate

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

# Template 1: Simple template
print("=== SIMPLE TEMPLATE ===")
template = "Tell me a {adjective} joke about {topic}"
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(adjective="funny", topic="programming")
print(f"Prompt: {formatted}")
response = model.generate(formatted)
print(f"Response: {response['results'][0]['generated_text']}\n")

# Template 2: Multi-variable template
print("=== MULTI-VARIABLE TEMPLATE ===")
template = """
You are a {role} expert.
Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(
    role="Python",
    question="How do decorators work?",
    context="In Python, decorators modify function behavior"
)
response = model.generate(formatted)
print(f"Response: {response['results'][0]['generated_text']}\n")

# Template 3: Few-shot template
print("=== FEW-SHOT TEMPLATE ===")
template = """
Classify the sentiment of these reviews:

Examples:
Review: "This product is amazing!" → Sentiment: Positive
Review: "Terrible quality" → Sentiment: Negative

Review: "{user_review}"
Sentiment:
"""
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(user_review="Pretty good, but expensive")
response = model.generate(formatted)
print(f"Response: {response['results'][0]['generated_text']}")