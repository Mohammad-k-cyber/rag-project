from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

# Template 1: Simple template
print("=== SIMPLE TEMPLATE ===")
template = "Tell me a {adjective} joke about {topic}"
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(adjective="funny", topic="programming")
print(f"Prompt: {formatted}")
response = model.invoke(formatted)
print(f"Response: {response.content}\n")

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
response = model.invoke(formatted)
print(f"Response: {response.content}\n")

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
response = model.invoke(formatted)
print(f"Response: {response.content}")