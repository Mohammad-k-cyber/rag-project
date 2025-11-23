from ollama import Client
from langchain.prompts import PromptTemplate

client = Client(host='http://localhost:11434')
MODEL = 'mistral'

print("=== SIMPLE TEMPLATE ===")
template = "Tell me a {adjective} joke about {topic}"
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(adjective="funny", topic="programming")
response = client.generate(model=MODEL, prompt=formatted, stream=False)
print(f"Joke: {response['response']}\n")

print("=== MULTI-VARIABLE TEMPLATE ===")
template = """
You are a {role} expert.
Question: {question}
Answer in 2-3 sentences:
"""
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(
    role="Python programmer",
    question="How do decorators work?"
)
response = client.generate(model=MODEL, prompt=formatted, stream=False)
print(f"Answer: {response['response']}\n")

print("=== SENTIMENT CLASSIFICATION ===")
template = """
Classify sentiment (Positive/Negative/Neutral):
Review: "{review}"
Sentiment:
"""
prompt = PromptTemplate.from_template(template)

formatted = prompt.format(review="Pretty good product, but expensive")
response = client.generate(model=MODEL, prompt=formatted, stream=False)
print(f"Sentiment: {response['response']}")