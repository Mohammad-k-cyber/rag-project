from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate

# Initialize model
model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={
        "max_new_tokens": 256,
        "temperature": 0.5,
    }
)

# 1. Basic Prompting (No examples)
print("=== ZERO-SHOT PROMPTING ===")
basic_prompt = "What is Retrieval Augmented Generation?"
response = model.invoke(basic_prompt)
print(f"Response: {response.content}\n")

# 2. One-Shot Prompting (One example)
print("=== ONE-SHOT PROMPTING ===")
one_shot = """
Example: What is Machine Learning? 
Answer: Machine Learning is a subset of AI...

Now answer: What is Deep Learning?
"""
response = model.invoke(one_shot)
print(f"Response: {response.content}\n")

# 3. Few-Shot Prompting (Multiple examples)
print("=== FEW-SHOT PROMPTING ===")
few_shot = """
Examples:
Q: What is NLP? A: Natural Language Processing is a field of AI...
Q: What is Computer Vision? A: Computer Vision enables machines to see...

Q: What is Reinforcement Learning?
"""
response = model.invoke(few_shot)
print(f"Response: {response.content}\n")

# 4. Chain-of-Thought (Step by step reasoning)
print("=== CHAIN-OF-THOUGHT ===")
cot_prompt = """
A store has 20 apples. They sell 7 apples and receive 5 more.
How many apples are there now?

Think step by step:
"""
response = model.invoke(cot_prompt)
print(f"Response: {response.content}")