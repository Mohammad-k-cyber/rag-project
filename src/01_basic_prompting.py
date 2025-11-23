from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Initialize model using direct API (more reliable)
params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}

model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    params=params,
    credentials={
        "url": "https://us-south.ml.cloud.ibm.com",
    },
    project_id="skills-network"
)

print("=== ZERO-SHOT PROMPTING ===")
basic_prompt = "What is Retrieval Augmented Generation?"
response = model.generate(basic_prompt)
print(f"Response: {response['results'][0]['generated_text']}\n")

print("=== ONE-SHOT PROMPTING ===")
one_shot = """
Example: What is Machine Learning? 
Answer: Machine Learning is a subset of AI...

Now answer: What is Deep Learning?
"""
response = model.generate(one_shot)
print(f"Response: {response['results'][0]['generated_text']}\n")

print("=== FEW-SHOT PROMPTING ===")
few_shot = """
Examples:
Q: What is NLP? A: Natural Language Processing is a field of AI...
Q: What is Computer Vision? A: Computer Vision enables machines to see...

Q: What is Reinforcement Learning?
"""
response = model.generate(few_shot)
print(f"Response: {response['results'][0]['generated_text']}\n")

print("=== CHAIN-OF-THOUGHT ===")
cot_prompt = """
A store has 20 apples. They sell 7 apples and receive 5 more.
How many apples are there now?

Think step by step:
"""
response = model.generate(cot_prompt)
print(f"Response: {response['results'][0]['generated_text']}")