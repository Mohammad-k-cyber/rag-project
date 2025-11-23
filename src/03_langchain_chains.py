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

# Chain 1: Simple sequential processing
print("=== CHAIN EXAMPLE 1 ===")
template = "Write a short poem about {topic}"
prompt = PromptTemplate.from_template(template)
formatted = prompt.format(topic="technology")
response = model.generate(formatted)
print(f"Poem: {response['results'][0]['generated_text']}\n")

# Chain 2: Multi-step processing
print("=== CHAIN EXAMPLE 2: Multi-step ===")
# Step 1: Generate topic
step1_prompt = "Generate a random tech topic in one word"
response1 = model.generate(step1_prompt)
topic = response1['results'][0]['generated_text'].strip()

print(f"Generated topic: {topic}")

# Step 2: Write about that topic
step2_template = "Write a brief paragraph about: {topic}"
step2_prompt = PromptTemplate.from_template(step2_template)
formatted2 = step2_prompt.format(topic=topic)
response2 = model.generate(formatted2)

print(f"Essay: {response2['results'][0]['generated_text']}")
