from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 200,
}

# Initialize different models
granite = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    params=params,
    credentials={"url": "https://us-south.ml.cloud.ibm.com"},
    project_id="skills-network"
)

llama = ModelInference(
    model_id="meta-llama/llama-3-2-1b-instruct",
    params=params,
    credentials={"url": "https://us-south.ml.cloud.ibm.com"},
    project_id="skills-network"
)

# Test prompt
prompt = "Explain what RAG is in one sentence"

print("=== MODEL COMPARISON ===\n")

print("GRANITE Response:")
response = granite.generate(prompt)
print(f"{response['results'][0]['generated_text']}\n")

print("LLAMA3 Response:")
response = llama.generate(prompt)
print(f"{response['results'][0]['generated_text']}")
