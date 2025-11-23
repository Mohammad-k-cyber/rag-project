from langchain_ibm import ChatWatsonx

# Initialize different models
granite = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

llama = ChatWatsonx(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

mixtral = ChatWatsonx(
    model_id="mistralai/mistral-large",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

# Test prompt
prompt = "Explain what RAG is in one sentence"

print("=== MODEL COMPARISON ===\n")

print("GRANITE Response:")
response = granite.invoke(prompt)
print(f"{response.content}\n")

print("LLAMA3 Response:")
response = llama.invoke(prompt)
print(f"{response.content}\n")

print("MIXTRAL Response:")
response = mixtral.invoke(prompt)
print(f"{response.content}")