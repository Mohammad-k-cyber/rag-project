from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

# Chain 1: Simple chain (template → LLM → output)
print("=== SIMPLE CHAIN ===")
template = "Write a short poem about {topic}"
prompt = PromptTemplate.from_template(template)

chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": "technology"})
print(f"Result: {result}\n")

# Chain 2: Multi-step chain with function
print("=== CHAIN WITH FUNCTION ===")
def format_input(inputs):
    return inputs["text"].upper()

chain = (
    RunnableLambda(format_input)
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke({"text": "artificial intelligence"})
print(f"Result: {result}\n")

# Chain 3: Sequential processing
print("=== SEQUENTIAL CHAIN ===")
# Step 1: Generate topic
step1_prompt = PromptTemplate.from_template("Generate a random tech topic in one word")
step1_chain = step1_prompt | model | StrOutputParser()

# Step 2: Write about that topic
step2_prompt = PromptTemplate.from_template("Write about: {topic}")
step2_chain = step2_prompt | model | StrOutputParser()

# Combine steps
topic = step1_chain.invoke({})
print(f"Generated topic: {topic}")
result = step2_chain.invoke({"topic": topic})
print(f"Essay: {result}")