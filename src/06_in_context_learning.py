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

print("=== IN-CONTEXT LEARNING ===\n")

# Without examples (model guesses)
print("WITHOUT EXAMPLES:")
prompt1 = "Translate 'Hello' to French"
response = model.generate(prompt1)
print(f"Response: {response['results'][0]['generated_text']}\n")

# With examples (few-shot learning)
print("WITH EXAMPLES (Few-shot):")
template = """
Learn from these examples:
English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French: Merci

Now translate: {word}
"""
prompt = PromptTemplate.from_template(template)
formatted = prompt.format(word="Good morning")
response = model.generate(formatted)
print(f"Response: {response['results'][0]['generated_text']}\n")

# Complex pattern learning
print("COMPLEX PATTERN LEARNING:")
template = """
Learn this pattern - convert names to professional titles:
John Developer → Senior Software Engineer
Sarah Manager → Director of Operations
Mike Designer → Lead UX Designer

Now convert: {name}
"""
prompt = PromptTemplate.from_template(template)
formatted = prompt.format(name="Alice Researcher")
response = model.generate(formatted)
print(f"Response: {response['results'][0]['generated_text']}")
