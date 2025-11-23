from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

# In-Context Learning: Teaching the model through examples
print("=== IN-CONTEXT LEARNING ===\n")

# Without examples (model guesses)
print("WITHOUT EXAMPLES:")
prompt1 = "Translate 'Hello' to French"
response = model.invoke(prompt1)
print(f"Response: {response.content}\n")

# With examples (model learns the pattern)
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
response = model.invoke(formatted)
print(f"Response: {response.content}\n")

# Complex in-context learning
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
response = model.invoke(formatted)
print(f"Response: {response.content}")