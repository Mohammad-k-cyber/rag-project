from langchain_ibm import ChatWatsonx
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 512}
)

# Define custom tools
@tool
def calculate(expression: str) -> str:
    """Calculate a math expression"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

@tool
def search_knowledge_base(query: str) -> str:
    """Search knowledge base for information"""
    knowledge = {
        "RAG": "Retrieval Augmented Generation combines retrieval with generation",
        "LLM": "Large Language Model trained on massive text data",
        "Agent": "AI system that can reason and use tools to solve tasks"
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return "No information found"

# Create tools list
tools = [calculate, search_knowledge_base]

# Create agent
agent_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Use the tools available to answer questions.\n\n{input}"
)

agent = create_react_agent(model, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("=== AGENT EXAMPLES ===\n")

# Test 1: Math calculation
print("Test 1: Math problem")
result = agent_executor.invoke({"input": "What is 25 * 4?"})
print(f"Answer: {result['output']}\n")

# Test 2: Knowledge lookup
print("Test 2: Knowledge lookup")
result = agent_executor.invoke({"input": "Tell me about RAG"})
print(f"Answer: {result['output']}")