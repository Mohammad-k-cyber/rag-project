from flask import Flask, request, jsonify
from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

app = Flask(__name__)

model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

class AIResponse(BaseModel):
    answer: str = Field(description="The AI's response")
    confidence: str = Field(description="Confidence level")

json_parser = JsonOutputParser(pydantic_object=AIResponse)

# Route 1: Simple question-answer
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    
    template = "Answer this question: {question}"
    prompt = PromptTemplate.from_template(template)
    formatted = prompt.format(question=question)
    
    response = model.invoke(formatted)
    return jsonify({"answer": response.content})

# Route 2: Structured JSON response
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text')
    
    template = """
    Analyze this text and respond in JSON:
    {format_instructions}
    
    Text: {text}
    """
    prompt = PromptTemplate.from_template(template)
    formatted = prompt.format(
        text=text,
        format_instructions=json_parser.get_format_instructions()
    )
    
    response = model.invoke(formatted)
    return jsonify({"analysis": response.content})

# Route 3: Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True)