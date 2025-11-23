from flask import Flask, request, jsonify
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.prompts import PromptTemplate

app = Flask(__name__)

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

# Route 1: Simple question-answer
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    
    template = "Answer this question: {question}"
    prompt = PromptTemplate.from_template(template)
    formatted = prompt.format(question=question)
    
    response = model.generate(formatted)
    return jsonify({"answer": response['results'][0]['generated_text']})

# Route 2: Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True)